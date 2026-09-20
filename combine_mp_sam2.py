# ====== Step 0: Import Libraries ====== #

# System
import os
import sys
import keyboard
import argparse
from tqdm import tqdm
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Computation and data handling
from matplotlib import patches
import numpy as np
import torch

# Visualization and image processing
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pandas as pd

# SAM2
from sam2.build_sam import build_sam2_video_predictor           #type:ignore   
from sam2.sam2_video_predictor import SAM2VideoPredictor        #type:ignore   
from sam2_resources import sam2_utils

# MediaPipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe_resources.mp_utils as mp_utils

# Processing SAM2 Masks and refineing Mediapipe landmarks
import mask_processing.mask_processor as mask_processor
import refine_landmarks.refinement_utils as refinement_utils
import refine_landmarks.data_manager as data_manager

UPPER_LIP_OBJ_ID = 1
LOWER_LIP_OBJ_ID = 2
TONGUE_OBJ_ID = 3

# ====== Step 1: Setup the environment ====== #

# Parse the command line arguments
parser = argparse.ArgumentParser(description='Refine MediaPipe mouth landmarks using SAM2 segmentation masks')
parser.add_argument('--v', '--video_filename', type=str, default='tc6.mp4', help='.mp4 filename of test video in mocapvids directory')
parser.add_argument('--debug', '--debug_stage', type=int, default=0, help='0: doe not use saved stage, 1: stage 1 saved state, 2: stage 2 saved state')
parser.add_argument('--npp', '--num_prompts_per_part', type=int, default=4, help='Number of prompts to add for each mouth part (integer)')
parser.add_argument('--fw', '--finetuned_weights', type=str, required=True, help="Name of weights file in finetuned_weights folder")
args = parser.parse_args() 

# Get the row index of the results for these finetuned weights
ft_res_df = pd.read_csv("finetuning_sam2/finetuning_results.csv")
weights_path = os.path.join("finetuning_sam2", "finetuned_weights", args.fw)

# Find the target finetuned weights and populate the base model and dataset variables
if weights_path not in ft_res_df['finetuned_weights'].values:
    raise ValueError(f"Finetuned weights '{args.fw}' not found in finetuning_results.csv")
else:
    fw_idx = ft_res_df.index[ft_res_df['finetuned_weights'] == weights_path][0]
    base_model = ft_res_df.loc[fw_idx, 'base_model']
    base_model_letter = str(base_model)[-1]
    dataset = str(ft_res_df.loc[fw_idx, 'dataset']).lower()
    print(f"Finetuned weights '{args.fw}' found in finetuning_results.csv at row {fw_idx} (base model: {base_model}, dataset: {dataset})")

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

# ====== Step 2: Initialize SAM2 predictor and Mediapipe Landmarker ====== #

sam2_predictor = sam2_utils.initialize_sam2(base_model, device, weights_path)
mp_detector = mp_utils.initialize_mp()

# ====== Step 3: Load and sort the video frames for SAM2 segmentation ====== #

# `vid_frames_dir` is a directory of JPEG frames with filenames like `<frame_index>.jpg`
vid_frames_dir = os.path.join("mocapvids/vid_frames", os.path.splitext(args.v)[0])

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(vid_frames_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# Load the original video to get the correct frame rate for the output video
vid_path = "mocapvids/" + args.v
cap = cv2.VideoCapture(vid_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {vid_path}")

# Get input frame rate (fps) to use for the output video
fps = cap.get(cv2.CAP_PROP_FPS)

cap.release()

# ====== Step 4: Prompt the SAM2 model and get MediaPipe predictions ====== #

if args.debug == 0:
    # SAM 2 requires stateful inference for interactive video segmentation, so we need to 
    # initialize an 'inference state' on this video. During initialization, it loads all 
    # the JPEG frames in `video_path` and stores their pixels in `inference_state`.
    inference_state = sam2_predictor.init_state(video_path=vid_frames_dir)
else:
    inference_state = None

obj_ids = []
prompts = {}

mouth_parts = {
    1: {
        "name": "Upper Lip",
        "color": np.array([255, 0, 0], dtype=np.uint8)   # Red
    },

    2: {
        "name": "Lower Lip",
        "color": np.array([0, 255, 0], dtype=np.uint8)   # Green
    },

    3: {
        "name": "Tongue",
        "color": np.array([0, 0, 255], dtype=np.uint8)   # Blue
    }
}

# ------ Generate Prompts for SAM2 Segmentation ------ #

# We will add `args.npp` prompts for each mouth part, evenly spaced throughout the video.
prompt_spacing = len(frame_names) // args.npp

if args.debug == 0:

    # ------ Upper Lip ------
    for prompt_idx in range(args.npp):
        ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
        ann_obj_id = UPPER_LIP_OBJ_ID 

        # Create/Select the bounding box
        box_ul = sam2_utils.generate_prompt_box(
            frame=np.array(Image.open(os.path.join(vid_frames_dir, frame_names[ann_frame_idx]))), 
            mouth_part="upper lip", 
            color = mouth_parts[1]["color"]
        )
        print("--> New ul box created: ", box_ul, '\n')

        # Try adding additional points as prompts
        points_ul, labels_ul = sam2_utils.generate_prompt_points(
            frame=np.array(Image.open(os.path.join(vid_frames_dir, frame_names[ann_frame_idx]))), 
            mouth_part="upper lip", 
            color = mouth_parts[1]["color"]
        )

        # Add prompts and get predictions
        if box_ul is not None:
            _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box_ul
            )

        if points_ul is not None and labels_ul is not None:
            # Add prompts and get predictions
            _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points_ul,
                labels=labels_ul
            )

    # ------ Bottom Lip ------
    for prompt_idx in range(args.npp):
        ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
        ann_obj_id = LOWER_LIP_OBJ_ID 

        # Create/Select the bounding box
        box_bl = sam2_utils.generate_prompt_box(
            frame=np.array(Image.open(os.path.join(vid_frames_dir, frame_names[ann_frame_idx]))), 
            mouth_part="bottom lip", 
            color = mouth_parts[2]["color"]
        )
        print("--> New bl box created: ", box_bl, "\n")
        
        # Try adding additional points as prompts
        points_bl, labels_bl = sam2_utils.generate_prompt_points(
            frame=np.array(Image.open(os.path.join(vid_frames_dir, frame_names[ann_frame_idx]))), 
            mouth_part="bottom lip", 
            color = mouth_parts[2]["color"]
        )

        if box_bl is not None:
            # Add prompts and get predictions
            _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box_bl,
            )

        if points_bl is not None and labels_bl is not None:
            # Add prompts and get predictions
            _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points_bl,
                labels=labels_bl,
            )

    # ------ Run SAM2 Video Object Segmentation ------ #

    # run propagation throughout the video and collect the results in a dict
    sam2_video_segs = {}  # sam2_video_segs contains the per-frame segmentation results

    # Storing the SAM2 Segmentation results in a memory-efficient way using indices and offsets
    upper_indices = []
    upper_offsets = [0]

    lower_indices = []
    lower_offsets = [0]

    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(
        inference_state
    ):
        for i, out_obj_id in enumerate(out_obj_ids):

            mask = (out_mask_logits[i] > 0.0).cpu().numpy()[0]

            indices = np.flatnonzero(mask).astype(np.uint32)

            if out_obj_id == 1:
                upper_indices.append(indices)
                upper_offsets.append(
                    upper_offsets[-1] + len(indices)
                )

            elif out_obj_id == 2:
                lower_indices.append(indices)
                lower_offsets.append(
                    lower_offsets[-1] + len(indices)
                )

    upper_indices = np.concatenate(upper_indices)
    lower_indices = np.concatenate(lower_indices)

    upper_offsets = np.asarray(upper_offsets, dtype=np.uint64)
    lower_offsets = np.asarray(lower_offsets, dtype=np.uint64)

    # ------ Generate MediaPipe Landmark Predictions ------ #
    mp_results_list = mp_utils.run_mp_for_video(os.path.join("mocapvids", args.v), mp_detector)
    mp_lip_coords = mp_utils.extract_lip_landmarks_coords(mp_results_list)

    save_path = f"saved_states/stage1/{args.v.split('.mp4')[0]}_{dataset}_sam2{base_model_letter}_fw{fw_idx}_{args.npp}pp_mp_results.npz"
    data_manager.save_state_stage1(upper_indices, lower_indices, upper_offsets, lower_offsets, mp_results_list, mp_lip_coords, save_path)

else:
    # If the user wants to load the saved state, we can skip the above steps and directly load the SAM2
    # results and MediaPipe predictions.
    save_path = f"saved_states/stage1/{args.v.split('.mp4')[0]}_{dataset}_sam2{base_model_letter}_fw{fw_idx}_{args.npp}pp_mp_results.npz"
    upper_indices, lower_indices, upper_offsets, lower_offsets, mp_results_list, mp_lip_coords = data_manager.load_saved_state_stage1(save_path)


# ====== Step 6: Refine MediaPipe Predictions with SAM2 results ====== #

# Access the first frame to save the height and width
first_frame = np.array(Image.open(os.path.join(vid_frames_dir, frame_names[0])))
height, width = first_frame.shape[:2]

if args.debug < 2:      # i.e. we are not using a stage 2 saved state

    # Save a copy of the original Mediapipe results
    sam2_mp_results = mp_results_list.copy()

    # Processing Steps (per-frame, per-mask):
    #   1. Clean the lip mask (top or bottom) by removing noise 
    #   2. Turn the lip mask into a smoothed spline to removing jagged lip contours
    #   3. 
    with tqdm(total=len(mp_lip_coords), desc="Refining MediaPipe Lip Landmarks with SAM2 Masks") as pbar:

        # Loop through each frame in the video
        for frame_idx in range(len(frame_names)):

            # Create a copy of the Mediapipe landmarks that will be updated
            frame_landmarks = sam2_mp_results[frame_idx]

            # Create MediaPipe landmark look-up table for this frame
            mp_lookup_table = refinement_utils.build_landmark_lookup(mp_lip_coords[frame_idx], width, height)

            # Find the Mediapipe-predicted mouth corners for anchoring
            inner_left_corner, outer_left_corner, inner_right_corner, outer_right_corner = refinement_utils.find_mp_corners(mp_lookup_table, height, width)

            # Create SAM2 upper and lower lip masks
            upper_mask = data_manager.reconstruct_mask(
                upper_indices,
                upper_offsets,
                frame_idx=frame_idx,
                height=height,
                width=width
            )

            lower_mask = data_manager.reconstruct_mask(
                lower_indices,
                lower_offsets,
                frame_idx=frame_idx,
                height=height,
                width=width
            )

            for obj_id, raw_mask in zip([UPPER_LIP_OBJ_ID, LOWER_LIP_OBJ_ID], [upper_mask, lower_mask]):

                # --------------------------------------
                # Mask Processing
                # --------------------------------------

                raw_mask = raw_mask.squeeze()

                # Clean the raw sam2 mask to remove holes and noise
                clean_mask_result = mask_processor.clean_mask(raw_mask, obj_id, inner_left_corner, outer_left_corner, inner_right_corner, outer_right_corner)

                # Convert the cleaned mask into a smooth spline
                spline = mask_processor.mask_to_spline(clean_mask_result)

                # --------------------------------------
                # Combining SAM2 and Mediapipe Results
                # --------------------------------------

                # Create points along the SAM2 lip contours
                upper_points, lower_points = refinement_utils.generate_sam2_guided_landmarks(obj_id, spline, inner_left_corner, outer_left_corner, 
                                                                                            inner_right_corner, outer_right_corner, mp_lookup_table)

                # Append Mediapipe mouth corners (only if a SAM2 mask is found) to upper SAM2 contour points
                if obj_id == 1:     # Top Lip
                    if upper_points is not None:
                        upper_points = np.vstack([
                            np.array(outer_left_corner), 
                            upper_points, 
                            np.array(outer_right_corner)
                        ])
                    if lower_points is not None:
                        lower_points = np.vstack([
                            np.array(inner_left_corner), 
                            lower_points, 
                            np.array(inner_right_corner)
                        ])

                # Replace the original Mediapipe ladnmarks with these SAM2-guided points
                frame_landmarks = mp_utils.replace_mp_lip_landmarks(frame_landmarks, obj_id, upper_points, lower_points, height, width)

            # Save the refined landmark results
            sam2_mp_results[frame_idx] = frame_landmarks
            
            pbar.update(1)

    save_path = f"saved_states/stage2/{args.v.split('.mp4')[0]}_{dataset}_sam2{base_model_letter}_fw{fw_idx}_{args.npp}pp_mp_results.npz"
    data_manager.save_state_stage2(sam2_mp_results, save_path)
else:                   
    # If the user wants to load the saved state, we can skip the above steps and directly load the refined mediapipe landmark results.
    save_path = f"saved_states/stage2/{args.v.split('.mp4')[0]}_{dataset}_sam2{base_model_letter}_fw{fw_idx}_{args.npp}pp_mp_results.npz"
    sam2_mp_results = data_manager.load_saved_state_stage2(save_path)

# ====== Step 7: Create Segmentation Video ====== #

# Output video path
# output_video_path = f"sam2_guided_mp_results/{base_model}_{dataset}_fw{fw_idx}_{args.npp}pp_{args.v}"
output_video_path = f"testing_visualization.mp4"

fourcc = cv2.VideoWriter.fourcc(*'mp4v')
video_writer = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps,
    (width, height)
)

mp_results_copy = mp_results_list.copy()

# ----- Generate Frames ------ #
for frame_idx in range(len(frame_names)):

    frame_path = os.path.join(vid_frames_dir, frame_names[frame_idx])

    frame = np.array(
        Image.open(frame_path).convert("RGB")
    )

    output_frame = frame.copy()

    # upper_mask = data_manager.reconstruct_mask(
    #     upper_indices,
    #     upper_offsets,
    #     frame_idx=frame_idx,
    #     height=height,
    #     width=width
    # )

    # lower_mask = data_manager.reconstruct_mask(
    #     lower_indices,
    #     lower_offsets,
    #     frame_idx=frame_idx,
    #     height=height,
    #     width=width
    # )

    # # Add lip segmentation masks to the output frame
    # alpha = 0.3
    # for obj_id, lip_mask in zip([UPPER_LIP_OBJ_ID, LOWER_LIP_OBJ_ID], [upper_mask, lower_mask]):
    #     color = mouth_parts[obj_id]["color"]

    #     # Apply transparent mask
    #     output_frame[lip_mask] = (
    #         alpha * color +
    #         (1 - alpha) * output_frame[lip_mask]
    #     ).astype(np.uint8)

    # Add SAM2-guided lip landmarks
    # for idx, point in enumerate(sam2_mp_results[frame_idx]):
    #     if idx not in mp_utils.lip_indices:
    #         continue

    #     x = round((point.x * width))
    #     y = round((point.y * height))

    #     cv2.circle(
    #         output_frame,
    #         (x, y),
    #         8,
    #         (255, 255, 0),  # Yellow
    #         -1
    #     )

    # output_frame = mp_utils.draw_lip_contours(output_frame, sam2_mp_results[frame_idx], contour_colour=(255, 255, 0))
    output_frame = mp_utils.add_lms_to_frame(output_frame, sam2_mp_results[frame_idx], lm_colour=(255, 255, 255))
    output_frame = mp_utils.draw_face_mesh(output_frame, sam2_mp_results[frame_idx])
    # Convert RGB -> BGR
    output_frame_bgr = cv2.cvtColor(
        output_frame,
        cv2.COLOR_RGB2BGR
    )

    video_writer.write(output_frame_bgr)
    
# ====== Finalize ====== #

video_writer.release()

print(f"Saved video to: {output_video_path}")