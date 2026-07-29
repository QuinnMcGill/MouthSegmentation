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

# ====== Step 1: Setup the environment ====== #

UPPER_LIP_OBJ_ID = 1
LOWER_LIP_OBJ_ID = 2
TONGUE_OBJ_ID = 3

# Parse the command line arguments
parser = argparse.ArgumentParser(description='Refine MediaPipe mouth landmarks using SAM2 segmentation masks')
parser.add_argument('--v', '--video_filename', type=str, default='tc6.mp4', help='.mp4 filename of test video in mocapvids directory')
parser.add_argument('--debug', '--debug_stage', type=int, default=0, help='0: doe not use saved stage, 1: stage 1 saved state, 2: stage 2 saved state')
parser.add_argument('--bm', '--base_model', type=str, default='s', help='SAM2.1 base model (options: s, b, l, t)')
parser.add_argument('--d', '--dataset', type=str, default='maya', help='Name of dataset to train on (options: "maya", "lapa" or "hybrid")')
parser.add_argument('--sv', '--save_video', type=int, default=0, help='Whether to save the output segmentation video (boolean)')
parser.add_argument('--npp', '--num_prompts_per_part', type=int, default=4, help='Number of prompts to add for each mouth part (integer)')
parser.add_argument('--cbb', '--create_bounding_boxes', type=int, default=0, help='Whether to create bounding boxes for each mouth part (1 or 0)')
parser.add_argument('--fw', '--finetuned_weights', type=str, default=None, help="Name of weights file in finetuned_weights folder")
args = parser.parse_args()

# Initialize some configuration parameters
base_model = "sam2.1" + args.bm
dataset = args.d
fw_idx = -1   
# If the user would like to use finetuned weights, get the index of the finetuning results for these weights
if args.fw is not None:
    # Get the row index of the results for these finetuned weights
    ft_res_df = pd.read_csv("finetuning_sam2/finetuning_results.csv")
    fw_idx = ft_res_df.index[ft_res_df['finetuned_weights'] == 'finetuning_sam2/finetuned_weights/' + args.fw][0]

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


# ====== Step 2: Define Helper Functions ====== #

# ------ 2.3: Visualization and Debugging Functions ------ #
def save_state_stage1(sam2_video_segs, mp_results_list, mp_lip_results_list, save_path):
    np.savez_compressed(
        save_path,
        sam2_video_segs=sam2_video_segs,
        mp_results_list=mp_results_list,
        mp_lip_results_list=mp_lip_results_list
    )
    print(f"Saved (Stage 1) state to {save_path}")

def save_state_stage2(sam2_mp_results, save_path):
    np.savez_compressed(
        save_path,
        sam2_mp_results=sam2_mp_results,
    )
    print(f"Saved (Stage 2) state to {save_path}")

def load_saved_state_stage1(load_path):
    data = np.load(load_path, allow_pickle=True)
    sam2_video_segs = data['sam2_video_segs'].item()
    mp_lip_results_list = data['mp_lip_results_list'].tolist()
    mp_results_list = data['mp_results_list']
    print(f"Loaded (Stage 1) state from {load_path}")
    return sam2_video_segs, mp_results_list, mp_lip_results_list


def load_saved_state_stage2(load_path):
    data = np.load(load_path, allow_pickle=True)
    sam2_mp_results = data['sam2_mp_results'].tolist()
    print(f"Loaded (Stage 2) state from {load_path}")
    return sam2_mp_results

# ====== Step 3: Initialize SAM2 predictor and Mediapipe Landmarker ====== #

sam2_predictor = sam2_utils.initialize_sam2(args.bm, device, args.fw)
mp_detector = mp_utils.initialize_mp()

# ====== Step 4: Load and sort the video frames for SAM2 segmentation ====== #

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = os.path.join("mocapvids/vid_frames", os.path.splitext(args.v)[0])

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# ====== Step 5: Add prompts for different mouth parts ====== #
if args.debug == 0:
    # SAM 2 requires stateful inference for interactive video segmentation, so we need to 
    # initialize an 'inference state' on this video. During initialization, it loads all 
    # the JPEG frames in `video_path` and stores their pixels in `inference_state`.
    inference_state = sam2_predictor.init_state(video_path=video_dir)
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

# We will add `args.npp` prompts for each mouth part, evenly spaced throughout the video.
prompt_spacing = len(frame_names) // args.npp

if args.debug == 0:
    # ------ Upper Lip ------
    decent_ul_boxes = []
    decent_ul_boxes_4 = [[408, 840, 628, 882], [377, 700, 608, 860], [428, 782, 684, 846], [446, 777, 692, 824]]
    decent_ul_boxes_5 = [[412, 851, 621, 882], [432, 680, 684, 817], [463, 689, 695, 795], [457, 749, 684, 806], None]
    decent_ul_boxes_6 = [[412, 838, 621, 882], [474, 780, 632, 858], [519, 713, 737, 811], [443, 769, 668, 851], [448, 757, 690, 795], [477, 744, 681, 800]]

    if args.cbb == 0:   # Load the pre-made bounding boxes for the upper and bottom lips
        if args.npp == 4:
            decent_ul_boxes = decent_ul_boxes_4
        elif args.npp == 5:
            decent_ul_boxes = decent_ul_boxes_5
        elif args.npp == 6:
            decent_ul_boxes = decent_ul_boxes_6
        else:
            raise ValueError(f"Unsupported number of prompts per part: {args.npp}. Supported values are 4, 5, or 6.")

    for prompt_idx in range(args.npp):
        ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        # Create/Select the bounding box
        if args.cbb == 1:
            box_ul = sam2_utils.generate_prompt_box(
                frame=np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))), 
                mouth_part="upper lip", 
                color = mouth_parts[1]["color"]
            )
            print("--> New ul box created: ", box_ul, '\n')
        else:
            if prompt_idx < len(decent_ul_boxes):
                box_ul = decent_ul_boxes[prompt_idx]
            else:
                raise ValueError(f"Not enough pre-made bounding boxes for the upper lip. Requested {args.npp}, but only {len(decent_ul_boxes)} available.")

        # Try adding additional points as prompts
        points_ul, labels_ul = sam2_utils.generate_prompt_points(
            frame=np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))), 
            mouth_part="upper lip"
        )

        # Add prompts and get predictions
        if box_ul is not None:
            _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box_ul,
                points=points_ul,
                labels=labels_ul
            )

    # ------ Bottom Lip ------
    decent_bl_boxes = []
    decent_bl_boxes_4 = [[406, 878, 626, 935], [392, 809, 583, 931], [414, 844, 708, 911], [434, 815, 704, 878]]
    decent_bl_boxes_5 = [[417, 886, 626, 931], [423, 824, 681, 962], [468, 780, 675, 842], None, [448, 820, 706, 886]]
    decent_bl_boxes_6 = [[417, 880, 630, 926], [486, 862, 626, 931], [539, 806, 746, 918], [423, 855, 712, 906], [450, 800, 699, 849], [470, 800, 695, 835]]

    if args.cbb == 0:   # Load the pre-made bounding boxes for the upper and bottom lips
        if args.npp == 4:
            decent_bl_boxes = decent_bl_boxes_4
        elif args.npp == 5:
            decent_bl_boxes = decent_bl_boxes_5
        elif args.npp == 6:
            decent_bl_boxes = decent_bl_boxes_6
        else:
            raise ValueError(f"Unsupported number of prompts per part: {args.npp}. Supported values are 4, 5, or 6.")

    for prompt_idx in range(args.npp):
        ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
        ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

        # Create/Select the bounding box
        if args.cbb == 1:
            box_bl = sam2_utils.generate_prompt_box(
                frame=np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))), 
                mouth_part="bottom lip", 
                color = mouth_parts[2]["color"]
            )
            print("--> New bl box created: ", box_bl, "\n")
        else:
            if prompt_idx < len(decent_bl_boxes):
                box_bl = decent_bl_boxes[prompt_idx]
            else:
                raise ValueError(f"Not enough pre-made bounding boxes for the bottom lip. Requested {args.npp}, but only {len(decent_bl_boxes)} available.")

        # Try adding additional points as prompts
        points_bl, labels_bl = sam2_utils.generate_prompt_points(
            frame=np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))), 
            mouth_part="bottom lip"
        )

        if box_bl is not None:
            # Add prompts and get predictions
            _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box_bl,
                points=points_bl,
                labels=labels_bl,
            )

    # ------ Tongue ------
    # for prompt_idx in range(args.npp):
    #     ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
    #     ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)

    #     # Create bounding box
    #     box_tg = sam2_utils.generate_prompt_box(
    #         frame=np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))), 
    #         mouth_part="tongue", 
    #         color = mouth_parts[3]["color"]
    #     )

    #     # Add prompts and get predictions
    #     if box_tg is not None:
    #         _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
    #             inference_state=inference_state,
    #             frame_idx=ann_frame_idx,
    #             obj_id=ann_obj_id,
    #             box=box_tg,
    #         )

    # ====== Step 5: Run SAM2 Video Object Segmentation and Tracking ====== #

    # run propagation throughout the video and collect the results in a dict
    sam2_video_segs = {}  # sam2_video_segs contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(inference_state):
        sam2_video_segs[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # ====== Step 6: Generate MediaPipe Landmark Predictions ====== #
    mp_results_list = mp_utils.get_mp_landmarks(os.path.join("mocapvids", args.v), mp_detector)
    mp_lip_results_list = mp_utils.isolate_lip_landmarks(mp_results_list)

    save_path = f"saved_states/stage1/{args.v.split('.mp4')[0]}_{args.d}_sam2{args.bm}_fw{fw_idx}_{args.npp}pp_mp_results.npz"
    save_state_stage1(sam2_video_segs, mp_results_list, mp_lip_results_list, save_path)
else:
    # If the user wants to load the saved state, we can skip the above steps and directly load the SAM2 
    # segmentation results and MediaPipe landmark predictions from a .npz file.

    # ====== Optional: Save the SAM2 segmentation results and MediaPipe landmark predictions ====== #
    save_path = f"saved_states/stage1/{args.v.split('.mp4')[0]}_{args.d}_sam2{args.bm}_fw{fw_idx}_{args.npp}pp_mp_results.npz"
    sam2_video_segs, mp_results_list, mp_lip_results_list = load_saved_state_stage1(save_path)

# ====== Step 7: Refine MediaPipe Predictions ====== #

# Access the first from to save the height and width
first_frame = np.array(Image.open(os.path.join(video_dir, frame_names[0])))

h, w = first_frame.shape[:2]

if args.debug < 2:      # i.e. we are not using a stage 2 saved state

    # Save a copy of the original Mediapipe results
    sam2_mp_results = mp_results_list.copy()

    # Processing Steps (per-frame, per-mask):
    #   1. Clean the lip mask (top or bottom) by removing noise 
    #   2. Turn the lip mask into a smoothed spline to removing jagged lip contours
    #   3. 
    with tqdm(total=len(mp_lip_results_list), desc="Refining MediaPipe Lip Landmarks with SAM2 Masks") as pbar:

        # Loop through each frame in the video
        for frame_idx in range(len(frame_names)):

            # Create a copy of the Mediapipe landmarks that will be updated
            frame_landmarks = sam2_mp_results[frame_idx]

            if frame_idx in sam2_video_segs:

                # Create MediaPipe landmark look-up table for this frame
                mp_lookup_table = refinement_utils.build_landmark_lookup(mp_lip_results_list[frame_idx], w, h)

                # Find the Mediapipe-predicted mouth corners for anchoring
                inner_left_corner, outer_left_corner, inner_right_corner, outer_right_corner = refinement_utils.find_mp_corners(mp_lookup_table, h, w)

                for obj_id, raw_mask in sam2_video_segs[frame_idx].items():
                    
                    if obj_id == UPPER_LIP_OBJ_ID or obj_id == LOWER_LIP_OBJ_ID:   # Create sam2-points for top and bottom lip

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
                        frame_landmarks = mp_utils.replace_mp_lip_landmarks(frame_landmarks, obj_id, upper_points, lower_points, h, w)

                    else:   # Tongue Mask
                        print("Tongue mask detected. Currently not processing tongue masks in this version of the code.")

                # Save the refined landmark results
                sam2_mp_results[frame_idx] = frame_landmarks
            
            pbar.update(1)

    save_path = f"saved_states/stage2/{args.v.split('.mp4')[0]}_{args.d}_sam2{args.bm}_fw{fw_idx}_{args.npp}pp_mp_results.npz"
    save_state_stage2(sam2_mp_results, save_path)
else:                   # Load the stage 2 saved state
    save_path = f"saved_states/stage2/{args.v.split('.mp4')[0]}_{args.d}_sam2{args.bm}_fw{fw_idx}_{args.npp}pp_mp_results.npz"
    sam2_mp_results = load_saved_state_stage2(save_path)

# ====== Step 8: Create Segmentation Video ====== #

if args.sv == 1:

    # Output video path
    output_video_path = f"sam2_guided_mp_results/DEBUG_{base_model}_{dataset}_fw{fw_idx}_{args.npp}pp_{args.v}"

    # Video writer
    fps = 30

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (w, h)
    )

    mp_results_copy = mp_results_list.copy()

    # ----- Generate Frames ------ #
    for frame_idx in range(len(frame_names)):

        frame_path = os.path.join(video_dir, frame_names[frame_idx])

        frame = np.array(
            Image.open(frame_path).convert("RGB")
        )

        output_frame = frame.copy()

        # Add the mediapipe landmarks for comparison
        output_frame = mp_utils.add_lms_to_frame(output_frame, mp_lip_results_list[frame_idx], False)

        # Add medaipipe lip contours
        output_frame = mp_utils.draw_lip_contours(output_frame, mp_results_list[frame_idx], (0, 0, 255))

        # Add SAM2-guided lip landmarks
        for idx, point in enumerate(sam2_mp_results[frame_idx]):
            if idx not in mp_utils.lip_indices:
                continue

            x = round((point.x * w))
            y = round((point.y * h))

            cv2.circle(
                output_frame,
                (x, y),
                3,
                (0, 0, 0),  # Black
                -1
            )

        # Add lip contours - > indented once
        output_frame = mp_utils.draw_lip_contours(output_frame, sam2_mp_results[frame_idx], (255, 0, 0)) #output_frame = mp_utils.draw_lip_contours2(output_frame, updated_landmarks) 
        
        # Convert RGB -> BGR
        output_frame_bgr = cv2.cvtColor(
            output_frame,
            cv2.COLOR_RGB2BGR
        )

        video_writer.write(output_frame_bgr)
        
    # ====== Finalize ====== #

    video_writer.release()

    print(f"Saved video to: {output_video_path}")