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
from sam2.build_sam import build_sam2_video_predictor       #type:ignore
from sam2.sam2_video_predictor import SAM2VideoPredictor    #type:ignore
from sam2_resources import sam2_utils as sam2_utils

# Processing SAM2 Masks
# import mediapipe_resources.mp_helpers as mph
import mask_processing.mask_processor as mask_processor
import refine_landmarks.data_manager as data_manager

# ====== Step 1: Setup the environment ====== #

UPPER_LIP_OBJ_ID = 1
LOWER_LIP_OBJ_ID = 2

# Parse the command line arguments
parser = argparse.ArgumentParser(description='Mouth Part Segmentation Script')
parser.add_argument('--v', '--video_filename', type=str, default='tc5.mp4', help='.mp4 filename of test video in mocapvids directory')
parser.add_argument('--bm', '--base_model', type=str, default='s', help='SAM2.1 base model (options: s, b, l, t)')
parser.add_argument('--d', '--dataset', type=str, default='maya', help='Name of dataset to train on (options: "maya", "lapa", or "hybrid")')
parser.add_argument('--crop_mouth',type=int,default=0,help='Whether to interactively crop the output video around the mouth (1 or 0)')
parser.add_argument('--npp', '--num_prompts_per_part', type=int, default=4, help='Number of prompts to add for each mouth part (integer)')
parser.add_argument('--gen_prompts', '--generate_prompts', type=int, default=0, help='Whether to generate prompts for each mouth part (1 or 0)')
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

# ====== Step 2: Load the SAM2 model ====== #

predictor = sam2_utils.initialize_sam2(args.bm, device, args.fw)

# ====== Step 3: Load the video and video frames to perform segmentation ====== #

# `vid_frames_dir` is a directory of JPEG frames with filenames like `<frame_index>.jpg`
vid_frames_dir = os.path.join("mocapvids/vid_frames", os.path.splitext(args.v)[0])

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(vid_frames_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

vid_path = "mocapvids/" + args.v

cap = cv2.VideoCapture(vid_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {vid_path}")

fps = cap.get(cv2.CAP_PROP_FPS)

cap.release()

# ====== Step 4: Add prompts for different mouth parts ====== #

# SAM 2 requires stateful inference for interactive video segmentation, so we need to 
# initialize an 'inference state' on this video. During initialization, it loads all 
# the JPEG frames in `video_path` and stores their pixels in `inference_state`.
if args.gen_prompts == 1: 
    inference_state = predictor.init_state(video_path=vid_frames_dir)
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

prompt_spacing = len(frame_names) // args.npp

# ------ Upper Lip ------
if args.gen_prompts == 1:

    print("---- Creating bounding box prompts for the upper lip ----")
    for prompt_idx in range(args.npp):
        ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        # Create/Select the bounding box
        box_ul = sam2_utils.generate_prompt_box(
            frame=np.array(Image.open(os.path.join(vid_frames_dir, frame_names[ann_frame_idx]))), 
            mouth_part="upper lip", 
            color = mouth_parts[1]["color"]
        )
        print("--> New ul box created: ", box_ul, '\n')

        # Create positive and negative points for the upper lip
        points_ul, labels_ul = sam2_utils.generate_prompt_points(
            frame=np.array(Image.open(os.path.join(vid_frames_dir, frame_names[ann_frame_idx]))), 
            mouth_part="upper lip",
            color = mouth_parts[1]["color"]
        )

        # Add prompts and get predictions
        if box_ul is not None:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box_ul,
            )

        if points_ul is not None and labels_ul is not None:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points_ul,
                labels=labels_ul
            )

    # ------ Bottom Lip ------
    print("---- Creating bounding box prompts for the bottom lip ----")
    for prompt_idx in range(args.npp):
        ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
        ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

        # Create/Select the bounding box
        box_bl = sam2_utils.generate_prompt_box(
            frame=np.array(Image.open(os.path.join(vid_frames_dir, frame_names[ann_frame_idx]))), 
            mouth_part="bottom lip", 
            color = mouth_parts[2]["color"]
        )
        print("--> New bl box created: ", box_bl, "\n")

        # Create positive and negative points for the bottom lip
        points_bl, labels_bl = sam2_utils.generate_prompt_points(
            frame=np.array(Image.open(os.path.join(vid_frames_dir, frame_names[ann_frame_idx]))), 
            mouth_part="bottom lip",
            color = mouth_parts[2]["color"]
        )

        if box_bl is not None:
            # Add prompts and get predictions
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box_bl,
            )

        if points_bl is not None and labels_bl is not None:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points_bl,
                labels=labels_bl
            )

# ====== Step 5: Run Video Object Segmentation and Save the Results ====== #

if args.gen_prompts == 1:

    upper_indices = []
    upper_offsets = [0]

    lower_indices = []
    lower_offsets = [0]

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
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

    save_path = f"saved_states/sam2_masks/sam2_masks_{args.v.split('.mp4')[0]}_{args.d}_sam2{args.bm}_fw{fw_idx}_{args.npp}pp.npz"
    data_manager.save_raw_sam2_masks(upper_indices, lower_indices, upper_offsets, lower_offsets, save_path)
else:
    save_path = f"saved_states/sam2_masks/sam2_masks_{args.v.split('.mp4')[0]}_{args.d}_sam2{args.bm}_fw{fw_idx}_{args.npp}pp.npz"
    upper_indices, lower_indices, upper_offsets, lower_offsets = data_manager.load_raw_sam2_masks(save_path)

# ====== Step 6: Create Segmentation Video ====== #

with tqdm(total=len(frame_names), desc="Creating Lip Segmentation Video") as pbar:

    # Load first frame to get video dimensions
    first_frame = np.array(
        Image.open(os.path.join(vid_frames_dir, frame_names[0]))
    )

    height, width = first_frame.shape[:2]

    # --------------------------------------------------
    # Optional mouth crop
    # --------------------------------------------------

    mouth_crop_box = None

    if args.crop_mouth == 1:

        mouth_crop_box = sam2_utils.generate_mouth_crop_box(first_frame)

        if mouth_crop_box is None:
            raise RuntimeError(
                "No mouth crop bounding box was selected."
            )

        x_min, y_min, x_max, y_max = mouth_crop_box

        # Clamp coordinates to valid image boundaries
        x_min = max(0, min(x_min, width))
        x_max = max(0, min(x_max, width))
        y_min = max(0, min(y_min, height))
        y_max = max(0, min(y_max, height))

        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"Invalid mouth crop box: "
                f"{[x_min, y_min, x_max, y_max]}"
            )

        print(
            f"Using mouth crop: "
            f"x={x_min}:{x_max}, "
            f"y={y_min}:{y_max}"
        )

        crop_width = x_max - x_min
        crop_height = y_max - y_min

    else:

        # No cropping
        x_min = 0
        y_min = 0
        x_max = width
        y_max = height

        crop_width = width
        crop_height = height

    # Output video path
    output_video_path = f"bb_seg_results/bb_{base_model}_{dataset}_fw{fw_idx}_{args.npp}pp_{args.v}"

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (crop_width, crop_height) 
    )

    alpha = 0.3

    # ----- Generate Frames ------ #

    for frame_idx in range(len(frame_names)):

        frame_path = os.path.join(vid_frames_dir, frame_names[frame_idx])

        frame = np.array(
            Image.open(frame_path).convert("RGB")
        )

        overlay = frame.copy()

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
            
            raw_mask = raw_mask.squeeze()

            # Smooth the raw SAM2 mask to remove noise and small artifacts
            smoothed_mask = mask_processor.smooth_mask(raw_mask)

            color = mouth_parts[obj_id]["color"]

            # Apply transparent mask
            overlay[smoothed_mask] = (
                alpha * color +
                (1 - alpha) * overlay[smoothed_mask]
            ).astype(np.uint8)

            # Optional: draw contours
            contours, _ = cv2.findContours(
                smoothed_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                overlay,
                contours,
                -1,
                (255, 255, 255),
                2
            )

            # Optional: add label text
            ys, xs = np.where(smoothed_mask)

            if len(xs) > 0 and len(ys) > 0:

                center_x = int(np.mean(xs))
                center_y = int(np.mean(ys))

                cv2.putText(
                    overlay,
                    mouth_parts[obj_id]["name"],
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        # Blend overlay with original frame
        output_frame = cv2.addWeighted(
            frame,
            1 - alpha,
            overlay,
            alpha,
            0
        )

        # Crop the output frame if mouth cropping is enabled
        output_frame = output_frame[
            y_min:y_max,
            x_min:x_max
        ]

        # Convert RGB -> BGR
        output_frame_bgr = cv2.cvtColor(
            output_frame,
            cv2.COLOR_RGB2BGR
        )

        video_writer.write(output_frame_bgr)

        pbar.update(1)

# ====== Finalize ====== #

video_writer.release()

print(f"Saved video to: {output_video_path}")