# ====== Step 0: Import Libraries ====== #

# System
import os
import argparse

from tqdm import tqdm
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Computation and data handling
import numpy as np
import torch

# Visualization and image processing
from PIL import Image
import cv2
import pandas as pd

# SAM2
from sam2.build_sam import build_sam2_video_predictor       #type:ignore
from sam2.sam2_video_predictor import SAM2VideoPredictor    #type:ignore
from sam2_resources import sam2_utils as sam2_utils

# Processing SAM2 Masks
# import mediapipe_resources.mp_helpers as mph
import refine_landmarks.data_manager as data_manager

# ====== Step 1: Setup the environment ====== #

UPPER_LIP_OBJ_ID = 1
LOWER_LIP_OBJ_ID = 2

# Parse the command line arguments
parser = argparse.ArgumentParser(description='Mouth Part Segmentation Script')
parser.add_argument('--v', '--video_filename', type=str, default='tc5.mp4', help='.mp4 filename of test video in mocapvids directory')
parser.add_argument('--fw', '--finetuned_weights', type=str, default=None, help="Name of weights file in finetuning_sam2/finetuned_weights directory")
parser.add_argument('--npp', '--num_prompts_per_part', type=int, default=4, help='Number of prompts to add for each mouth part (integer)')
parser.add_argument('--crop_mouth',type=int,default=0,help='Whether to interactively crop the output video around the mouth (1 or 0)')
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
    run_tag = f"{args.v.split('.mp4')[0]}_{dataset}_sam2{base_model_letter}_fw{fw_idx}_{args.npp}pp"
    print(f"Finetuned weights '{args.fw}' found in finetuning_results.csv at row {fw_idx} (base model: {base_model}, dataset: {dataset})")

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# ====== Step 2: Load the SAM2 model ====== #

sam2_predictor = sam2_utils.initialize_sam2(base_model, device, weights_path)

# ====== Step 3: Load the video ====== #

# `vid_frames_dir` is a directory of JPEG frames with filenames like `<frame_index>.jpg`
tc_name = args.v.split('.mp4')[0]
vid_frames_dir = os.path.join("mocapvids/vid_frames", os.path.splitext(args.v)[0])

# Load the original video to get the correct frame rate for the output video
vid_path = "mocapvids/" + args.v
cap = cv2.VideoCapture(vid_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {vid_path}")

# Get input frame rate (fps) to use for the output video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()

# Get the first frame for mouth crop later
frame_names = [
    p for p in os.listdir(vid_frames_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
first_frame = np.array(Image.open(os.path.join(vid_frames_dir, frame_names[0])))

# ====== Step 4: Run SAM2 Segmentation with Finetuned Weights ====== #

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

# Prompt the model with bounding boxes and prompt points for the upper and lower lip
sam2_predictor, inference_state = sam2_utils.run_sam2_lip_seg(num_parts_per_object=args.npp, 
                                                                vid_frames_dir=vid_frames_dir, 
                                                                sam2_predictor=sam2_predictor)

# Propagate predictions between frames to achieve VOS of lips
(upper_indices, lower_indices, 
upper_offsets, lower_offsets) = sam2_utils.create_lip_masks(sam2_predictor=sam2_predictor, inference_state=inference_state)

# ====== Step 6: Create Segmentation Video ====== #

with tqdm(total=num_frames, desc="Creating Lip Segmentation Video") as pbar:

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
        x_min = max(0, min(x_min, frame_width))
        x_max = max(0, min(x_max, frame_width))
        y_min = max(0, min(y_min, frame_height))
        y_max = max(0, min(y_max, frame_height))

        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"Invalid mouth crop box: "
                f"{[x_min, y_min, x_max, y_max]}"
            )

        crop_width = x_max - x_min
        crop_height = y_max - y_min

    else:

        # No cropping
        x_min = 0
        y_min = 0
        x_max = frame_width
        y_max = frame_height

        crop_width = frame_width
        crop_height = frame_height

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
            height=frame_height,
            width=frame_width
        )

        lower_mask = data_manager.reconstruct_mask(
            lower_indices,
            lower_offsets,
            frame_idx=frame_idx,
            height=frame_height,
            width=frame_width
        )

        for obj_id, raw_mask in zip([UPPER_LIP_OBJ_ID, LOWER_LIP_OBJ_ID], [upper_mask, lower_mask]):

            color = mouth_parts[obj_id]["color"]

            # Apply transparent mask
            overlay[raw_mask] = (
                alpha * color +
                (1 - alpha) * overlay[raw_mask]
            ).astype(np.uint8)

            # Optional: draw contours
            contours, _ = cv2.findContours(
                raw_mask.astype(np.uint8),
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
            ys, xs = np.where(raw_mask)

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