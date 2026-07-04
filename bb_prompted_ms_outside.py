# ====== Step 0: Import Libraries ====== #

# System
import os
import sys
import keyboard
import argparse
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


# ====== Step 1: Setup the environment ====== #

# Parse the command line arguments
parser = argparse.ArgumentParser(description='Mouth Part Segmentation Script')
parser.add_argument('--v', '--video_filename', type=str, default='tc5.mp4', help='.mp4 filename of test video in mocapvids directory')
parser.add_argument('--sc', '--sam_checkpoint', type=str, default='sam2.1_hiera_large.pt', help='Name of SAM checkpoint file')
parser.add_argument('--d', '--dataset', type=str, default='maya', help='Name of dataset to train on (options: "maya" or "lapa")')
parser.add_argument('--sv', '--save_video', type=int, default=0, help='Whether to save the output segmentation video (boolean)')
parser.add_argument('--npp', '--num_prompts_per_part', type=int, default=4, help='Number of prompts to add for each mouth part (integer)')
parser.add_argument('--fw', '--finetuned_weights', type=str, default=None, help="Name of weights file in finetuned_weights folder")
args = parser.parse_args()

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

# ====== Step 3: Load the SAM2 model ====== #

sam2_checkpoint = os.path.join("external/sam2/checkpoints", args.sc)
#sam2_checkpoint = "finetuned_weights/sam2_lapa_step_8000.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device.type) # Build the SAM2 model using pretrained weights

# Load the fine-tuned model weights
fw_idx = -1   # Only used if the user specifies the finetuned weights
base_model = "sam2.1s"
dataset = args.d

if args.fw is not None:
    # Get the row index of the results for these finetuned weights
    ft_res_df = pd.read_csv("finetuning_sam2/finetuning_results.csv")
    fw_idx = ft_res_df.index[ft_res_df['finetuned_weights'] == 'finetuning_sam2/finetuned_weights/' + args.fw][0]

    # Get the model type
    base_model = ft_res_df.loc[fw_idx, 'base_model']

    # Get the dataset
    dataset = ft_res_df.loc[fw_idx, 'dataset']

    fw_checkpoint = torch.load(
        f='finetuning_sam2/finetuned_weights/' + args.fw,
        map_location=device,
        weights_only=False
    )

    predictor.load_state_dict(
        fw_checkpoint["model_state_dict"]
    )

    # state_dict = torch.load(
    #     f"finetuned_weights/{args.fw}",
    #     map_location=device,
    #     weights_only=True
    # )
    # predictor.load_state_dict(state_dict)

# ====== Step 4: Load the video and perform segmentation ====== #

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = os.path.join("mocapvids/vid_frames", os.path.splitext(args.v)[0])

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# ====== Step 5: Add prompts for different mouth parts ====== #

# SAM 2 requires stateful inference for interactive video segmentation, so we need to 
# initialize an 'inference state' on this video. During initialization, it loads all 
# the JPEG frames in `video_path` and stores their pixels in `inference_state`.
inference_state = predictor.init_state(video_path=video_dir)

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
decent_ul_boxes = [[408, 840, 628, 882], [377, 700, 608, 860], [428, 782, 684, 846], [446, 777, 692, 824]]

for prompt_idx in range(args.npp):
    ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # Add positive and negative clicks
    # box_ul = get_prompt_box(
    #     frame=np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))), 
    #     mouth_part="upper lip", color = mouth_parts[1]["color"]
    # )

    box_ul = decent_ul_boxes[prompt_idx]

    # Add prompts and get predictions
    if box_ul is not None:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box_ul,
        )

# ------ Bottom Lip ------
decent_bl_boxes = [[406, 878, 626, 935], [392, 809, 583, 931], [414, 844, 708, 911], [434, 815, 704, 878]]

for prompt_idx in range(args.npp):
    ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
    ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

    # Add positive and negative clicks
    # box_bl = get_prompt_box(
    #     frame=np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))), 
    #     mouth_part="bottom lip", color = mouth_parts[2]["color"]
    # )

    box_bl = decent_bl_boxes[prompt_idx]

    if box_bl is not None:
        # Add prompts and get predictions
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box_bl,
        )

# # ------ Tongue ------
# for prompt_idx in range(args.npp):
#     ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
#     ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)

#     # Add positive and negative clicks
#     box_tg = get_prompt_box(
#         frame=np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))), 
#         mouth_part="tongue", color = mouth_parts[3]["color"]
#     )

#     # Add prompts and get predictions
#     if box_tg is not None:
#         _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#             inference_state=inference_state,
#             frame_idx=ann_frame_idx,
#             obj_id=ann_obj_id,
#             box=box_tg,
#         )

# ====== Step 6: Run Video Object Segmentation and Tracking ====== #

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


# ====== STEP 7: Visualize some of the segmented frames ====== #
# frame_stride = 30
# for out_frame_idx in range(0, len(frame_names), frame_stride):
#     plt.figure(figsize=(6, 4))
#     plt.title(f"frame {out_frame_idx}")
#     plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
#     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
#         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

#     plt.axis('off')
#     plt.show()



# ====== Step 8 (Optional): Create Segmentation Video ====== #

if args.sv == 1:
    # Load first frame to get video dimensions
    first_frame = np.array(
        Image.open(os.path.join(video_dir, frame_names[0]))
    )

    height, width = first_frame.shape[:2]

    # Output video path
    output_video_path = f"bb_seg_results/bb_{base_model}_{dataset}_fw{fw_idx}_{args.v}"

    # Video writer
    fps = 30

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    alpha = 0.5

    # ----- Generate Frames ------ #
    example_frames = len(frame_names) // 5

    for frame_idx in range(len(frame_names)):

        frame_path = os.path.join(video_dir, frame_names[frame_idx])

        frame = np.array(
            Image.open(frame_path).convert("RGB")
        )

        overlay = frame.copy()

        # Draw masks
        if frame_idx in video_segments:

            for obj_id, raw_mask in video_segments[frame_idx].items():
                
                raw_mask = raw_mask.squeeze()
                # if (frame_idx % example_frames == 0):
                #     clean_mask_result = mask_processor.clean_mask(raw_mask)
                #     if clean_mask_result is not None:
                #         mask_processor.visualize_mask_cleanup(
                #             frame,
                #             raw_mask,
                #             clean_mask_result
                #         )

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

        # Convert RGB -> BGR
        output_frame_bgr = cv2.cvtColor(
            output_frame,
            cv2.COLOR_RGB2BGR
        )

        video_writer.write(output_frame_bgr)

    # ====== Finalize ====== #

    video_writer.release()

    print(f"Saved video to: {output_video_path}")