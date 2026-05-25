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


# ====== Step 1: Setup the environment ====== #

# Parse the command line arguments
parser = argparse.ArgumentParser(description='Mouth Part Segmentation Script')
parser.add_argument('--v', '--video_filename', type=str, default='tc5.mp4', help='.mp4 filename of test video in mocapvids directory')
parser.add_argument('--sc', '--sam_checkpoint', type=str, default='sam2.1_hiera_large.pt', help='Name of SAM checkpoint file')
parser.add_argument('--sv', '--save_video', type=bool, default=False, help='Whether to save the output segmentation video (boolean)')
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


# ====== Step 2: Define Helper Functions ====== #

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(patches.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def get_prompt_box(frame, mouth_part, color):

    print(
        f"Please draw a bounding box for {mouth_part}.\n"
        "Instructions:\n"
        "  - Click TOP LEFT corner\n"
        "  - Click BOTTOM RIGHT corner\n"
        "  - Press 'r' to redraw\n"
        "  - Press 'n' if the object is NOT visible\n"
        "  - Press ESC to confirm and close\n"
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(frame)
    ax.set_title(f"Bounding Box Prompt: {mouth_part}", color=tuple(color / 255.0))

    clicked_points = []
    point_artists = []

    bbox = [None]
    object_missing = [False]

    rectangle_patch = [None]

    # ---------------------------------
    # Mouse click event
    # ---------------------------------
    def onclick(event):

        if event.inaxes != ax:
            return

        # Ignore additional clicks once a box exists.
        # User must press 'r' to redraw.
        if bbox[0] is not None:
            print("Bounding box already exists. Press 'r' to redraw.")
            return

        x = int(event.xdata)
        y = int(event.ydata)

        clicked_points.append((x, y))

        print(f"Clicked: ({x}, {y})")

        point = ax.scatter(x, y, c="yellow", s=50)
        point_artists.append(point)

        fig.canvas.draw()

        if len(clicked_points) == 2:

            (x1, y1), (x2, y2) = clicked_points

            x_min = min(x1, x2)
            y_min = min(y1, y2)

            x_max = max(x1, x2)
            y_max = max(y1, y2)

            bbox[0] = [x_min, y_min, x_max, y_max]

            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor="lime",
                facecolor="none"
            )

            rectangle_patch[0] = rect
            ax.add_patch(rect)

            fig.canvas.draw()

            print(f"Bounding box: {bbox[0]}")
            print("Press ESC to confirm or 'r' to redraw.")

    # ---------------------------------
    # Keyboard event
    # ---------------------------------
    def onkey(event):

        # Mark object absent
        if event.key == "n":

            object_missing[0] = True

            print(f"{mouth_part} marked as NOT VISIBLE.")

            plt.close(fig)

        # Reset box and points
        elif event.key == "r":

            clicked_points.clear()
            bbox[0] = None

            if rectangle_patch[0] is not None:
                rectangle_patch[0].remove()
                rectangle_patch[0] = None

            for artist in point_artists:
                artist.remove()

            point_artists.clear()

            fig.canvas.draw()

            print("Selection cleared. Draw a new bounding box.")

        # Confirm and close
        elif event.key == "escape":

            if object_missing[0]:
                print("Object marked as not visible.")

            elif bbox[0] is None:
                print("No bounding box selected.")

            else:
                print("Bounding box confirmed.")

            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)

    plt.show()

    if object_missing[0]:
        return None

    if bbox[0] is None:
        return None

    return np.array(bbox[0], dtype=np.float32)

# ====== Step 3: Load the SAM2 model ====== #

sam2_checkpoint = os.path.join("external/sam2/checkpoints", args.sc)
#sam2_checkpoint = "finetuned_weights/sam2_lapa_step_8000.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device.type) # Build the SAM2 model using pretrained weights

# Load the fine-tuned model weights
fw_idx = -1   # Only used if the user specifies the finetuned weights
base_model = "sam2.1s"
dataset = "lapa"
if args.fw is not None:
    # Get the row index of the results for these finetuned weights
    ft_res_df = pd.read_csv("finetuning/finetuning_results.csv")
    fw_idx = ft_res_df.index[ft_res_df['finetuned_weights'] == 'finetuned_weights/' + args.fw][0]

    # Get the model type
    base_model = ft_res_df.loc[fw_idx, 'base_model']

    # Get the dataset
    dataset_name = ft_res_df.loc[fw_idx, 'dataset']

    fw_checkpoint = torch.load(
        f='finetuned_weights/' + args.fw,
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
for prompt_idx in range(args.npp):
    ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # Add positive and negative clicks
    box_ul = get_prompt_box(
        frame=np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))), 
        mouth_part="upper lip", color = mouth_parts[1]["color"]
    )

    # Add prompts and get predictions
    if box_ul is not None:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box_ul,
        )

# ------ Bottom Lip ------
for prompt_idx in range(args.npp):
    ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
    ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

    # Add positive and negative clicks
    box_bl = get_prompt_box(
        frame=np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))), 
        mouth_part="bottom lip", color = mouth_parts[2]["color"]
    )

    if box_bl is not None:
        # Add prompts and get predictions
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box_bl,
        )

# ------ Tongue ------
for prompt_idx in range(args.npp):
    ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
    ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)

    # Add positive and negative clicks
    box_tg = get_prompt_box(
        frame=np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))), 
        mouth_part="tongue", color = mouth_parts[3]["color"]
    )

    # Add prompts and get predictions
    if box_tg is not None:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box_tg,
        )

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

if args.sv is True:
    # Load first frame to get video dimensions
    first_frame = np.array(
        Image.open(os.path.join(video_dir, frame_names[0]))
    )

    height, width = first_frame.shape[:2]

    # Output video path
    output_video_path = f"bb_seg_results/bb_{base_model}_{dataset}_fw{fw_idx}_tongue_{args.v}"

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

    for frame_idx in range(len(frame_names)):

        frame_path = os.path.join(video_dir, frame_names[frame_idx])

        frame = np.array(
            Image.open(frame_path).convert("RGB")
        )

        overlay = frame.copy()

        # Draw masks
        if frame_idx in video_segments:

            for obj_id, mask in video_segments[frame_idx].items():

                mask = mask.squeeze()

                color = mouth_parts[obj_id]["color"]

                # Apply transparent mask
                overlay[mask] = (
                    alpha * color +
                    (1 - alpha) * overlay[mask]
                ).astype(np.uint8)

                # Optional: draw contours
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
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
                ys, xs = np.where(mask)

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