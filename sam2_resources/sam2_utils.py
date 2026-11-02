import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional

# SAM2
from sam2.build_sam import build_sam2_video_predictor           #type:ignore   
from sam2.sam2_video_predictor import SAM2VideoPredictor        #type:ignore   

MODEL_DICT = {'s': 'small', 'b': 'base_plus', 'l': 'large', 't': 'tiny'}

# CONTROL PARAMETERS
UPPER_LIP_OBJ_ID = 1
LOWER_LIP_OBJ_ID = 2

MOUTH_PARTS = {
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

# ====== SAM2 Configuration ====== #

def initialize_sam2(base_model, device, finetuned_weights=None):
    # Load the base SAM2 model with the original weights (no fine-tuning)
    base_model_letter = base_model[-1]  # Extract the last character to determine the model type (s, b, l, t)
    sam2_checkpoint = os.path.join("external/sam2/checkpoints", "sam2.1_hiera_" + MODEL_DICT[base_model_letter] + ".pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_" + base_model_letter + ".yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device.type) # Build the SAM2 model using pretrained weights

    fw_idx = -1   # Only used if the user specifies the finetuned weights

    # Load the fine-tuned model weights (optional)
    if finetuned_weights is not None:

        fw_checkpoint = torch.load(
            f=finetuned_weights,
            map_location=device,
            weights_only=False
        )

        predictor.load_state_dict(
            fw_checkpoint["model_state_dict"]
        )

    return predictor

# ====== Generating SAM2 Promts ====== #

def run_sam2_lip_seg(num_parts_per_object, vid_frames_dir, sam2_predictor):

    # Inialize SAM2 inference state
    inference_state = sam2_predictor.init_state(video_path=vid_frames_dir)

    # Scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(vid_frames_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    prompt_spacing = len(frame_names) // num_parts_per_object

    # ------ Upper Lip ------
    for prompt_idx in range(num_parts_per_object):
        ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
        ann_obj_id = UPPER_LIP_OBJ_ID 

        # Create/Select the bounding box
        box_ul = generate_prompt_box(
            frame=np.array(Image.open(os.path.join(vid_frames_dir, frame_names[ann_frame_idx]))), 
            mouth_part="upper lip", 
            color = MOUTH_PARTS[1]["color"]
        )
        print("--> New ul box created: ", box_ul, '\n')

        # Try adding additional points as prompts
        points_ul, labels_ul = generate_prompt_points(
            frame=np.array(Image.open(os.path.join(vid_frames_dir, frame_names[ann_frame_idx]))), 
            mouth_part="upper lip", 
            color = MOUTH_PARTS[1]["color"]
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
    for prompt_idx in range(num_parts_per_object):
        ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
        ann_obj_id = LOWER_LIP_OBJ_ID 

        # Create/Select the bounding box
        box_bl = generate_prompt_box(
            frame=np.array(Image.open(os.path.join(vid_frames_dir, frame_names[ann_frame_idx]))), 
            mouth_part="bottom lip", 
            color = MOUTH_PARTS[2]["color"]
        )
        print("--> New bl box created: ", box_bl, "\n")
        
        # Try adding additional points as prompts
        points_bl, labels_bl = generate_prompt_points(
            frame=np.array(Image.open(os.path.join(vid_frames_dir, frame_names[ann_frame_idx]))), 
            mouth_part="bottom lip", 
            color = MOUTH_PARTS[2]["color"]
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

    # Return the model and the prompted inference state
    return sam2_predictor, inference_state


def create_lip_masks(sam2_predictor, inference_state):
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

    return upper_indices, lower_indices, upper_offsets, lower_offsets


def generate_mouth_crop_box(frame):
    """
    Ask the user to draw a bounding box around the mouth region.

    Instructions:
        - Click TOP LEFT corner
        - Click BOTTOM RIGHT corner
        - Press 'r' to redraw
        - Press ESC to confirm
    """

    print(
        "\nPlease draw a bounding box around the mouth region.\n"
        "Instructions:\n"
        "  - Click TOP LEFT corner\n"
        "  - Click BOTTOM RIGHT corner\n"
        "  - Press 'r' to redraw\n"
        "  - Press ESC to confirm and close\n"
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame)
    ax.set_title("Mouth Crop Bounding Box")

    clicked_points = []
    bbox = [None]
    rectangle_patch = [None]

    # ---------------------------------
    # Mouse click event
    # ---------------------------------
    def onclick(event):

        if event.inaxes != ax:
            return

        # Ignore additional clicks once a box exists
        if bbox[0] is not None:
            print("Bounding box already exists. Press 'r' to redraw.")
            return

        if event.xdata is None or event.ydata is None:
            return

        x = int(event.xdata)
        y = int(event.ydata)

        clicked_points.append((x, y))

        print(f"Clicked: ({x}, {y})")

        # Wait until two opposite corners have been selected
        if len(clicked_points) < 2:
            print("Select the opposite corner.")
            return

        (x1, y1), (x2, y2) = clicked_points

        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        y_max = max(y1, y2)

        bbox[0] = [x_min, y_min, x_max, y_max]  # type:ignore

        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )

        rectangle_patch[0] = rect   # type:ignore
        ax.add_patch(rect)

        fig.canvas.draw()

        print(f"Mouth crop bounding box: {bbox[0]}")
        print("Press ESC to confirm or 'r' to redraw.")

    # ---------------------------------
    # Keyboard event
    # ---------------------------------
    def onkey(event):

        # Redraw
        if event.key == "r":

            clicked_points.clear()
            bbox[0] = None

            if rectangle_patch[0] is not None:
                rectangle_patch[0].remove()
                rectangle_patch[0] = None

            fig.canvas.draw()

            print("Selection cleared. Draw a new bounding box.")

        # Confirm
        elif event.key == "escape":

            if bbox[0] is None:
                print("No bounding box selected.")
            else:
                print("Mouth crop bounding box confirmed.")

            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)

    plt.show()

    if bbox[0] is None:
        return None

    return np.array(bbox[0], dtype=np.int32)


def generate_prompt_box(frame, mouth_part, color):

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
        if bbox[0] is not None:
            print("Bounding box already exists. Press 'r' to redraw.")
            return

        x = int(event.xdata)
        y = int(event.ydata)

        clicked_points.append((x, y))

        print(f"Clicked: ({x}, {y})")

        # Wait until two opposite corners have been selected
        if len(clicked_points) < 2:
            print("Select the opposite corner.")
            return

        (x1, y1), (x2, y2) = clicked_points

        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        y_max = max(y1, y2)

        bbox[0] = [x_min, y_min, x_max, y_max]  # type:ignore

        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )

        rectangle_patch[0] = rect   # type:ignore
        ax.add_patch(rect)

        fig.canvas.draw()

        print(f"Bounding box: {bbox[0]}")
        print("Press ESC to confirm or 'r' to redraw.")

    # ---------------------------------
    # Keyboard events
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


def generate_prompt_points(frame, mouth_part, color):

    print(
        f"Please select prompt points for {mouth_part}.\n"
        "Instructions:\n"
        "  - Left click to choose a location\n"
        "  - Press 'p' for a positive point\n"
        "  - Press 'n' for a negative point\n"
        "  - Press 'r' to redraw\n"
        "  - Press ESC when finished\n"
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(frame)
    ax.set_title(f"Point Prompts: {mouth_part}", color=tuple(color / 255.0))

    current_point: list[Optional[tuple[int, int]]] = [None]

    points = []
    labels = []

    point_artists = []

    # ---------------------------------
    # Mouse click event
    # ---------------------------------
    def onclick(event):

        if event.inaxes != ax:
            return

        x = int(event.xdata)
        y = int(event.ydata)

        current_point[0] = (x, y)

        print(f"Selected ({x}, {y})")
        print("Press 'p' or 'n'.")

    # ---------------------------------
    # Keyboard event
    # ---------------------------------
    def onkey(event):

        # Finish
        if event.key == "escape":
            plt.close(fig)
            return

        # Restart
        elif event.key == "r":

            points.clear()
            labels.clear()
            current_point[0] = None

            for artist in point_artists:
                artist.remove()

            point_artists.clear()

            fig.canvas.draw()

            print("All prompt points cleared.")

            return

        if current_point[0] is None:
            return

        x, y = current_point[0]

        if event.key == "p":

            points.append([x, y])
            labels.append(1)

            artist = ax.scatter(
                x,
                y,
                c="lime",
                marker="o",
                s=80,
            )

            point_artists.append(artist)

            fig.canvas.draw()

            print(f"Added POSITIVE point ({x}, {y})")

            current_point[0] = None

        elif event.key == "n":

            points.append([x, y])
            labels.append(0)

            artist = ax.scatter(
                x,
                y,
                c="red",
                marker="x",
                s=80,
            )

            point_artists.append(artist)

            fig.canvas.draw()

            print(f"Added NEGATIVE point ({x}, {y})")

            current_point[0] = None

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)

    plt.show()

    return np.array(points), np.array(labels)

# ====== Visualization ====== #

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