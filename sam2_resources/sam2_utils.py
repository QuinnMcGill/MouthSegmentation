import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional

# SAM2
from sam2.build_sam import build_sam2_video_predictor           #type:ignore   
from sam2.sam2_video_predictor import SAM2VideoPredictor        #type:ignore   

MODEL_DICT = {'s': 'small', 'b': 'base_plus', 'l': 'large', 't': 'tiny'}

# ====== SAM2 Configuration ====== #

def initialize_sam2(base_model_letter, device, finetuned_weights=None):
    # Load the base SAM2 model with the original weights (no fine-tuning)
    sam2_checkpoint = os.path.join("external/sam2/checkpoints", "sam2.1_hiera_" + MODEL_DICT[base_model_letter] + ".pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_" + base_model_letter + ".yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device.type) # Build the SAM2 model using pretrained weights

    fw_idx = -1   # Only used if the user specifies the finetuned weights

    # Load the fine-tuned model weights (optional)
    if finetuned_weights is not None:
        # Get the row index of the results for these finetuned weights
        ft_res_df = pd.read_csv("finetuning_sam2/finetuning_results.csv")
        fw_idx = ft_res_df.index[ft_res_df['finetuned_weights'] == 'finetuning_sam2/finetuned_weights/' + finetuned_weights][0]

        # Get the model type
        base_model = ft_res_df.loc[fw_idx, 'base_model']

        # Get the dataset
        dataset_name = ft_res_df.loc[fw_idx, 'dataset']

        fw_checkpoint = torch.load(
            f='finetuning_sam2/finetuned_weights/' + finetuned_weights,
            map_location=device,
            weights_only=False
        )

        predictor.load_state_dict(
            fw_checkpoint["model_state_dict"]
        )

    return predictor

# ====== Generating SAM2 Promts ====== #

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

            bbox[0] = [x_min, y_min, x_max, y_max]  # type:ignore

            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor="lime",
                facecolor="none"
            )

            rectangle_patch[0] = rect   # type:ignore
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

def generate_prompt_points(frame, mouth_part):

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
    ax.set_title(f"Prompt Points: {mouth_part}")

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