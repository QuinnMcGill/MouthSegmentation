import os
from pathlib import Path
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class LaPaLipDataset(Dataset):
    """
    Dataset for lip segmentation using the LaPa dataset.

    Outputs:
        image: Tensor [3, H, W] float32 in range [0,1]
        mask:  Tensor [2, H, W]
        landmarks: Tensor [N, 2] (optional, only if landmarks=True)
        lip_corners: Tensor [2, 2] (optional, only if lip_corners=True) - (x,y) coordinates of left and right lip corners (left first, then right)

    Mask Channels: 
        Channel 0 = upper lip
        Channel 1 = lower lip
    """

    # LaPa label IDs
    UPPER_LIP_CLASS = 7
    LOWER_LIP_CLASS = 9

    def __init__(
        self,
        root_dir,
        split="train",
        image_size=None,
        transform=None,
        landmarks=False, 
        lip_corners=False
    ):
        """
        Args:
            root_dir: Path to LaPa root directory
            split: 'train', 'val', or 'test'
            image_size: tuple (W, H) or None
            transform: optional augmentation function
            landmarks: whether to load landmarks
            lip_corners: whether to load lip corner coordinates
        """

        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.landmarks = landmarks
        self.lip_corners = lip_corners
        self.image_dir = self.root_dir / split / "images"
        self.label_dir = self.root_dir / split / "labels"
        self.landmarks_dir = self.root_dir / split / "landmarks"

        self.image_paths = sorted(self.image_dir.glob("*.jpg"))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        # Example:
        # 12345.jpg -> 12345.png
        label_path = self.label_dir / f"{image_path.stem}.png"

        if not label_path.exists():
            raise FileNotFoundError(f"Missing label: {label_path}")

        # -----------------------------
        # Load image
        # -----------------------------
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # -----------------------------
        # Load segmentation mask
        # IMPORTANT:
        # Read as grayscale / unchanged
        # -----------------------------
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)

        if label is None:
            raise RuntimeError(f"Failed to load label: {label_path}")

        # If label accidentally loads as multi-channel,
        # keep only first channel
        if len(label.shape) == 3:
            label = label[:, :, 0]

        # -----------------------------
        # Create binary masks
        # -----------------------------
        upper_lip_mask = (label == self.UPPER_LIP_CLASS).astype(np.float32)
        lower_lip_mask = (label == self.LOWER_LIP_CLASS).astype(np.float32)

        # Stack into [2, H, W]
        mask = np.stack(
            [upper_lip_mask, lower_lip_mask],
            axis=0
        )

        # -----------------------------
        # Optional: load landmarks
        # -----------------------------
        landmarks = None

        landmark_path = self.landmarks_dir / f"{image_path.stem}.txt"

        if not self.landmarks_dir.exists():
            raise FileNotFoundError(
                f"Missing landmarks: {landmark_path}"
            )

        if self.landmarks:
            
            # --- Lip Landmark Cheatsheet: ---
            #   84 = left lip corner and start of mouth range (from image perspective)
            #   90 = right lip corner
            #   84-90 = upper lip outer contour, corners included (7 points)
            #   93 = center of lower lip outer boundary 
            #   91-95 = lower lip outer contour, corners not included (5 points)
            #   98 = center of upper lip inner contour
            #   96-100 = upper lip inner contour (5 points)
            #   102 = center of bottom lip inner contour
            #   101-103 = lower lip inner contour (3 points)

            mouth_landmark_range = range(84, 104)

            with open(landmark_path, "r") as f:
                lines = [line.strip() for line in f.readlines()]

            landmarks = []

            for i, line in enumerate(lines[1:], start=0):

                x, y = map(float, line.split())

                if i in mouth_landmark_range:
                    landmarks.append([x, y])

            landmarks = np.array(
                landmarks,
                dtype=np.float32
            )

            if len(landmarks) != len(mouth_landmark_range):
                raise RuntimeError(
                    f"Expected {len(mouth_landmark_range)} landmarks but "
                    f"found {len(landmarks)} in {landmark_path}"
                )
            
        # -----------------------------
        # Optional: load mouth corners
        # -----------------------------
        with open(landmark_path, "r") as f:
                lines = [line.strip() for line in f.readlines()]

        right_lip_corner = None
        left_lip_corner = None
        lip_corners = None

        if self.lip_corners:
            lx, ly = map(float, lines[85].split())
            rx, ry = map(float, lines[91].split())
            left_lip_corner = [lx, ly]
            right_lip_corner = [rx, ry]
            lip_corners = np.array(
                [left_lip_corner, right_lip_corner],
                dtype=np.float32
            )

        # -----------------------------
        # Resize if needed
        # -----------------------------
        orig_h, orig_w = image.shape[:2]

        if self.image_size is not None:

            w, h = self.image_size

            image = cv2.resize(
                image,
                (w, h),
                interpolation=cv2.INTER_LINEAR
            )

            resized_masks = []

            for m in mask:
                resized_m = cv2.resize(
                    m,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )
                resized_masks.append(resized_m)

            mask = np.stack(resized_masks, axis=0)

            if landmarks is not None:
                sx = w/orig_w
                sy = h/orig_h

                landmarks[:, 0] *= sx
                landmarks[:, 1] *= sy

            if lip_corners is not None:
                sx = w/orig_w
                sy = h/orig_h

                lip_corners[:, 0] *= sx
                lip_corners[:, 1] *= sy

        # -----------------------------
        # Normalize image
        # -----------------------------
        image = image.astype(np.float32) / 255.0

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        # -----------------------------
        # Convert to tensors
        # -----------------------------
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        if landmarks is not None: 
            landmarks = torch.from_numpy(landmarks).float()

        # -----------------------------
        # Optional augmentations
        # -----------------------------
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # -----------------------------
        # Return sample
        # -----------------------------
        sample =  {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
        }

        if landmarks is not None:
            sample["landmarks"] = landmarks

        # TODO: consider returning visibility labels as well, but for now we can compute them on the fly in the training loop if needed
        # if visibility_labels is not None:
        #     sample["visibility_labels"] = visibility_labels

        if self.lip_corners:
            sample["lip_corners"] = lip_corners

        return sample

def resample_contour(contour, n_points):
    """
    Resample a polyline to exactly n_points.

    contour:
        [N,2]

    returns:
        [n_points,2]
    """

    contour = np.asarray(contour, dtype=np.float32)

    segment_lengths = np.linalg.norm(
        contour[1:] - contour[:-1],
        axis=1
    )

    cumulative = np.concatenate(
        [[0.0], np.cumsum(segment_lengths)]
    )

    total_length = cumulative[-1]

    target_distances = np.linspace(
        0,
        total_length,
        n_points
    )

    result = []

    for d in target_distances:

        idx = np.searchsorted(cumulative, d) - 1
        idx = np.clip(idx, 0, len(segment_lengths)-1)

        seg_start = cumulative[idx]
        seg_end = cumulative[idx+1]

        t = (d - seg_start) / max(seg_end - seg_start, 1e-8)

        p = (
            (1 - t) * contour[idx]
            + t * contour[idx + 1]
        )

        result.append(p)

    return np.array(result, dtype=np.float32)

def lapa_to_mediapipe_mouth(landmarks):
    """
    landmarks:
        [20,2] LaPa mouth landmarks

    returns:
        dict:
            mediapipe landmark id -> (x,y)
    """

    # -------------------
    # Extract contours
    # -------------------

    # upper_outer = landmarks[0:7]
    upper_outer_left = landmarks[0:4]
    upper_outer_right = landmarks[3:7]

    # lower_outer = np.array([
    #     landmarks[0],    # 84
    #     landmarks[11],   # 95
    #     landmarks[10],   # 94
    #     landmarks[9],    # 93
    #     landmarks[8],    # 92
    #     landmarks[7],    # 91
    #     landmarks[6],    # 90
    # ])
    lower_outer_left = np.array([
        landmarks[0],    # 84
        landmarks[11],   # 95
        landmarks[10],   # 94
        landmarks[9],    # 93
    ])

    lower_outer_right = np.array([
        landmarks[9],    # 93
        landmarks[8],    # 92
        landmarks[7],    # 91
        landmarks[6],    # 90
    ])

    # upper_inner = landmarks[12:17]
    upper_inner_left = landmarks[12:15]
    upper_inner_right = landmarks[14:17]

    # lower_inner = np.array([
    #     landmarks[12],   # 96
    #     landmarks[19],   # 103
    #     landmarks[18],   # 102
    #     landmarks[17],   # 101
    #     landmarks[16],   # 100
    # ])
    lower_inner_left = np.array([
        landmarks[12],   # 96
        landmarks[19],   # 103
        landmarks[18],   # 102
    ])


    lower_inner_right = np.array([
        landmarks[18],   # 102
        landmarks[17],   # 101
        landmarks[16],   # 100
    ])

    # -------------------
    # Resample
    # -------------------

    # upper_outer_mp = resample_contour(
    #     upper_outer,
    #     11
    # )

    upper_outer_left_mp = resample_contour(
        upper_outer_left,
        6
    )
    
    upper_outer_right_mp = resample_contour(
        upper_outer_right,
        6
    )

    # lower_outer_mp = resample_contour(
    #     lower_outer,
    #     11
    # )

    lower_outer_left_mp = resample_contour(
        lower_outer_left,
        6
    )
    lower_outer_right_mp = resample_contour(
        lower_outer_right,
        6
    )

    # upper_inner_mp = resample_contour(
    #     upper_inner,
    #     11
    # )

    upper_inner_left_mp = resample_contour(
        upper_inner_left,
        6
    )

    upper_inner_right_mp = resample_contour(
        upper_inner_right,
        6
    )

    # lower_inner_mp = resample_contour(
    #     lower_inner,
    #     11
    # )

    lower_inner_left_mp = resample_contour(
        lower_inner_left,
        6
    )

    lower_inner_right_mp = resample_contour(
        lower_inner_right,
        6
    )

    # -------------------
    # Assign MP ids
    # -------------------

    mp = {}

    # upper_outer_ids = [
    #     61,185,40,39,37,
    #     0,
    #     267,269,270,409,291
    # ]

    upper_outer_left_ids = [
        61,185,40,39,37,
        0,
    ]
    
    upper_outer_right_ids = [
        0,
        267,269,270,409,291
    ]

    # upper_inner_ids = [
    #     78,191,80,81,82,
    #     13,
    #     312,311,310,415,308
    # ]

    upper_inner_left_ids = [
        78,191,80,81,82,
        13
    ]

    upper_inner_right_ids = [
        13,
        312,311,310,415,308
    ]

    # lower_outer_ids = [
    #     61,146,91,181,84,
    #     17,
    #     314,405,321,375,291
    # ]

    lower_outer_left_ids = [
        61,146,91,181,84,
        17
    ]

    lower_outer_right_ids = [
        17,
        314,405,321,375,291
    ]

    # lower_inner_ids = [
    #     78,95,88,178,87,
    #     14,
    #     317,402,318,324,308
    # ]

    lower_inner_left_ids = [
        78,95,88,178,87,
        14
    ]

    lower_inner_right_ids = [
        14,
        317,402,318,324,308
    ]

    # for idx, mp_id in enumerate(upper_outer_ids):
    #     mp[mp_id] = upper_outer_mp[idx]

    # for idx, mp_id in enumerate(upper_inner_ids):
    #     mp[mp_id] = upper_inner_mp[idx]

    # for idx, mp_id in enumerate(lower_outer_ids):
    #     mp[mp_id] = lower_outer_mp[idx]

    # for idx, mp_id in enumerate(lower_inner_ids):
    #     mp[mp_id] = lower_inner_mp[idx]

    for idx, mp_id in enumerate(upper_outer_left_ids):
        mp[mp_id] = upper_outer_left_mp[idx]

    for idx, mp_id in enumerate(upper_outer_right_ids):
        mp[mp_id] = upper_outer_right_mp[idx]

    for idx, mp_id in enumerate(upper_inner_left_ids):
        mp[mp_id] = upper_inner_left_mp[idx]

    for idx, mp_id in enumerate(upper_inner_right_ids):
        mp[mp_id] = upper_inner_right_mp[idx]

    for idx, mp_id in enumerate(lower_outer_left_ids):
        mp[mp_id] = lower_outer_left_mp[idx]

    for idx, mp_id in enumerate(lower_outer_right_ids):
        mp[mp_id] = lower_outer_right_mp[idx]

    for idx, mp_id in enumerate(lower_inner_left_ids):
        mp[mp_id] = lower_inner_left_mp[idx]

    for idx, mp_id in enumerate(lower_inner_right_ids):
        mp[mp_id] = lower_inner_right_mp[idx]

    return mp

# MP_IDS = [
#     # upper outer
#     61,185,40,39,37,0,267,269,270,409,291,

#     # lower outer
#     61,146,91,181,84,17,314,405,321,375,291,

#     # upper inner
#     78,191,80,81,82,13,312,311,310,415,308,

#     # lower inner
#     78,95,88,178,87,14,317,402,318,324,308
# ]

MP_IDS = [
    # upper outer left
    61,185,40,39,37,0,
    # upper outer right
    0,267,269,270,409,291,
    # lower outer left
    61,146,91,181,84,17,
    # lower outer right
    17,314,405,321,375,291,
    # upper inner left
    78,191,80,81,82,13,
    # upper inner right
    13,312,311,310,415,308,
    # lower inner left
    78,95,88,178,87,14,
    # lower inner right
    14,317,402,318,324,308
]


def mp_dict_to_array(mp_dict):
    return np.array(
        [mp_dict[idx] for idx in MP_IDS],
        dtype=np.float32
    )

def draw_contour(ax, points, color, closed=False):

    ax.plot(
        points[:,0],
        points[:,1],
        color=color,
        linewidth=2
    )

    if closed:
        ax.plot(
            [points[-1,0], points[0,0]],
            [points[-1,1], points[0,1]],
            color=color,
            linewidth=2
        )

def visualize_lapa_vs_mp(image, lapa_landmarks):

    mp_dict = lapa_to_mediapipe_mouth(lapa_landmarks)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14,7)
    )

    # ------------------------
    # LEFT: LaPa
    # ------------------------

    axes[0].imshow(image)

    axes[0].scatter(
        lapa_landmarks[:,0],
        lapa_landmarks[:,1],
        c="red",
        s=25
    )

    # for i, (x,y) in enumerate(lapa_landmarks):
    #     axes[0].text(
    #         x,
    #         y,
    #         str(i+84),
    #         fontsize=8,
    #         color="yellow"
    #     )

    axes[0].set_title("LaPa (20 landmarks)")
    axes[0].axis("off")

    # ------------------------
    # RIGHT: Generated MP
    # ------------------------

    axes[1].imshow(image)

    upper_outer = np.array([
        mp_dict[idx]
        for idx in [61,185,40,39,37,0,267,269,270,409,291]
    ])

    lower_outer = np.array([
        mp_dict[idx]
        for idx in [61,146,91,181,84,17,314,405,321,375,291]
    ])

    upper_inner = np.array([
        mp_dict[idx]
        for idx in [78,191,80,81,82,13,312,311,310,415,308]
    ])

    lower_inner = np.array([
        mp_dict[idx]
        for idx in [78,95,88,178,87,14,317,402,318,324,308]
    ])

    draw_contour(
        axes[1],
        upper_outer,
        "red"
    )

    draw_contour(
        axes[1],
        lower_outer,
        "blue"
    )

    draw_contour(
        axes[1],
        upper_inner,
        "lime"
    )

    draw_contour(
        axes[1],
        lower_inner,
        "yellow"
    )

    mp_array = mp_dict_to_array(mp_dict)

    axes[1].scatter(
        mp_array[:,0],
        mp_array[:,1],
        c="white",
        s=12
    )

    axes[1].set_title("Generated MediaPipe Layout")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

# ------------ SCRIPT FOR TESTING DATASET ------------
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import numpy as np

# # from datasets.lapa_dataset import LaPaLipDataset
# dataset_path = "/home/quinnm/.cache/downloaded_datasets/LaPa"

# train_dataset = LaPaLipDataset(
#     root_dir=dataset_path,
#     split="train",
#     image_size=(512, 512),
#     landmarks=True,
#     lip_corners=True
# )

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=4,
#     shuffle=True,
#     num_workers=4,
# )

# batch = next(iter(train_loader))

# print("Image shape:", batch["image"].shape)
# print("Mask shape:", batch["mask"].shape)
# print("Landmarks shape:", batch["landmarks"].shape)

# for _ in range(10):

#     idx = np.random.randint(len(train_dataset))

#     sample = train_dataset[idx]

#     image = sample["image"].permute(1,2,0).numpy()

#     lapa_landmarks = sample["landmarks"].numpy()

#     visualize_lapa_vs_mp(
#         image,
#         lapa_landmarks
#     )


# # # --------------------------------------------------
# # # Visualize first sample in batch
# # # --------------------------------------------------

# for idx in range(batch["image"].shape[0]):
#     # CHW -> HWC
#     image = batch["image"][idx].permute(1, 2, 0).numpy()

#     # [2, H, W]
#     mask = batch["mask"][idx].numpy()

#     # [N, 2]
#     landmarks = batch["landmarks"][idx].numpy()

#     # [2, 2]
#     lip_corners = batch["lip_corners"][idx].numpy()

#     # [N]
#     # visibility_labels = batch["visibility_labels"][idx].numpy()

#     # Create visualization mask
#     upper_lip = mask[0]
#     lower_lip = mask[1]

#     mask_vis = np.zeros(
#         (mask.shape[1], mask.shape[2], 3),
#         dtype=np.float32
#     )

#     # Red = upper lip
#     mask_vis[..., 0] = upper_lip

#     # Green = lower lip
#     mask_vis[..., 1] = lower_lip

#     # --------------------------------------------------
#     # Plot
#     # --------------------------------------------------

#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))

#     # Original image
#     axes[0].imshow(image)

#     # colors = ["blue" if vis == 1.0 else "yellow" for vis in visibility_labels]

#     axes[0].scatter(
#         landmarks[:, 0],
#         landmarks[:, 1],
#         s=10,
#         # c=colors
#     )

#     axes[0].set_title("Image + Landmarks")
#     axes[0].axis("off")

#     # Mask image
#     axes[1].imshow(mask_vis)

#     axes[1].scatter(
#         landmarks[:, 0],
#         landmarks[:, 1],
#         s=10,
#         # c=colors
#     )

#     axes[1].set_title("Lip Masks + Landmarks")
#     axes[1].axis("off")

#     plt.tight_layout()
#     plt.show()

# ------------ END OF SCRIPT ------------

# good_sample = train_dataset.__getitem__(np.random.randint(len(train_dataset)))
# good_img = good_sample["image"].permute(1, 2, 0).numpy()
# good_mask = good_sample["mask"].numpy()
# good_landmarks = good_sample["landmarks"].numpy()

# landmark_range = range(84,104)
# # 104 and 105 are pupils
# # 84 = left lip corner and start of mouth range (from image perspective)
# # 90 = right lip corner
# # 84-90 = upper lip outer contour, corners included (7 points)
# # 93 = center of lower lip outer boundary 
# # 91-95 = lower lip outer contour, corners not included (5 points)
# # 98 = center of upper lip inner contour
# # 96-100 = upper lip inner contour (5 points)
# # 102 = center of bottom lip inner contour
# # 101-103 = lower lip inner contour (3 points)

# # mouth + pupils = range(83, 106)

# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# # Original image
# axes[0].imshow(good_img)

# for i, (x, y) in enumerate(good_landmarks):
#     if i in landmark_range:
#         axes[0].scatter(x, y, s=20)
#         axes[0].text(x, y, str(i), fontsize=8)

# axes[0].set_title("Image + Landmarks")
# axes[0].axis("off")

# # Get masks
# # Create visualization mask
# upper_lip = good_mask[0]
# lower_lip = good_mask[1]

# mask_vis = np.zeros(
#     (good_mask.shape[1], good_mask.shape[2], 3),
#     dtype=np.float32
# )

# # Red = upper lip
# mask_vis[..., 0] = upper_lip

# # Green = lower lip
# mask_vis[..., 1] = lower_lip

# # Mask image
# axes[1].imshow(mask_vis)

# for i, (x, y) in enumerate(good_landmarks):
#     if i in landmark_range:
#         axes[1].scatter(x, y, s=20)
#         axes[1].text(x, y, str(i), fontsize=8)

# axes[1].set_title("Lip Masks + Landmarks")
# axes[1].axis("off")

# plt.tight_layout()
# plt.show()
