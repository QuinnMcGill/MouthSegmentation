import os
from pathlib import Path

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
        visibility_labels = None

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

            # mouth_landmark_range = range(84, 104)
            mouth_landmark_range = range(84, 104)

            with open(landmark_path, "r") as f:
                lines = [line.strip() for line in f.readlines()]

            num_landmarks = int(lines[0])

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
            
            #visibility_labels = self.create_visibility_labels(landmarks, [upper_lip_mask, lower_lip_mask])

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
    
    # TODO: this is a very simple heuristic for visibility - does
    # NOT work consistently. Consider adding a threshold for the distance betwen the 
    # landmark and the nearest point in the mask, or some other more robust method.
    # def create_visibility_labels(self, landmarks, masks):
    #     """
    #     Create binary visibility labels for each landmark based on whether
    #     it falls within the lip masks.

    #     Args:
    #         landmarks: Tensor [N, 2]
    #         masks: Tensor [2, H, W]

    #     Returns:
    #         visibilities: Tensor [N]
    #     """
    #     visibilities = torch.zeros(landmarks.shape[0], dtype=torch.float32, device=landmarks.device)    # Assume all are visible

    #     for i, (x, y) in enumerate(landmarks):
    #         # Check if the landmark falls within any of the lip masks
    #         for mask in masks:
    #             if mask[int(y), int(x)] == 1:
    #                 visibilities[i] = 1  
    #                 break

    #         if visibilities[i] != 1:
    #             print(f"Landmark {i} on image {self.image_paths[i]} is not a part of any lip mask.")
    #     return visibilities

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
# print("Lip corners shape:", batch["lip_corners"].shape)

# print("Lip corners:", batch["lip_corners"])
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
#         lip_corners[:, 0],
#         lip_corners[:, 1],
#         s=10,
#         # c=colors
#     )

#     axes[0].set_title("Image + Landmarks")
#     axes[0].axis("off")

#     # Mask image
#     axes[1].imshow(mask_vis)

#     axes[1].scatter(
#         lip_corners[:, 0],
#         lip_corners[:, 1],
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
