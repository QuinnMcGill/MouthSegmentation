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
    ):
        """
        Args:
            root_dir: Path to LaPa root directory
            split: 'train', 'val', or 'test'
            image_size: tuple (W, H) or None
            transform: optional augmentation function
        """

        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.transform = transform

        self.image_dir = self.root_dir / split / "images"
        self.label_dir = self.root_dir / split / "labels"

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
        # Resize if needed
        # -----------------------------
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

        # -----------------------------
        # Optional augmentations
        # -----------------------------
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
        }
    
from torch.utils.data import DataLoader

# from datasets.lapa_dataset import LaPaLipDataset
dataset_path = "/home/quinnm/.cache/downloaded_datasets/LaPa"

train_dataset = LaPaLipDataset(
    root_dir=dataset_path,
    split="train",
    image_size=(512, 512),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
)

batch = next(iter(train_loader))

print(batch["image"].shape)
print(batch["mask"].shape)