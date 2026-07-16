import cv2
import torch
import numpy as np
from torch.utils.data import random_split

from pathlib import Path
from torch.utils.data import Dataset

import albumentations as A


class MayaFacesDataset(Dataset):
    """
    Returns:
        image: Tensor [3,H,W] float32 in [0,1]
        mask:  Tensor [2,H,W] float32
            mask[0] = upper lip mask
            mask[1] = lower lip mask
    """

    def __init__(
        self,
        root_dir,
        transform=None,
    ):
        self.root_dir = Path(root_dir)

        self.image_dir = self.root_dir / "images"
        self.ul_dir = self.root_dir / "ul_masks"
        self.ll_dir = self.root_dir / "ll_masks"

        self.transform = transform

        self.samples = []

        for img_file in sorted(self.image_dir.glob("img_*.png")):

            img_id = img_file.stem.split("_")[-1]

            ul_file = self.ul_dir / f"ul_{img_id}.png"
            ll_file = self.ll_dir / f"ll_{img_id}.png"

            if not ul_file.exists():
                continue

            if not ll_file.exists():
                continue

            self.samples.append(
                {
                    "image": img_file,
                    "ul": ul_file,
                    "ll": ll_file,
                }
            )

        print(
            f"Loaded {len(self.samples)} MayaFace samples"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        # --------------------------------------------------
        # Load RGB image
        # --------------------------------------------------

        image = cv2.imread(
            str(sample["image"]),
        )

        if image is None:
            raise RuntimeError(f"Failed to load image: {str(sample["image"])}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --------------------------------------------------
        # Load masks
        # --------------------------------------------------

        ul_mask = cv2.imread(
            str(sample["ul"]),
            cv2.IMREAD_GRAYSCALE,
        )

        ll_mask = cv2.imread(
            str(sample["ll"]),
            cv2.IMREAD_GRAYSCALE,
        )

        if ul_mask is None:
            raise RuntimeError(f"Failed to load image: {str(sample["ul"])}")

        if ll_mask is None:
            raise RuntimeError(f"Failed to load image: {str(sample["ll"])}")
        
        # convert to binary masks
        ul_mask = (ul_mask > 127).astype(np.float32)
        ll_mask = (ll_mask > 127).astype(np.float32)

        # --------------------------------------------------
        # Apply transforms
        # --------------------------------------------------

        if self.transform is not None:

            transformed = self.transform(
                image=image,
                masks=[ul_mask, ll_mask],
            )

            image = transformed["image"]

            ul_mask = transformed["masks"][0]
            ll_mask = transformed["masks"][1]

        # --------------------------------------------------
        # Convert to tensors
        # --------------------------------------------------

        image = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        mask = torch.stack(
            [
                torch.from_numpy(ul_mask),
                torch.from_numpy(ll_mask),
            ],
            dim=0,
        ).float()

        return {
            "image": image,
            "mask": mask,
        }
    
# ------------ SCRIPT FOR TESTING DATASET ------------
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import numpy as np

# # from datasets.lapa_dataset import LaPaLipDataset
# dataset_path = "datasets/MayaFaces"

# train_transform = A.Compose(
#     [
#         A.HorizontalFlip(p=0.5),

#         A.Rotate(
#             limit=15,
#             border_mode=cv2.BORDER_CONSTANT,
#             p=0.5,
#         ),

#         A.RandomBrightnessContrast(
#             p=0.3,
#         ),

#         A.ShiftScaleRotate(
#             shift_limit=0.05,
#             scale_limit=0.10,
#             rotate_limit=10,
#             p=0.5,
#         ),
#     ]
# )

# maya_dataset = MayaFacesDataset(
#     root_dir=dataset_path,
#     transform=train_transform
# )

# print(len(maya_dataset))

# train_size = int(0.8 * len(maya_dataset))
# val_size = len(maya_dataset) - train_size

# train_dataset, val_dataset = random_split(
#     maya_dataset,
#     [train_size, val_size],
# )

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=8,
#     shuffle=True,
#     num_workers=4,
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=8,
#     shuffle=False,
#     num_workers=4
# )

# print("Training size (samples): ", len(train_loader))
# print("Validation size (samples): ", len(val_loader))

# batch = next(iter(train_loader))

# print("Image shape:", batch["image"].shape)
# print("Mask shape:", batch["mask"].shape)

# # # --------------------------------------------------
# # # Visualize first sample in batch
# # # --------------------------------------------------

# for idx in range(batch["image"].shape[0]):
#     # CHW -> HWC
#     image = batch["image"][idx].permute(1, 2, 0).numpy()

#     # [2, H, W]
#     mask = batch["mask"][idx].numpy()

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

#     overlay = image.copy()

#     alpha = 0.5

#     overlay = (
#         (1 - alpha) * overlay
#         + alpha * mask_vis
#     )

#     overlay = np.clip(
#         overlay,
#         0,
#         1
#     )

#     fig, ax = plt.subplots(
#         1,
#         3,
#         figsize=(15, 5)
#     )

#     ax[0].imshow(image)
#     ax[0].set_title("Image")
#     ax[0].axis("off")

#     ax[1].imshow(mask_vis)
#     ax[1].set_title(
#         "Masks\n(Red=Upper, Green=Lower)"
#     )
#     ax[1].axis("off")

#     ax[2].imshow(overlay)
#     ax[2].set_title("Overlay")
#     ax[2].axis("off")

#     plt.tight_layout()
#     plt.show()

# # ------------ END OF SCRIPT ------------