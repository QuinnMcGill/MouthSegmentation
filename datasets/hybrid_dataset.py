import os
import cv2
import torch
import pandas as pd
import random
import numpy as np
from pathlib import Path

from torch.utils.data import random_split
from torch.utils.data import Dataset

import albumentations as A

class HybridDataset(Dataset):
    """
    Dataset composed of both LaPa samples and Maya Faces (synthetic). This dataset
    is designed to be customizable for easier testing (optional landmarks, compatible with
    different splits of Lapa/MayaFaces).

    Outputs:
        image: Tensor [3, H, W] float32 in range [0,1]
        mask:  Tensor [2, H, W]
        landmarks: Tensor [N, 2] (optional, only if landmarks=True)

    Mask Channels: 
        Channel 0 = upper lip
        Channel 1 = lower lip
    """
    def __init__(
            self, 
            lapa_root_dir,
            maya_root_dir,
            maya_split=0.5,
            image_size=(1024,1024),
            transform=None,
            landmarks=False, 
    ):
        self.lapa_root_dir = Path(lapa_root_dir)
        self.maya_root_dir = Path(maya_root_dir)
        self.maya_split = maya_split
        self.image_size = image_size
        self.transform = transform
        self.landmarks = landmarks

        self.lapa_image_paths = self.build_lapa_paths()
        self.maya_samples = self.build_maya_samples()
        self.lapa_samples = self.build_lapa_samples()

        self.all_samples = self.maya_samples + self.lapa_samples

    def build_lapa_paths(self):
        splits = ["train", "test"]
        image_paths = []

        for split in splits:
            image_dir = self.lapa_root_dir / split / "images"
            image_paths.extend(image_dir.glob("*.jpg"))

        return sorted(image_paths)
    
    def build_maya_samples(self):
        """
        Create all the samples from the MayaFaces dataset. Since this is a much smaller dataset,
        all samples should be used. The parameter "maya_split" controls the percentage of the 
        final dataset which these samples account for.
        """

        samples = []
        image_dir = self.maya_root_dir / "images"
        ul_dir = self.maya_root_dir / "ul_masks"
        ll_dir = self.maya_root_dir / "ll_masks"
        lm_dir = self.maya_root_dir / "landmarks"

        inner_lm_colors = self.maya_root_dir / "landmark_colors_inner.txt"
        outer_lm_colors = self.maya_root_dir / "landmark_colors_outer.txt"

        if self.landmarks is True:
            inner_color_dict = self.load_landmark_colors(inner_lm_colors)
            outer_color_dict = self.load_landmark_colors(outer_lm_colors)

        for img_file in sorted(image_dir.glob("img_*.png")):

            img_id = img_file.stem.split("_")[-1]

            ul_file = ul_dir / f"ul_{img_id}.png"
            ll_file = ll_dir / f"ll_{img_id}.png"
            inner_lm_file = lm_dir / f"inner_lm_{img_id}.png"
            outer_lm_file = lm_dir / f"inner_lm_{img_id}.png"

            # -------------------------------------------------
            # Load Image and Convert to RGB
            # -------------------------------------------------

            # image = cv2.imread(img_file)

            # if image is None:
            #     raise RuntimeError(f"Failed to load image: {image}")
            
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # -------------------------------------------------
            # Check for UL, LL, and Landmarks
            # -------------------------------------------------

            if not ul_file.exists() or not ll_file.exists():
                continue

            if self.landmarks is True and (not inner_lm_file.exists() or not outer_lm_file.exists()):
                continue

            # --------------------------------------------------
            # Load masks
            # --------------------------------------------------

            ul_mask = cv2.imread(
                ul_file,
                cv2.IMREAD_GRAYSCALE,
            )

            ll_mask = cv2.imread(
                ll_file,
                cv2.IMREAD_GRAYSCALE,
            )

            if ul_mask is None:
                raise RuntimeError(f"Failed to load image: {ul_file}")

            if ll_mask is None:
                raise RuntimeError(f"Failed to load image: {ll_file}")
            
            # convert to binary masks
            ul_mask = (ul_mask > 127).astype(np.float32)
            ll_mask = (ll_mask > 127).astype(np.float32)

            # --------------------------------------------------
            # Apply transforms
            # --------------------------------------------------

            # if self.transform is not None:

            #     transformed = self.transform(
            #         image=image,
            #         masks=[ul_mask, ll_mask],
            #     )

            #     image = transformed["image"]

            #     ul_mask = transformed["masks"][0]
            #     ll_mask = transformed["masks"][1]

            # --------------------------------------------------
            # Convert to tensors
            # --------------------------------------------------

            # image = (
            #     torch.from_numpy(image)
            #     .permute(2, 0, 1)
            #     .float()
            #     / 255.0
            # )

            mask = np.stack(
                [
                    ul_mask,
                    ll_mask,
                ],
                axis=0,
            ).astype(np.float32)

            # --------------------------------------------------
            # Create a sample
            # --------------------------------------------------
            sample = {
                "image": img_file,
                "mask": mask
            }

            # if self.landmarks is True:
            #     inner_landmarks = self.find_landmarks(inner_lm_file, inner_color_dict)
            #     outer_landmarks = self.find_landmarks(outer_lm_file, outer_color_dict)
            #     # TODO: Add landmarks to sample

            samples.append(sample)

        return samples

    def load_landmark_colors(self, txt_file): 
        """ 
        Reads: 
            LandmarkID,R,G,B 

        Returns: 
            dict: landmark_id -> (R,G,B) 
        """ 
        df = pd.read_csv(txt_file) 
        color_dict = {} 

        for _, row in df.iterrows(): 
            color_dict[int(row["LandmarkID"])] = ( int(row["R"]), int(row["G"]), int(row["B"]), ) 

        return color_dict 
        
    def find_landmarks(self, image, color_dict, tol=3): 
        """ 
        Parameters 
        ---------- 
            image : ndarray RGB or RGBA image 
            color_dict : dict landmark_id -> (R,G,B) 

        Returns 
        ------- 
            dict landmark_id -> (x,y) centroid 
        """ 
        # Ignore alpha channel if present 
        rgb = image[:, :, :3] 
        landmarks = {} 

        for landmark_id, color in color_dict.items(): 
            color = np.array(color, dtype=np.int16) # Match pixels whose R,G,B values are all within +/- tol 
            mask = np.all( np.abs(rgb - color) <= tol, axis=2 ) 
            ys, xs = np.where(mask) 

            if len(xs) == 0:
                landmarks[landmark_id] = None 
                continue 
            
            x = round(xs.mean()) 
            y = round(ys.mean()) 
            landmarks[landmark_id] = (x, y) 
            
        return landmarks
    
    def build_lapa_samples(self):
        lapa_split = 1.0 - self.maya_split
        num_lapa_samples = round(lapa_split * len(self.maya_samples)/self.maya_split)

        indices = random.sample(range(len(self.lapa_image_paths)), k=num_lapa_samples)

        samples = []

        for idx in indices:
            img_path = self.lapa_image_paths[idx]

            img_stem = img_path.stem
            label_dir = str(img_path).split("images")[0] + '/labels/'
            landmarks_dir = str(img_path).split("images")[0] + '/landmarks/'

            label_path = label_dir + img_stem + ".png"
            lm_path = landmarks_dir + img_stem + ".txt"

            # -------------------------------------------------
            # Load Image and Convert to RGB
            # -------------------------------------------------

            # image = cv2.imread(img_path)

            # if image is None:
            #     raise RuntimeError(f"Failed to load image: {image}")
            
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # -----------------------------
            # Load segmentation mask
            # -----------------------------
            label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)

            if label is None:
                raise RuntimeError(f"Failed to load label: {label_path}")

            # -----------------------------
            # Create binary masks
            # -----------------------------
            ul_mask = (label == 7).astype(np.float32)
            ll_mask = (label == 9).astype(np.float32)

            # --------------------------------------------------
            # Convert to tensors
            # --------------------------------------------------

            # image = (
            #     torch.from_numpy(image)
            #     .permute(2, 0, 1)
            #     .float()
            #     / 255.0
            # )

            mask = np.stack(
                [
                    ul_mask,
                    ll_mask,
                ],
                axis=0,
            ).astype(np.float32)

            # --------------------------------------------------
            # Create a sample
            # --------------------------------------------------
            sample = {
                "image": img_path,
                "mask": mask
            }

            # if self.landmarks is True:
            #     # TODO: Add landmarks to sample

            samples.append(sample)
    
        return samples
    
    def __getitem__(self, idx):
        sample = self.all_samples[idx]

        # --------------------------------------------------
        # Load RGB image
        # --------------------------------------------------

        image = cv2.imread(str(sample["image"]))

        if image is None:
            raise RuntimeError(
                f"Failed to load image: {sample['image']}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mask is already a numpy array (2,H,W)
        mask = sample["mask"].copy()

        # -----------------------------
        # Resize if needed
        # -----------------------------
        if self.image_size is not None:

            orig_h, orig_w = image.shape[:2]
            w, h = self.image_size

            if (w, h) != (orig_w, orig_h):

                image = cv2.resize(
                    image,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )

                mask = np.stack(
                    [
                        cv2.resize(
                            m,
                            (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        for m in mask
                    ],
                    axis=0,
                )
        
        # --------------------------------------------------
        # Apply transforms
        # --------------------------------------------------

        if self.transform is not None:

            transformed = self.transform(
                image=image,
                masks=[mask[0], mask[1]],
            )

            image = transformed["image"]
            mask = np.stack(
                transformed["masks"],
                axis=0,
            )

        # --------------------------------------------------
        # Create Tensors
        # --------------------------------------------------
        image = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        mask = torch.from_numpy(mask).float()

        return {
            "image": image,
            "mask": mask,
        } 

    def __len__(self):
        return len(self.all_samples)   
    
# ------------ SCRIPT FOR TESTING DATASET ------------
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

lapa_root_dir = "/home/quinnm/.cache/downloaded_datasets/LaPa"
maya_root_dir = "datasets/MayaFaces"

hybrid_dataset = HybridDataset(
    lapa_root_dir,
    maya_root_dir,
    maya_split=0.5,
    image_size=(1024, 1024),
    transform=None,
    landmarks=False, 
)

train_loader = DataLoader(dataset=hybrid_dataset, batch_size=16, shuffle=True)

print("Length of dataloader: ", len(train_loader))
print("Length of dataset: ", len(hybrid_dataset))

batch = next(iter(train_loader))

print("---- First Batch ----")
print("Image shape:", batch["image"].shape)
print("Mask shape:", batch["mask"].shape)

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


