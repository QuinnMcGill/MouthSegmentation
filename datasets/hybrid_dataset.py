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

        self.maya_lip_corner_ids = [61, 291, 78, 308]
        self.lapa_lip_corner_ids = [84, 90, 96, 100]
        self.maya_meaningful_lm_ids = [37, 0, 267, 17, 13, 14]
        self.lapa_meaningful_lm_ids = [86, 87, 88, 93, 98, 102]

        self.lapa_image_paths = self.build_lapa_paths()
        self.maya_samples = self.build_maya_samples2()
        self.lapa_samples = self.build_lapa_samples()
        self.all_samples = self.maya_samples + self.lapa_samples

    def build_lapa_paths(self):
        splits = ["train", "test"]
        image_paths = []

        for split in splits:
            image_dir = self.lapa_root_dir / split / "images"
            image_paths.extend(image_dir.glob("*.jpg"))

        return sorted(image_paths)
    
    def build_maya_samples2(self):
        """
        Create all the samples from the MayaFaces dataset. Since this is a much smaller dataset,
        all samples should be used. The parameter "maya_split" controls the percentage of the 
        final dataset which these samples account for.
        """

        samples = []
        image_dir = self.maya_root_dir / "images"

        landmark_dir = self.maya_root_dir / "landmark_txt"

        required_landmark_ids = set(
            self.maya_lip_corner_ids + self.maya_meaningful_lm_ids
        )

        for img_file in sorted(image_dir.glob("img_*.png")):

            img_id = img_file.stem.split("_")[-1]

            ul_mask_path = self.maya_root_dir / "ul_masks" / f"ul_{img_id}.png"
            ll_mask_path = self.maya_root_dir / "ll_masks" / f"ll_{img_id}.png"

            lm_path = str(landmark_dir) + "/landmarks_" + img_id + ".txt"

            # -------------------------------------------------
            # Check for UL and LL masks
            # -------------------------------------------------

            if not ul_mask_path.exists() or not ll_mask_path.exists():
                continue

            # -------------------------------------------------
            # Check landmark files
            # -------------------------------------------------

            if self.landmarks:

                if not Path(lm_path).exists():
                    continue

                found_landmark_ids = set()

                with open(lm_path, "r") as f:

                    next(f)  # Skip header

                    for line in f:
                        line = line.strip()

                        if not line:
                            continue

                        landmark_id = int(
                            line.split(",")[0]
                        )

                        found_landmark_ids.add(
                            landmark_id
                        )

                # Skip sample if any required landmark is missing
                if not required_landmark_ids.issubset(found_landmark_ids):
                    continue

            # --------------------------------------------------
            # Create a sample
            # --------------------------------------------------

            sample = {
                "image": img_file,
                "mask_path": self.maya_root_dir
            }

            if self.landmarks is True:
                sample["lm_path"] = Path(lm_path)

            samples.append(sample)
        
        return samples
    
    # def build_maya_samples(self):
    #     """
    #     Create all the samples from the MayaFaces dataset. Since this is a much smaller dataset,
    #     all samples should be used. The parameter "maya_split" controls the percentage of the 
    #     final dataset which these samples account for.
    #     """

    #     samples = []
    #     image_dir = self.maya_root_dir / "images"

    #     inner_lm_colors = self.maya_root_dir / "landmark_colors_inner.txt"
    #     outer_lm_colors = self.maya_root_dir / "landmark_colors_outer.txt"

    #     if self.landmarks is True:
    #         self.inner_color_dict = self.load_landmark_colors(inner_lm_colors)
    #         self.outer_color_dict = self.load_landmark_colors(outer_lm_colors)

    #     for img_file in sorted(image_dir.glob("img_*.png")):

    #         img_id = img_file.stem.split("_")[-1]

    #         ul_mask_path = self.maya_root_dir / "ul_masks" / f"ul_{img_id}.png"
    #         ll_mask_path = self.maya_root_dir / "ll_masks" / f"ll_{img_id}.png"

    #         inner_lm_file = self.maya_root_dir / "landmarks" / f"inner_lm_{img_id}.png"
    #         outer_lm_file = self.maya_root_dir / "landmarks" / f"outer_lm_{img_id}.png"

    #         # -------------------------------------------------
    #         # Check for UL and LL masks
    #         # -------------------------------------------------

    #         if not ul_mask_path.exists() or not ll_mask_path.exists():
    #             continue

    #         # -------------------------------------------------
    #         # Check landmark files
    #         # -------------------------------------------------

    #         if self.landmarks:

    #             if (
    #                 not inner_lm_file.exists()
    #                 or not outer_lm_file.exists()
    #             ):
    #                 print(
    #                     f"Skipping {img_file.name}: "
    #                     f"missing landmark image"
    #                 )
    #                 continue

    #             # Find landmarks
    #             inner_landmarks = self.find_landmarks(
    #                 inner_lm_file,
    #                 self.inner_color_dict
    #             )

    #             outer_landmarks = self.find_landmarks(
    #                 outer_lm_file,
    #                 self.outer_color_dict
    #             )

    #             landmarks = (
    #                 inner_landmarks
    #                 | outer_landmarks
    #             )

    #             if len(landmarks) < (len(self.maya_lip_corner_ids) + len(self.maya_meaningful_lm_ids)):
    #                 # print(
    #                 #     f"Skipping {img_file.name}: "
    #                 #     f"incomplete landmarks. Found {len(landmarks)} but expected "
    #                 #     f"{len(self.maya_lip_corner_ids) + len(self.maya_meaningful_lm_ids)}."
    #                 # )
    #                 continue

    #         # --------------------------------------------------
    #         # Create a sample
    #         # --------------------------------------------------

    #         sample = {
    #             "image": img_file,
    #             "mask_path": self.maya_root_dir
    #         }

    #         if self.landmarks is True:
    #             sample["lm_path"] = self.maya_root_dir

    #         samples.append(sample)

    #     return samples

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
        
    def find_landmarks(self, image_path, color_dict, tol=3):
        """
        Find landmark locations in a color-coded landmark image.

        Parameters
        ----------
        image_path : Path
            Path to the landmark image.
        color_dict : dict
            Dictionary mapping landmark IDs to RGB colors.
        tol : int
            Color matching tolerance.

        Returns
        -------
        dict
            Dictionary mapping landmark IDs to (x, y) coordinates.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing landmark image: {image_path}"
            )

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Failed to load landmark image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        rgb = image[:, :, :3]

        landmarks = {}

        for landmark_id, color in color_dict.items():

            color = np.array(
                color,
                dtype=np.int16
            )

            mask = np.all(
                np.abs(rgb.astype(np.int16) - color) <= tol,
                axis=2
            )

            ys, xs = np.where(mask)

            if len(xs) == 0:
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
            mask_dir = str(img_path).split("images")[0] + '/labels/'
            landmarks_dir = str(img_path).split("images")[0] + '/landmarks/'

            mask_path = mask_dir + img_stem + ".png"
            lm_path = landmarks_dir + img_stem + ".txt"

            # --------------------------------------------------
            # Create a sample
            # --------------------------------------------------
            sample = {
                "image": img_path,
                "mask_path": mask_path
            }

            if self.landmarks is True:
                if not Path(lm_path).exists():
                    raise FileNotFoundError(
                        f"Missing landmarks: {lm_path}"
                    )
                else:
                    sample["lm_path"] = lm_path

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
        orig_h, orig_w = image.shape[:2]

        # --------------------------------------------------
        # Load the mask images
        # --------------------------------------------------

        if sample["mask_path"] == self.maya_root_dir:   # This is a Maya sample, load the masks from the Maya dataset

            img_id = sample['image'].stem.split("_")[-1]

            # Load Maya masks
            ul_mask_path = self.maya_root_dir / "ul_masks" / f"ul_{img_id}.png"
            ll_mask_path = self.maya_root_dir / "ll_masks" / f"ll_{img_id}.png"

            ul_mask = cv2.imread(str(ul_mask_path), cv2.IMREAD_GRAYSCALE)
            ll_mask = cv2.imread(str(ll_mask_path), cv2.IMREAD_GRAYSCALE)

            if ul_mask is None:
                raise RuntimeError(f"Failed to load upper lip mask: {ul_mask_path}")

            if ll_mask is None:
                raise RuntimeError(f"Failed to load lower lip mask: {ll_mask_path}")

            # Convert to binary masks
            ul_mask = (ul_mask > 127).astype(np.float32)
            ll_mask = (ll_mask > 127).astype(np.float32)

            mask = np.stack([ul_mask, ll_mask], axis=0)

        else:   # This is a LaPa sample, load the masks from the LaPa dataset

            mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_UNCHANGED)

            if mask is None:
                raise RuntimeError(f"Failed to load label: {str(sample["mask_path"])}")

            # Create binary masks of upper lip and lower lip
            ul_mask = (mask == 7).astype(np.float32)
            ll_mask = (mask == 9).astype(np.float32)

            mask = np.stack([ul_mask,ll_mask], axis=0,)

        # --------------------------------------------------
        # Load the mouth landmarks
        # --------------------------------------------------
        landmarks = None
        lip_corners = None
        meaningful_landmarks = None

        if self.landmarks is True: 

            if sample["mask_path"] == self.maya_root_dir:     # This is a Maya sample, load the landmarks from two separate landmark images
                with open(sample["lm_path"], "r") as f:
                    lines = [line.strip() for line in f.readlines()]

                landmarks = {}

                # Skip header
                for line in lines[1:]:

                    if not line:
                        continue

                    landmark_id, x, y = line.split(",")

                    landmarks[int(landmark_id)] = [
                        float(x),
                        float(y)
                    ]
                
                lip_corners = np.array(
                    [
                        landmarks[landmark_id]
                        for landmark_id in self.maya_lip_corner_ids
                    ],
                    dtype=np.float32
                )

                meaningful_landmarks = np.array(
                    [
                        landmarks[landmark_id]
                        for landmark_id in self.maya_meaningful_lm_ids
                    ],
                    dtype=np.float32
                )

                # Normalize landmark coordinates to [0, 1] range based on original image dimensions
                lip_corners[:,0] = lip_corners[:,0] / (orig_w - 1)
                lip_corners[:,1] = lip_corners[:,1] / (orig_h - 1)
                meaningful_landmarks[:,0] = meaningful_landmarks[:,0] / (orig_w - 1)
                meaningful_landmarks[:,1] = meaningful_landmarks[:,1] / (orig_h - 1)

                print("\nIdx: ", idx, ", Type: MAYA")
                print("lip_corner for idx ", idx, lip_corners)
                print("meaningful_landmarks for idx ", idx, meaningful_landmarks)

            else:   # This is a LaPa sample, load the landmarks directly from the .txt file
                mouth_landmark_ids = list(range(84, 104))

                with open(sample["lm_path"], "r") as f:
                    lines = [line.strip() for line in f.readlines()]

                landmarks = []

                for landmark_id in mouth_landmark_ids:
                    line = lines[landmark_id + 1]

                    x, y = map(float, line.split())

                    landmarks.append([x, y])

                landmarks = np.array(
                    landmarks,
                    dtype=np.float32
                )

                if len(landmarks) != len(mouth_landmark_ids):
                    raise RuntimeError(
                        f"Expected {len(mouth_landmark_ids)} landmarks but "
                        f"found {len(landmarks)} in {sample["lm_path"]}"
                    )

                lip_corners = np.array(
                    [
                        landmarks[
                            landmark_id - 84
                        ]
                        for landmark_id in self.lapa_lip_corner_ids
                    ],
                    dtype=np.float32
                )

                meaningful_landmarks = np.array(
                    [
                        landmarks[
                            landmark_id - 84
                        ]
                        for landmark_id in self.lapa_meaningful_lm_ids
                    ],
                    dtype=np.float32
                )

                # Normalize landmark coordinates to [0, 1] range based on original image dimensions
                lip_corners[:,0] = lip_corners[:,0] / (orig_w - 1)
                lip_corners[:,1] = lip_corners[:,1] / (orig_h - 1)
                meaningful_landmarks[:,0] = meaningful_landmarks[:,0] / (orig_w - 1)
                meaningful_landmarks[:,1] = meaningful_landmarks[:,1] / (orig_h - 1)

                print("\nIdx: ", idx, ", Type: LAPA")
                print("lip_corner for idx ", idx, lip_corners)
                print("meaningful_landmarks for idx ", idx, meaningful_landmarks)
                    
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

                # if meaningful_landmarks is not None:
                #     sx = w/orig_w
                #     sy = h/orig_h

                #     meaningful_landmarks[:, 0] *= sx
                #     meaningful_landmarks[:, 1] *= sy

                # if lip_corners is not None:
                #     sx = w/orig_w
                #     sy = h/orig_h

                #     lip_corners[:, 0] *= sx
                #     lip_corners[:, 1] *= sy
        
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

        instance = {
            "image": image,
            "mask": mask,
        } 


        if self.landmarks is True:
            if (lip_corners is not None) and (meaningful_landmarks is not None):
                lip_corners = torch.from_numpy(lip_corners).float()
                meaningful_landmarks = torch.from_numpy(meaningful_landmarks).float()
                instance["lip_corners"] = lip_corners
                instance["meaningful_landmarks"] = meaningful_landmarks

        return instance


    def __len__(self):
        return len(self.all_samples)   
    
# ------------ SCRIPT FOR TESTING DATASET ------------
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

lapa_root_dir = "/home/quinnm/.cache/downloaded_datasets/LaPa"
maya_root_dir = "datasets/MayaFaces"

image_size = (1024, 1024)

hybrid_dataset = HybridDataset(
    lapa_root_dir,
    maya_root_dir,
    maya_split=0.9,
    image_size=image_size,
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

# # --------------------------------------------------
# # Visualize first sample in batch
# # --------------------------------------------------

for idx in range(batch["image"].shape[0]):
    # CHW -> HWC
    image = batch["image"][idx].permute(1, 2, 0).numpy()
    h, w = image.shape[:2]

    # Get landmarks if available
    lip_corners = None
    meaningful_landmarks = None
    if batch.get("lip_corners") is not None and batch.get("meaningful_landmarks") is not None:
        lip_corners = (
            batch["lip_corners"][idx]
            .numpy()
        )

        meaningful_landmarks = (
            batch["meaningful_landmarks"][idx]
            .numpy()
        )

    # [2, H, W]
    mask = batch["mask"][idx].numpy()

    # # Create visualization mask
    # upper_lip = mask[0]
    # lower_lip = mask[1]

    upper_lip = mask[0].astype(bool)
    lower_lip = mask[1].astype(bool)

    # Assign overlapping pixels to the upper lip
    lower_lip = lower_lip & (~upper_lip)

    # Convert back to float for visualization if needed
    upper_lip = upper_lip.astype(np.float32)
    lower_lip = lower_lip.astype(np.float32)

    mask_vis = np.zeros(
        (mask.shape[1], mask.shape[2], 3),
        dtype=np.float32
    )

    # Red = upper lip
    mask_vis[..., 0] = upper_lip

    # Green = lower lip
    mask_vis[..., 1] = lower_lip

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    overlay = image.copy()

    alpha = 0

    overlay = (
        (1 - alpha) * overlay
        + 0.5 * mask_vis
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    # Save the overlay photo for paper writing
    # Convert from float [0,1] to uint8 [0,255]
    output_dir = "visuals_for_papers"
    overlay_uint8 = (overlay * 255).astype(np.uint8)

    # Convert RGB -> BGR for OpenCV
    overlay_bgr = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)

    # Save
    output_path = os.path.join(output_dir, f"overlay_{idx}.png")
    cv2.imwrite(output_path, overlay_bgr)
    print(f"Saved overlay to: {output_path}")

    fig, ax = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(mask_vis)
    ax[1].set_title(
        "Masks\n(Red=Upper, Green=Lower)"
    )
    ax[1].axis("off")

    ax[2].imshow(overlay)
    ax[2].set_title("Overlay")
    ax[2].axis("off")


    ax[3].imshow(image)
    ax[3].axis("off")

    if lip_corners is not None and meaningful_landmarks is not None:

        # Lip corners
        ax[3].scatter(
            lip_corners[:, 0] * w,
            lip_corners[:, 1] * h,
            s=50,
            marker="o",
            label="Lip Corners"
        )

        # Meaningful landmarks
        ax[3].scatter(
            meaningful_landmarks[:, 0] * w,
            meaningful_landmarks[:, 1] * h,
            s=50,
            marker="x",
            label="Meaningful Landmarks"
        )

        ax[3].set_title(
            "Lip Landmarks"
        )

        ax[3].legend()

    plt.tight_layout()
    plt.show()


