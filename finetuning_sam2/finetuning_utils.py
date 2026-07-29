import torch
import torch.nn.functional as F
from torch.utils.data import random_split


from datasets.lapa_dataset import LaPaLipDataset
from datasets.mayaFaces_dataset import MayaFacesDataset
from datasets.hybrid_dataset import HybridDataset 

from torch.utils.data import DataLoader

def create_dataloaders(dataset_name, dataset_root, image_size, batch_size):
    if dataset_name == "lapa":
        train_dataset = LaPaLipDataset(
            root_dir=dataset_root,
            split="train",
            image_size=image_size,
            lip_corners=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=False,
        )

        val_dataset = LaPaLipDataset(
            root_dir=dataset_root,
            split="val",
            image_size=image_size,
            lip_corners=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

    elif dataset_name == "maya": 
        maya_dataset = MayaFacesDataset(
            root_dir=dataset_root,
            transform=None
        )

        # Assume 80/20 split
        train_size = int(0.8 * len(maya_dataset))
        val_size = len(maya_dataset) - train_size

        train_dataset, val_dataset = random_split(
            maya_dataset,
            [train_size, val_size],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

    else:
        lapa_root_dir = dataset_root[0]
        maya_root_dir = dataset_root[1]

        hybrid_dataset = HybridDataset(
            lapa_root_dir,
            maya_root_dir,
            maya_split=0.3,
            image_size=image_size,
            transform=None,
            landmarks=False, 
        )

        # Assume 80/20 split
        train_size = int(0.8 * len(hybrid_dataset))
        val_size = len(hybrid_dataset) - train_size

        train_dataset, val_dataset = random_split(
            hybrid_dataset,
            [train_size, val_size],
        )

        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
    
    return train_loader, val_loader

def soft_lip_corners(
    prob_masks: torch.Tensor,
    corner_type: str = "upper",
):
    """
    Estimate lip corners from probability masks.

    Args:
        prob_masks: [B,H,W] probability masks in [0,1]
        corner_type:
            "upper" -> bias toward lower edge of mask
            "lower" -> bias toward upper edge of mask

    Returns:
        pred_corners: [B,2,2]
            [:,0,:] = left corner (x,y)
            [:,1,:] = right corner (x,y)
    """

    B, H, W = prob_masks.shape
    device = prob_masks.device

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    xs = xs.unsqueeze(0)  # [1,H,W]
    ys = ys.unsqueeze(0)  # [1,H,W]

    if corner_type == "upper":
        vertical_bias = ys / max(H - 1, 1)

    elif corner_type == "lower":
        vertical_bias = 1.0 - ys / max(H - 1, 1)

    else:
        raise ValueError(
            f"Unknown corner_type: {corner_type}"
        )

    # --------------------------------------------------
    # Left corner
    # --------------------------------------------------

    left_weights = (
        prob_masks
        * (W - xs)
        * vertical_bias
    )

    left_weights = left_weights / (
        left_weights.sum(dim=(1, 2), keepdim=True)
        + 1e-6
    )

    left_x = (
        left_weights * xs
    ).sum(dim=(1, 2))

    left_y = (
        left_weights * ys
    ).sum(dim=(1, 2))

    # --------------------------------------------------
    # Right corner
    # --------------------------------------------------

    right_weights = (
        prob_masks
        * xs
        * vertical_bias
    )

    right_weights = right_weights / (
        right_weights.sum(dim=(1, 2), keepdim=True)
        + 1e-6
    )

    right_x = (
        right_weights * xs
    ).sum(dim=(1, 2))

    right_y = (
        right_weights * ys
    ).sum(dim=(1, 2))

    pred_corners = torch.stack(
        [
            torch.stack([left_x, left_y], dim=1),
            torch.stack([right_x, right_y], dim=1),
        ],
        dim=1,
    )

    return pred_corners


def soft_corner_loss(
    prob_masks: torch.Tensor,
    gt_corners: torch.Tensor,
    corner_type: str = "upper",
):
    """
    Args:
        prob_masks: [B,H,W]
        gt_corners: [B,2,2]
            [[left_x,left_y],
             [right_x,right_y]]

    Returns:
        loss
        pred_corners
    """

    pred_corners = soft_lip_corners(
        prob_masks,
        corner_type=corner_type,
    )

    loss = F.smooth_l1_loss(
        pred_corners,
        gt_corners.float(),
    )

    return loss, pred_corners