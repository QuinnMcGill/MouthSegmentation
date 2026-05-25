import os
import random
import numpy as np
import cv2
import torch
import pandas as pd

from torch.utils.data import DataLoader

from sam2.build_sam import build_sam2                       # type: ignore               
from sam2.sam2_image_predictor import SAM2ImagePredictor    # type: ignore

from datasets.lapa_dataset import LaPaLipDataset

# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda"

DATASET_ROOT = "/home/quinnm/.cache/downloaded_datasets/LaPa"
DATASET_NAME = DATASET_ROOT.split("/")[-1].lower()

SAM2_CHECKPOINT = "external/sam2/checkpoints/sam2.1_hiera_small.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM_MODEL = MODEL_CFG.split("/")[-1].split("_")[0] + MODEL_CFG.split("/")[-1].split(".yaml")[0][-1]

BATCH_SIZE = 4
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 4e-5
JITTER_VAL = 10

MAX_EPOCHS = 10

IMAGE_SIZE = (1024, 1024)

SAVE_DIR = "finetuned_weights"

os.makedirs(SAVE_DIR, exist_ok=True)
print("All configurations set. Starting training...")

# ============================================================
# Datasets
# ============================================================

train_dataset = LaPaLipDataset(
    root_dir=DATASET_ROOT,
    split="train",
    image_size=IMAGE_SIZE,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=False,
)

val_dataset = LaPaLipDataset(
    root_dir=DATASET_ROOT,
    split="val",
    image_size=IMAGE_SIZE,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# ============================================================
# Build SAM2
# ============================================================

sam2_model = build_sam2(
    MODEL_CFG,
    SAM2_CHECKPOINT,
    device=DEVICE,
)

predictor = SAM2ImagePredictor(sam2_model)


# ============================================================
# Training Mode
# ============================================================

predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)

# Freeze image encoder initially
predictor.model.image_encoder.eval()

# OPTIONAL:
# Uncomment to finetune full encoder
#
# predictor.model.image_encoder.train(True)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    list(predictor.model.sam_mask_decoder.parameters()) +
    list(predictor.model.sam_prompt_encoder.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu") # type: ignore


# ============================================================
# Utility Functions
# ============================================================

def mask_to_box(mask, padding=5):
    """
    Convert binary mask to XYXY box.
    """

    coords = np.argwhere(mask > 0)

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    x1 = max(0, x_min - padding)
    y1 = max(0, y_min - padding)

    x2 = x_max + padding
    y2 = y_max + padding

    return [x1, y1, x2, y2]


def jitter_box(box, image_shape, jitter=10):
    """
    Randomly perturb box coordinates.
    """

    x1, y1, x2, y2 = box

    h, w = image_shape[:2]

    x1 += np.random.randint(-jitter, jitter + 1)
    y1 += np.random.randint(-jitter, jitter + 1)

    x2 += np.random.randint(-jitter, jitter + 1)
    y2 += np.random.randint(-jitter, jitter + 1)

    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)

    x2 = np.clip(x2, 0, w - 1)
    y2 = np.clip(y2, 0, h - 1)

    return [x1, y1, x2, y2]


def dice_loss(pred, target, smooth=1e-6):
    """
    Soft Dice Loss.
    """

    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    intersection = (pred * target).sum(dim=1)

    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()


@torch.no_grad()
def validate(predictor, val_loader):

    predictor.model.eval()

    ious = []

    for batch in val_loader:

        images = batch["image"]
        masks = batch["mask"]

        for b in range(images.shape[0]):

            image = images[b]
            mask_stack = masks[b]

            for cls_idx in [0, 1]:

                gt_mask = mask_stack[cls_idx].numpy()

                if gt_mask.sum() == 0:
                    continue

                image_np = (
                    image.permute(1, 2, 0).numpy() * 255
                ).astype(np.uint8)

                box = mask_to_box(gt_mask)

                if box is None:
                    continue

                predictor.set_image(image_np)

                masks_pred, scores, _ = predictor.predict(
                    box=np.array(box),
                    multimask_output=False,
                )

                pred = masks_pred[0].astype(np.float32)

                inter = (pred * gt_mask).sum()

                union = (
                    pred.sum()
                    + gt_mask.sum()
                    - inter
                )

                iou = inter / (union + 1e-6)

                ious.append(iou)

    predictor.model.train()

    return np.mean(ious)


def save_results_to_csv(
    training_params,
    results,
    filename="finetuning/finetuning_results.csv",
):
    """
    Append finetuning experiment results to a CSV file.

    Args:
        training_params (dict):
            Dictionary of hyperparameters/settings.

        results (dict):
            Dictionary of evaluation metrics/results.

        filename (str):
            Output CSV path.
    """

    # --------------------------------------------------------
    # Merge dictionaries into single experiment entry
    # --------------------------------------------------------

    row_data = {
        **training_params,
        **results,
    }

    # Convert to one-row DataFrame
    df = pd.DataFrame([row_data])

    # --------------------------------------------------------
    # Append if file exists
    # --------------------------------------------------------

    if os.path.exists(filename):

        existing_df = pd.read_csv(filename)

        updated_df = pd.concat(
            [existing_df, df],
            ignore_index=True,
        )

        updated_df.to_csv(filename, index=False)

    else:

        df.to_csv(filename, index=False)

    print(f"Results saved to: {filename}")

# ============================================================
# Training Loop
# ============================================================

global_step = 0
mean_iou = 0.0
best_val_iou = 0.0

# File to save best model checkpoint
checkpoint_path = os.path.join(
                    SAVE_DIR,
                    f"best_{SAM_MODEL}_{DATASET_NAME}.pt"
                )

if os.path.exists(checkpoint_path):
    ckpt_idx = 1
    while os.path.exists(os.path.join(SAVE_DIR, f"best_{SAM_MODEL}_{DATASET_NAME}_v{ckpt_idx}.pt")):
        ckpt_idx += 1
    checkpoint_path = os.path.join(SAVE_DIR, f"best_{SAM_MODEL}_{DATASET_NAME}_v{ckpt_idx}.pt")

for epoch in range(MAX_EPOCHS):

    print(f"\n===== EPOCH {epoch} =====")

    epoch_iou_sum = 0.0
    epoch_iou_count = 0

    for batch in train_loader:

        images = batch["image"]
        masks = batch["mask"]

        batch_images = []
        batch_masks = []
        batch_boxes = []

        # ----------------------------------------------------
        # Build SAM-style training batch
        # ----------------------------------------------------

        for b in range(images.shape[0]):

            image = images[b]
            mask_stack = masks[b]

            # Randomly choose:
            # 0 = upper lip
            # 1 = lower lip

            cls_idx = random.randint(0, 1)

            mask = mask_stack[cls_idx]

            mask_np = mask.numpy()

            # Skip empty masks
            if mask_np.sum() == 0:
                continue

            # Convert image tensor -> uint8 numpy
            image_np = (
                image.permute(1, 2, 0).numpy() * 255
            ).astype(np.uint8)

            # Create bbox prompt
            box = mask_to_box(mask_np)

            if box is None:
                continue

            # Add random perturbation
            box = jitter_box(
                box,
                image_np.shape,
                jitter=JITTER_VAL,
            )

            batch_images.append(image_np)
            batch_masks.append(mask_np)
            batch_boxes.append(box)

        # Skip invalid batches
        if len(batch_images) == 0:
            continue

        input_boxes = np.array(batch_boxes)

        gt_mask = torch.tensor(
            np.array(batch_masks),
            dtype=torch.float32,
            device=DEVICE,
        )

        # ====================================================
        # Forward Pass
        # ====================================================

        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"): # type: ignore

            predictor.set_image_batch(batch_images)

            (
                mask_input,
                unnorm_coords,
                labels,
                unnorm_box,
            ) = predictor._prep_prompts(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                mask_logits=None,
                normalize_coords=True,
            )

            sparse_embeddings, dense_embeddings = (
                predictor.model.sam_prompt_encoder(
                    points=None,
                    boxes=unnorm_box,
                    masks=None,
                )
            )

            high_res_features = [
                feat_level[-1].unsqueeze(0)
                for feat_level in predictor._features["high_res_feats"]
            ]

            low_res_masks, prd_scores, _, _ = (
                predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"],
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features,
                )
            )

            # Upscale masks
            prd_masks = predictor._transforms.postprocess_masks(
                low_res_masks,
                predictor._orig_hw[-1],
            )

            # Probability masks
            prd_logits = prd_masks[:, 0]
            prd_mask = torch.sigmoid(prd_logits)

            # ====================================================
            # BCE Segmentation Loss
            # ====================================================

            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                prd_logits,
                gt_mask,
            )

            # ====================================================
            # Dice Loss
            # ====================================================

            d_loss = dice_loss(
                prd_mask,
                gt_mask,
            )

            seg_loss = bce_loss + d_loss

            # ====================================================
            # IoU Calculation
            # ====================================================

            pred_binary = (prd_mask > 0.5).float()

            inter = (
                pred_binary * gt_mask
            ).sum(dim=(1, 2))

            union = (
                pred_binary.sum(dim=(1, 2))
                + gt_mask.sum(dim=(1, 2))
                - inter
            )

            iou = inter / (union + 1e-6)

            # ====================================================
            # Score Loss
            # ====================================================

            score_loss = torch.abs(
                prd_scores[:, 0] - iou
            ).mean()

            # ====================================================
            # Final Loss
            # ====================================================

            loss = seg_loss + 0.05 * score_loss

        # ====================================================
        # Backpropagation
        # ====================================================

        optimizer.zero_grad()

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        # ====================================================
        # Metrics
        # ====================================================

        batch_iou = np.mean(
            iou.detach().cpu().numpy()
        )

        epoch_iou_sum += batch_iou
        epoch_iou_count += 1

        mean_iou = epoch_iou_sum / epoch_iou_count

        print(
            f"Step {global_step} | "
            f"Loss: {loss.item():.4f} | "
            f"BCE: {bce_loss.item():.4f} | "
            f"Dice: {d_loss.item():.4f} | "
            f"Mean IoU: {mean_iou:.4f}"
        )

        global_step += 1

    # ====================================================
    # Run Validation and Save New Checkpoints
    # ====================================================

    val_iou = validate(
        predictor,
        val_loader,
    )

    print(f"Validation IoU: {val_iou:.4f}")

    if val_iou > best_val_iou:

        best_val_iou = val_iou

        torch.save(
            {
                "model_state_dict":
                    predictor.model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "step": global_step,

                "best_val_iou": best_val_iou,
            },
            checkpoint_path,
        )

    print(
        f"New best model saved! "
        f"Validation IoU = {best_val_iou:.4f}"
    )

print("Training complete.")

# ====================================================
# Save the Finetuning Results to the csv
# ====================================================

training_params = {
    "dataset": DATASET_ROOT.split("/")[-1],
    "base_model": SAM_MODEL,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "epochs": MAX_EPOCHS,
    "image_size": IMAGE_SIZE,
    "finetuned_weights": checkpoint_path,
    "bbox_jitter": JITTER_VAL,
}

results = {
    "final_mean_train_iou": mean_iou,
    "best_mean_val_iou": best_val_iou,
    "notes": "",
}

save_results_to_csv(
    training_params,
    results,
)