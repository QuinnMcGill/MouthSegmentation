import numpy as np
import os 
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2 # type: ignore
from sam2.sam2_image_predictor import SAM2ImagePredictor # type: ignore

import kagglehub

# Download latest version
dataset_path = kagglehub.dataset_download("remainaplomb/lips-segmentation-dataset")

print("Path to dataset files:", dataset_path)
# Lip Dataset location: /home/quinnm/.cache/kagglehub/datasets/remainaplomb/lips-segmentation-dataset/versions/1/Lips_Dataset

# Read the data
data = []
for ff, name in enumerate(os.listdir(dataset_path+"/Lips_Dataset/original")):
    if name.endswith(".png"):
        pic_idx = int(name.split("_")[1].split(".")[0])
        data.append({"image": dataset_path+"/Lips_Dataset/original/"+name, "annotation": dataset_path+"/Lips_Dataset/mask/"+"mask_"+str(pic_idx)+".png"})

def read_batch(data):
    # Select image
    ent = data[np.random.randint(0, len(data))]
    img = cv2.imread(ent["image"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # type: ignore # Convert BGR to RGB
    mask = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)  # Read annotation as grayscale

    # Resize image
    r = np.min([1024/img.shape[1], 1024/img.shape[0]]) # scaling factor
    img = cv2.resize(img, (int(img.shape[1]*r), int(img.shape[0]*r)))

    # Resize annotation map
    mask = cv2.resize(src=mask, dsize=(int(mask.shape[1]*r), int(mask.shape[0]*r)), interpolation=cv2.INTER_NEAREST) # type: ignore

    # Convert to binary mask
    mask = (mask > 0).astype(np.uint8)

    if mask is None:
        print("Failed to load:", ent["annotation"])
        return None, None, None, None

    # Get a random in the foreground (lip) region
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        print("No foregroud pixels found for image:", ent["image"])
        print("Unique values in mask:", np.unique(mask))
        return None, None, None, None
    yx = coords[np.random.randint(len(coords))]
    
    # SAM expects shape:
    # masks -> [N, H, W]
    # points -> [N, 1, 2]

    masks = np.array([mask])
    points = np.array([[[yx[1], yx[0]]]])

    # labels: 1 = positive point
    labels = np.array([[1]])

    return img, masks, points, labels

# Load SAM2 Model
sam2_checkpoint = "external/sam2/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters
predictor.model.sam_mask_decoder.train(True)    # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True)  # enable training of prompt encoder

optimizer = torch.optim.Adam(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu") # type: ignore

# Training Loop
num_iterations = 10000
for it in range(num_iterations):
    with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"): # type: ignore
        img, mask, point, label = read_batch(data)

        if img is None:
            continue

        predictor.set_image(img)

        # Prompt encoding
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(point,label, box=None, 
                                                                                mask_logits=None, 
                                                                                normalize_coords=True)
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None,)

        # Mask decoder
        batched_mode = unnorm_coords.shape[0] > 1   # Multi object prediction
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0), 
                                                                            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                                                                            sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, 
                                                                            multimask_output=True, repeat_image=batched_mode,
                                                                            high_res_features=high_res_features,)
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])  # Upscale the masks to the original image size

        # Segmentation Loss Calculation
        gt_mask  = torch.tensor(mask.astype(np.float32), device=device)
        prd_mask = torch.sigmoid(prd_masks[:,0])  # urn logit map into probability map
        seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1-prd_mask) + 0.00001)).mean()    # Cross Entropy Loss

        # Score Loss Calculation (Intersection over Union) IoU
        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter/(gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss= seg_loss + score_loss*0.05

        # Back propagation
        predictor.model.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if it%1000 == 0:
            torch.save(predictor.model.state_dict(), "finetuned_weights/new_2.1s_lipseg.torch")
            print("Save model at iteration", it)

        # Display some intermediate results
        if it == 0: 
            mean_iou =0

        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        if it%200 == 0:
            print(f"Iteration {it}, Loss: {loss.item():.4f}, Seg Loss: {seg_loss.item():.4f}, Score Loss: {score_loss.item():.4f}, Mean IoU: {mean_iou:.4f}")






        

