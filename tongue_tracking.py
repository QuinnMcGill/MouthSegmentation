# System imports
import os
join = os.path.join
import sys
import argparse
sys.path.append("external/TongueSAM")  # adjust path as needed

# Computation and data handling
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, jaccard_score

# Visualization and image processing
from PIL import ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, io, segmentation
import monai
import cv2

# TongueSAM imports
from external.TongueSAM.segment_anything import SamPredictor, sam_model_registry
from external.TongueSAM.segment_anything.utils.transforms import ResizeLongestSide
from external.TongueSAM.utils.SurfaceDice import compute_dice_coefficient
from  external.TongueSAM.utils_metrics import *
from external.TongueSAM.segment.yolox import YOLOX

# set seeds
torch.manual_seed(2023)
np.random.seed(2023)
import random
import warnings

## ---- Argument Parsing ---- ##
parser = argparse.ArgumentParser(description='Testing out TongueSAM for tongue segmentation')
parser.add_argument('--v', '--video_filename', type=str, default='Testcase3.mp4', help='.mp4 filename of test video in mocapvids directory')
args = parser.parse_args()

## ---- Initialize TongueSAM model for tongue segmentation ---- ##
ts_img_path = './data/test_in/'
model_type = 'vit_b'
checkpoint = 'external/TongueSAM/pretrained_model/tonguesam.pth'
device = 'cuda:0'
path_out='./data/test_out/'
segment=YOLOX(classes_path='external/TongueSAM/segment/tongue_classes.txt', model_path='./external/TongueSAM/segment/yolox.pth')

sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
sam_model.eval()
print("Model loaded successfully.")

## ---- Load the test video ---- ##
video_path = os.path.join('./mocapvids/', args.v)
cap = cv2.VideoCapture(video_path)

# Control flags
out = None
save_video = True

if save_video:
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(
        'output.mp4',
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (400, 400)
    )

if save_video and out is not None and not out.isOpened():
    raise RuntimeError("Failed to open video writer")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached or cannot read the video.")
        break

    with torch.no_grad():                 
        if frame.shape[-1] > 3 and len(frame.shape) == 3:
            frame = frame[:, :, :3]
        if len(frame.shape) == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=-1)
        
        # ---- Preprocess ---- #
        lower_bound, upper_bound = np.percentile(frame, 0.5), np.percentile(frame, 99.5)
        image_data_pre = np.clip(frame, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
        image_data_pre[frame == 0] = 0
        
        img = cv2.resize(image_data_pre.astype(np.uint8), (400, 400))
        
        # ---- SAM encoding ---- #
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image_data_pre)

        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
        input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])     

        image_embedding = sam_model.image_encoder(input_image)

        # ---- Prompt (bounding box) ---- #
        boxes = segment.get_prompt(img)
                
        if boxes is None:
            # No tongue detected → just show frame
            overlay = img.copy()          
        else:
            box = sam_transform.apply_boxes(boxes, (400, 400))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)

            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )

            # ---- Segmentation ---- #
            seg_prob, _ = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            seg_prob = seg_prob.cpu().numpy().squeeze()
            seg_mask = (seg_prob > 0.5).astype(np.uint8)

            seg_mask = cv2.resize(
                seg_mask.astype(np.uint8),
                (400, 400),
                interpolation=cv2.INTER_NEAREST
            )

            # ---- Confidence check ---- #
            mask_area = np.sum(seg_mask)

            if mask_area < 500:  # ADJUST AS NEEDED:
                # Too small -> likely no tongue
                overlay = img.copy()
            else:
                # ---- Overlay mask ---- #
                overlay = img.copy()

                # Blue mask
                overlay[seg_mask == 1] = (
                    0.6 * overlay[seg_mask == 1] +
                    0.4 * np.array([255, 0, 0])
                )

                # ---- Draw edges ---- #
                edges = cv2.Canny(seg_mask * 255, 100, 200)
                overlay[edges != 0] = [0, 0, 255]

                # ---- Draw bounding box ---- #
                x1, y1, x2, y2 = map(int, boxes)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # ---- Show video ---- #
        cv2.imshow("Tongue Segmentation", overlay.astype(np.uint8))

        if save_video:
            if out is not None:
                out.write(overlay.astype(np.uint8))

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()