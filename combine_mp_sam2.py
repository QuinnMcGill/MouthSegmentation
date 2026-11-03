# ====== Step 0: Import Libraries ====== #

# System
import os
import argparse
from tqdm import tqdm
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Computation and data handling
import numpy as np
import torch

# Visualization and image processing
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# SAM2
from sam2.build_sam import build_sam2_video_predictor           #type:ignore   
from sam2.sam2_video_predictor import SAM2VideoPredictor        #type:ignore   
from sam2_resources import sam2_utils

# MediaPipe
import mediapipe_resources.mediapipe_utils as mediapipe_utils
import mediapipe_resources.blendshape_model as blendshape_model

# Processing SAM2 Masks and refineing Mediapipe landmarks
import refine_landmarks.refinement_utils as refinement_utils
import refine_landmarks.data_manager as data_manager
import sam2_resources.sam2_lip_seg_processor as sam2_lip_seg_processor
import refine_landmarks.visualization_utils as vis_utils

# ====== Step 1: Setup the environment ====== #

# Parse the command line arguments
parser = argparse.ArgumentParser(description='Refine MediaPipe mouth landmarks using SAM2 segmentation masks')
parser.add_argument('--v', '--video_filename', type=str, default='tc6.mp4', help='.mp4 filename of test video in mocapvids directory')
parser.add_argument('--fw', '--finetuned_weights', type=str, required=True, help="Name of weights file in finetuned_weights folder")
parser.add_argument('--npp', '--num_prompts_per_part', type=int, default=5, help='Number of prompts to add for each mouth part (integer)')
parser.add_argument('--debug', '--debug_stage', type=int, default=0, help='0: doe not use saved stage, 1: stage 1 saved state, 2: stage 2 saved state')
parser.add_argument('--mp_input', type=str, default='gray', choices=['gray', 'colour'],
                    help="Frames given to MediaPipe: 'gray' (avoids lipstick interference) or 'colour'")
args = parser.parse_args() 

# Get the row index of the results for these finetuned weights
ft_res_df = pd.read_csv("finetuning_sam2/finetuning_results.csv")
weights_path = os.path.join("finetuning_sam2", "finetuned_weights", args.fw)

# Find the target finetuned weights and populate the base model and dataset variables
if weights_path not in ft_res_df['finetuned_weights'].values:
    raise ValueError(f"Finetuned weights '{args.fw}' not found in finetuning_results.csv")
else:
    fw_idx = ft_res_df.index[ft_res_df['finetuned_weights'] == weights_path][0]
    base_model = ft_res_df.loc[fw_idx, 'base_model']
    base_model_letter = str(base_model)[-1]
    dataset = str(ft_res_df.loc[fw_idx, 'dataset']).lower()
    mp_input_suffix = "" if args.mp_input == "colour" else "_mpgray"
    run_tag = f"{args.v.split('.mp4')[0]}_{dataset}_sam2{base_model_letter}_fw{fw_idx}_{args.npp}pp{mp_input_suffix}"
    print(f"Finetuned weights '{args.fw}' found in finetuning_results.csv at row {fw_idx} (base model: {base_model}, dataset: {dataset})")

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# ====== Step 2: Initialize SAM2, MediaPipe Blendshape Predictor ====== #

sam2_predictor = sam2_utils.initialize_sam2(base_model, device, weights_path)
bs_predictor = blendshape_model.BlendshapePredictor("mediapipe_resources/face_landmarker.task")

# ====== Step 3: Load the video ====== #

# `vid_frames_dir` is a directory of JPEG frames with filenames like `<frame_index>.jpg`
tc_name = args.v.split('.mp4')[0]
vid_frames_dir = os.path.join("input/vid_frames", os.path.splitext(args.v)[0])

# Load the original video to get the correct frame rate for the output video
input_vid_path = "input/input_vids/" + args.v
cap = cv2.VideoCapture(input_vid_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {input_vid_path}")

# Get input frame rate (fps) to use for the output video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.release()

# ====== Step 4: Prompt the SAM2 model and get MediaPipe predictions ====== #

# ------ Generate Prompts for SAM2 Segmentation ------ #
save_path = f"saved_states/stage1/{run_tag}_mp_results.npz" 

if args.debug == 0:    # Generate new prompts (no saved state)

    # ------ Run SAM2 Lip Segmentation ------ #
    # Prompt the model with bounding boxes and prompt points for the upper and lower lip
    sam2_predictor, inference_state = sam2_utils.run_sam2_lip_seg(num_parts_per_object=args.npp, 
                                                                  vid_frames_dir=vid_frames_dir, 
                                                                  sam2_predictor=sam2_predictor)
    
    # Propagate predictions between frames to achieve VOS of lips
    (upper_indices, lower_indices, 
    upper_offsets, lower_offsets) = sam2_utils.create_lip_masks(sam2_predictor=sam2_predictor, inference_state=inference_state)

    # ------ Run Mediapipe Face Landmarker ------ #
    with mediapipe_utils.initialize_mp() as mp_detector:
        mp_results_list = mediapipe_utils.run_mp_for_video(os.path.join("mocapvids", args.v), mp_detector, grayscale=(args.mp_input == "gray"))
    mp_lip_coords = mediapipe_utils.extract_lip_landmarks_coords(mp_results_list)

    data_manager.save_state_stage1(upper_indices, lower_indices, upper_offsets, lower_offsets, mp_results_list, mp_lip_coords, save_path)
else:               # Use a saved state
    upper_indices, lower_indices, upper_offsets, lower_offsets, mp_results_list, mp_lip_coords = data_manager.load_saved_state_stage1(save_path)

# ------ Blendshapes BEFORE SAM2 refinement ------ #
bs_dir = "blendshape_csvs"
os.makedirs(bs_dir, exist_ok=True)

bs_before_df = bs_predictor.create_bs_df(mp_results_list, frame_width, frame_height)
bs_before_path = os.path.join(bs_dir, f"{run_tag}_mp_blendshapes.csv")
bs_before_df.to_csv(bs_before_path)
print(f"Saved pre-refinement blendshapes to: {bs_before_path}")

# ====== Step 5: Refine MediaPipe Predictions with SAM2 results ====== #

# Save create a copy of the original MediaPipe results to save refined results
sam2_mp_results = mp_results_list.copy()

# Process the SAM2 results (corner prediction, contour smoothing, etc.) for the entire clip.
sam2_contours, sam2_corners = sam2_lip_seg_processor.get_sam2_contours_and_corners(save_path, 
                                                                        frame_height=frame_height, 
                                                                        frame_width=frame_width)

# Refinement Steps:
#   1. Replace MediaPipe lip contours with SAM2-derived lip corners
#   2. Project each original MediaPipe landmark onto the SAM2-derived lip contours.
#   3. Save the refined landmarks into sam2_mp_results
with tqdm(total=len(mp_lip_coords), desc="Refining MediaPipe Lip Landmarks with SAM2 Masks") as pbar:

    # Loop through each frame in the video
    for frame_idx in range(len(mp_lip_coords)):

        # Create a copy of the Mediapipe landmarks that will be updated
        frame_landmarks = sam2_mp_results[frame_idx]
        original_frame_landmarks = frame_landmarks.copy()

        # Create mp lookup table for quicker MediaPipe landmark look-up
        mp_lookup_table = refinement_utils.build_landmark_lookup(mp_lip_coords[frame_idx], frame_width, frame_height)

        # Grab the SAM2 contours and corners for this frame
        sam2_contours_frame, sam2_corners_frame = sam2_contours[frame_idx], sam2_corners[frame_idx]

        for obj_id in [sam2_utils.UPPER_LIP_OBJ_ID, sam2_utils.LOWER_LIP_OBJ_ID]:

            # Call to function in refinement_utils.py to generate refined MediaPipe landmarks (excluding
            # lip corners, since this should come from the SAM2 masks) 
            upper_points, lower_points = refinement_utils.generate_sam2_contour_landmarks(obj_id, mp_lookup_table,
                                                                            sam2_contours_frame, sam2_corners_frame)

            # Replace the original Mediapipe ladnmarks with these SAM2-guided points
            frame_landmarks = mediapipe_utils.replace_mp_lip_landmarks(frame_landmarks, obj_id, upper_points,
                                                                lower_points, frame_height, frame_width)
        
        # Save the refined landmark results
        sam2_mp_results[frame_idx] = frame_landmarks
        
        pbar.update(1)

# Smooth the refined lip landmarks over consecutive frames (frames where SAM2 failed are left out)
valid = ~np.isnan(sam2_corners).any(axis=(1, 2))
sam2_mp_results = refinement_utils.smooth_refined_landmarks(sam2_mp_results, valid)

# Move the middle lip rings and the landmarks around the lips to follow the refined contours
sam2_mp_results = [refinement_utils.adjust_nearby_landmarks(frame_landmarks, original_frame_landmarks,
                                                            frame_width, frame_height)
                for frame_landmarks, original_frame_landmarks in zip(sam2_mp_results, mp_results_list)]

# ------ Blendshapes AFTER SAM2 refinement ------ #
bs_after_df = bs_predictor.create_bs_df(sam2_mp_results, frame_width, frame_height)
bs_after_path = os.path.join(bs_dir, f"{run_tag}_sam2_mp_blendshapes.csv")
bs_after_df.to_csv(bs_after_path)
print(f"Saved post-refinement blendshapes to: {bs_after_path}")

# ====== Step 6: Create Segmentation Video ====== #

# Output video path
output_video_dir = f"output/sam2_guided_mp_results/"
output_video_path = os.path.join(output_video_dir, f"fw{fw_idx}_{args.v}")

fourcc = cv2.VideoWriter.fourcc(*'mp4v')
video_writer = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps,
    (frame_width, frame_height)
)

max_dim = max(frame_height, frame_width)
large_point_radius = round(max_dim*vis_utils.LARGE_POINT_RADIUS_RATIO)
small_point_radius = round(max_dim*vis_utils.SMALL_POINT_RADIUS_RATIO)

# Load the input video file
cap = cv2.VideoCapture(input_vid_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {input_vid_path}")

frame_idx = 0

with tqdm(total=len(mp_lip_coords), desc="Creating Video of Refined Lip Landmarks") as pbar:

    while cap.isOpened():
        # cv2 reads frames as BGR, which is what the video writer expects (no colour conversion needed)
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = frame.copy()

        # Draw the refined lip landmarks (frames past the end of the results are written unchanged)
        if frame_idx < len(sam2_mp_results):
            output_frame = vis_utils.draw_face_mesh(output_frame, sam2_mp_results[frame_idx])
            output_frame = vis_utils.draw_lip_contours(output_frame, sam2_mp_results[frame_idx], contour_colour=(255, 0, 0), contour_thickness=6)
            output_frame = vis_utils.add_lms_to_frame(output_frame, sam2_mp_results[frame_idx],
                                                    only_lips=True, point_radius=large_point_radius,
                                                    bgr_lm_colour=(255, 255, 255), draw_indices=False)
            output_frame = vis_utils.add_lms_to_frame(output_frame, sam2_mp_results[frame_idx],
                                                    only_lips=False, point_radius=small_point_radius,
                                                    bgr_lm_colour=(255, 255, 255), draw_indices=False)

        video_writer.write(output_frame)
        frame_idx += 1

        pbar.update(1)

cap.release()
video_writer.release()
print(f"Saved video to: {output_video_path}")