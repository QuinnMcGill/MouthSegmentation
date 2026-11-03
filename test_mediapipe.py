import cv2
import os
from tqdm import tqdm
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import argparse

import mediapipe_resources.mp_utils as mp_utils

import refine_landmarks.visualization_utils as vis_utils

# ==== Configuration ==== #
INPUT_VIDEO_PATH = "mocapvids/"
OUTPUT_PATH_ROOT = "mediapipe_resources/baseline_mp_results/"

# ==== Parse command-line arguments ==== #
parser = argparse.ArgumentParser(description="Running Mediapipe Face Landmarker on Performance Videos")
parser.add_argument("--v", help=".mp4 file in mocap_vids directory", default="tc6.mp4")
parser.add_argument('--mp_input', type=str, default='gray', choices=['gray', 'colour'],
                    help="Frames given to MediaPipe: 'gray' (avoids lipstick interference) or 'colour'")

args = parser.parse_args()

# ==== Load the input video and prepare output video writer ==== #
tc_name = args.v.split(".mp4")[0]
input_video_path = INPUT_VIDEO_PATH + args.v
output_video_path = OUTPUT_PATH_ROOT + tc_name + "_mediapipe.mp4"

# Get input video properties
cap = cv2.VideoCapture(input_video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create MP4 writer
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps,
    (frame_width, frame_height)
)

# ==== Tracking with MediaPipe ==== #
with mp_utils.initialize_mp() as mp_detector:
    mp_results_list = mp_utils.run_mp_for_video(input_video_path, mp_detector, grayscale=(args.mp_input == "gray"))

max_dim = max(frame_height, frame_width)
large_point_radius = round(max_dim*vis_utils.LARGE_POINT_RADIUS_RATIO)
small_point_radius = round(max_dim*vis_utils.SMALL_POINT_RADIUS_RATIO)

frame_idx = 0
with tqdm(total=num_frames, desc="Creating Video of Refined Lip Landmarks") as pbar:

    while cap.isOpened():
        # cv2 reads frames as BGR, which is what the video writer expects (no colour conversion needed)
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = frame.copy()

        # Draw the refined lip landmarks (frames past the end of the results are written unchanged)
        if frame_idx < len(mp_results_list):
            output_frame = vis_utils.draw_face_mesh(output_frame, mp_results_list[frame_idx])
            output_frame = vis_utils.draw_lip_contours(output_frame, mp_results_list[frame_idx], contour_colour=(255, 0, 0), contour_thickness=6)
            output_frame = vis_utils.add_lms_to_frame(output_frame, mp_results_list[frame_idx],
                                                    only_lips=True, point_radius=large_point_radius,
                                                    bgr_lm_colour=(255, 255, 255), draw_indices=False)
            output_frame = vis_utils.add_lms_to_frame(output_frame, mp_results_list[frame_idx],
                                                    only_lips=False, point_radius=small_point_radius,
                                                    bgr_lm_colour=(255, 255, 255), draw_indices=False)

        video_writer.write(output_frame)
        frame_idx += 1

        pbar.update(1)

cap.release()
video_writer.release()
print(f"Saved video to: {output_video_path}")