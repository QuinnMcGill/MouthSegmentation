import cv2
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import argparse

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import mediapipe_resources.mp_utils as mph

# ==== Configuration ==== #
INPUT_VIDEO_PATH = "mocapvids/"
OUTPUT_PATH_ROOT = "mediapipe_resources/video_outputs/"

# ==== Parse command-line arguments ==== #
parser = argparse.ArgumentParser(description="Messing around with MediaPipe Face Landmarker")
parser.add_argument("--v", help=".mp4 file in mocap_vids directory", default="tc6.mp4")

args = parser.parse_args()

# ==== Variable Initialization ==== #
# ---- Control paramters ---- #
frame_idx = 0

# ---- Variables for output video ---- #
tc_name = args.v.split(".mp4")[0]
output_video_path = OUTPUT_PATH_ROOT + tc_name + "_mediapipe.mp4"

# ==== Tracking with MediaPipe ==== #
# Load the .task model and set up the options
base_options = python.BaseOptions(model_asset_path="mediapipe_resources/face_landmarker.task")
options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, output_face_blendshapes=False, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Load the video file
cv2.namedWindow("MediaPipe Facial Motion Capture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MediaPipe Facial Motion Capture", 640, 1440)
cap = cv2.VideoCapture(INPUT_VIDEO_PATH + args.v)

# Get input video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create MP4 writer
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps,
    (frame_width, frame_height)
)

print("Output path:", output_video_path)
print("VideoWriter opened:", video_writer.isOpened())
mp_results_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp_ms = int((frame_idx / fps) * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    # ==== Visualize the tracking ==== #
    # ---- Landmarks and mesh ---- #
    annotated_image = mph.draw_lip_contour_landmarks(mp_image.numpy_view(), result)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    if frame_idx == 30:     # Random frame choice for sanity check
        mph.print_lip_landmark_coordinates(result)

    cv2.imshow("MediaPipe Facial Motion Capture", annotated_image)
    video_writer.write(annotated_image)

    frame_idx += 1

cap.release()
video_writer.release()
cv2.destroyAllWindows()