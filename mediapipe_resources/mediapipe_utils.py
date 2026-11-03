import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# MediaPipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

from mediapipe.tasks.python.components.containers import NormalizedLandmark

# Constants from mediapipe docs
FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])

lip_indices = sorted(set(idx for connection in FACEMESH_LIPS for idx in connection))

# Medapipe landmark indices along lip contours
MEDIAPIPE_MOUTH_LAYOUT = {
    "top_outer": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],           # Left outer corner to right outer corner (includes corners)
    "top_inner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],          # Left inner corner to left inner corner (includes corners)
    "bottom_inner": [324, 318, 402, 317, 14, 87, 178, 88, 95],                # Right inner corner to left inner corner (excludes corners)
    "bottom_outer": [375, 321, 405, 314, 17, 84, 181, 91, 146]                # Right outer corner to left outer corner (excludes corners)
}

# ====== MediaPipe Setup and Usage ====== #

def initialize_mp():
    base_options = python.BaseOptions(model_asset_path="mediapipe_resources/face_landmarker.task")
    options = vision.FaceLandmarkerOptions(base_options=base_options, 
                                           running_mode=vision.RunningMode.VIDEO, 
                                           output_face_blendshapes=True, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector


def run_mp_for_video(path_to_video, mp_detector, grayscale=False):
    """
    Runs the MediaPipe face landmarker on every frame of a video.

    grayscale : bool
        If True, each frame is converted to grayscale (then back to 3 channels,
        since MediaPipe expects SRGB) before detection. This stops coloured
        lipstick from interfering with the lip landmarks.
    """

    # Initialize frame index
    frame_idx = 0

    # Load the video file
    cap = cv2.VideoCapture(path_to_video)

    # Get input video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Generate MediaPipe landmark predictions for each frame and store results in a list
    mp_results_list = []

    desc = "Generating MediaPipe Landmarks (grayscale)" if grayscale else "Generating MediaPipe Landmarks"
    with tqdm(total=total_frames, desc=desc, unit=" frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Increment the progress bar manually
            pbar.update(1)

            # Convert to RGB (optionally via grayscale)
            if grayscale:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int((frame_idx / fps) * 1000)
            result = mp_detector.detect_for_video(mp_image, timestamp_ms)

            face_landmarks = result.face_landmarks[0]   # Assuming only one face is detected for simplicity
            mp_results_list.append(face_landmarks)

            frame_idx += 1

    cap.release()
    return mp_results_list


def extract_lip_landmarks_coords(mp_results_list):
    mp_lip_results_list = []

    for frame in mp_results_list:
        frame_lip_landmarks = np.array([
            [lip_idx, frame[lip_idx].x, frame[lip_idx].y, frame[lip_idx].z]
            for lip_idx in lip_indices
        ])

        mp_lip_results_list.append(frame_lip_landmarks)

    return mp_lip_results_list


def replace_mp_lip_landmarks(mp_landmarks, lip_id, upper_points, lower_points, h, w):
    """
    Replace MediaPipe lip landmark x,y coordinates with
    SAM2-derived landmark coordinates.

    Parameters
    ----------
    mp_landmarks : list of NormalizedLandmark
        MediaPipe landmarks for a given frame (assuming one face/frame).

    lip_id : int
        1 = upper lip
        2 = lower lip

    upper_points : ndarray, shape (N, 2)
        SAM2-derived points along the upper arc of the
        selected lip region, in pixel coordinates [x, y].

    lower_points : ndarray, shape (N, 2)
        SAM2-derived points along the lower arc of the
        selected lip region, in pixel coordinates [x, y].

    h : int
        Image height in pixels.

    w : int
        Image width in pixels.

    Returns
    -------
    updated_landmarks : list of NormalizedLandmark
        MediaPipe landmarks with selected lip landmark
        coordinates replaced by SAM2-derived coordinates.
    """

    # --------------------------------------------------
    # SIMPLIFICATION: if no SAM2 mask is found, collapse
    # the lip to the inner contour.
    # --------------------------------------------------

    if upper_points is None or lower_points is None:

        if lip_id == 1:
            # Upper lip is folded inward.
            # Collapse upper outer contour onto upper inner contour.

            inner_indices = MEDIAPIPE_MOUTH_LAYOUT["top_inner"]
            outer_indices = MEDIAPIPE_MOUTH_LAYOUT["top_outer"]
        else:
            # Lower lip is folded inward.
            # Collapse lower outer contour onto lower inner contour.

            inner_indices = MEDIAPIPE_MOUTH_LAYOUT["bottom_inner"]
            outer_indices = MEDIAPIPE_MOUTH_LAYOUT["bottom_outer"]


        for outer_idx, inner_idx in zip(
            outer_indices,
            inner_indices
        ):

            # Copy the x,y coordinates of the inner
            # contour landmark to the outer contour landmark.

            mp_landmarks[outer_idx].x = (
                mp_landmarks[inner_idx].x
            )

            mp_landmarks[outer_idx].y = (
                mp_landmarks[inner_idx].y
            )

        return mp_landmarks

    # --------------------------------------------------
    # Select the appropriate MediaPipe landmark layout
    # --------------------------------------------------

    if lip_id == 1:     # Top lip
        # Upper lip
        upper_indices = MEDIAPIPE_MOUTH_LAYOUT["top_outer"]
        lower_indices = MEDIAPIPE_MOUTH_LAYOUT["top_inner"]

    else:               # Bottom lip
        # Lower lip
        upper_indices = MEDIAPIPE_MOUTH_LAYOUT["bottom_inner"]
        lower_indices = MEDIAPIPE_MOUTH_LAYOUT["bottom_outer"]

    # --------------------------------------------------
    # Make a copy so original MediaPipe landmarks
    # are not modified
    # --------------------------------------------------

    updated_landmarks = [
        NormalizedLandmark(
            x=landmark.x,
            y=landmark.y,
            z=landmark.z,
            visibility=landmark.visibility,
            presence=landmark.presence,
            name=landmark.name
        )
        for landmark in mp_landmarks
    ]

    # --------------------------------------------------
    # Helper function for pixel -> normalized coords
    # --------------------------------------------------

    def pixel_to_normalized(point):

        x_pixel = float(point[0])
        y_pixel = float(point[1])

        x_norm = x_pixel / w
        y_norm = y_pixel / h

        return x_norm, y_norm

    # --------------------------------------------------
    # Replace upper arc landmarks
    # --------------------------------------------------

    for mp_idx, point in zip(
        upper_indices,
        upper_points
    ):

        x_norm, y_norm = pixel_to_normalized(
            point
        )

        old_landmark = updated_landmarks[mp_idx]

        updated_landmarks[mp_idx] = (
            NormalizedLandmark(
                x=x_norm,
                y=y_norm,
                z=old_landmark.z,
                visibility=old_landmark.visibility,
                presence=old_landmark.presence,
                name=old_landmark.name
            )
        )

    # --------------------------------------------------
    # Replace lower arc landmarks
    # --------------------------------------------------

    for mp_idx, point in zip(
        lower_indices,
        lower_points
    ):

        x_norm, y_norm = pixel_to_normalized(
            point
        )

        old_landmark = updated_landmarks[mp_idx]

        updated_landmarks[mp_idx] = (
            NormalizedLandmark(
                x=x_norm,
                y=y_norm,
                z=old_landmark.z,
                visibility=old_landmark.visibility,
                presence=old_landmark.presence,
                name=old_landmark.name
            )
        )

    return updated_landmarks