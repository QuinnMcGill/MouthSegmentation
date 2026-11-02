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

# Import MediaPipe Lip landmark indices for easy reference in visualization and analysis
# Outer contour
OUTER_LEFT_CORNER = 61
OUTER_RIGHT_CORNER = 291
OUTER_TOP_MIDDLE = 0
OUTER_BOTTOM_MIDDLE = 17

# Inner contour
INNER_LEFT_CORNER = 78
INNER_RIGHT_CORNER = 308
INNER_TOP_MIDDLE = 13
INNER_BOTTOM_MIDDLE = 14

# Medapipe landmark indices along lip contours
MEDIAPIPE_MOUTH_LAYOUT = {
    "top_outer": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],           # Left outer corner to right outer corner (includes corners)
    "top_inner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],          # Left inner corner to left inner corner (includes corners)
    "bottom_inner": [324, 318, 402, 317, 14, 87, 178, 88, 95],                # Right inner corner to left inner corner (excludes corners)
    "bottom_outer": [375, 321, 405, 314, 17, 84, 181, 91, 146]                # Right outer corner to left outer corner (excludes corners)
}

# For Visualization
RED_COLOR = (255, 0, 0)
MIN_POINT_RADIUS_RATIO = 0.0025

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


# ====== MediaPipe Visualization ====== #

# Does not work in combine_mp_sam2.py because of the way the MediaPipe results are stored in a list of NormalizedLandmark objects.
#  File "/home/quinnm/repo/MouthSegmentation/.venv_ms/lib/python3.12/site-packages/mediapipe/tasks/python/vision/drawing_utils.py", line 157, in draw_landmarks
    # if not landmark_list:
    #        ^^^^^^^^^^^^^
# ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
def draw_mp_face_mesh(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    thick_contour_drawing_spec = drawing_utils.DrawingSpec(
        color=(255,255,255), thickness=5
    )

    thin_contour_drawing_spec = drawing_utils.DrawingSpec(
        color=(255,255,255), thickness=2
    )

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=thin_contour_drawing_spec)
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=thick_contour_drawing_spec)
        # drawing_utils.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks,
        #     connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
        # drawing_utils.draw_landmarks(
        #     image=annotated_image,
        #     landmark_list=face_landmarks,
        #     connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
            landmark_drawing_spec=None,
            connection_drawing_spec=thick_contour_drawing_spec)

    return annotated_image

# Likely, a similar issue as above
def draw_mediapipe_lip_contours(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    landmark_drawing_spec = drawing_utils.DrawingSpec(
        color=RED_COLOR, thickness=5
    )

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
    
        drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
                landmark_drawing_spec=landmark_drawing_spec,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
        
    return annotated_image


def draw_mp_lip_landmarks(rgb_image, height, width, point_radius, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    face_landmarks = face_landmarks_list[0]  # Assuming only one face is detected for simplicity
    for idx in lip_indices:
        landmark = face_landmarks[idx]
        x_pixel = int(landmark.x * width)
        y_pixel = int(landmark.y * height)

        cv2.circle(
            annotated_image,
            (x_pixel, y_pixel),
            radius=point_radius,
            color=(255, 255, 255),
            thickness=-1
        )

    # Add connections between lip landmarks
    lip_connections = (
        vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS
    )

    for connection in lip_connections:

        start_idx = connection.start
        end_idx = connection.end

        if start_idx not in lip_indices or end_idx not in lip_indices:
            continue

        start_landmark = face_landmarks[start_idx]
        end_landmark = face_landmarks[end_idx]

        # Convert normalized coordinates to pixels
        start_x = int(
            start_landmark.x * annotated_image.shape[1]
        )

        start_y = int(
            start_landmark.y * annotated_image.shape[0]
        )

        end_x = int(
            end_landmark.x * annotated_image.shape[1]
        )

        end_y = int(
            end_landmark.y * annotated_image.shape[0]
        )

        cv2.line(
            annotated_image,
            (start_x, start_y),
            (end_x, end_y),
            (255, 255, 255),
            thickness=2
        )

    return annotated_image


def draw_lip_contours(rgb_image, lm_results, contour_colour, contour_thickness):
    ''''
    Draw lines connecting the lip landmarks on the image. The lip landmarks can either
    be the MediaPipe predicted landmarks or the SAM2-influenced landmarks. 
    '''

    annotated_image = np.copy(rgb_image)

    # MediaPipe lip connections
    lip_connections = (
        vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS
    )

    for connection in lip_connections:

        start_idx = connection.start
        end_idx = connection.end

        start_landmark = lm_results[start_idx]
        end_landmark = lm_results[end_idx]

        # Convert normalized coordinates to pixels
        start_x = int(
            start_landmark.x * annotated_image.shape[1]
        )

        start_y = int(
            start_landmark.y * annotated_image.shape[0]
        )

        end_x = int(
            end_landmark.x * annotated_image.shape[1]
        )

        end_y = int(
            end_landmark.y * annotated_image.shape[0]
        )

        cv2.line(
            annotated_image,
            (start_x, start_y),
            (end_x, end_y),
            contour_colour,
            thickness=contour_thickness
        )

    return annotated_image


def add_lms_to_frame(frame, frame_lm_list, only_lips=False, point_radius=6, bgr_lm_colour=(255, 255, 255), draw_indices = True):
    """
    Takes any frame and adds annotated landmarks. Landmarks can be straight from MediaPipe or can 
    be the sam2-influenced landmarks.

    Parameters
    ----------
    frame : ndarray (RGB or BGR)
        Video frame corresponding to frame_idx

    frame_lm_list : list of normalized lip landmark locations (3D), along with their associated 
                    MediaPipe landmark index
    """

    h, w = frame.shape[:2]

    output_frame = frame.copy()

    for idx, point in enumerate(frame_lm_list):

        if only_lips and idx not in lip_indices:
            continue

        px = int(point.x * w)
        py = int(point.y * h)

        cv2.circle(
            output_frame,
            (px, py),
            point_radius,
            bgr_lm_colour,
            -1
        )

        if draw_indices:
            cv2.putText(
                output_frame,
                str(idx),
                (px + 4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                1
            )

    return output_frame


def add_masks_to_frame(frame, mask_list, mask_colours, alpha=0.5):
    for lip_mask, color in zip(mask_list, mask_colours):
        # Apply transparent mask
        frame[lip_mask] = (
            alpha * color +
            (1 - alpha) * frame[lip_mask]
        ).astype(np.uint8)

    return frame


def draw_face_mesh(frame, frame_lm_list):
    """
    Takes any frame and adds annotated landmarks. Landmarks can be straight from MediaPipe or can 
    be the sam2-influenced landmarks.

    Parameters
    ----------
    frame : ndarray (RGB or BGR)
        Video frame corresponding to frame_idx

    frame_lm_list : list of normalized lip landmark locations (3D), along with their associated 
                    MediaPipe landmark index
    """

    h, w = frame.shape[:2]

    output_frame = frame.copy()

    # MediaPipe lip connections
    lip_connections = (
        vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS
    )
    face_connections = (
        vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    )
    face_contour_connections = (
        vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
    )

    for connection_list, contour_colour in zip([face_connections, face_contour_connections, lip_connections],[(255,255,255), (255,255,255), (0,0,255)]):
        for connection in connection_list:

            start_idx = connection.start
            end_idx = connection.end

            start_landmark = frame_lm_list[start_idx]
            end_landmark = frame_lm_list[end_idx]

            # Convert normalized coordinates to pixels
            start_x = int(
                start_landmark.x * w
            )

            start_y = int(
                start_landmark.y * h
            )

            end_x = int(
                end_landmark.x * w
            )

            end_y = int(
                end_landmark.y * h
            )

            cv2.line(
                output_frame,
                (start_x, start_y),
                (end_x, end_y),
                contour_colour,
                thickness=2
            )

    return output_frame


# ======== Debugging ========= #


def print_lip_landmark_coordinates(detection_result):

    face_landmarks = detection_result.face_landmarks[0]

    lip_points = []

    for idx in lip_indices:
        lm = face_landmarks[idx]

        lip_points.append(lm)

    lip_points = np.array(lip_points)

    print(lip_points.shape)
    print(lip_points)