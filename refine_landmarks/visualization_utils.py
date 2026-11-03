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

from sam2_resources.sam2_utils import MOUTH_PARTS

# ===== VISUALIZATION PARAMETERS ===== #
LARGE_POINT_RADIUS_RATIO = 0.003
SMALL_POINT_RADIUS_RATIO = 0.0015

# From MediaPipe documentation
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

# ===== METHODS ===== #

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


def add_masks_to_frame(frame, mask_list, alpha=0.5):

    output_frame = frame.copy()

    for lip_id, lip_mask in enumerate(mask_list, start=1):

        color = MOUTH_PARTS[lip_id]["color"]

        # Apply transparent mask
        output_frame[lip_mask] = (
            alpha * color +
            (1 - alpha) * output_frame[lip_mask]
        ).astype(np.uint8)

        # Optional: draw contours
        contours, _ = cv2.findContours(
            lip_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            output_frame,
            contours,
            -1,
            (255, 255, 255),
            2
        )

        ys, xs = np.where(lip_mask)

        if len(xs) > 0 and len(ys) > 0:

            center_x = int(np.mean(xs))
            center_y = int(np.mean(ys))

            cv2.putText(
                output_frame,
                MOUTH_PARTS[lip_id]["name"],
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    # Blend overlay with original frame
    output_frame = cv2.addWeighted(
        frame,
        1 - alpha,
        output_frame,
        alpha,
        0
    )

    return output_frame


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