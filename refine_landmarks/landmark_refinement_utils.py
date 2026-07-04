import numpy as np

UPPER_LIP_OBJ_ID = 1
LOWER_LIP_OBJ_ID = 2
TONGUE_OBJ_ID = 3

# Assigning MediaPipe Landmarks to Lip Contours
UPPER_LIP_OUTER_CONTOUR = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
UPPER_LIP_INNER_CONTROR = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_LIP_OUTER_CONTOUR = [146, 91, 181, 84, 17, 314, 314, 405, 321, 375]
LOWER_LIP_INNER_CONTOUR = [95, 88, 178, 87, 14, 317, 402, 318, 324]

def nearest_spline_point(
    landmark,
    spline
):

    diff = spline - landmark

    distances = np.sum(
        diff**2,
        axis=1
    )

    idx = np.argmin(distances)

    return spline[idx]

def refine_landmark(
    landmark,
    spline,
    pull_strength=0.3
):
    """
    Move landmark partially toward spline.
    """

    target = nearest_spline_point(
        landmark,
        spline
    )

    refined = (
        (1-pull_strength)*landmark
        +
        pull_strength*target
    )

    return refined

def refine_landmarks(
    landmarks,
    indices,
    spline,
    pull_strength=0.3
):

    refined = landmarks.copy()

    for idx in indices:

        refined[idx,:2] = refine_landmark(
            landmarks[idx,:2],
            spline,
            pull_strength
        )

    return refined

# ======= Visualization Utils ======= #

import cv2
import numpy as np


def visualize_mp_sam_frame(
    frame,
    frame_idx,
    sam2_video_segs,
    mp_lip_results_list,
    alpha=0.5,
    draw_indices=True
):
    """
    Visualize:
        - Upper lip SAM mask
        - Lower lip SAM mask
        - MediaPipe lip landmarks

    Parameters
    ----------
    frame : ndarray (RGB or BGR)
        Video frame corresponding to frame_idx

    frame_idx : int

    sam2_video_segs : dict

    mp_lip_results_list : list
    """

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    overlay = frame.copy()

    h, w = frame.shape[:2]

    # --------------------------------------------------
    # Draw upper lip mask
    # --------------------------------------------------

    if frame_idx in sam2_video_segs:    # Check if SAM2 produced any segments for this frame

        frame_masks = sam2_video_segs[frame_idx]

        if UPPER_LIP_OBJ_ID in frame_masks:

            upper_mask = frame_masks[
                UPPER_LIP_OBJ_ID
            ].squeeze().astype(bool)

            upper_color = np.array(
                [0, 255, 0],  # Green
                dtype=np.uint8
            )

            overlay[upper_mask] = (
                alpha * upper_color +
                (1 - alpha) * overlay[upper_mask]
            ).astype(np.uint8)

            contours, _ = cv2.findContours(
                upper_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                overlay,
                contours,
                -1,
                (255, 255, 255),
                2
            )

        # --------------------------------------------------
        # Draw lower lip mask
        # --------------------------------------------------

        if LOWER_LIP_OBJ_ID in frame_masks:

            lower_mask = frame_masks[
                LOWER_LIP_OBJ_ID
            ].squeeze().astype(bool)

            lower_color = np.array(
                [255, 0, 0],  # Blue
                dtype=np.uint8
            )

            overlay[lower_mask] = (
                alpha * lower_color +
                (1 - alpha) * overlay[lower_mask]
            ).astype(np.uint8)

            contours, _ = cv2.findContours(
                lower_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                overlay,
                contours,
                -1,
                (255, 255, 255),
                2
            )

    # --------------------------------------------------
    # Draw MediaPipe landmarks
    # --------------------------------------------------

    lip_landmarks = mp_lip_results_list[frame_idx]

    for lm in lip_landmarks:

        mp_idx = int(lm[0])

        px = int(lm[1] * w)
        py = int(lm[2] * h)

        cv2.circle(
            overlay,
            (px, py),
            3,
            (0, 0, 255),  # Red
            -1
        )

        if draw_indices:

            cv2.putText(
                overlay,
                str(mp_idx),
                (px + 4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1
            )

    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    cv2.putText(
        overlay,
        f"Frame {frame_idx}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.namedWindow(
        "SAM2 + MediaPipe",
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        "SAM2 + MediaPipe",
        overlay
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()