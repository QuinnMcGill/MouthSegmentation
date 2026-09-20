import cv2
import numpy as np

from mask_processing.mask_processing_utils import *

# ===== Mask Cleaning ===== #

def mask_to_spline(mask):

    contour = get_largest_contour(mask)

    if contour is None:
        return None

    return contour_to_spline(contour)

def clip_upper_lip_mask(
    mask,
    inner_left_corner,
    outer_left_corner,
    inner_right_corner,
    outer_right_corner,
    padding=0
):
    """
    Prevent an upper lip mask from extending below
    the MediaPipe corner boundary lines.

    Parameters
    ----------
    mask : ndarray [H,W]
        Binary upper lip mask.

    inner_left_corner : (x,y)
        MediaPipe inner left lip corner.

    outer_left_corner : (x,y)
        MediaPipe outer left lip corner.

    inner_right_corner : (x,y)
        MediaPipe inner right lip corner.

    outer_right_corner : (x,y)
        MediaPipe outer right lip corner.

    padding : int
        Number of pixels allowed below the boundary.

    Returns
    -------
    clipped_mask : bool ndarray [H,W]
    """

    mask = mask.astype(bool)

    h, w = mask.shape

    # Convert to arrays
    inner_left = np.asarray(
        inner_left_corner,
        dtype=np.float32
    )

    outer_left = np.asarray(
        outer_left_corner,
        dtype=np.float32
    )

    inner_right = np.asarray(
        inner_right_corner,
        dtype=np.float32
    )

    outer_right = np.asarray(
        outer_right_corner,
        dtype=np.float32
    )

    # ------------------------------------------------
    # Create image coordinate grid
    # ------------------------------------------------

    # Image coordinates
    yy, xx = np.mgrid[
        0:h,
        0:w
    ]

    yy = yy.astype(np.float32)
    xx = xx.astype(np.float32)

    # ------------------------------------------------
    # Left boundary
    #
    # outer_left -> inner_left
    # ------------------------------------------------

    x1, y1 = outer_left
    x2, y2 = inner_left

    dx_left = x2 - x1

    if abs(dx_left) < 1e-6:

        # Degenerate case
        left_boundary_y = np.full_like(
            xx,
            max(y1, y2)
        )

    else:

        left_boundary_y = (
            y1
            + (
                (xx - x1)
                * (y2 - y1)
                / dx_left
            )
        )

    # ------------------------------------------------
    # Right boundary
    #
    # inner_right -> outer_right
    # ------------------------------------------------

    x1, y1 = inner_right
    x2, y2 = outer_right

    dx_right = x2 - x1

    if abs(dx_right) < 1e-6:

        right_boundary_y = np.full_like(
            xx,
            max(y1, y2)
        )

    else:

        right_boundary_y = (
            y1
            + (
                (xx - x1)
                * (y2 - y1)
                / dx_right
            )
        )

    # ------------------------------------------------
    # Determine horizontal clipping range
    # ------------------------------------------------

    # min_x = outer_left[0]
    # max_x = outer_right[0]  

    # # Pixels inside the horizontal range
    # within_corner_range = (
    #     (xx >= min_x) &
    #     (xx <= max_x)
    # )

    # ------------------------------------------------
    # Combine the two boundaries
    #
    # The upper lip must be above BOTH
    # boundary lines.
    # ------------------------------------------------

    boundary_y = np.maximum(
        left_boundary_y,
        right_boundary_y
    )

    # Allow optional padding
    boundary_y += padding

    # Upper lip must be ABOVE boundary
    above_boundary = (
        yy <= boundary_y
    )

    # ------------------------------------------------
    # Only clip within corner range
    #
    # Outside the range:
    #     keep original mask
    #
    # Inside the range:
    #     enforce upper-lip constraint
    # ------------------------------------------------

    # valid_region = (
    #     (~within_corner_range) |
    #     above_boundary
    # )

    clipped_mask = (
        mask &
        above_boundary
    )

    return clipped_mask

def clip_lower_lip_mask(
    mask,
    inner_left_corner,
    outer_left_corner,
    inner_right_corner,
    outer_right_corner,
    padding=0
):
    """
    Prevent a lower lip mask from extending above
    the MediaPipe corner boundary lines.
    """

    mask = mask.astype(bool)

    h, w = mask.shape

    inner_left = np.asarray(
        inner_left_corner,
        dtype=np.float32
    )

    outer_left = np.asarray(
        outer_left_corner,
        dtype=np.float32
    )

    inner_right = np.asarray(
        inner_right_corner,
        dtype=np.float32
    )

    outer_right = np.asarray(
        outer_right_corner,
        dtype=np.float32
    )

    # ------------------------------------------------
    # Create image coordinate grid
    # ------------------------------------------------

    yy, xx = np.mgrid[
        0:h,
        0:w
    ]

    yy = yy.astype(np.float32)
    xx = xx.astype(np.float32)

    # Left boundary
    x1, y1 = outer_left
    x2, y2 = inner_left

    dx_left = x2 - x1

    if abs(dx_left) < 1e-6:

        left_boundary_y = np.full_like(
            xx,
            max(y1, y2)
        )

    else:

        left_boundary_y = (
            y1
            + (
                (xx - x1)
                * (y2 - y1)
                / dx_left
            )
        )

    # Right boundary
    x1, y1 = inner_right
    x2, y2 = outer_right

    dx_right = x2 - x1

    if abs(dx_right) < 1e-6:

        right_boundary_y = np.full_like(
            xx,
            max(y1, y2)
        )

    else:

        right_boundary_y = (
            y1
            + (
                (xx - x1)
                * (y2 - y1)
                / dx_right
            )
        )

    # ------------------------------------------------
    # Determine horizontal clipping range
    # ------------------------------------------------

    # min_x = outer_left[0]
    # max_x = outer_right[0]  

    # within_corner_range = (
    #     (xx >= min_x) &
    #     (xx <= max_x)
    # )

    # ------------------------------------------------
    # Determine clipping boundary
    # ------------------------------------------------

    # Lower lip should be below BOTH boundaries
    boundary_y = np.minimum(
        left_boundary_y,
        right_boundary_y
    )

    boundary_y -= padding

    # Lower lip must be BELOW boundary
    below_boundary = (
        yy >= boundary_y
    )

    # ------------------------------------------------
    # Only clip within corner range
    # ------------------------------------------------

    # valid_region = (
    #     (~within_corner_range) |
    #     below_boundary
    # )

    clipped_mask = (
        mask &
        below_boundary
    )

    return clipped_mask

def clean_mask(mask,
                lip_id,
                inner_left_corner,
                outer_left_corner,
                inner_right_corner,
                outer_right_corner,
                close_kernel_size=5,
                open_kernel_size=4, 
                corner_padding=2):
    """
    Clean a binary lip mask.

    Args:
        mask: bool or binary numpy array [H,W]

    Returns:
        cleaned binary mask (bool)
    """

    mask = mask.astype(np.uint8)

    # -----------------------------
    # Morphological closing
    # -----------------------------

    # Fill small holes
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_kernel_size, close_kernel_size)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    # -----------------------------
    # Morphological opening
    # -----------------------------

    # Remove isolated noise
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (open_kernel_size, open_kernel_size)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel
    )

    # -----------------------------
    # Largest connected component
    # TODO: problably change this in cases where the tongue is out
    # -----------------------------

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels > 1:  # More than just background

        largest_label = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        mask = (labels == largest_label).astype(np.uint8)   # keep only the largest connected component

    mask = mask.astype(bool)

    # return mask

    # -----------------------------
    # Lip-specific geometric cleanup
    # -----------------------------

    if lip_id == 1:

        # Upper lip
        mask = clip_upper_lip_mask(
            mask,
            inner_left_corner,
            outer_left_corner,
            inner_right_corner,
            outer_right_corner,
            padding=corner_padding
        )

    elif lip_id == 2:

        # Lower lip
        mask = clip_lower_lip_mask(
            mask,
            inner_left_corner,
            outer_left_corner,
            inner_right_corner,
            outer_right_corner,
            padding=corner_padding
        )

    return mask

def smooth_mask(mask,
                close_kernel_size=5,
                open_kernel_size=4):
    """
    Clean a binary lip mask.

    Args:
        mask: bool or binary numpy array [H,W]

    Returns:
        cleaned binary mask (bool)
    """

    mask = mask.astype(np.uint8)

    # -----------------------------
    # Morphological closing
    # -----------------------------

    # Fill small holes
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_kernel_size, close_kernel_size)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    # -----------------------------
    # Morphological opening
    # -----------------------------

    # Remove isolated noise
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (open_kernel_size, open_kernel_size)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel
    )

    # -----------------------------
    # Largest connected component
    # -----------------------------

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels > 1:  # More than just background

        largest_label = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        mask = (labels == largest_label).astype(np.uint8)   # keep only the largest connected component

    mask = mask.astype(bool)

    return mask

# def process_video_segments(video_segments):
#     """
#     Process all masks in a frame.

#     Args:
#         video_segments:
#             {
#                 obj_id: mask
#             }

#     Returns:
#         processed dictionary
#     """

#     processed = {}

#     for obj_id, mask in video_segments.items():

#         mask = np.squeeze(mask)

#         processed[obj_id] = clean_mask(mask)

    return processed

# ===== Visualization ===== #

def visualize_mask_cleanup(
    frame,
    raw_mask,
    cleaned_mask,
    max_width=1600,
    max_height=900,
    alpha=0.5
):
    """
    Display side-by-side comparison of:
        frame + raw mask
        frame + cleaned mask
    """

    # Ensure masks are boolean
    raw_mask = raw_mask.astype(bool)
    cleaned_mask = cleaned_mask.astype(bool)

    # Covert frame to BGR for visualization
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    # Create copies of frame
    raw_overlay = frame.copy()
    clean_overlay = frame.copy()

    # Colors
    raw_color = np.array([0, 0, 255], dtype=np.uint8)      # Red
    clean_color = np.array([0, 255, 0], dtype=np.uint8)    # Green

    # Overlay raw mask
    raw_overlay[raw_mask] = (
        alpha * raw_color +
        (1 - alpha) * raw_overlay[raw_mask]
    ).astype(np.uint8)

    # Overlay cleaned mask
    clean_overlay[cleaned_mask] = (
        alpha * clean_color +
        (1 - alpha) * clean_overlay[cleaned_mask]
    ).astype(np.uint8)

    # Draw contours for easier comparison
    raw_contours, _ = cv2.findContours(
        raw_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        raw_overlay,
        raw_contours,
        -1,
        (255, 255, 255),
        2
    )

    clean_contours, _ = cv2.findContours(
        cleaned_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(clean_contours) > 0:
        clean_contour = max(
            clean_contours,
            key=cv2.contourArea
        )

        epsilon = (
            0.002 *
            cv2.arcLength(clean_contour, True)
        )

        smooth_clean_contour = cv2.approxPolyDP(
            clean_contour,
            epsilon,
            True
        )

        cv2.drawContours(
            clean_overlay,
            [smooth_clean_contour],
            -1,
            (255, 255, 255),
            2
        )
    else:

        cv2.putText(
            clean_overlay,
            "No contour found",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # Labels
    cv2.putText(
        raw_overlay,
        "Raw Mask",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        clean_overlay,
        "Cleaned Mask",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Side-by-side comparison
    comparison = np.hstack([raw_overlay, clean_overlay])

    h, w = comparison.shape[:2]

    scale = min(
        max_width / w,
        max_height / h,
        1.0
    )

    display = cv2.resize(
        comparison,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    cv2.namedWindow(
        "Mask Cleanup Comparison",
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        "Mask Cleanup Comparison",
        display
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_spline(
    frame,
    spline,
):
    """
    Draw a smooth spline on top of an image.

    Args:
        image : RGB or BGR image
        spline : (N,2) array returned by mask_to_spline()
    """

    if spline is None:
        return

    # Covert frame to BGR for visualization
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    vis = frame.copy()

    # Convert spline coordinates to integer pixels
    pts = np.round(spline).astype(np.int32)

    # Draw closed polyline
    cv2.polylines(
        vis,
        [pts],
        isClosed=True,
        color=(255, 255, 0),   # Cyan (BGR)
        thickness=2,
        lineType=cv2.LINE_AA
    )

    cv2.namedWindow(
        "Lip Spline",
        cv2.WINDOW_NORMAL
    )

    cv2.imshow("Lip Spline", vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


