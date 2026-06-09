import cv2
import numpy as np

from mask_processing.utils import *

# ===== Mask Cleaning ===== #

def mask_to_spline(mask):

    contour = get_largest_contour(mask)

    if contour is None:
        return None

    return contour_to_spline(contour)

def clean_mask(mask,
               close_kernel_size=5,
               open_kernel_size=3):
    """
    Clean a binary lip mask.

    Args:
        mask: bool or binary numpy array [H,W]

    Returns:
        cleaned binary mask (bool)
    """

    mask = mask.astype(np.uint8)

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

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels > 1:

        largest_label = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        mask = (labels == largest_label).astype(np.uint8)

    return mask.astype(bool)

def process_video_segments(video_segments):
    """
    Process all masks in a frame.

    Args:
        video_segments:
            {
                obj_id: mask
            }

    Returns:
        processed dictionary
    """

    processed = {}

    for obj_id, mask in video_segments.items():

        mask = np.squeeze(mask)

        processed[obj_id] = clean_mask(mask)

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

    clean_contours, _ = cv2.findContours(
        cleaned_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if len(clean_contours) > 0:
        clean_contour = max(clean_contours, key=cv2.contourArea)
    else:
        clean_contour = clean_contours[0]  # empty contour to avoid errors in drawing

    # Smooth the clean contours
    epsilon = 0.002 * cv2.arcLength(clean_contour, True)
    smooth_clean_contour = cv2.approxPolyDP(
        clean_contour,
        epsilon,
        True
    )

    cv2.drawContours(
        raw_overlay,
        raw_contours,
        -1,
        (255, 255, 255),
        2
    )

    cv2.drawContours(
        clean_overlay,
        smooth_clean_contour,
        -1,
        (255, 255, 255),
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