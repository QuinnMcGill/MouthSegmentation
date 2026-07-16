import numpy as np
import cv2
from scipy.interpolate import splprep
from scipy.interpolate import splev

mouth_parts = {
    1: {
        "name": "Upper Lip",
        "color": np.array([255, 0, 0], dtype=np.uint8)   # Red
    },

    2: {
        "name": "Lower Lip",
        "color": np.array([0, 255, 0], dtype=np.uint8)   # Green
    },

    3: {
        "name": "Tongue",
        "color": np.array([0, 0, 255], dtype=np.uint8)   # Blue
    }
}

# ===== Util Functions ===== #

def get_largest_contour(mask):

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        return None

    return max(contours, key=cv2.contourArea)


def contour_to_spline(
    contour,
    num_points=500,
    smoothness=15
):
    """
    Convert contour to smooth closed spline.
    """

    contour = contour.squeeze()

    x = contour[:,0]
    y = contour[:,1]

    tck, _ = splprep(
        [x,y],
        s=smoothness,
        per=True
    )

    u_new = np.linspace(
        0,
        1,
        num_points,
        endpoint=False
    )

    x_new, y_new = splev(
        u_new,
        tck
    )

    spline = np.column_stack(
        [x_new, y_new]
    )

    return spline
