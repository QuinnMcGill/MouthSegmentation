import numpy as np
from scipy.interpolate import splprep
from scipy.interpolate import splev
from scipy.interpolate import interp1d
import cv2

import numpy as np

def find_mp_corners(mp_lip_results, h, w):
    """
    Find the left and right inner and outer Mediapipe lip corners

    Uses:
        Outer left  = 61
        Outer right = 291
        Inner left  = 78
        Inner right = 308

    Parameters
    ----------
    mp_lip_results : ndarray or list
        Array of shape (N,4):

            [mediapipe_index, x, y, z]

        where x and y are normalized to [0,1].

    w : int
        Image width.

    h : int
        Image height.

    Returns
    -------
    inner_left_corner : ndarray (2,)
        Pixel coordinates (x,y)

    outer_left_corner : ndarray (2,)
        Pixel coordinates (x,y)

    inner_right_corner : ndarray (2,)
        Pixel coordinates (x,y)

    outer_right_corner : ndarray (2,)
        Pixel coordinates (x,y)
    """

    # Build lookup table:
    # mediapipe index -> (x,y)
    landmark_lookup = {}

    for row in mp_lip_results:
        mp_idx = int(row[0])

        landmark_lookup[mp_idx] = np.array([
            row[1] * w,
            row[2] * h
        ], dtype=np.float32)

    required = [61, 291, 78, 308]

    missing = [idx for idx in required if idx not in landmark_lookup]

    if len(missing) > 0:
        raise RuntimeError(
            f"Missing MediaPipe landmarks: {missing}"
        )

    inner_left = landmark_lookup[78]
    outer_left = landmark_lookup[61]

    inner_right = landmark_lookup[308]
    outer_right = landmark_lookup[291]

    inner_left_corner = (
        int(round(inner_left[0])),
        int(round(inner_left[1]))
    )

    outer_left_corner = (
        int(round(outer_left[0])),
        int(round(outer_left[1]))
    )

    inner_right_corner = (
        int(round(inner_right[0])),
        int(round(inner_right[1]))
    )

    outer_right_corner = (
        int(round(outer_right[0])),
        int(round(outer_right[1]))
    )

    return inner_left_corner, outer_left_corner, inner_right_corner, outer_right_corner

def nearest_point_index(spline, point):
    """
    Find the spline sample closest to a point.

    Parameters
    ----------
    spline : (N,2)
    point : (2,)

    Returns
    -------
    int
    """

    d = np.sum((spline - point) ** 2, axis=1)

    return np.argmin(d)

def rotate_closed_spline(spline, start_idx):
    """
    Rotate a cyclic spline so that sample start_idx becomes sample 0.
    """

    return np.roll(spline, -start_idx, axis=0)

def split_closed_spline(spline, left_idx, right_idx):
    """
    Splits a closed spline into two arcs connecting the corners.

    Returns
    -------
    arc1
    arc2
    """

    N = len(spline)

    spline = rotate_closed_spline(spline, left_idx)

    right_idx = (right_idx - left_idx) % N

    arc1 = spline[:right_idx + 1]

    arc2 = np.vstack([
        spline[right_idx:],
        spline[:1]
    ])

    return arc1, arc2

def classify_upper_lower(arc1, arc2):

    if arc1[:,1].mean() < arc2[:,1].mean():
        return arc1, arc2

    return arc2, arc1

def anchor_arc_endpoints(
    lip_id,
    upper_arc,
    lower_arc,
    inner_left_corner, 
    outer_left_corner, 
    inner_right_corner, 
    outer_right_corner
):
    """
    Replace spline endpoints with MediaPipe corners.
    """
    if lip_id == 1: # Upper lip
        upper_arc[0] = outer_right_corner
        upper_arc[-1] = outer_left_corner

        lower_arc[0] = inner_left_corner
        lower_arc[-1] = inner_right_corner
    else: # Lower Lip
        upper_arc[0] = inner_right_corner
        upper_arc[-1] = inner_left_corner

        lower_arc[0] = outer_left_corner
        lower_arc[-1] = outer_right_corner

    return upper_arc, lower_arc

def arc_length_parameterization(points):
    """
    Compute normalized cumulative arc length.

    Returns
    -------
    s : values in [0,1]
    """

    segment_lengths = np.linalg.norm(
        np.diff(points, axis=0),
        axis=1,
    )

    s = np.concatenate([
        [0],
        np.cumsum(segment_lengths)
    ])

    s /= s[-1]

    return s

def sample_arc(points, num_points):
    """
    Sample equally spaced points along an arc.
    """

    s = arc_length_parameterization(points)

    target = np.linspace(
        0,
        1,
        num_points,
    )

    fx = interp1d(
        s,
        points[:,0],
        kind="linear"
    )

    fy = interp1d(
        s,
        points[:,1],
        kind="linear"
    )

    sampled = np.stack([
        fx(target),
        fy(target)
    ], axis=1)

    return sampled

def refine_lip_contour(
    lip_id,
    spline,
    inner_left_corner, 
    outer_left_corner, 
    inner_right_corner, 
    outer_right_corner,
    num_landmarks,
    frame
):
    """
    Refine one lip contour using MediaPipe corners.

    Parameters
    ----------
    spline : (1000,2)

    mp_left_corner : (2,)

    mp_right_corner : (2,)

    num_landmarks :
        Number of desired MediaPipe landmarks.

    Returns
    -------
    upper_points
    lower_points
    """
    # Check to see if no lip was detected
    if spline is None:
        return None, None

    outer_left_idx = nearest_point_index(
        spline,
        outer_left_corner,
    )
    
    outer_right_idx = nearest_point_index(
        spline,
        outer_right_corner,
    )

    arc1, arc2 = split_closed_spline(
        spline,
        outer_left_idx,
        outer_right_idx,
    )
    
    upper_arc, lower_arc = classify_upper_lower(
        arc1,
        arc2,
    )

    upper_arc, lower_arc = anchor_arc_endpoints(
        lip_id,
        upper_arc,
        lower_arc,
        inner_left_corner, 
        outer_left_corner, 
        inner_right_corner, 
        outer_right_corner
    )

    # return upper_arc, lower_arc
    upper_points = sample_arc(
        upper_arc,
        num_landmarks,
    )

    lower_points = sample_arc(
        lower_arc,
        num_landmarks,
    )

    return upper_points, lower_points