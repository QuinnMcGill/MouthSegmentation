import numpy as np
from scipy.interpolate import splprep
from scipy.interpolate import splev
from scipy.interpolate import interp1d

import numpy as np

# Mediapipe indices of inner and outer contour landmark pairs. Always inner, then outer contour index.
TOP_LIP_LANDMARK_PAIRS = [(185, 191), (40, 80), (39, 81), (37, 82), (0, 13), (267, 312), (269, 311), (270, 310), (409, 415)]
BOTTOM_LIP_LANDMARK_PAIRS = [(375, 324), (321, 318), (405, 402), (314, 317), (17, 14), (84, 87), (181, 178), (91, 88), (146, 95)]

def build_landmark_lookup(mp_lip_results, w, h):
    """
    Convert MediaPipe landmark array into a lookup dictionary.

    Parameters
    ----------
    mp_lip_results : ndarray
        Shape (N, 4):
        [mediapipe_index, x, y, z]

    w : int
        Image width.

    h : int
        Image height.

    Returns
    -------
    landmark_lookup : dict
        Maps MediaPipe index -> (x, y)
    """

    landmark_lookup = {}

    for row in mp_lip_results:

        mp_idx = int(row[0])

        x = row[1] * w
        y = row[2] * h

        landmark_lookup[mp_idx] = np.array(
            [x, y],
            dtype=np.float32
        )

    return landmark_lookup

def find_mp_corners(mp_lookup_table, h, w):
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

    required = [61, 291, 78, 308]

    missing = [idx for idx in required if idx not in mp_lookup_table]

    if len(missing) > 0:
        raise RuntimeError(
            f"Missing MediaPipe landmarks: {missing}"
        )

    inner_left = mp_lookup_table[78]
    outer_left = mp_lookup_table[61]

    inner_right = mp_lookup_table[308]
    outer_right = mp_lookup_table[291]

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

# Option 1: Sample equally spaced points along the arc
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

# Option 2: Solve cross-contour correspondence problem
def line_segment_intersection(
    p,
    direction,
    a,
    b
):
    """
    Find intersection between an infinite line and
    a contour segment.

    Infinite line:
        p + t * direction

    Segment:
        a + u * (b - a)

    Returns
    -------
    intersection : ndarray (2,) or None
    t : float or None
    """

    v = b - a

    cross = (
        direction[0] * v[1]
        - direction[1] * v[0]
    )

    # Parallel lines
    if abs(cross) < 1e-8:
        return None, None

    w = a - p

    t = (
        w[0] * v[1]
        - w[1] * v[0]
    ) / cross

    u = (
        w[0] * direction[1]
        - w[1] * direction[0]
    ) / cross

    # Intersection must be on contour segment
    if 0 <= u <= 1:

        intersection = (
            p +
            t * direction
        )

        return intersection, t

    return None, None

def find_line_contour_intersections(
    p1,
    p2,
    contour
):
    """
    Find intersections between an infinite line
    through p1 and p2 and a closed contour.

    Parameters
    ----------
    p1 : (2,)
        First MediaPipe point.

    p2 : (2,)
        Second MediaPipe point.

    contour : (N,2)
        Closed SAM2 contour.

    Returns
    -------
    intersections : list of ndarray
    """

    p1 = np.asarray(
        p1,
        dtype=np.float32
    )

    p2 = np.asarray(
        p2,
        dtype=np.float32
    )

    direction = p2 - p1

    intersections = []

    N = len(contour)

    for i in range(N):

        a = contour[i]

        b = contour[
            (i + 1) % N
        ]

        intersection, t = (
            line_segment_intersection(
                p1,
                direction,
                a,
                b
            )
        )

        if intersection is not None:

            intersections.append(
                intersection
            )

    return intersections

def closest_intersection(
    intersections,
    reference_point
):
    """
    Select the intersection closest to
    a reference MediaPipe landmark.
    """

    if len(intersections) == 0:
        return None

    reference_point = np.asarray(
        reference_point,
        dtype=np.float32
    )

    distances = [
        np.linalg.norm(
            p - reference_point
        )
        for p in intersections
    ]

    return intersections[
        np.argmin(distances)
    ]

def find_corresponding_contour_points(
    mp_first_point,
    mp_second_point,
    first_contour,
    second_contour
):
    """
    Find contour points corresponding to two MediaPipe
    landmarks connected by a cross-contour line.

    The first MediaPipe landmark is matched to the
    first contour.

    The second MediaPipe landmark is matched to the
    second contour.
    """

    first_intersections = (
        find_line_contour_intersections(
            mp_first_point,
            mp_second_point,
            first_contour
        )
    )

    second_intersections = (
        find_line_contour_intersections(
            mp_first_point,
            mp_second_point,
            second_contour
        )
    )

    first_point = closest_intersection(
        first_intersections,
        mp_first_point
    )

    second_point = closest_intersection(
        second_intersections,
        mp_second_point
    )

    return (
        first_point,
        second_point
    )

def generate_cross_contour_landmarks(
    lip_id,
    mp_lookup,
    contour_pairs,
    upper_arc,
    lower_arc
):
    """
    Generate SAM2-based landmarks from MediaPipe
    cross-contour landmark pairs.
    """

    outer_points = []
    inner_points = []

    for i, (outer_idx, inner_idx) in enumerate(contour_pairs):

        mp_outer = mp_lookup[
            outer_idx
        ]

        mp_inner = mp_lookup[
            inner_idx
        ]

        if lip_id == 1:     # Top lip

            # MP outer → upper arc
            # MP inner → lower arc

            outer_point, inner_point = (
                find_corresponding_contour_points(
                    mp_outer,
                    mp_inner,
                    upper_arc,
                    lower_arc
                )
            )

        else:               # Bottom lip

            # MP inner → upper arc
            # MP outer → lower arc

            inner_point, outer_point = (
                find_corresponding_contour_points(
                    mp_inner,
                    mp_outer,
                    upper_arc,
                    lower_arc
                )
            )

        if inner_point is None or outer_point is None:
            print(
                f"\nFailed pair {i}: "
                f"outer={outer_idx}, inner={inner_idx}"
            )
            print("Is upper_arc None?: ", upper_arc is None)
            print("Is lower_arc None?: ", lower_arc is None)    

        outer_points.append(
            outer_point
        )

        inner_points.append(
            inner_point
        )

    for i, p in enumerate(inner_points):
        if np.shape(p) != (2,):
            print(f"Bad inner point {i}:")
            print(p)
            print(type(p))
            print(np.shape(p))

    for i, p in enumerate(outer_points):
        if np.shape(p) != (2,):
            print(f"Bad outer point {i}:")
            print(p)
            print(type(p))
            print(np.shape(p))

    return (
        np.array(outer_points),
        np.array(inner_points)
    )

def generate_sam2_guided_landmarks(
    lip_id,
    spline,
    inner_left_corner, 
    outer_left_corner, 
    inner_right_corner, 
    outer_right_corner,
    mp_lookup_table,
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

    # Anchor the arcs to the MediaPipe corners
    upper_arc, lower_arc = anchor_arc_endpoints(
        lip_id,
        upper_arc,
        lower_arc,
        inner_left_corner, 
        outer_left_corner, 
        inner_right_corner, 
        outer_right_corner
    )

    # return upper_arc, lower_arc # Temporary return for testing purposes

    # Create points along the inner and outer contours
    if lip_id == 1: # Top lip    
        upper_points, lower_points = (
            generate_cross_contour_landmarks(
                lip_id,
                mp_lookup_table,
                TOP_LIP_LANDMARK_PAIRS,
                upper_arc,
                lower_arc
            )
        )
    else: # Bottom Lip
        upper_points, lower_points = (
            generate_cross_contour_landmarks(
                lip_id,
                mp_lookup_table,
                BOTTOM_LIP_LANDMARK_PAIRS,
                upper_arc,
                lower_arc
            )
        )

    # # return upper_arc, lower_arc
    # upper_points = sample_arc(
    #     upper_arc,
    #     num_landmarks,
    # )

    # lower_points = sample_arc(
    #     lower_arc,
    #     num_landmarks,
    # )

    return upper_points, lower_points