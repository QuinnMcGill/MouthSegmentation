import dataclasses

import numpy as np
from scipy.interpolate import splprep
from scipy.interpolate import splev
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from mediapipe.tasks.python import vision

import numpy as np

LANDMARK_SMOOTH = 3.0   # frames; std of the Gaussian that smooths the refined lip landmarks over time

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

# def find_mp_corners(mp_lookup_table, h, w):
#     """
#     Find the left and right inner and outer Mediapipe lip corners

#     Uses:
#         Outer left  = 61
#         Outer right = 291
#         Inner left  = 78
#         Inner right = 308

#     Parameters
#     ----------
#     mp_lip_results : ndarray or list
#         Array of shape (N,4):

#             [mediapipe_index, x, y, z]

#         where x and y are normalized to [0,1].

#     w : int
#         Image width.

#     h : int
#         Image height.

#     Returns
#     -------
#     inner_left_corner : ndarray (2,)
#         Pixel coordinates (x,y)

#     outer_left_corner : ndarray (2,)
#         Pixel coordinates (x,y)

#     inner_right_corner : ndarray (2,)
#         Pixel coordinates (x,y)

#     outer_right_corner : ndarray (2,)
#         Pixel coordinates (x,y)
#     """

#     required = [61, 291, 78, 308]

#     missing = [idx for idx in required if idx not in mp_lookup_table]

#     if len(missing) > 0:
#         raise RuntimeError(
#             f"Missing MediaPipe landmarks: {missing}"
#         )

#     inner_left = mp_lookup_table[78]
#     outer_left = mp_lookup_table[61]

#     inner_right = mp_lookup_table[308]
#     outer_right = mp_lookup_table[291]

#     inner_left_corner = (
#         int(round(inner_left[0])),
#         int(round(inner_left[1]))
#     )

#     outer_left_corner = (
#         int(round(outer_left[0])),
#         int(round(outer_left[1]))
#     )

#     inner_right_corner = (
#         int(round(inner_right[0])),
#         int(round(inner_right[1]))
#     )

#     outer_right_corner = (
#         int(round(outer_right[0])),
#         int(round(outer_right[1]))
#     )

#     return inner_left_corner, outer_left_corner, inner_right_corner, outer_right_corner

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

# def rotate_closed_spline(spline, start_idx):
#     """
#     Rotate a cyclic spline so that sample start_idx becomes sample 0.
#     """

#     return np.roll(spline, -start_idx, axis=0)

# def split_closed_spline(spline, left_idx, right_idx):
#     """
#     Splits a closed spline into two arcs connecting the corners.

#     Returns
#     -------
#     arc1
#     arc2
#     """

#     N = len(spline)

#     spline = rotate_closed_spline(spline, left_idx)

#     right_idx = (right_idx - left_idx) % N

#     arc1 = spline[:right_idx + 1]

#     arc2 = np.vstack([
#         spline[right_idx:],
#         spline[:1]
#     ])

#     return arc1, arc2

# def classify_upper_lower(arc1, arc2):

#     if arc1[:,1].mean() < arc2[:,1].mean():
#         return arc1, arc2

#     return arc2, arc1

# def anchor_arc_endpoints(
#     lip_id,
#     upper_arc,
#     lower_arc,
#     inner_left_corner, 
#     outer_left_corner, 
#     inner_right_corner, 
#     outer_right_corner
# ):
#     """
#     Replace spline endpoints with MediaPipe corners.
#     """
#     if lip_id == 1: # Upper lip
#         upper_arc[0] = outer_right_corner
#         upper_arc[-1] = outer_left_corner

#         lower_arc[0] = inner_left_corner
#         lower_arc[-1] = inner_right_corner
#     else: # Lower Lip
#         upper_arc[0] = inner_right_corner
#         upper_arc[-1] = inner_left_corner

#         lower_arc[0] = outer_left_corner
#         lower_arc[-1] = outer_right_corner

#     return upper_arc, lower_arc

# def arc_length_parameterization(points):
#     """
#     Compute normalized cumulative arc length.

#     Returns
#     -------
#     s : values in [0,1]
#     """

#     segment_lengths = np.linalg.norm(
#         np.diff(points, axis=0),
#         axis=1,
#     )

#     s = np.concatenate([
#         [0],
#         np.cumsum(segment_lengths)
#     ])

#     s /= s[-1]

#     return s

# # Option 1: Sample equally spaced points along the arc
# def sample_arc(points, num_points):
#     """
#     Sample equally spaced points along an arc.
#     """

#     s = arc_length_parameterization(points)

#     target = np.linspace(
#         0,
#         1,
#         num_points,
#     )

#     fx = interp1d(
#         s,
#         points[:,0],
#         kind="linear"
#     )

#     fy = interp1d(
#         s,
#         points[:,1],
#         kind="linear"
#     )

#     sampled = np.stack([
#         fx(target),
#         fy(target)
#     ], axis=1)

#     return sampled

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
    contour,
    closed=True
):
    """
    Find intersections between an infinite line
    through p1 and p2 and a contour.

    Parameters
    ----------
    p1 : (2,)
        First MediaPipe point.

    p2 : (2,)
        Second MediaPipe point.

    contour : (N,2)
        SAM2 contour.

    closed : bool
        True if the contour is closed (the last point
        connects back to the first), False for an open arc.

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

    for i in range(N if closed else N - 1):

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
    second_contour,
    closed=True
):
    """
    Find contour points corresponding to two MediaPipe
    landmarks connected by a cross-contour line.

    The first MediaPipe landmark is matched to the
    first contour.

    The second MediaPipe landmark is matched to the
    second contour.

    closed is passed to find_line_contour_intersections()
    (False for open arcs).
    """

    first_intersections = (
        find_line_contour_intersections(
            mp_first_point,
            mp_second_point,
            first_contour,
            closed
        )
    )

    second_intersections = (
        find_line_contour_intersections(
            mp_first_point,
            mp_second_point,
            second_contour,
            closed
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

# def generate_cross_contour_landmarks(
#     lip_id,
#     mp_lookup,
#     contour_pairs,
#     upper_arc,
#     lower_arc
# ):
#     """
#     Generate SAM2-based landmarks from MediaPipe
#     cross-contour landmark pairs.
#     """

#     outer_points = []
#     inner_points = []

#     for i, (outer_idx, inner_idx) in enumerate(contour_pairs):

#         mp_outer = mp_lookup[
#             outer_idx
#         ]

#         mp_inner = mp_lookup[
#             inner_idx
#         ]

#         if lip_id == 1:     # Top lip

#             # MP outer → upper arc
#             # MP inner → lower arc

#             outer_point, inner_point = (
#                 find_corresponding_contour_points(
#                     mp_outer,
#                     mp_inner,
#                     upper_arc,
#                     lower_arc
#                 )
#             )

#         else:               # Bottom lip

#             # MP inner → upper arc
#             # MP outer → lower arc

#             inner_point, outer_point = (
#                 find_corresponding_contour_points(
#                     mp_inner,
#                     mp_outer,
#                     upper_arc,
#                     lower_arc
#                 )
#             )

#         if inner_point is None or outer_point is None:
#             print(
#                 f"\nFailed pair {i}: "
#                 f"outer={outer_idx}, inner={inner_idx}"
#             )
#             print("Is upper_arc None?: ", upper_arc is None)
#             print("Is lower_arc None?: ", lower_arc is None)    

#         outer_points.append(
#             outer_point
#         )

#         inner_points.append(
#             inner_point
#         )

#     for i, p in enumerate(inner_points):
#         if np.shape(p) != (2,):
#             print(f"Bad inner point {i}:")
#             print(p)
#             print(type(p))
#             print(np.shape(p))

#     for i, p in enumerate(outer_points):
#         if np.shape(p) != (2,):
#             print(f"Bad outer point {i}:")
#             print(p)
#             print(type(p))
#             print(np.shape(p))

#     if lip_id == 1:     # Top lip: outer contour is the upper arc
#         return np.array(outer_points), np.array(inner_points)
#     else:               # Bottom lip: inner contour is the upper arc
#         return np.array(inner_points), np.array(outer_points)


def generate_sam2_contour_landmarks(
    lip_id,
    mp_lookup,
    sam2_contours,
    sam2_corners
):
    """
    Generate landmarks along the four SAM2 lip contours returned by
    extract_sam2_lip_results.get_sam2_contours_and_corners().

    As in generate_cross_contour_landmarks(), each MediaPipe cross-contour
    landmark pair (outer, inner) defines a line through the two landmarks;
    the new outer / inner landmark is where that line crosses the SAM2
    outer / inner contour, closest to the MediaPipe landmark. If the line
    misses a contour, the contour point nearest to the MediaPipe landmark
    is used instead. The lip corners are the SAM2-predicted corners.

    Parameters
    ----------
    lip_id : int
        1 = upper lip, 2 = lower lip

    mp_lookup : dict
        MediaPipe index -> (x, y) pixel coordinates,
        from build_landmark_lookup().

    sam2_contours : (4, N, 2)
        One frame's SAM2 contours in pixel coordinates, each running
        image-left -> image-right: upper outer, upper inner,
        lower outer, lower inner.

    sam2_corners : (4, 2)
        One frame's SAM2 corners in pixel coordinates: outer left,
        outer right, inner left, inner right (image-left / right).

    Returns
    -------
    upper_points, lower_points
        In the order replace_mp_lip_landmarks() expects:
            upper lip: top_outer (11, incl. corners), top_inner (11, incl. corners)
            lower lip: bottom_inner (9), bottom_outer (9); the corners are
                       set with the upper lip
        None, None if SAM2 has no result for this frame.
    """

    # No SAM2 result for this frame
    if np.isnan(sam2_contours).any() or np.isnan(sam2_corners).any():
        return None, None

    if lip_id == 1:     # Top lip
        outer_contour, inner_contour = sam2_contours[0], sam2_contours[1]
        contour_pairs = TOP_LIP_LANDMARK_PAIRS
    else:               # Bottom lip
        outer_contour, inner_contour = sam2_contours[2], sam2_contours[3]
        contour_pairs = BOTTOM_LIP_LANDMARK_PAIRS

    outer_points = []
    inner_points = []

    for outer_idx, inner_idx in contour_pairs:

        mp_outer = mp_lookup[outer_idx]
        mp_inner = mp_lookup[inner_idx]

        outer_point, inner_point = find_corresponding_contour_points(
            mp_outer,
            mp_inner,
            outer_contour,
            inner_contour,
            closed=False
        )

        # Line missed the contour: fall back to the nearest contour point
        if outer_point is None:
            outer_point = outer_contour[nearest_point_index(outer_contour, mp_outer)]
        if inner_point is None:
            inner_point = inner_contour[nearest_point_index(inner_contour, mp_inner)]

        outer_points.append(outer_point)
        inner_points.append(inner_point)

    if lip_id == 1:     # Top lip: outer contour is the upper arc
        # Add the SAM2 corners. MediaPipe corners 61 (outer) / 78 (inner) are on
        # the same side as 185 / 191, the start of the pairs; match them to the
        # SAM2 image-left or image-right corners by MediaPipe's own corner positions.
        outer_left, outer_right, inner_left, inner_right = sam2_corners
        if mp_lookup[61][0] > mp_lookup[291][0]:
            outer_left, outer_right = outer_right, outer_left
            inner_left, inner_right = inner_right, inner_left

        outer_points = [outer_left, *outer_points, outer_right]
        inner_points = [inner_left, *inner_points, inner_right]

        return np.array(outer_points), np.array(inner_points)
    else:               # Bottom lip: inner contour is the upper arc
        return np.array(inner_points), np.array(outer_points)


def smooth_refined_landmarks(
    sam2_mp_results,
    valid=None,
    sigma=LANDMARK_SMOOTH
):
    """
    Remove jitter from the refined lip landmarks by smoothing their x, y
    positions over consecutive frames with a Gaussian of std sigma frames,
    like extract_sam2_lip_results.smooth_corners().

    Only the landmarks in TOP_LIP_LANDMARK_PAIRS and BOTTOM_LIP_LANDMARK_PAIRS
    are smoothed; the lip corners (61, 291, 78, 308) are already smoothed by
    smooth_corners() and are left as they are.

    Parameters
    ----------
    sam2_mp_results : sequence of length F
        Each frame's MediaPipe landmarks (NormalizedLandmark).

    valid : (F,) bool or None
        Frames whose landmarks were refined with SAM2. Other frames are left
        out of the average and left unchanged. None = all frames are valid.

    sigma : float
        Gaussian std in frames.

    Returns
    -------
    smoothed_results : list of length F
        Each frame's landmarks, with the smoothed lip landmarks replaced by
        new NormalizedLandmark objects (the input is not changed).
    """

    indices = [idx for pair in TOP_LIP_LANDMARK_PAIRS + BOTTOM_LIP_LANDMARK_PAIRS for idx in pair]

    # (F, L, 2) normalized x, y of every smoothed landmark in every frame
    xy = np.array([[(frame[idx].x, frame[idx].y) for idx in indices] for frame in sam2_mp_results])

    ok = np.ones(len(xy), bool) if valid is None else np.asarray(valid, bool)
    xy = np.where(ok[:, None, None], xy, 0)

    # Weighted average over time: dividing by the smoothed weights ignores the invalid frames
    xy = (gaussian_filter1d(xy, sigma, axis=0, mode="nearest") /
          gaussian_filter1d(ok.astype(float), sigma, mode="nearest")[:, None, None])

    smoothed_results = []

    for f, frame in enumerate(sam2_mp_results):

        frame = list(frame)     # new list, so the input results are not changed

        if ok[f]:
            for n, idx in enumerate(indices):
                frame[idx] = dataclasses.replace(frame[idx], x=float(xy[f, n, 0]), y=float(xy[f, n, 1]))

        smoothed_results.append(frame)

    return smoothed_results


# def generate_sam2_guided_landmarks(
#     lip_id,
#     spline,
#     inner_left_corner, 
#     outer_left_corner, 
#     inner_right_corner, 
#     outer_right_corner,
#     mp_lookup_table,
# ):
#     """
#     Refine one lip contour using MediaPipe corners.

#     Parameters
#     ----------
#     spline : (1000,2)

#     mp_left_corner : (2,)

#     mp_right_corner : (2,)

#     num_landmarks :
#         Number of desired MediaPipe landmarks.

#     Returns
#     -------
#     upper_points
#     lower_points
#     """
#     # Check to see if no lip was detected
#     if spline is None:
#         return None, None

#     outer_left_idx = nearest_point_index(
#         spline,
#         outer_left_corner,
#     )
    
#     outer_right_idx = nearest_point_index(
#         spline,
#         outer_right_corner,
#     )

#     arc1, arc2 = split_closed_spline(
#         spline,
#         outer_left_idx,
#         outer_right_idx,
#     )
    
#     upper_arc, lower_arc = classify_upper_lower(
#         arc1,
#         arc2,
#     )

#     # Anchor the arcs to the MediaPipe corners
#     upper_arc, lower_arc = anchor_arc_endpoints(
#         lip_id,
#         upper_arc,
#         lower_arc,
#         inner_left_corner, 
#         outer_left_corner, 
#         inner_right_corner, 
#         outer_right_corner
#     )

#     # return upper_arc, lower_arc # Temporary return for testing purposes

#     # Create points along the inner and outer contours
#     if lip_id == 1: # Top lip    
#         upper_points, lower_points = (
#             generate_cross_contour_landmarks(
#                 lip_id,
#                 mp_lookup_table,
#                 TOP_LIP_LANDMARK_PAIRS,
#                 upper_arc,
#                 lower_arc
#             )
#         )
#     else: # Bottom Lip
#         upper_points, lower_points = (
#             generate_cross_contour_landmarks(
#                 lip_id,
#                 mp_lookup_table,
#                 BOTTOM_LIP_LANDMARK_PAIRS,
#                 upper_arc,
#                 lower_arc
#             )
#         )

#     # # return upper_arc, lower_arc
#     # upper_points = sample_arc(
#     #     upper_arc,
#     #     num_landmarks,
#     # )

#     # lower_points = sample_arc(
#     #     lower_arc,
#     #     num_landmarks,
#     # )

#     return upper_points, lower_points

# ================== Fixing Nearby Landmarks ================== #

# Spokes through the lip mesh, one per (outer, inner) pair in TOP/BOTTOM_LIP_LANDMARK_PAIRS:
# (outer contour, semi-outer ring, semi-inner ring, inner contour)
TOP_LIP_SPOKES = [(185, 184, 183, 191), (40, 74, 42, 80), (39, 73, 41, 81), (37, 72, 38, 82), (0, 11, 12, 13),
                  (267, 302, 268, 312), (269, 303, 271, 311), (270, 304, 272, 310), (409, 408, 407, 415)]
BOTTOM_LIP_SPOKES = [(375, 307, 325, 324), (321, 320, 319, 318), (405, 404, 403, 402), (314, 315, 316, 317), (17, 16, 15, 14),
                     (84, 85, 86, 87), (181, 180, 179, 178), (91, 90, 89, 88), (146, 77, 96, 95)]
# Mouth corners (outer corner, semi-outer, semi-inner, inner corner)
CORNER_SPOKES = [(61, 76, 62, 78), (291, 306, 292, 308)]

LIP_SPOKES = np.array(TOP_LIP_SPOKES + BOTTOM_LIP_SPOKES + CORNER_SPOKES)     # (20, 4)

NEIGHBOR_HOPS = 3       # mesh connections out from the outer lip contour that are moved
NEIGHBOR_DECAY = 0.7    # each hop after the first moves this fraction of the previous hop's shift


def build_neighbor_rings(max_hops=NEIGHBOR_HOPS):
    """
    Rings of landmarks around the outer lip contour, following the face mesh
    (FACE_LANDMARKS_TESSELATION). Ring 1 is every non-lip landmark connected to
    an outer contour landmark, ring 2 every new landmark connected to ring 1, etc.

    Returns
    -------
    rings : list of length max_hops
        Each ring is a list of (landmark index, indices of its connected
        landmarks in the previous ring).
    """

    neighbors = {i: set() for i in range(478)}
    for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
        neighbors[c.start].add(c.end)
        neighbors[c.end].add(c.start)

    seen = set(LIP_SPOKES.ravel().tolist())
    previous = set(LIP_SPOKES[:, 0].tolist())     # outer contour
    rings = []

    for _ in range(max_hops):
        ring = sorted({j for i in previous for j in neighbors[i]} - seen)
        rings.append([(j, np.array(sorted(neighbors[j] & previous))) for j in ring])
        seen |= set(ring)
        previous = set(ring)

    return rings


NEIGHBOR_RINGS = build_neighbor_rings()


def landmarks_to_pixels(landmarks, w, h):
    """(478, 2) pixel x, y of one frame's MediaPipe landmarks."""
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks])


def adjust_nearby_landmarks(
    frame_landmarks,
    original_frame_landmarks,
    w,
    h,
    decay=NEIGHBOR_DECAY,
    rings=NEIGHBOR_RINGS
):
    """
    Move the landmarks around the refined lip contours so they follow the
    contours, after the contour landmarks have been replaced.

    1. Between the contours (the two middle lip rings): each point keeps its
       position relative to its spoke (outer -> inner contour landmark), i.e.
       the same fraction along the spoke and the same sideways offset,
       both relative to the spoke length.
    2. Outside the lips: the shift (refined - original) of each outer contour
       landmark spreads out through the face mesh. A landmark in ring k
       (k connections from the outer contour) moves by the average shift of
       its connected landmarks in ring k - 1, weighted by closeness
       (1 / distance). Ring 1 gets this full shift; each later ring is scaled
       by decay, so ring k moves decay^(k-1) as much as the contour.
       Landmarks beyond the last ring stay put.

    Only x, y change; z is left as it is.

    Parameters
    ----------
    frame_landmarks : list of NormalizedLandmark
        One frame's landmarks with the refined lip contours.

    original_frame_landmarks : list of NormalizedLandmark
        The same frame's landmarks before refinement.

    w, h : int
        Image width and height (the adjustment is done in pixels).

    decay : float
        Fraction of the previous ring's shift passed on at each hop.

    rings : list
        From build_neighbor_rings(); its length sets how many hops are moved.

    Returns
    -------
    adjusted_landmarks : list of NormalizedLandmark
        New list with the adjusted landmarks replaced (the input is not changed).
    """

    old = landmarks_to_pixels(original_frame_landmarks, w, h)
    new = landmarks_to_pixels(frame_landmarks, w, h)

    outer_idx, inner_idx = LIP_SPOKES[:, 0], LIP_SPOKES[:, 3]

    # ---- 1. Middle lip rings: same (along, sideways) spoke coordinates ----
    def spoke_axes(points):
        start = points[outer_idx]
        along = points[inner_idx] - start
        sideways = np.stack([-along[:, 1], along[:, 0]], axis=1)    # along rotated 90 degrees
        return start, along, sideways

    old_start, old_along, old_sideways = spoke_axes(old)
    new_start, new_along, new_sideways = spoke_axes(new)
    length2 = np.sum(old_along ** 2, axis=1)

    for ring in (1, 2):
        idx = LIP_SPOKES[:, ring]
        offset = old[idx] - old_start
        a = np.sum(offset * old_along, axis=1) / length2        # fraction along the spoke
        b = np.sum(offset * old_sideways, axis=1) / length2     # sideways, relative to spoke length
        new[idx] = new_start + a[:, None] * new_along + b[:, None] * new_sideways

    # ---- 2. Landmarks outside the lips: spread the outer contour shifts ring by ring ----
    shift = np.zeros_like(old)
    shift[outer_idx] = new[outer_idx] - old[outer_idx]

    for k, ring in enumerate(rings):
        scale = 1.0 if k == 0 else decay        # ring 1 gets the full shift of the contour
        for idx, connected in ring:
            # Average over the connected landmarks, weighted by closeness (1 / original distance)
            weights = 1.0 / np.maximum(np.linalg.norm(old[connected] - old[idx], axis=1), 1e-6)
            shift[idx] = scale * (weights @ shift[connected]) / weights.sum()

    ring_idx = np.array([idx for ring in rings for idx, _ in ring], dtype=int)
    new[ring_idx] = old[ring_idx] + shift[ring_idx]

    # ---- Write back only the landmarks that moved ----
    moved = np.concatenate([LIP_SPOKES[:, 1:3].ravel(), ring_idx])

    adjusted_landmarks = list(frame_landmarks)
    for idx in moved:
        adjusted_landmarks[idx] = dataclasses.replace(
            adjusted_landmarks[idx], x=float(new[idx, 0] / w), y=float(new[idx, 1] / h))

    return adjusted_landmarks




