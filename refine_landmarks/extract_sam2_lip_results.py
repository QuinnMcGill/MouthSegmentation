"""
    Extract SAM2 Lip Contour Results
    ==============================
    Turns one performance clip's SAM2 lip masks into the lip-contour targets that the fit
    matches the character's lips against.

    Input: the clip's results .npz, which holds
        upper_indices / upper_offsets   SAM2 upper-lip mask: all foreground pixels of all frames as
        lower_indices / lower_offsets   flat indices (y * width + x); frame f's pixels are
                                        indices[offsets[f]:offsets[f + 1]]

    For every frame:
        1. decode the upper and lower lip masks
        2. take each mask's outer boundary (largest blob)
        3. find the two mouth corners: the pinch points where the upper and lower lip meet
        4. split each boundary at the corners into outer and inner contours
        5. predict four corner points: outer = midpoint of the pinch points, inner = CORNER_INSET px inwards
    Then smooth the outer corners over consecutive frames (inner corners follow), and for every frame:
        6. bend the contours so the outer contours meet at the outer corners and the inner contours
           at the inner corners

    Output:
        <clip>_lip_targets.npz
            contours    (F, 4, n_samples, 2)  pixel coordinates, each contour runs image-left -> image-right;
                                              order: upper outer, upper inner, lower outer, lower inner.
                                              NaN for frames that failed.
            corners     (F, 4, 2)             pixel coordinates; order: outer left, outer right,
                                              inner left, inner right. NaN for frames that failed.
"""

import argparse
import os
from tqdm import tqdm

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree

CORNER_TOL = 3.0                 # px; lip points within (closest gap + tol) count as touching at a corner
CORNER_INSET = 25.0              # px; inner corner distance inwards from the outer corner
CORNER_FADE = 0.3                # fraction of a contour's length, from each end, bent to meet its corner
CORNER_SMOOTH = 5.0              # frames; std of the Gaussian that smooths the outer corners over time
CONTOUR_SMOOTH = 16.0            # px; std of the Gaussian that smooths each contour along its length

'''
Load the SAM2 mask and Mediapipe landmarks in the .npz file and split them for easier handling.
Reads two arrays per lip, upper_indices/upper_offsets and lower_indices/lower_offsets.
'''
def load_sam2_masks(path):
    z = np.load(path, allow_pickle=False)
    # Load SAM2 mask information for each frame
    masks = {lip: (z[f"{lip}_indices"], z[f"{lip}_offsets"]) for lip in ("upper", "lower")}
    return masks


'''
Rebuild frame f's SAM2 lip mask as a full-frame (H, W) uint8 image.
'''
def decode_mask(idx, off, f, W, H):
    m = np.zeros(H * W, np.uint8)
    m[idx[off[f]:off[f + 1]]] = 1
    return m.reshape(H, W)


'''
Outer boundary of the mask's largest blob as (N, 2) pixel coordinates.
'''
def mask_boundary(m):
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))   # seal 1-2 px gaps
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        raise ValueError("empty mask")
    return max(cs, key=cv2.contourArea)[:, 0].astype(float)


'''
Mouth corners as pinch points (as above), plus the four corner points:
    outer corner = midpoint of the upper / lower pinch points
    inner corner = outer corner moved inset px towards the other outer corner
Returns (idx, points):
    idx      [(upper_idx, lower_idx) left, (upper_idx, lower_idx) right], for splitting the boundaries
    points   (4, 2) pixel coordinates: outer left, outer right, inner left, inner right
'''
def find_corners(cu, cl, tol=CORNER_TOL, inset=CORNER_INSET):
    d, nearest = map(np.asarray, KDTree(cl).query(cu))   # gap from each upper point to the lower lip
    mid = np.concatenate([cu, cl])[:, 0].mean()          # split the lips into left and right halves
    idx = []
    for sgn in (-1, 1):                         # left, then right
        side = sgn * (cu[:, 0] - mid) >= 0
        cand = np.flatnonzero(side & (d <= d[side].min() + tol))
        i = cand[np.argmax(sgn * cu[cand, 0])]
        idx.append((i, nearest[i]))
    (ul, ll), (ur, lr) = idx
    left, right = ((cu[ul] + cl[ll]) / 2)-7, ((cu[ur] + cl[lr]) / 2)+7
    step = inset * (right - left) / np.linalg.norm(right - left)
    return idx, np.array([left, right, left + step, right - step])


'''
Cut closed boundary c at indices i (left corner) and j (right corner); both arcs run left -> right.
'''
def split_at_corners(c, i, j):
    c = np.roll(c, -i, axis=0)
    j = (j - i) % len(c)
    return c[:j + 1], np.vstack([c[j:], c[:1]])[::-1]


'''
Bend a contour so it starts at corner a and ends at corner b. Each end moves onto its corner, and the
shift fades out smoothly over the first / last `fade` fraction of the contour's length, so the middle
of the contour stays where it is.
'''
def meet_corners(curve, a, b, fade=CORNER_FADE):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
    t = s / s[-1]                                               # 0 at the left end, 1 at the right end
    weight = lambda x: (1 - np.clip(x / fade, 0, 1)) ** 2       # 1 at the end, 0 from `fade` onwards
    return curve + np.outer(weight(t), a - curve[0]) + np.outer(weight(1 - t), b - curve[-1])


'''
Smooth out bumps along an open contour: Gaussian smoothing of its points with a std of sigma px,
after resampling them 1 px apart so sigma is a distance along the contour. The ends shrink in a little;
meet_corners() moves them back onto the corners afterwards.
'''
def smooth_contour(curve, sigma=CONTOUR_SMOOTH):
    length = np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()
    curve = resample(curve, max(int(length), 2))
    return gaussian_filter1d(curve, sigma, axis=0, mode="nearest")


'''
n evenly spaced points along a polyline, by arc length.
'''
def resample(curve, n):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
    return interp1d(s, curve, axis=0)(np.linspace(0, s[-1], n))


'''
Smooth the outer corners over consecutive frames with a Gaussian of std sigma frames, then place the
inner corners inset px inwards from the smoothed outer corners again. Failed (NaN) frames are left out
of the average and stay NaN. corners is (F, 4, 2): outer left, outer right, inner left, inner right.
'''
def smooth_corners(corners, sigma=CORNER_SMOOTH, inset=CORNER_INSET):
    ok = ~np.isnan(corners).any(axis=(1, 2))
    outer = np.where(ok[:, None, None], corners[:, :2], 0)
    # Weighted average over time: dividing by the smoothed weights ignores the failed frames
    outer = (gaussian_filter1d(outer, sigma, axis=0, mode="nearest") /
             gaussian_filter1d(ok.astype(float), sigma, mode="nearest")[:, None, None])
    left, right = outer[:, 0], outer[:, 1]
    step = inset * (right - left) / np.linalg.norm(right - left, axis=1, keepdims=True)
    smoothed = np.stack([left, right, left + step, right - step], axis=1)
    smoothed[~ok] = np.nan
    return smoothed


'''
Split a single frame's lip boundaries cu, cl into outer and inner contours that meet at the given
corners (4, 2): outer left, outer right, inner left, inner right. Returns (4, n_samples, 2) pixel
coordinates: upper outer, upper inner, lower outer, lower inner.
'''
def extract_frame(cu, cl, corners, n_samples):
    (ul, ll), (ur, lr) = find_corners(cu, cl)[0]                             # pinch points to split at
    by_height = lambda arc: arc[:, 1].mean()                                  # image y points down
    u_top, u_bot = sorted(split_at_corners(cu, ul, ur), key=by_height)
    l_top, l_bot = sorted(split_at_corners(cl, ll, lr), key=by_height)
    u_top, u_bot, l_top, l_bot = map(smooth_contour, (u_top, u_bot, l_top, l_bot))
    # Outer contours meet at the outer corners, inner contours at the inner corners
    outer, inner = corners[:2], corners[2:]
    arcs = [meet_corners(u_top, *outer), meet_corners(u_bot, *inner),
            meet_corners(l_bot, *outer), meet_corners(l_top, *inner)]
    return np.stack([resample(a, n_samples) for a in arcs])


'''
Write predicted_corners_{video_name}.mp4 to out_dir: the original video overlaid with each frame's
4 lip contours and 4 corner points (outer = filled dot, inner = ring). Frames that failed are left bare.
'''
def write_overlay_video(video_path, contours, corners, out_dir):
    colors = [(0, 0, 255), (0, 165, 255), (255, 0, 0), (255, 255, 0)]   # BGR: UO red, UI orange, LO blue, LI cyan
    name = os.path.splitext(os.path.basename(video_path))[0]
    path = os.path.join(out_dir, f"predicted_corners_{name}.mp4")

    cap = cv2.VideoCapture(video_path)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS), (W, H))

    for c, k in zip(contours, corners):
        ok, img = cap.read()
        if not ok:
            break
        if not np.isnan(c).any():
            pts = c.round().astype(np.int32)
            for curve, color in zip(pts, colors):
                cv2.polylines(img, [curve], False, color, 7, cv2.LINE_AA)
            for n, (x, y) in enumerate(k.round().astype(int)):            # outer L, outer R, inner L, inner R
                cv2.circle(img, (int(x), int(y)), 8, (0, 255, 0), -1 if n < 2 else 3, cv2.LINE_AA)
        writer.write(img)

    cap.release()
    writer.release()
    print("wrote", path)


def get_sam2_contours_and_corners(saved_state_path, frame_height, frame_width):
    # Load the upper and lower lip masks
    masks = load_sam2_masks(saved_state_path)
    (ui, uo), (li, lo) = masks["upper"], masks["lower"]
    F = len(uo) - 1

    # Pass 1: each frame's lip boundaries and pinch corners
    boundaries: list[tuple[np.ndarray, np.ndarray] | None] = [None] * F    # None for failed frames
    corners = np.full((F, 4, 2), np.nan)
    with tqdm(total=len(uo), desc="Extracting Contours and Corners from SAM2 Masks") as pbar:
        for f in range(F):
            try:
                cu = mask_boundary(decode_mask(ui, uo, f, frame_width, frame_height))
                cl = mask_boundary(decode_mask(li, lo, f, frame_width, frame_height))
                corners[f] = find_corners(cu, cl)[1]
                boundaries[f] = (cu, cl)
            except Exception as e:
                print(f"frame {f}: {e}")
            pbar.update(1)

    # Smooth the corners over consecutive frames to remove jitter
    corners = smooth_corners(corners)

    # Pass 2: split each frame's boundaries into contours that meet at the smoothed corners
    contours = np.full((F, 4, 64, 2), np.nan, np.float32)
    for f, b in enumerate(boundaries):
        if b is None:
            continue
        try:
            cu, cl = b
            contours[f] = extract_frame(cu, cl, corners[f], 64)
        except Exception as e:
            print(f"frame {f}: {e}")

    return contours, corners


# def main():
#     ap = argparse.ArgumentParser(description="Turn a clip's SAM2 lip masks into lip-contour targets.")
#     ap.add_argument("--video", required=True, help="Input video file path")
#     ap.add_argument("--input", default=None,
#                     help="clip results .npz with upper/lower_indices, upper/lower_offsets and mp_results_list")
#     ap.add_argument("--output", default=None,
#                     help="output .npz for all frames (default: <results name>_lip_targets.npz next to the input)")
#     ap.add_argument("--n-samples", type=int, default=64,
#                     help="points per contour after resampling (default: 64)")
#     args = ap.parse_args()

#     cap = cv2.VideoCapture(args.video)
#     if not cap.isOpened():
#         raise RuntimeError(f"Could not open video: {args.video}")

#     # Get input frame dimensions
#     W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()

#     out = args.output or os.path.splitext(args.input)[0] + "_lip_targets.npz"

#     # Load the upper and lower lip masks
#     masks = load_sam2_masks(args.input)
#     (ui, uo), (li, lo) = masks["upper"], masks["lower"]
#     F = len(uo) - 1

#     # Pass 1: each frame's lip boundaries and pinch corners
#     boundaries: list[tuple[np.ndarray, np.ndarray] | None] = [None] * F    # None for failed frames
#     corners = np.full((F, 4, 2), np.nan)
#     for f in range(F):
#         try:
#             cu = mask_boundary(decode_mask(ui, uo, f, W, H))
#             cl = mask_boundary(decode_mask(li, lo, f, W, H))
#             corners[f] = find_corners(cu, cl)[1]
#             boundaries[f] = (cu, cl)
#         except Exception as e:
#             print(f"frame {f}: {e}")
#         if f % 100 == 0:
#             print(f"frame {f}/{F}")

#     # Smooth the corners over consecutive frames to remove jitter
#     corners = smooth_corners(corners)

#     # Pass 2: split each frame's boundaries into contours that meet at the smoothed corners
#     contours = np.full((F, 4, args.n_samples, 2), np.nan, np.float32)
#     for f, b in enumerate(boundaries):
#         if b is None:
#             continue
#         try:
#             cu, cl = b
#             contours[f] = extract_frame(cu, cl, corners[f], args.n_samples)
#         except Exception as e:
#             print(f"frame {f}: {e}")

#     np.savez_compressed(out, contours=contours, corners=corners)
#     print(f"{(~np.isnan(contours[:, 0, 0, 0])).sum()}/{F} frames extracted -> {out}")

#     write_overlay_video(args.video, contours, corners, os.path.dirname(os.path.abspath(out)))


# if __name__ == "__main__":
#     main()
