# ====== Step 0: Import Libraries ====== #

# System
import os
import sys
import keyboard
import argparse
from tqdm import tqdm
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Computation and data handling
from matplotlib import patches
import numpy as np
import torch

# Visualization and image processing
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pandas as pd

# SAM2
from sam2.build_sam import build_sam2_video_predictor           #type:ignore   
from sam2.sam2_video_predictor import SAM2VideoPredictor        #type:ignore   
from sam2_resources import sam2_utils

# MediaPipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe_resources.mp_utils as mp_utils

# Processing SAM2 Masks and refineing Mediapipe landmarks
import mask_processing.mask_processor as mask_processor
import refine_landmarks.landmark_refinement_utils as refinement_utils
import refine_landmarks.op1_constrained_spline as con_spline

# ====== Step 1: Setup the environment ====== #

UPPER_LIP_OBJ_ID = 1
LOWER_LIP_OBJ_ID = 2
TONGUE_OBJ_ID = 3

# Parse the command line arguments
parser = argparse.ArgumentParser(description='Refine MediaPipe mouth landmarks using SAM2 segmentation masks')
parser.add_argument('--v', '--video_filename', type=str, default='tc6.mp4', help='.mp4 filename of test video in mocapvids directory')
parser.add_argument('--debug', '--debug_mode', type=int, default=0, help='Enter 1 if you want to load the saved state (default: 0)')
parser.add_argument('--bm', '--base_model', type=str, default='s', help='SAM2.1 base model (options: s, b, l, t)')
parser.add_argument('--d', '--dataset', type=str, default='maya', help='Name of dataset to train on (options: "maya", "lapa" or "hybrid")')
parser.add_argument('--sv', '--save_video', type=int, default=0, help='Whether to save the output segmentation video (boolean)')
parser.add_argument('--npp', '--num_prompts_per_part', type=int, default=4, help='Number of prompts to add for each mouth part (integer)')
parser.add_argument('--cbb', '--create_bounding_boxes', type=int, default=0, help='Whether to create bounding boxes for each mouth part (1 or 0)')
parser.add_argument('--fw', '--finetuned_weights', type=str, default=None, help="Name of weights file in finetuned_weights folder")
args = parser.parse_args()

# Initialize some configuration parameters
base_model = "sam2.1" + args.bm
dataset = args.d
fw_idx = -1   
# If the user would like to use finetuned weights, get the index of the finetuning results for these weights
if args.fw is not None:
    # Get the row index of the results for these finetuned weights
    ft_res_df = pd.read_csv("finetuning_sam2/finetuning_results.csv")
    fw_idx = ft_res_df.index[ft_res_df['finetuned_weights'] == 'finetuning_sam2/finetuned_weights/' + args.fw][0]

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


# ====== Step 2: Define Helper Functions ====== #

# ------ 2.3: Visualization and Debugging Functions ------ #
def save_state(sam2_video_segs, mp_lip_results_list, save_path):
    np.savez_compressed(
        save_path,
        sam2_video_segs=sam2_video_segs,
        mp_lip_results_list=mp_lip_results_list
    )
    print(f"Saved state to {save_path}")

def load_saved_state(load_path):
    data = np.load(load_path, allow_pickle=True)
    sam2_video_segs = data['sam2_video_segs'].item()
    mp_lip_results_list = data['mp_lip_results_list'].tolist()
    print(f"Loaded state from {load_path}")
    return sam2_video_segs, mp_lip_results_list

# ====== Step 3: Initialize SAM2 predictor and Mediapipe Landmarker ====== #

sam2_predictor = sam2_utils.initialize_sam2(args.bm, device, args.fw)
mp_detector = mp_utils.initialize_mp()

# ====== Step 4: Load and sort the video frames for SAM2 segmentation ====== #

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = os.path.join("mocapvids/vid_frames", os.path.splitext(args.v)[0])

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# ====== Step 5: Add prompts for different mouth parts ====== #
if not args.debug:
    # SAM 2 requires stateful inference for interactive video segmentation, so we need to 
    # initialize an 'inference state' on this video. During initialization, it loads all 
    # the JPEG frames in `video_path` and stores their pixels in `inference_state`.
    inference_state = sam2_predictor.init_state(video_path=video_dir)
else:
    inference_state = None

obj_ids = []
prompts = {}

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

# We will add `args.npp` prompts for each mouth part, evenly spaced throughout the video.
prompt_spacing = len(frame_names) // args.npp

if not args.debug:
    # ------ Upper Lip ------
    decent_ul_boxes = []
    decent_ul_boxes_4 = [[408, 840, 628, 882], [377, 700, 608, 860], [428, 782, 684, 846], [446, 777, 692, 824]]
    decent_ul_boxes_5 = [[412, 851, 621, 882], [432, 680, 684, 817], [463, 689, 695, 795], [457, 749, 684, 806], None]
    decent_ul_boxes_6 = [[412, 838, 621, 882], [474, 780, 632, 858], [519, 713, 737, 811], [443, 769, 668, 851], [448, 757, 690, 795], [477, 744, 681, 800]]

    if args.cbb == 0:   # Load the pre-made bounding boxes for the upper and bottom lips
        if args.npp == 4:
            decent_ul_boxes = decent_ul_boxes_4
        elif args.npp == 5:
            decent_ul_boxes = decent_ul_boxes_5
        elif args.npp == 6:
            decent_ul_boxes = decent_ul_boxes_6
        else:
            raise ValueError(f"Unsupported number of prompts per part: {args.npp}. Supported values are 4, 5, or 6.")

    for prompt_idx in range(args.npp):
        ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        # Create/Select the bounding box
        if args.cbb == 1:
            box_ul = sam2_utils.generate_prompt_box(
                frame=np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))), 
                mouth_part="upper lip", 
                color = mouth_parts[1]["color"]
            )
            print("--> New ul box created: ", box_ul, '\n')
        else:
            if prompt_idx < len(decent_ul_boxes):
                box_ul = decent_ul_boxes[prompt_idx]
            else:
                raise ValueError(f"Not enough pre-made bounding boxes for the upper lip. Requested {args.npp}, but only {len(decent_ul_boxes)} available.")

        # Try adding additional points as prompts
        points_ul, labels_ul = sam2_utils.generate_prompt_points(
            frame=np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))), 
            mouth_part="upper lip"
        )

        # Add prompts and get predictions
        if box_ul is not None:
            _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box_ul,
                points=points_ul,
                labels=labels_ul
            )

    # ------ Bottom Lip ------
    decent_bl_boxes = []
    decent_bl_boxes_4 = [[406, 878, 626, 935], [392, 809, 583, 931], [414, 844, 708, 911], [434, 815, 704, 878]]
    decent_bl_boxes_5 = [[417, 886, 626, 931], [423, 824, 681, 962], [468, 780, 675, 842], None, [448, 820, 706, 886]]
    decent_bl_boxes_6 = [[417, 880, 630, 926], [486, 862, 626, 931], [539, 806, 746, 918], [423, 855, 712, 906], [450, 800, 699, 849], [470, 800, 695, 835]]

    if args.cbb == 0:   # Load the pre-made bounding boxes for the upper and bottom lips
        if args.npp == 4:
            decent_bl_boxes = decent_bl_boxes_4
        elif args.npp == 5:
            decent_bl_boxes = decent_bl_boxes_5
        elif args.npp == 6:
            decent_bl_boxes = decent_bl_boxes_6
        else:
            raise ValueError(f"Unsupported number of prompts per part: {args.npp}. Supported values are 4, 5, or 6.")

    for prompt_idx in range(args.npp):
        ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
        ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

        # Create/Select the bounding box
        if args.cbb == 1:
            box_bl = sam2_utils.generate_prompt_box(
                frame=np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))), 
                mouth_part="bottom lip", 
                color = mouth_parts[2]["color"]
            )
            print("--> New bl box created: ", box_bl, "\n")
        else:
            if prompt_idx < len(decent_bl_boxes):
                box_bl = decent_bl_boxes[prompt_idx]
            else:
                raise ValueError(f"Not enough pre-made bounding boxes for the bottom lip. Requested {args.npp}, but only {len(decent_bl_boxes)} available.")

        # Try adding additional points as prompts
        points_bl, labels_bl = sam2_utils.generate_prompt_points(
            frame=np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))), 
            mouth_part="bottom lip"
        )

        if box_bl is not None:
            # Add prompts and get predictions
            _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box_bl,
                points=points_bl,
                labels=labels_bl,
            )

    # ------ Tongue ------
    # for prompt_idx in range(args.npp):
    #     ann_frame_idx = prompt_idx * prompt_spacing  # the frame index we interact with
    #     ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)

    #     # Create bounding box
    #     box_tg = sam2_utils.generate_prompt_box(
    #         frame=np.array(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))), 
    #         mouth_part="tongue", 
    #         color = mouth_parts[3]["color"]
    #     )

    #     # Add prompts and get predictions
    #     if box_tg is not None:
    #         _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
    #             inference_state=inference_state,
    #             frame_idx=ann_frame_idx,
    #             obj_id=ann_obj_id,
    #             box=box_tg,
    #         )

    # ====== Step 6: Run SAM2 Video Object Segmentation and Tracking ====== #

    # run propagation throughout the video and collect the results in a dict
    sam2_video_segs = {}  # sam2_video_segs contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(inference_state):
        sam2_video_segs[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # ====== Step 7: Generate MediaPipe Landmark Predictions ====== #
    mp_results_list = mp_utils.get_mp_landmarks(os.path.join("mocapvids", args.v), mp_detector)
    mp_lip_results_list = mp_utils.isolate_lip_landmarks(mp_results_list)

    save_path = f"saved_states/{args.v.split('.mp4')[0]}_{args.d}_sam2{args.bm}_fw{fw_idx}_{args.npp}pp_mp_results.npz"
    save_state(sam2_video_segs, mp_lip_results_list, save_path)
else:
    # If the user wants to load the saved state, we can skip the above steps and directly load the SAM2 
    # segmentation results and MediaPipe landmark predictions from a .npz file.

    # ====== Optional: Save the SAM2 segmentation results and MediaPipe landmark predictions ====== #
    save_path = f"saved_states/{args.v.split('.mp4')[0]}_{args.d}_sam2{args.bm}_fw{fw_idx}_{args.npp}pp_mp_results.npz"
    sam2_video_segs, mp_lip_results_list = load_saved_state(save_path)

num_example_frames = 6
example_frame_inc = len(frame_names) // num_example_frames

# ====== Step 8: Refine MediaPipe Predictions ====== #
'''
with tqdm(total=len(mp_lip_results_list), desc="Refining MediaPipe Lip Landmarks with SAM2 Masks") as pbar:
    for frame_idx, (frame_mp_lms, frame_sam2_segs) in enumerate(zip(mp_lip_results_list, sam2_video_segs.values())):
        if frame_idx % example_frame_inc == 0:
            print(f"Refining frame {frame_idx} / {len(mp_lip_results_list)}")

            for obj_id, raw_mask in frame_sam2_segs.items():
                raw_mask = raw_mask.squeeze()
                raw_mask = mask_processor.clean_mask(raw_mask)

            refinement_utils.visualize_mp_sam_frame(
                frame=np.array(Image.open(os.path.join(video_dir, frame_names[frame_idx]))),
                frame_idx=frame_idx,
                sam2_video_segs=sam2_video_segs,
                mp_lip_results_list=mp_lip_results_list,
                alpha=0.5,
                draw_indices=True
            )
            
        # # Get the SAM2 masks for this frame
        # upper_lip_mask = frame_sam2_segs.get(1, None)
        # lower_lip_mask = frame_sam2_segs.get(2, None)

        # # Clean masks using morphological operations and keep largest connected component
        # if upper_lip_mask is not None:
        #     upper_lip_mask = mask_processor.clean_mask(upper_lip_mask)
        # if lower_lip_mask is not None:
        #     lower_lip_mask = mask_processor.clean_mask(lower_lip_mask)
        
        # # Turn clean masks to splines
        # upper_lip_spline = None if upper_lip_mask is None else mask_processor.mask_to_spline(upper_lip_mask)
        # lower_lip_spline = None if lower_lip_mask is None else mask_processor.mask_to_spline(lower_lip_mask)

        # Refine the MediaPipe landmarks using the SAM2 masks
        # refined_landmarks = refinement_utils.refine_landmarks_with_masks(
        #     mp_landmarks=mp_lip_landmarks,
        #     sam2_masks=sam2_masks,
        #     mouth_parts=mouth_parts
        # )

        # # Update the refined landmarks in the list
        # mp_lip_results_list[frame_idx] = refined_landmarks

        pbar.update(1)'''

# ====== STEP 9: Visualize some of the segmented frames ====== #
# frame_stride = 30
# for out_frame_idx in range(0, len(frame_names), frame_stride):
#     plt.figure(figsize=(6, 4))
#     plt.title(f"frame {out_frame_idx}")
#     plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
#     for out_obj_id, out_mask in sam2_video_segs[out_frame_idx].items():
#         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

#     plt.axis('off')
#     plt.show()



# ====== Step 10 (Optional): Create Segmentation Video ====== #

if args.sv == 1:
    # Load first frame to get video dimensions
    first_frame = np.array(
        Image.open(os.path.join(video_dir, frame_names[0]))
    )

    height, width = first_frame.shape[:2]

    # Output video path
    output_video_path = f"sam2_guided_mp_results/{base_model}_{dataset}_fw{fw_idx}_{args.npp}pp_{args.v}"

    # Video writer
    fps = 30

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    alpha = 0.5

    # ----- Generate Frames ------ #
    example_frames = len(frame_names) // 5

    for frame_idx in range(len(frame_names)):

        frame_path = os.path.join(video_dir, frame_names[frame_idx])

        frame = np.array(
            Image.open(frame_path).convert("RGB")
        )

        output_frame = frame.copy()
        h, w = frame.shape[:2]

        # Draw masks
        if frame_idx in sam2_video_segs:

            for obj_id, raw_mask in sam2_video_segs[frame_idx].items():
                
                if obj_id == UPPER_LIP_OBJ_ID or obj_id == LOWER_LIP_OBJ_ID:   # Create sam2-points for upper and bottom lips
                    raw_mask = raw_mask.squeeze()

                    # Clean the raw sam2 mask to remove holes and noise
                    clean_mask_result = mask_processor.clean_mask(raw_mask)

                    # Convert the cleaned mask into a smooth spline
                    spline = mask_processor.mask_to_spline(clean_mask_result)

                    # Find the Mediapipe-predicted mouth corners for anchoring
                    inner_left_corner, outer_left_corner, inner_right_corner, outer_right_corner = con_spline.find_mp_corners(mp_lip_results_list[frame_idx], h, w)

                    # Sample points along the predicted SAM2 lip contours
                    upper_points, lower_points = con_spline.refine_lip_contour(obj_id, spline, inner_left_corner, outer_left_corner, inner_right_corner, outer_right_corner, 11, output_frame)

                    # Visualize some sample frames
                    if (frame_idx % example_frames == 0):
                        vis_frame = frame.copy()
                        # Convert RGB -> BGR
                        vis_frame = cv2.cvtColor(
                            vis_frame,
                            cv2.COLOR_RGB2BGR
                        )
                        '''
                        # Visulaize the mask cleanup
                        mask_processor.visualize_mask_cleanup(
                            frame,
                            raw_mask, 
                            clean_mask_result
                        )

                        # Visualize the spline
                        mask_processor.visualize_spline(
                            frame,
                            spline
                        )
                        '''
                        #left_corner, right_corner = con_spline.find_mp_corner_midpoint(mp_lip_results_list[frame_idx], h, w)
                        # inner_left_corner, outer_left_corner, inner_right_corner, outer_right_corner = con_spline.find_mp_corners(mp_lip_results_list[frame_idx], h, w)
                        
                        #upper_arc, lower_arc = con_spline.refine_lip_contour(spline, left_corner, right_corner, 11)
                        # upper_points, lower_points = con_spline.refine_lip_contour_4(obj_id, spline, inner_left_corner, outer_left_corner, inner_right_corner, outer_right_corner, 11)

                        # corners = [inner_left_corner, outer_left_corner, inner_right_corner, outer_right_corner]
                        # for corner in corners:
                        #     cv2.circle(
                        #         vis_frame,
                        #         (corner[0], corner[1]),
                        #         3,
                        #         (0, 0, 0),  # Black
                        #         -1
                        #     )

                        # if upper_arc is not None: # Only add spline if a lip was located
                        #     pts = np.round(upper_arc).astype(np.int32)

                        #     # Draw closed polyline
                        #     cv2.polylines(
                        #         vis_frame,
                        #         [pts],
                        #         isClosed=False,
                        #         color=(255, 0, 0),  # Green  
                        #         thickness=2,
                        #         lineType=cv2.LINE_AA
                        #     )
                        
                        # if lower_arc is not None: # Only add spline if a lip was located
                        #     pts = np.round(lower_arc).astype(np.int32)

                        #     # Draw closed polyline
                        #     cv2.polylines(
                        #         vis_frame,
                        #         [pts],
                        #         isClosed=False,
                        #         color=(0, 0, 255),   # Red
                        #         thickness=2,
                        #         lineType=cv2.LINE_AA
                        #     )

                        # ========== Visualize the points ============ 
                        # cv2.namedWindow(
                        #     "Average Corners",
                        #     cv2.WINDOW_NORMAL
                        # )

                        # if upper_points is not None:
                        #     for p in upper_points:
                        #         cv2.circle(
                        #             vis_frame,
                        #             tuple(np.round(p).astype(int)),
                        #             3,
                        #             (0, 0, 255),   # Red
                        #             -1
                        #         )

                        # if lower_points is not None:
                        #     for p in lower_points:
                        #         cv2.circle(
                        #             vis_frame,
                        #             tuple(np.round(p).astype(int)),
                        #             3,
                        #             (255, 0, 0),   # Blue
                        #             -1
                        #         )

                        # cv2.imshow("Average Corners", vis_frame)

                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        
                    # Add results to output video frame
                    top_colors = [(0, 0, 255), (255, 0, 0)]
                    bottom_colors = [(0, 255, 155), (255, 178, 0)]

                    if obj_id == 1:
                        colors = top_colors
                    else:
                        colors = bottom_colors

                    # Add mask results
                    mask_uint8 = clean_mask_result.astype(np.uint8)

                    # Overlay the cleaned mask for that lip
                    output_frame[mask_uint8 > 0] = (
                        0.8 * output_frame[mask_uint8 > 0] +
                        0.2 * np.array(colors[0])
                    ).astype(np.uint8)

                    # Add predicted SAM points
                    if upper_points is not None:
                        for p in upper_points:
                            cv2.circle(
                                output_frame,
                                tuple(np.round(p).astype(int)),
                                3,
                                colors[0],   
                                -1
                            )

                    if lower_points is not None:
                        for p in lower_points:
                            cv2.circle(
                                output_frame,
                                tuple(np.round(p).astype(int)),
                                3,
                                colors[1],   # Blue
                                -1
                            )
                else:   # Tongue Mask
                    raw_mask = raw_mask.squeeze()

                    # Clean the raw sam2 mask to remove holes and noise
                    clean_mask_result = mask_processor.clean_mask(raw_mask)

                    # Convert bool mask to uint8
                    mask_uint8 = clean_mask_result.astype(np.uint8)

                    # Overlay the cleaned mask for that lip
                    output_frame[mask_uint8 > 0] = (
                        0.2 * output_frame[mask_uint8 > 0] +
                        0.8 * np.array([0, 0, 0])
                    ).astype(np.uint8)
                
                # Add the mediapipe landmarks for comparison
                output_frame = mp_utils.add_lms_to_frame(output_frame, mp_lip_results_list[frame_idx], False)
                
        # Convert RGB -> BGR
        output_frame_bgr = cv2.cvtColor(
            output_frame,
            cv2.COLOR_RGB2BGR
        )

        video_writer.write(output_frame_bgr)
        

    # ====== Finalize ====== #

    video_writer.release()

    print(f"Saved video to: {output_video_path}")