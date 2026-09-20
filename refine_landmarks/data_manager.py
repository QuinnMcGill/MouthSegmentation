import numpy as np

# ======= Stage 1: SAM2 Segmentation results and MediaPipe landmarks ======= #

def save_state_stage1(upper_indices, lower_indices, upper_offsets, lower_offsets, mp_results_list, mp_lip_results_list, save_path):
    np.savez_compressed(
        save_path,
        upper_indices=upper_indices,
        lower_indices=lower_indices,
        upper_offsets=upper_offsets,
        lower_offsets=lower_offsets,
        mp_results_list=mp_results_list,
        mp_lip_results_list=mp_lip_results_list
    )
    print(f"Saved (Stage 1) state to {save_path}")

def save_raw_sam2_masks(upper_indices, lower_indices, upper_offsets, lower_offsets, save_path):
    np.savez_compressed(
        save_path,
        upper_indices=upper_indices,
        lower_indices=lower_indices,
        upper_offsets=upper_offsets,
        lower_offsets=lower_offsets,
    )
    print(f"Saved raw SAM2 masks to {save_path}") 

def save_mp_results(mp_results_list, mp_lip_results_list, save_path):
    np.savez_compressed(
        save_path,
        mp_results_list=mp_results_list,
        mp_lip_results_list=mp_lip_results_list
    )
    print(f"Saved Mediapipe landmark results to {save_path}")

def load_saved_state_stage1(load_path):
    data = np.load(load_path, allow_pickle=True)
    upper_indices = data['upper_indices']
    lower_indices = data['lower_indices']
    upper_offsets = data['upper_offsets']
    lower_offsets = data['lower_offsets']
    mp_lip_results_list = data['mp_lip_results_list'].tolist()
    mp_results_list = data['mp_results_list']
    print(f"Loaded (Stage 1) state from {load_path}")
    return upper_indices, lower_indices, upper_offsets, lower_offsets, mp_results_list, mp_lip_results_list

def load_raw_sam2_masks(load_path):
    data = np.load(load_path, allow_pickle=True)
    upper_indices = data['upper_indices']
    lower_indices = data['lower_indices']
    upper_offsets = data['upper_offsets']
    lower_offsets = data['lower_offsets']
    print(f"Loaded raw sam2 masks from {load_path}")
    return upper_indices, lower_indices, upper_offsets, lower_offsets

def load_mp_results(load_path):
    data = np.load(load_path, allow_pickle=True)
    mp_lip_results_list = data['mp_lip_results_list'].tolist()
    mp_results_list = data['mp_results_list']
    print(f"Loaded Mediapipe landmarks from {load_path}")
    return mp_results_list, mp_lip_results_list

# ======= Stage 2: Results of combinging SAM2 Masks and Mediapipe landmarks ======= #

def save_state_stage2(sam2_mp_results, save_path):
    np.savez_compressed(
        save_path,
        sam2_mp_results=sam2_mp_results,
    )
    print(f"Saved (Stage 2) state to {save_path}")

def load_saved_state_stage2(load_path):
    data = np.load(load_path, allow_pickle=True)
    sam2_mp_results = data['sam2_mp_results'].tolist()
    print(f"Loaded (Stage 2) state from {load_path}")
    return sam2_mp_results

# ======= Helper Functions ======= #

def reconstruct_mask(indices, offsets, frame_idx, height, width):
    ''''
    Reconstructs a binary mask for a specific frame using the provided indices and offsets. 
    The results of stage are sparse, so we need to reconstruct the full mask for each frame.
    '''
    start = offsets[frame_idx]
    end = offsets[frame_idx + 1]

    mask = np.zeros(height * width, dtype=bool)
    mask[indices[start:end]] = True

    return mask.reshape(height, width)
