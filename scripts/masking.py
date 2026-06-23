#!/usr/bin/env python3
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def isolate_class_in_depth_map(npz_path, id2label_path, depth_map_path, target_class="car"):
    """
    Uses Mask2Former panoptic segmentation output to mask a depth map,
    isolating pixels belonging only to a specific target class.
    """
    # 1. Load the panoptic segmentation data
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Panoptic NPZ file not found: {npz_path}")
        
    print(f"Loading panoptic data from: {npz_path}")
    panoptic_data = np.load(npz_path)
    
    panoptic_seg = panoptic_data["panoptic_seg"]  # Shape: (H, W)
    segment_ids  = panoptic_data["segment_ids"]   # Shape: (N,)
    label_ids    = panoptic_data["label_ids"]     # Shape: (N,)
    
    # 2. Load the class lookup dictionary
    if not os.path.exists(id2label_path):
        raise FileNotFoundError(f"Label map JSON not found: {id2label_path}")
        
    with open(id2label_path, "r") as f:
        id2label = json.load(f)  # Format: {"0": "road", "1": "car", ...}

    # Inverse lookup: Find all numeric label_ids that match the target class string
    # (e.g., finding that 'car' corresponds to label_id 13)
    target_label_ids = [
        int(lbl_id) for lbl_id, class_name in id2label.items() 
        if class_name.lower() == target_class.lower()
    ]
    
    if not target_label_ids:
        print(f"Warning: Class '{target_class}' not found in id2label.json.")
        return None

    # 3. Find which specific segment instances belong to the target class
    # segment_ids represent the specific instances in the image (e.g., Instance #1002, #1005)
    matching_segment_indices = np.isin(label_ids, target_label_ids)
    target_segment_ids = segment_ids[matching_segment_indices]
    
    if len(target_segment_ids) == 0:
        print(f"No instances of '{target_class}' detected in this specific frame.")
        # Return an empty mask matching the frame dimensions
        car_mask = np.zeros_like(panoptic_seg, dtype=bool)
    else:
        print(f"Found {len(target_segment_ids)} instances of '{target_class}' in this frame.")
        # Create a boolean mask: True where pixels match any of our target segment IDs
        car_mask = np.isin(panoptic_seg, target_segment_ids)

    # 4. Load your actual depth map (Assuming a single-channel image or numpy array)
    # Note: Replace this with how your specific depth files are formatted (e.g., .png or .npy/.npz)
    if not os.path.exists(depth_map_path):
        raise FileNotFoundError(f"Depth map file not found: {depth_map_path}")
        
    print(f"Loading depth map from: {depth_map_path}")
    if depth_map_path.endswith('.npy') or depth_map_path.endswith('.npz'):
        # If depth is stored as raw float values in a numpy structure
        depth_data = np.load(depth_map_path)
        depth_map = depth_data['depth'] if depth_map_path.endswith('.npz') else depth_data
    else:
        # If depth is stored as a grayscale PNG/TIFF image
        depth_img = Image.open(depth_map_path)
        depth_map = np.array(depth_img, dtype=np.float32)

    # Sanity check for matching dimensions
    if depth_map.shape[:2] != panoptic_seg.shape:
        raise ValueError(
            f"Dimension mismatch! Depth map is {depth_map.shape[:2]}, "
            f"but panoptic segmentation is {panoptic_seg.shape}"
        )

    # 5. Apply the mask to the depth map
    # Pixels that are NOT cars are set to 0.0 (or np.nan depending on your pipeline)
    masked_depth_map = np.where(car_mask, depth_map, 0.0)
    
    return car_mask, depth_map, masked_depth_map


# --- Example usage visualization ---
if __name__ == "__main__":
    # Define paths based on your script outputs
    NPZ_FILE       = "000000.npz"
    LABEL_JSON     = "/usr/prakt/s0043/panoptic_predictions/id2label.json"
    DEPTH_FILE     = "000000_old.npz" # Placeholder path
    
    try:
        car_mask, original_depth, isolated_depth = isolate_class_in_depth_map(
            npz_path=NPZ_FILE,
            id2label_path=LABEL_JSON,
            depth_map_path=DEPTH_FILE,
            target_class="car"
        )
        
        # Plotting the results to verify visually
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(car_mask, cmap='gray')
        axes[0].set_title("Isolated Class Mask ('car')")
        axes[0].axis('off')
        
        axes[1].imshow(original_depth, cmap='plasma')
        axes[1].set_title("Original Depth Map")
        axes[1].axis('off')
        
        axes[2].imshow(isolated_depth, cmap='plasma')
        axes[2].set_title("Masked Depth (Cars Only)")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        #save the masked depth map if needed
        np.savez("000000_masked_depth.npz", depth=isolated_depth)
        
    except FileNotFoundError as e:
        print(f"\n[Execution stopped]: {e}")
        print("Please update the file paths at the bottom of the script to test with your real data.")