import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# 1. Load the model (Download the checkpoint file 'sam_vit_h_4b8939.pth' first)
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
print("Model loaded successfully.")
mask_generator = SamAutomaticMaskGenerator(sam)

# 2. Read the image
original_bgr = cv2.imread("dog.jpg")
image = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
print("Image loaded successfully.")

# 3. Generate masks
masks = mask_generator.generate(image)
print(f"Generated {len(masks)} masks.")
#save masks as separate images
for i, mask in enumerate(masks):
    seg = mask["segmentation"].astype(bool)
    mask_image = (seg * 255).astype("uint8")  # Convert boolean mask to uint8
    colored_obj = np.zeros_like(original_bgr)
    colored_obj[seg] = original_bgr[seg]
    b, g, r = cv2.split(original_bgr)
    alpha = mask_image
    rgba = cv2.merge([b, g, r, alpha])
    cv2.imwrite(f"mask_color_{i}_alpha.png", rgba)
print("Masks saved successfully as separate images (color+alpha).")

# 4. save masks as one image
# 4. save masks as one image containing only the masked objects in original color
combined_objects = np.zeros_like(original_bgr)
for mask in masks:
    seg = mask["segmentation"].astype(bool)
    combined_objects[seg] = original_bgr[seg]
cv2.imwrite("whole_image_mask.png", combined_objects)
print("Combined mask image saved (objects in original color, background black).")
"""

from PIL import Image

# 1. Open the two mask images
mask1 = Image.open('mask_color_4_alpha.png')
mask2 = Image.open('mask_color_9_alpha.png')

# --- Option 1: Simple Union (Logical OR) ---

# Make sure they are the same size for this operation
if mask1.size != mask2.size:
    print("Warning: Masks have different sizes. Resizing mask2 to match mask1 for Union.")
    mask2 = mask2.resize(mask1.size)

# Convert to binary for a cleaner union if they aren't already. 
# This assumes the object is anything that isn't black.
# We'll work with the images directly for now.
# NOTE: This can have strange effects on color, as it ORs the pixel values.
#union_mask = Image.blend(mask1, mask2, alpha=0.5) # Blending is one simple way to "union" with color. 0.5 makes them equal weight.

# A better way for binary masks would be using numpy:
import numpy as np
mask1_np = np.array(mask1)
mask2_np = np.array(mask2)
union_mask_np = np.maximum(mask1_np, mask2_np) # Take the pixel-wise max
union_mask = Image.fromarray(union_mask_np)

# Save the blended union mask
union_mask.save('union_mask.png')
print("Saved union mask to 'union_mask.png'")
"""