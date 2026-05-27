import cv2
import numpy as np
from rembg import remove

# 1. Load the original image
input_path = '000000_cropped.png'
img = cv2.imread(input_path)

# 2. Use AI to remove the background (returns an RGBA image)
# rembg expects RGB, so we flip channels
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
output_rgba = remove(img_rgb)

# 3. Separate the car (RGB) and the mask/alpha channel (A)
car_rgb = output_rgba[:, :, :3]
mask = output_rgba[:, :, 3]

# 4. Create your solid background (Match the original image size)
# For a WHITE background: use np.ones(...) * 255
# For a BLACK background: use np.zeros(...)
background_color = "white" # Change to "black" if desired

if background_color == "white":
    background = np.ones_like(car_rgb) * 255
else:
    background = np.zeros_like(car_rgb)

# 5. Blend the car onto the solid background using the mask
mask_3d = np.expand_dims(mask, axis=2) / 255.0
final_image = (car_rgb * mask_3d) + (background * (1 - mask_3d))

# 6. Save the final result (Convert back to BGR for OpenCV)
final_image_bgr = cv2.cvtColor(final_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite('isolated_car.jpg', final_image_bgr)