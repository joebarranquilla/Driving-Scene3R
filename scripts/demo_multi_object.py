#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Copyright (c) Meta Platforms, Inc. and affiliates.


# ## 1. Imports and Model Loading

# In[ ]:


import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import torch, gc
torch.cuda.empty_cache()
gc.collect()
print("CUDA memory cleared.")


# In[ ]:


import os
import uuid
import imageio
import numpy as np
from IPython.display import Image as ImageDisplay

from inference import Inference, ready_gaussian_for_video_rendering, load_image, load_masks, display_image, make_scene, render_video, interactive_visualizer
print("Modules imported successfully.")


# In[3]:


PATH = os.getcwd()
config_path = f"{PATH}/notebook/checkpoints/pipeline.yaml"
inference = Inference(config_path, compile=False)


# ## 2. Load input image to lift to 3D (multiple objects)

# In[4]:


PATH = os.getcwd()
IMAGE_PATH = f"{PATH}/notebook/images/new_mask_seq4/000000.png"
IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))

image = load_image(IMAGE_PATH)
masks = load_masks(os.path.dirname(IMAGE_PATH), extension=".png")
#display_image(image, masks)

print(f"Loaded image: {IMAGE_PATH}")


# ## 3. Generate Gaussian Splats

# In[ ]:


outputs = [inference(image, mask, seed=42) for mask in masks]

print("Gaussian splats generated for all objects.")


# ## 4. Visualize Gaussian Splat of the Scene
# ### a. Animated Gif

# In[ ]:


scene_gs = make_scene(*outputs)
print("Scene Gaussian splats created.")
#scene_gs.scale_to_target_size()
# export posed gaussian splatting (as point cloud)
scene_gs.save_ply(f"{PATH}/notebook/gaussians/{IMAGE_NAME}_posed.ply")
print(f"Posed Gaussian splats saved as PLY: {IMAGE_NAME}_posed.ply")
scene_gs = ready_gaussian_for_video_rendering(scene_gs)
scene_gs.save_ply(f"{PATH}/notebook/gaussians/multi/{IMAGE_NAME}.ply")
