# import os
# import cv2
# import numpy as np

# # Path to your training masks
# mask_dir = 'C:\\Users\\JARVIS\\LandCover\\data\\train\\masks';

# unique_values = set()

# for fname in os.listdir(mask_dir):
#     if not fname.lower().endswith(('.png', '.jpg', '.tif')):
#         continue
#     mask = cv2.imread(os.path.join(mask_dir, fname), 0)
#     if mask is None:
#         continue
#     unique_values.update(np.unique(mask))

# print("🟩 Unique class values across dataset:", sorted(unique_values))
# print("🧮 Total unique classes found:", len(unique_values))
import cv2, numpy as np
from dataset import SegmentationDataset
from constants import Constants
import torch


import os, cv2, numpy as np

mask_dir = "C:\\Users\\JARVIS\\LandCover\\data\\train\\masks"
unique_classes = set()
for f in os.listdir(mask_dir):
    if f.endswith(".tif"):
        m = cv2.imread(os.path.join(mask_dir, f), 0)
        unique_classes.update(np.unique(m).tolist())

print("All unique mask values across training set:", sorted(unique_classes))

# images_dir = "C:\\Users\\JARVIS\\LandCover\\data\\train\\images"
# masks_dir = "C:\\Users\\JARVIS\\LandCover\\data\\train\\masks"

# ds = SegmentationDataset(
#     images_dir,
#     masks_dir,
#     all_classes=Constants.CLASSES.value,
#     classes=Constants.CLASSES.value
# )

# for i in range(3):
#     raw = cv2.imread(ds.masks[i], cv2.IMREAD_UNCHANGED)
#     print(ds.masks[i], "unique:", np.unique(raw)[:15])
#     img, mask = ds[i]
#     print("processed mask unique:", np.unique(mask.numpy())[:15])


# ds = SegmentationDataset(
#     r"C:\Users\JARVIS\LandCover\data\train\images",
#     r"C:\Users\JARVIS\LandCover\data\train\masks",
#     all_classes=["background", "building", "woodland", "water", "road"],
#     classes=["background", "building", "woodland", "water", "road"]
# )

# img, mask = ds[0]
# print(img.shape, img.dtype)
# print(mask.shape, mask.dtype)
# print(torch.unique(mask))
