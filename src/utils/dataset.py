import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    """
    LandCover.ai dataset loader (train only foreground classes)
    Classes used:
        raw: 
            0 = background
            1 = building
            2 = woodland
            3 = water
            4 = road

        remapped:
            building -> 0
            woodland -> 1
            water -> 2
            road -> 3

    Background is mapped to IGNORE_INDEX for loss.
    """

    IGNORE_INDEX = 255   # ignored during training

    def __init__(self, images_dir, masks_dir, all_classes, classes=None, augmentation=None, preprocessing=None):
        self.ids = sorted(os.listdir(images_dir))
        self.images = [os.path.join(images_dir, f) for f in self.ids]
        self.masks = [os.path.join(masks_dir, f) for f in self.ids]

        # Mapping original dataset labels to training labels
        # include background again
        # raw → new index mapping
        self.mapping = {0:0, 1:1, 2:2, 3:3, 4:4}
        self.num_classes = 5


        self.valid_raw_labels = set(self.mapping.keys())

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # ---- Load Image ----
        img = cv2.imread(self.images[i], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image missing: {self.images[i]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ---- Load Mask ----
        mask = cv2.imread(self.masks[i], cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask missing: {self.masks[i]}")

        if mask.ndim > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # ---- Clean invalid values ----
        # Any label not in {0–4} becomes background
        mask = np.where(np.isin(mask, [0,1,2,3,4]), mask, 0)

        # ---- Remap raw classes to training indices ----
        mapped_mask = np.full_like(mask, self.IGNORE_INDEX, dtype=np.uint8)
        for raw, new in self.mapping.items():
            mapped_mask[mask == raw] = new

        # background stays IGNORE_INDEX (255)

        # ---- Apply Augmentation ----
        if self.augmentation:
            augmented = self.augmentation(image=img, mask=mapped_mask)
            img, mapped_mask = augmented["image"], augmented["mask"]

        # ---- Apply Preprocessing ----
        if self.preprocessing:
            processed = self.preprocessing(image=img, mask=mapped_mask)
            img, mapped_mask = processed["image"], processed["mask"]
        else:
            img = img.astype("float32") / 255.0

        # ---- Tensor conversion ----
        if img.ndim == 3 and img.shape[-1] in (1, 3):
            img = img.transpose(2, 0, 1)  # HWC -> CHW

        # Final tensors
        img_tensor = torch.tensor(img, dtype=torch.float32)
        mask_tensor = torch.tensor(mapped_mask, dtype=torch.long)

        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.ids)
