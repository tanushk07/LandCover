import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    """
    LandCover.AI dataset.
    Reads images and segmentation masks, applies augmentations and preprocessing.

    Args:
        images_dir (str): Path to images folder
        masks_dir (str): Path to segmentation masks folder
        all_classes (list): List of all available classes
        classes (list): Subset of classes to train on
        augmentation (albumentations.Compose, optional): Data augmentation pipeline
        preprocessing (albumentations.Compose, optional): Data preprocessing pipeline
    """

    def __init__(self, images_dir, masks_dir, all_classes, classes=None, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # Convert class names to index positions
        self.class_values = [all_classes.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        # Masks are already encoded as 0–4 (background, building, woodland, water, road)
        self.id_to_class = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}


    def __getitem__(self, i):
        image = cv2.imread(self.images[i], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32') / 255.0

        raw_mask = cv2.imread(self.masks[i], cv2.IMREAD_UNCHANGED)
        if raw_mask is None:
            raise FileNotFoundError(self.masks[i])

        # Convert 0–14 raw mask → 0–4 semantic mask
        if raw_mask.ndim == 2:
            mask_index = np.vectorize(lambda v: self.id_to_class.get(int(v), 0))(raw_mask)
        else:
            rgb_flat = raw_mask.reshape(-1, 3)
            unique_rgbs = np.unique(rgb_flat, axis=0)
            palette_map = {tuple(rgb.tolist()): idx for idx, rgb in enumerate(unique_rgbs)}
            mask_single = np.zeros(raw_mask.shape[:2], dtype=np.uint8)
            flat_out = mask_single.reshape(-1)
            for rgb, idxval in palette_map.items():
                matches = np.all(rgb_flat == rgb, axis=1)
                flat_out[matches] = idxval
            mask_index = mask_single

        # ✅ Important: ensure dtype is discrete integer
        mask_index = mask_index.astype(np.uint8)

        # Apply augmentations and preprocessing
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_index)
            image, mask_index = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask_index)
            image, mask_index = sample['image'], sample['mask']

        # To tensor
        image = torch.as_tensor(image, dtype=torch.float32)
        if image.ndim == 3 and image.shape[2] in (1, 3):
            image = image.permute(2, 0, 1).contiguous()
        if image.ndim == 2:
            image = image.unsqueeze(0)

        mask_index = torch.as_tensor(mask_index, dtype=torch.long).squeeze()

        return image, mask_index

    def __len__(self):
        return len(self.ids)
