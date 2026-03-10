import io
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from tqdm import tqdm

def resize_images(images: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resizes multiple images to some target size.

    Assumes that the resulting images will be square.

    Args:
        images (np.ndarray): Input images with shape (T, H, W, C)
        target_size (int): Target image size (square)

    Returns:
        np.ndarray: Resized images with shape (T, target_size, target_size, C)
    """
    assert len(images.shape) == 4, f"Expected 4 dimensions in images but got: {len(images.shape)}"

    # If the images are already the target size, return them as is
    if images.shape[-3:] == (target_size, target_size, 3):
        return images.copy()

    # Get the number of images
    num_images = images.shape[0]

    # Create an empty array for the resized images
    # We assume the channel dimension C remains the same
    C = images.shape[3]
    resized_images = np.empty((num_images, target_size, target_size, C), dtype=images.dtype)

    # Resize each image
    for i in range(num_images):
        resized_images[i] = np.array(Image.fromarray(images[i]).resize((target_size, target_size)))

    return resized_images


def apply_image_aug(images: torch.Tensor, stronger: bool = False) -> torch.Tensor:
    """
    Apply image augmentations to a batch of images represented as a torch.Tensor of shape (C, T, H, W).

    Args:
        images: A torch.Tensor of shape (C, T, H, W) and dtype torch.uint8 representing a set of images.
        stronger (bool): Whether to apply stronger augmentations

    Returns:
        A torch.Tensor of the same shape and dtype with augmentations applied.
    """
    # Get dimensions
    _, _, H, W = images.shape
    assert H == W, "Image height and width must be equal"
    assert images.dtype == torch.uint8, f"Expected images dtype == torch.uint8 but got: {images.dtype}"

    # Convert to (T, C, H, W) format for compatibility with torchvision transforms
    images = images.permute(1, 0, 2, 3)

    # Detect consecutive duplicate images to avoid redundant augmentations
    # Build a list of (start_idx, end_idx, is_duplicate_group) tuples
    unique_groups = []
    num_images = len(images)
    i = 0
    while i < num_images:
        # Check if this image is the same as the next one
        group_start = i
        while i + 1 < num_images and torch.equal(images[i], images[i + 1]):
            i += 1
        group_end = i + 1  # end is exclusive
        group_size = group_end - group_start
        unique_groups.append((group_start, group_end, group_size))
        i += 1

    # Define augmentations with the same parameters for all images
    # 1. Random resized crop
    i, j, h, w = T.RandomResizedCrop.get_params(
        img=torch.zeros(H, W),  # Dummy tensor for getting params
        scale=(0.9, 0.9),  # 90% area
        ratio=(1.0, 1.0),  # Always maintain square aspect ratio
    )

    # 2. Random rotation (only for stronger augmentations)
    if stronger:
        angle = torch.FloatTensor(1).uniform_(-5, 5).item()
    else:
        angle = 0.0  # No rotation in default aug pipeline

    # 3. Color jitter – use wider ranges when `stronger` is True
    if stronger:
        brightness_factor = torch.FloatTensor(1).uniform_(0.7, 1.3).item()  # ±0.3
        contrast_factor = torch.FloatTensor(1).uniform_(0.6, 1.4).item()  # ±0.4
        saturation_factor = torch.FloatTensor(1).uniform_(0.5, 1.5).item()  # ±0.5
    else:
        brightness_factor = torch.FloatTensor(1).uniform_(0.8, 1.2).item()  # ±0.2
        contrast_factor = torch.FloatTensor(1).uniform_(0.8, 1.2).item()  # ±0.2
        saturation_factor = torch.FloatTensor(1).uniform_(0.8, 1.2).item()  # ±0.2
    hue_factor = torch.FloatTensor(1).uniform_(-0.05, 0.05).item()  # 0.05 hue

    # Apply the same transformations to unique images only
    results = []

    for group_idx, (group_start, group_end, group_size) in enumerate(unique_groups):
        # Only augment the first image in each group
        img = images[group_start]

        # 1. Apply random crop and resize back
        img = F.resized_crop(img, i, j, h, w, size=[H, W], antialias=True)

        # 2. Apply random rotation (skip if angle == 0)
        if stronger:
            img = F.rotate(img, angle, expand=False)

        # 3. Apply color jitter
        img = F.adjust_brightness(img, brightness_factor)
        img = F.adjust_contrast(img, contrast_factor)
        img = F.adjust_saturation(img, saturation_factor)
        img = F.adjust_hue(img, hue_factor)

        # Replicate the augmented image for all duplicates in this group
        for _ in range(group_size):
            results.append(img)

    # Combine results and revert to original shape (C, T, H, W)
    augmented_images = torch.stack(results)
    augmented_images = augmented_images.permute(1, 0, 2, 3)

    return augmented_images


def preprocess_image(
    images: np.ndarray,
    final_image_size: int,
    normalize_images: bool = False,
    use_image_aug: bool = True,
    stronger_image_aug: bool = False,
) -> torch.Tensor:
    """
    Preprocesses images for training.

    Resizes to final_image_size, permutes from (T, H, W, C) to (C, T, H, W),
    converts to torch.Tensor, optionally applies image augmentations, and optionally normalizes (no need to
    normalize if, e.g., the dataloader logic will normalize later).

    Args:
        images (np.ndarray): Images to be preprocessed
        final_image_size (int): Target size for resized images (square)
        normalize_images (bool): Whether the images should be normalized in the end
        stronger_image_aug (bool): Whether to apply stronger image augmentations

    Returns:
        torch.Tensor: Preprocessed images
    """
    assert isinstance(images, np.ndarray), f"Images are not of type `np.ndarray`! Got type: {type(images)}"
    assert images.dtype == np.uint8, f"Images do not have dtype `np.uint8`! Got dtype: {images.dtype}"
    assert len(images.shape) == 4 and images.shape[-1] == 3, (
        f"Unexpected images shape! Expected (T, H, W, 3) but got: {images.shape}"
    )

    images = resize_images(images, final_image_size)

    images = np.transpose(images, (3, 0, 1, 2))  # (T, H, W, C) -> (C, T, H, W)
    # images = np.transpose(images, (0, 3, 1, 2))  # (T, H, W, C) -> (C, T, H, W)
    images = torch.from_numpy(images)
    if use_image_aug:
        images = apply_image_aug(images, stronger=stronger_image_aug)
    if normalize_images:
        # Normalize images and return as dtype torch.float32
        images = images.to(torch.float32)
        images = images / 255.0
        # images = images.permute(1, 0, 2, 3)
        # print(images.shape)
        norm_func = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        images = norm_func(images)
    else:
        # Keep images as dtype torch.uint8
        images = images.to(torch.uint8)
    return images
