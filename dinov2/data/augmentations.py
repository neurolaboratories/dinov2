# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms
import torch

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

logger = logging.getLogger("dinov2")

def gauss_noise_tensor(img, probability=0.25):
    assert isinstance(img, torch.Tensor)
    
    # Check if Gaussian noise should be applied based on probability
    if torch.rand(1).item() > probability:
        return img
    
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    
    sigma = 0.05
    
    out = img + sigma * torch.randn_like(img)
    
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out

class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = A.Compose(
            [
                A.RandomResizedCrop(
                    height=global_crops_size, width=global_crops_size, scale=global_crops_scale, 
                    interpolation=cv2.INTER_CUBIC
                ),
                A.HorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = A.Compose(
            [   
                A.RandomResizedCrop(
                    height=local_crops_size, width=local_crops_size, scale=local_crops_scale, 
                    interpolation=cv2.INTER_CUBIC
                ),
                A.HorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = A.Compose(
            [
                A.OneOf(
                    [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)],
                    p=0.8,
                ),
                A.ToGray(p=0.1),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = A.Compose(
            [
                GaussianBlur(p=0.1),
                A.Solarize(threshold=200, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = A.Compose(
            [   
                make_normalize_transform(),
                A.GaussNoise(var_limit=0.25, p=0.25),
                ToTensorV2()
            ]
        )

        self.global_transfo1 = A.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = A.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = A.Compose([color_jittering, local_transfo_extra, self.normalize])
        
    def __call__(self, image):
        output = {}

        image = np.array(image)

        # global crops:
        im1_base = self.geometric_augmentation_global(image=image)['image']
        global_crop_1 = self.global_transfo1(image=im1_base)['image']

        im2_base = self.geometric_augmentation_global(image=image)['image']
        global_crop_2 = self.global_transfo2(image=im2_base)['image']

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        local_crops = []
        for _ in range(self.local_crops_number):
            local_image = self.geometric_augmentation_local(image=image)['image']
            local_crops.append(self.local_transfo(image=local_image)['image'])
        
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
