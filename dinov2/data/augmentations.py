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
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering_old = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.1),
            ]
        )
        
        color_jittering_new = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra_old = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )
        
        global_transfo2_extra_new = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=200, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize_new = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
                gauss_noise_tensor,
            ]
        )
        
        self.normalize_old = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1_old = transforms.Compose([color_jittering_old, global_transfo1_extra, self.normalize_old])
        self.global_transfo2_old = transforms.Compose([color_jittering_old, global_transfo2_extra_old, self.normalize_old])
        self.local_transfo_old = transforms.Compose([color_jittering_old, local_transfo_extra, self.normalize_old])
        
        self.global_transfo1_new = transforms.Compose([color_jittering_new, global_transfo1_extra, self.normalize_new])
        self.global_transfo2_new = transforms.Compose([color_jittering_new, global_transfo2_extra_new, self.normalize_new])
        self.local_transfo_new = transforms.Compose([color_jittering_new, local_transfo_extra, self.normalize_new])
        
    def __call__(self, image):
        output = {}
        probability = 0.5

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        if torch.rand(1).item() < probability:
            global_crop_1 = self.global_transfo1_old(im1_base)
        else:
            global_crop_1 = self.global_transfo1_new(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        if torch.rand(1).item() < probability:
            global_crop_2 = self.global_transfo2_old(im2_base)
        else:
            global_crop_2 = self.global_transfo2_new(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = []
        for _ in range(self.local_crops_number):
            if torch.rand(1).item() < probability:
                local_crop = self.local_transfo_old(self.geometric_augmentation_local(image))
            else:
                local_crop = self.local_transfo_new(self.geometric_augmentation_local(image))
            local_crops.append(local_crop)
        # local_crops = [
        #     self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        # ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
