import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from random import random
from torchvision import transforms
from torchvision.utils import save_image

def rand_rotate(x, ratio=0.3):
    if ratio > random():
        return x
    else:  
        size = [int(i) for i in x.shape]
        h, w = size[2], size[3]
        # for debug
        #print("TEST: ", x)

        aug = transforms.Compose([
            transforms.Pad(10, padding_mode = 'reflect'),
            transforms.RandomRotation(5, expand=False),
            transforms.CenterCrop([h,w])
        ])  
        x = aug(x)

        # save image
        #img = x[0] 
        #save_image(img, 'result.png')

        #plt.imshow(x.numpy()[0], cmap='gray')
        #cv2.imwrite("result.png", x)
        return x

def rand_affine(x, ratio=0.5):
    if ratio > random():
        return x
    else:  
        size = [int(i) for i in x.shape]
        h, w = size[2], size[3]

        aug = transforms.Compose([
            transforms.Pad(10, padding_mode = 'reflect'),
            transforms.RandomAffine(degrees=10, translate=(0.2, 0.2),
            scale=(0.8, 1.2), shear=15, resample=Image.BILINEAR),
            transforms.CenterCrop([h,w])
        ])  
        x = aug(x)

        return x
        
def DiffAugment(x, policy='vflip,hflip', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x

def rand_hflip(x, ratio=0.5):
    if ratio > random():
        return x
    else:
        x = torch.flip(x, dims=(3,))
        return x

def rand_vflip(x, ratio=0.3):
    if ratio > random():
        return x
    else:
        x = torch.flip(x, dims=(2,))
        return x

def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 1, 1, 1, 1])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
    'hflip': [rand_hflip],
    'vflip': [rand_vflip],
    'rotate': [rand_rotate],
    'affine' : [rand_affine],
}
