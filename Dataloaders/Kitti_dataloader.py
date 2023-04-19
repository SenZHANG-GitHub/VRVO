import os
import torch
import numpy as np
from PIL import Image
from torch.utils import data

from torchvision import transforms 
from torchvision import utils
import random

class MonoKittiDataset(data.Dataset):
    def __init__(self, 
                 root_dir, 
                 height,
                 width,
                 frame_ids,
                 num_scales,
                 phase,
                 folder):

        # e.g. "data/kitti/kitti_raw"
        self.root_dir = root_dir
        
        # "train", "val", or "test"
        self.phase = phase 

        # "Kitti" or "Kitti-Zhan"
        self.folder = folder 
        
        self.height = height 
        self.width = width 
        self.num_scales = num_scales 
        
        # since it is monocular dataset, "s" should not be included
        self.frame_ids = frame_ids 
        
        # NOTE: Following monodepth2 practice
        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        # Antialias renamed to Lanczos as in https://pillow.readthedocs.io/en/3.0.x/releasenotes/2.7.0.html
        self.interp = Image.ANTIALIAS
        
        assert self.phase in ["train", "val", "test", "odom"]
        assert self.folder in ["Kitti", "Kitti-Zhan"]
        self.file = "dataset_files/{}/{}.txt".format(self.folder, self.phase)
        
        with open(self.file,'r') as f:
            # e.g. "2011_09_28/"2011_09_28_drive_0002_sync 0000000030 l\n"
            self.filepaths = f.readlines()
            
        self.to_tensor = transforms.ToTensor()
        
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
            
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
        
        self.K_pyramid = {}
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            self.K_pyramid[("K", scale)] = K
            self.K_pyramid[("inv_K", scale)] = inv_K
            
                                    
    def __len__(self):
        return len(self.filepaths)

    
    def get_color(self, folder, frame_index, side, do_flip):
        assert side in ["l", "r"]
        side_path = "image_02" if side == "l" else "image_03"
        
        color = Image.open(os.path.join(self.root_dir, folder, side_path, "data/{:010d}.jpg".format(frame_index))).convert("RGB")
        
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        return color

    
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    # resize from last scale -> computation consideration
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
        
        
            
    def __getitem__(self, index):
        
        inputs = {}
        
        do_color_aug = self.phase == "train" and random.random() > 0.5
        do_flip = self.phase == "train" and random.random() > 0.5
        
        # [2011_09_28/"2011_09_28_drive_0002_sync", "0000000030", "l"]
        line = self.filepaths[index].strip().split() 
        
        folder = line[0]
        frame_index = int(line[1])
        side = line[2] 
        
        # only load left images
        assert side == "l" 
                
        # NOTE: ("color", frame_id, scale)
        assert "s" not in self.frame_ids
        for i in self.frame_ids:
            inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
        
        
        # NOTE: we also calculate the temporal consistency in real domain
        for scale in range(self.num_scales):
            K = self.K_pyramid[("K", scale)].copy()
            inv_K = self.K_pyramid[("inv_K", scale)].copy()

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
        
        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        
        self.preprocess(inputs, color_aug)
        
        for i in self.frame_ids:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        return inputs