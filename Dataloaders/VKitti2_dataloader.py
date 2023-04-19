from __future__ import print_function, division
import os
import glob
import time

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from torchvision import utils
import cv2
from tqdm import tqdm
from skimage.morphology import binary_closing, disk
import torchvision.transforms.functional as F
import random
import matplotlib.pyplot as plt
from PIL import Image

use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
if use_cuda:
    gpu = 0
    seed=250
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def pil_loader(path, convert_rgb):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if convert_rgb:
                return img.convert('RGB')
            else:
                return img
        

class Paired_transform():
    def __call__(self, img1, img2, depth1, depth2):
        # Do we need to modify fb here?
        # No need since we also swap img1_ and img2_
        # If we do so, we also need to swap depth1 and depth2
        flip = random.random()
        if flip > 0.5:
            img1_ = img1
            img2_ = img2
            img1 = F.hflip(img2_)
            img2 = F.hflip(img1_)
            
            depth1_ = depth1 
            depth2_ = depth2 
            depth1 = F.hflip(depth2_)
            depth2 = F.hflip(depth1_)
        
        n_rotation = random.random()
        if n_rotation > 0.5:
            degree = random.randrange(-500, 500)/100
            img1 = F.rotate(img1, degree, Image.BICUBIC)
            img2 = F.rotate(img2, degree, Image.BICUBIC)
            if depth1 is not None:
                depth1 = F.rotate(depth1, degree, Image.BILINEAR, fill=(0,))
            if depth2 is not None:
                depth2 = F.rotate(depth2, degree, Image.BILINEAR, fill=(0,))
        return img1, img2, depth1, depth2



class depth_to_tensor():
    def __init__(self):
        self.ToTensor = transforms.ToTensor()
    def __call__(self, depth):
        depth_tensor = self.ToTensor(depth).float()
        depth_tensor[depth_tensor>8000.0] = 8000.0
        return depth_tensor/8000.0


class VKitti2(Dataset):
    def __init__(self, 
                 root_dir, 
                 height,
                 width,
                 baseline,
                 frame_ids,
                 num_scales,
                 phase,
                 preload):
        
        self.root_dir = root_dir
        
        assert phase in ["train", "val", "test"]
        self.is_train = True if phase == "train" else False
        
        self.preload = preload
        
        self.height = height 
        self.width = width 
        self.num_scales = num_scales 
        
        # self.baseline = 0.532725
        self.baseline = baseline
        
        ## In KITTI the intrinsics is given directly!!!
        # # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)
        
        self.K = np.array([[0.5837429, 0, 0.4995974, 0],
                           [0, 1.9333565, 0.4986667, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        
        # Antialias renamed to Lanczos as in https://pillow.readthedocs.io/en/3.0.x/releasenotes/2.7.0.html
        self.interp = Image.ANTIALIAS
        
        if self.is_train:
            self.file = "dataset_files/vKitti2/train_syn_vkitti2.txt"
        else:
            self.file = "dataset_files/vKitti2/test_syn_vkitti2.txt"
        
        with open(self.file,'r') as f:
            self.filepaths = f.readlines() # e.g. vKitti2/Scene01/15-deg-left 1\n
        
        self.frame_ids = frame_ids 
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
        
        # NOTE: Normalize here means (depth - 0.5) / 0.5
        # that's why later we need to (img + 1.0) / 2.0 or img * 0.5 + 0.5
        # and for depth: (pred + 1.0) / 2.0 or pred * 0.5 + 0.5
        self.depth_tensor_transform = [depth_to_tensor(),transforms.Normalize((0.5,), (0.5,))]
        self.depth_transform = transforms.Compose(self.depth_tensor_transform)
        
        
        # adjusting intrinsics to match each scale in the pyramid
        # K: In KITTI it is given directly
        self.K_pyramid = {}
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            self.K_pyramid[("K", scale)] = K
            self.K_pyramid[("inv_K", scale)] = inv_K
            
        
        if self.preload:
            raise ValueError("Not supported now: Will use too many memory!")
            self.colors = {}
            self.depths = {}
            self.preload_data()
        

    def __len__(self):
        return len(self.filepaths)

    
    
    def get_depth(self, folder, frame_index, side, do_flip):
        assert side in ["l", "r"]
        if self.preload:
            scene = "/".join(folder.split("/")[-2:])
            depth_key = (scene, frame_index, side)
            if depth_key not in self.depths.keys():
                side_path = "Camera_0" if side == "l" else "Camera_1"
                self.depths[depth_key] = pil_loader(os.path.join(folder, "frames/depth", side_path, "depth_{:05d}.png".format(frame_index)), convert_rgb=False)
            depth = self.depths[depth_key]
        else:
            side_path = "Camera_0" if side == "l" else "Camera_1"
            depth_path = os.path.join(folder, "frames/depth", side_path, "depth_{:05d}.png".format(frame_index))
            depth = Image.open(depth_path)
        
        if do_flip:
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        
        return depth
    
    
    def get_color(self, folder, frame_index, side, do_flip):
        assert side in ["l", "r"]
        if self.preload:
            scene = "/".join(folder.split("/")[-2:])
            color_key = (scene, frame_index, side)
            if color_key not in self.depths.keys():
                side_path = "Camera_0" if side == "l" else "Camera_1"
                self.colors[color_key] = pil_loader(os.path.join(folder, "frames/rgb", side_path, "rgb_{:05d}.jpg".format(frame_index)), convert_rgb=True)
            color = self.colors[color_key]
        else:
            side_path = "Camera_0" if side == "l" else "Camera_1"
            img_path = os.path.join(folder, "frames/rgb", side_path, "rgb_{:05d}.jpg".format(frame_index))
            color = Image.open(img_path).convert("RGB")
        
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        
        return color
    
    
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
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
        
        # For depth we only obtain the first scale ground-truth
        for depth_side in ["l", "r"]:
            inputs[("depth", depth_side, 0)] = self.resize[0](inputs[("depth", depth_side, -1)])
            # NOTE: here depth has been normalized to 0.5!!!
            inputs[("depth", depth_side, 0)] = self.depth_transform(inputs[("depth", depth_side, 0)])
        
    
    def __getitem__(self,idx):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            ("depth", "l" or "r", 0)                for ground truth depth maps for frame_id 0

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair. Now processed individually (Not inside frame_ids)

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        
        inputs = {}
        
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        
        # e.g. img_name: $root_dir/vKitti2/Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00000.jpg
        # e.g. vKitti2/Scene01/15-deg-left 1\n
        line = self.filepaths[idx].strip().split()
        folder = os.path.join(self.root_dir, line[0])
        frame_index = int(line[1])
        side = "l"
        
        # Get color images
        for i in self.frame_ids:
            inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
        inputs[("color", "s", -1)] = self.get_color(folder, frame_index, "r", do_flip)

        # Get depths
        for depth_side in ["l", "r"]:
            inputs[("depth", depth_side, -1)] = self.get_depth(folder, frame_index, depth_side, do_flip)
        
        
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
        
        # Preprocess both color, color_aug, and depth here!
        self.preprocess(inputs, color_aug)
        
        for i in self.frame_ids:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
        del inputs[("color", "s", -1)]
        del inputs[("color_aug", "s", -1)]
                
        for depth_side in ["l", "r"]:
            del inputs[("depth", depth_side, -1)]
        
        # stereo_T = np.eye(4, dtype=np.float32)
        # side_sign = -1 if side == "l" else 1
        # stereo_T[0, 3] = side_sign * baseline_sign * self.baseline
        # inputs["stereo_T"] = torch.from_numpy(stereo_T)
        
        # NOTE: we don't need side_sign since in SharinGAN and monodepth, baseline > 0!
        # NOTE: here the baseline is fixed to 0.1! 
        # NOTE: we actually only need to store the normalized fb
        # self.K[0,0] = 0.5837429

        # NOTE: some explanations
        # (1) We do not swap img_left and img_right, thus we take -1*baseline if do_flip
        # (2) Important to be aware that translation_x from left to right is -1 * baseline
        # (3) monodepth2 handles the -1 here while we use the bilinear_sampler_1d_h() from 
        #     sharinGAN and monodepth which handle the -1 there
        # used for opt.stereo_mode == "sharinGAN"
        baseline_sign = -1 if do_flip else 1
        new_baseline = baseline_sign * self.baseline
        assert self.K[0,0] <= 1 
        inputs["fb"] = self.K[0,0] * new_baseline
        if not do_flip:
            assert inputs["fb"] >= 0
        
        # used for opt.stereo_mode == "monodepth2"
        stereo_T = np.eye(4, dtype=np.float32)
        side_sign = -1 if side == "l" else 1
        stereo_T[0, 3] = side_sign * baseline_sign * self.baseline
        inputs["stereo_T"] = torch.from_numpy(stereo_T)
        
        # used for opt.predict_right_disp and "monodepth2"
        stereo_T_right = np.eye(4, dtype=np.float32)
        stereo_T_right[0, 3] = -1 * side_sign * baseline_sign * self.baseline
        inputs["stereo_T_right"] = torch.from_numpy(stereo_T_right)

        
        # # NOTE: here is the pixel-level fb for each scale
        # for scale in range(self.num_scales):
        #     # here is the pixel level focal length x from K
        #     fb = inputs[("K", scale)][0,0] * stereo_T[0, 3]
        #     inputs[("fb", scale)] = torch.from_numpy(fb)
        
        return inputs


    def preload_data(self):
        for idx in tqdm(range(len(self.filepaths))):
            line = self.filepaths[idx].strip().split()
            folder = os.path.join(self.root_dir, line[0])
            frame_index = int(line[1])
            
            # e.g. Scene01/fog
            scene = "/".join(folder.split("/")[-2:])

            self.colors[(scene, frame_index, "l")] = pil_loader(os.path.join(folder, "frames/rgb/Camera_0/rgb_{:05d}.jpg".format(frame_index)), convert_rgb=True)
            
            self.colors[(scene, frame_index, "r")] = pil_loader(os.path.join(folder, "frames/rgb/Camera_1/rgb_{:05d}.jpg".format(frame_index)), convert_rgb=True)
            
            self.depths[(scene, frame_index, "l")] = pil_loader(os.path.join(folder, "frames/depth/Camera_0/depth_{:05d}.png".format(frame_index)), convert_rgb=False)
            
            self.depths[(scene, frame_index, "r")] = pil_loader(os.path.join(folder, "frames/depth/Camera_1/depth_{:05d}.png".format(frame_index)), convert_rgb=False)
        
    
    