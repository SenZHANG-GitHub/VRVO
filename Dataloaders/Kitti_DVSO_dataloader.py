import os
import pdb
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from glob import glob
import shutil

from torchvision import transforms 
from torchvision import utils
from transforms3d import euler 
from transforms3d import quaternions as quat
import random


class depth_to_tensor():
    """Transform (depth * 256) Image object to tensor
    Return (0,1) by /= 80.0
    """
    def __init__(self):
        self.ToTensor = transforms.ToTensor()
    def __call__(self, depth):
        depth_tensor = self.ToTensor(depth).float()
        
        depth_tensor /= 256.0
        depth_tensor[depth_tensor>80.0] = 80.0
        return depth_tensor/80.0


class MonoKittiDVSODataset(data.Dataset):
    def __init__(self, 
                 root_dir, 
                 height,
                 width,
                 frame_ids,
                 num_scales,
                 phase,
                 seqs,
                 exp,
                 param,
                 use_dvso_depth):
        """
        NOTE: the depth from DVSO is normalized by (0.5, 0.5)
        => depths are read from DVSO_Finetune
        """
        # e.g. "data/kitti/odometry/dataset/sequences"
        self.root_dir = root_dir
        
        assert phase in ["train", "val", "test"]
        self.phase = phase
        
        # e.g. ["01", "09", "10"]
        self.seqs = seqs
        
        # NOTE: used for loading dvso depth and pose results
        self.epoch = None
        self.abs_poses = dict()

        # The experiment name for current finetuning model
        self.exp = exp 
        
        self.height = height 
        self.width = width 
        self.num_scales = num_scales 
        self.use_dvso_depth = use_dvso_depth
        
        # since it is monocular dataset, "s" should not be included
        self.frame_ids = frame_ids 
        
        # Antialias renamed to Lanczos as in https://pillow.readthedocs.io/en/3.0.x/releasenotes/2.7.0.html
        self.interp = Image.ANTIALIAS
        
        self.filepaths = []
        self.param_DVSO = {}
        self.Ks = {}
        self.K_pyramids = {}
        for seq in self.seqs:
            # (1) load filenames
            fileToLoad = "dataset_files/Kitti-Odom/{}.txt".format(seq)
            with open(fileToLoad,'r') as f:
                # e.g. "01/image_2/000008.jpg 01/image_3/000008.jpg\n"
                filepaths = f.readlines()
                # remove the first and last frame for temporal processing
                self.filepaths.extend(filepaths[1:-1])
            
            # (2) load DVSO param
            # NOTE: Currently use a single param for all seqs
            self.param_DVSO[seq] = param
            
            # (3) load K
            self.Ks[seq] = self.get_K(seq)
            
            self.K_pyramids[seq] = {}
            for scale in range(self.num_scales):
                K = self.Ks[seq].copy()

                K[0, :] *= self.width // (2 ** scale)
                K[1, :] *= self.height // (2 ** scale)

                inv_K = np.linalg.pinv(K)

                self.K_pyramids[seq][("K", scale)] = K
                self.K_pyramids[seq][("inv_K", scale)] = inv_K
            
        self.to_tensor = transforms.ToTensor()
        
        # NOTE: Normalize here means (depth - 0.5) / 0.5
        # thus for depth: (pred + 1.0) / 2.0 or pred * 0.5 + 0.5
        self.depth_tensor_transform = [depth_to_tensor(),transforms.Normalize((0.5,), (0.5,))]
        self.depth_transform = transforms.Compose(self.depth_tensor_transform)
        
        
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
        
        
    def __len__(self):
        return len(self.filepaths)


    def get_K(self, seq):
        """Read the intrinsics matrix from dataset_files/Kitti-Odom/calib
        (1) Following monodepth2 practice
        (2) Intrinsics matrix is *normalized* by the original image size    
        """
        K_file = "dataset_files/Kitti-Odom/calib/camera_{}x{}_{}.txt".format(self.width, self.height, seq)
        assert os.path.isfile(K_file)
        with open(K_file, "r") as f:
            K_lines = f.readlines() 
        K_values = [float(x) for x in K_lines[0].strip().split()]
        K = np.eye(4, dtype=np.float32)
        K[0,0] = K_values[0]
        K[1,1] = K_values[1]
        K[0,2] = K_values[2]
        K[1,2] = K_values[3]
        return K
    
    def get_color(self, seq, frame_index, side, do_flip):
        assert side == "image_2"
        
        color = Image.open(os.path.join(self.root_dir, seq, side, "{:06d}.jpg".format(frame_index))).convert("RGB")
        
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
                    # Sen -> resize from last scale -> computation consideration
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
        
        if self.use_dvso_depth:
            inputs[("depth", "l", 0)] = self.resize[0](inputs[("depth", "l", -1)])
            inputs[("depth", "l", 0)] = self.depth_transform(inputs[("depth", "l", 0)])
    

    def update_epoch(self, epoch):
        """Update self.epoch for loading depths
        => Will also check the number of valid keyframe depths that can be accessed
        """
        self.epoch = epoch
        if self.use_dvso_depth:
            total_num = {
                "keyframe": 0,
                "allframe": 0
            }
            for line in self.filepaths:
                # ["01/image_2/000008.jpg 01/image_3/000008.jpg"]
                line = line.strip().split()[0]
                line = line.split("/")
                
                seq = line[0]
                side = line[1]
                frame_index = int(line[2].split(".")[0])

                assert side == "image_2"
                assert seq in self.seqs

                if seq not in total_num.keys():
                    total_num[seq] = {
                        "keyframe": 0,
                        "allframe": 0
                    }
                
                _, isKeyframe = self.get_depth(seq, frame_index, False)

                if isKeyframe:
                    total_num["keyframe"] += 1
                    total_num[seq]["keyframe"] += 1
                total_num["allframe"] += 1
                total_num[seq]["allframe"] += 1
            
            print("=> we are now using the dvso depths from epoch {}".format(self.epoch))
            print("=> total number of keyframes / total number of all frames: {} / {} ({:.2f}%)".format(
                total_num["keyframe"], total_num["allframe"], 100 * total_num["keyframe"] / total_num["allframe"]
            ))
            for seq in total_num.keys():
                if seq in ["keyframe", "allframe"]: continue
                print("=> for seq {}: total number of keyframes / total number of all frames: {} / {} ({:.2f}%)".format(
                    seq, total_num[seq]["keyframe"], total_num[seq]["allframe"], 100 * total_num[seq]["keyframe"] / total_num[seq]["allframe"]
                ))
    

    def update_loaded_poses(self, seq):
        """Load the absolute poses from DVSO
        (1) e.g DVSO_Finetune/results_dvso/exp/epoch_{}/seq/param/result.txt
        (2) format: "id r00 r01 r02 t0 r10 r11 r12 t1 r20 r21 r22 t2\n"
        """
        pose_path = os.path.join("DVSO_Finetune/results_dvso", self.exp, "epoch_{}".format(self.epoch), seq, self.param_DVSO[seq], "result.txt")
        assert self.epoch is not None
        assert os.path.isfile(pose_path)
        self.abs_poses[seq] = {}
        with open(pose_path, "r") as f:
            for line in f.readlines():
                line = line.strip().split()
                pid = int(line[0])
                pose = [float(x) for x in line[1:]]
                trans = np.array(pose[:3]).reshape((3,1))
                qx, qy, qz, qw = pose[3:]
                rot = quat.quat2mat([qw, qx, qy, qz])
                T = np.hstack([rot, trans])
                T = np.vstack([T, np.array([0., 0., 0., 1.])])
                self.abs_poses[seq][pid] = T


    def get_pose(self, seq, prev_index, curr_index, do_flip):
        """Load the poses from DVSO
        (1) e.g DVSO_Finetune/results_dvso/exp/epoch_{}/seq/param/result.txt
        (2) format: "id r00 r01 r02 t0 r10 r11 r12 t1 r20 r21 r22 t2\n"
        (3) output: ("axisangle/translation", 0, -1) means rel_pose from -1 to 0 
                    ("axisangle/translation", 0, 1) means rel_pose frmo 0 to 1
        """
        if do_flip:
            raise NotImplementedError("error")
        prev_p = self.abs_poses[seq][prev_index]
        curr_p = self.abs_poses[seq][curr_index]
        T = np.linalg.inv(curr_p) @ prev_p
        rel_rot = np.array(euler.mat2euler(T[:3, :3]))
        rel_trans = T[:3, 3]
        return rel_rot, rel_trans

        
    def get_depth(self, seq, frame_index, do_flip):
        """Load the depths from DVSO
        (1) e.g DVSO_Finetune/results_dvso/exp/epoch_{}/seq/param/depths/000000.png
        (2) format: CV_U16, (depth * 256)
        (3) currently only support size 640x192
        """
        # depth_path = os.path.join("results_DVSO", seq, self.exp, self.param_DVSO[seq], "depths", "{:06d}.png".format(frame_index))
        assert self.epoch is not None
        assert self.use_dvso_depth
        depth_path = os.path.join("DVSO_Finetune/results_dvso", self.exp, "epoch_{}".format(self.epoch), seq, self.param_DVSO[seq], "depths/{:06d}.png".format(frame_index))

        isKeyframe = False
        if os.path.isfile(depth_path): 
            # mode=I, size=640x192
            depth = Image.open(depth_path)
            isKeyframe = True
            assert depth.size[0] == self.width
            assert depth.size[1] == self.height
        else:
            # non-keyframe depths may not be available
            # NOTE: black depths for non-keyframes
            depth = Image.new("I;16", (self.width, self.height), 0)
        
        if do_flip:
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            
        return depth, isKeyframe
    
                
    def __getitem__(self, index):
        
        inputs = {}
        
        do_color_aug = self.phase == "train" and random.random() > 0.5
        # do_flip = self.phase == "train" and random.random() > 0.5
        
        # Disable flip augmentation due to potential pose changes
        do_flip = False
        
        # ["01/image_2/000008.jpg 01/image_3/000008.jpg"]
        line = self.filepaths[index].strip().split()[0]
        line = line.split("/")
        
        seq = line[0]
        side = line[1]
        frame_index = int(line[2].split(".")[0])
        inputs["frame_index"] = frame_index
        
        # only load left images
        assert side == "image_2" 
        assert seq in self.seqs
                
        # NOTE: ("color", frame_id, scale)
        assert "s" not in self.frame_ids
        for i in self.frame_ids:
            inputs[("color", i, -1)] = self.get_color(seq, frame_index + i, side, do_flip)
            if i == -1: 
                inputs[("axisangle", 0, -1)], inputs[("translation", 0, -1)] = self.get_pose(seq, frame_index - 1, frame_index, do_flip)
            if i == 1:
                inputs[("axisangle", 0, 1)], inputs[("translation", 0, 1)] = self.get_pose(seq, frame_index, frame_index + 1, do_flip)
            
        # NOTE: obtain depths from DVSO (which use image_2 by default)
        if self.use_dvso_depth:
            inputs[("depth", "l", -1)], isKeyframe = self.get_depth(seq, frame_index, do_flip)
        
        # NOTE: we also calculate the temporal consistency in real domain
        for scale in range(self.num_scales):
            K = self.K_pyramids[seq][("K", scale)].copy()
            inv_K = self.K_pyramids[seq][("inv_K", scale)].copy()

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

        if self.use_dvso_depth:
            del inputs[("depth", "l", -1)]

        return inputs
