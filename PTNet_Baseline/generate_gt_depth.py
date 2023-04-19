import os
import os.path as osp
import glob
import pdb
import random
from tqdm import tqdm
import imageio
import cv2

import torch
from torch import nn
import torch.optim as Optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as tr
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from networks import all_networks

from Dataloaders.Kitti_dataloader import MonoKittiDataset as real_dataset
from Dataloaders.transform import *
from Dataloaders.Kitti_dataset_util import generate_depth_map


def get_gt_depth_from_velo(mode="val"):

    # Seed
    seed = 1729
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    joint_transform_list = [RandomImgAugment(no_flip=True, no_rotation=True, no_augment=True, size=(192,640))]
    img_transform_list = [tr.ToTensor(), tr.Normalize([.5, .5, .5], [.5, .5, .5])]

    batch_size = 16
    joint_transform = tr.Compose(joint_transform_list)
    img_transform = tr.Compose(img_transform_list)
    depth_transform = tr.Compose([DepthToTensor()])

    # phase = "test" if mode in ["val", "test"] else "train"
    # Always use "test" phase for depth generation
    dataset = real_dataset(data_file="{}.txt".format(mode), phase="test", img_transform=img_transform, joint_transform=joint_transform, depth_transform=depth_transform)
    real_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=4)

    # e.g. "../data/kitti/kitti_raw"
    root_path = dataset.root
    root_path = os.path.join(root_path, "depth_from_velodyne")
    if not os.path.isdir(root_path):
        raise ValueError("Need to mkdir depth_from_velodyne dir manually")
    for i, (data, depth_filenames) in tqdm(enumerate(real_dataloader)):
        # assert len(depth_filenames) == batch_size # Not hold for last batch
        for t_id in range(len(depth_filenames)):
            t_id_global = (i * batch_size) + t_id 
            # e.g. "2011_09_30/2011_09_30_drive_0033_sync/velodyne_points/data/0000000200.bin"
            depth_file = dataset.files[t_id_global]["depth"]
            scene = depth_file.split("/")[0]
            sub_scene = depth_file.split("/")[1]
            img_idx = depth_file.split("/")[-1].split(".")[0]

            tmp_path = os.path.join(root_path, scene)
            if not os.path.isdir(tmp_path):
                os.mkdir(tmp_path)
            
            tmp_path = os.path.join(tmp_path, sub_scene)
            if not os.path.isdir(tmp_path):
                os.mkdir(tmp_path)

            calib_dir = os.path.join(dataset.root, scene)
            velo_file_name = os.path.join(dataset.root, depth_file)

            # left_img_file = dataset.files[t_id_global]["l_rgb"]
            # left_img_file = os.path.join(dataset.root, left_img_file)
            # left_img = Image.open(left_img_file)
            # im_shape = left_img.size
            # depth, depth_interp = kitti.get_depth(calib_dir, velo_file_name, im_shape, cam=2, interp=True, vel_depth=True)

            depth = generate_depth_map(calib_dir, velo_file_name, 2, True)
            depth = (depth * 256).astype(np.uint16)
            depth_png = Image.fromarray(depth) 
            depth_png.save(os.path.join(tmp_path, "{}.png".format(img_idx)))



if __name__ == "__main__":
    get_gt_depth_from_velo("val")
    get_gt_depth_from_velo("test")
