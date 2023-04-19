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

import numpy as np
from collections import Counter
from PIL import Image
import itertools
#import png

def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    Sen -> This func is from monodepth2
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth


def get_gt_depth_from_velo(phase="val"):
    print("=> Processing phase: {}".format(phase))

    # Seed
    seed = 1729
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # phase = "test" if mode in ["val", "test"] else "train"
    # Always use "test" phase for depth generation
    dataset = real_dataset(
            root_dir="data/kitti/kitti_raw",
            height=192,
            width=640,
            frame_ids=[0],
            num_scales=4,
            phase=phase,
            folder="Kitti-Zhan"
        )

    # e.g. "../data/kitti/kitti_raw"
    root_path = os.path.join("data/kitti/kitti_raw/depth_from_velodyne")
    if not os.path.isdir(root_path):
        raise ValueError("Need to mkdir depth_from_velodyne dir manually")

    for i, data in tqdm(enumerate(dataset.filepaths)):
        # e.g. "2011_09_30/2011_09_30_drive_0033_sync/velodyne_points/data/0000000200.bin"
        folder = data.strip().split()[0] # 2011_09_30/2011_09_30_drive_0018_sync
        date = folder.split("/")[0] # 2011_09_30
        img_idx = data.strip().split()[1] # 0000001678

        tmp_path = os.path.join(root_path, folder)
        if not os.path.isdir(tmp_path):
            os.makedirs(tmp_path)

        calib_dir = os.path.join("data/kitti/kitti_raw", date)
        velo_file_name = os.path.join("data/kitti/kitti_raw", folder, "velodyne_points/data/{}.bin".format(img_idx))

        depth = generate_depth_map(calib_dir, velo_file_name, 2, True)
        depth = (depth * 256).astype(np.uint16)
        depth_png = Image.fromarray(depth) 
        depth_png.save(os.path.join(tmp_path, "{}.png".format(img_idx)))



if __name__ == "__main__":
    get_gt_depth_from_velo("val")
    get_gt_depth_from_velo("test")
