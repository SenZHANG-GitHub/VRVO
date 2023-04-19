import os
import pdb
import os.path as osp
import glob
import numpy as np
import random
from tqdm import tqdm
import argparse
import matplotlib
import matplotlib.cm
from PIL import Image
import cv2
import time
from options import SharinOptions

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
from networks import networks
from bilinear_sampler import bilinear_sampler_1d_h
from networks.layers import disp_to_depth 
from Dataloaders.Kitti_dataloader import MonoKittiDataset as real_dataset # for val
import Dataloaders.transform as transf
from transforms3d import euler


MIN_DEPTH = 1e-3
MAX_DEPTH = 80


def readCalib(cpath, seq, width, height):
    """
    cpath: "dataset_files/Kitti-Odom/calib"
    """
    lines = []
    with open(os.path.join(cpath, "camera_{}x{}_{}.txt".format(width, height, seq)), "r") as f:
        lines = f.readlines() 
    fx = float(lines[0].strip().split()[0])
    w = int(lines[3].strip().split()[0])
    h = int(lines[3].strip().split()[1])
    baseline = float(lines[4].strip())
    assert w == width
    assert h == height
    return fx, baseline, w, h


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    => from monodepth2
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


class Solver():
    def __init__(self, opt):
        self.root_dir = '.'
        self.opt = opt
        
        # Seed
        self.seed = 1729
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        
        # NOTE: Now frame_ids are only used for temporal consistency
        # NOTE: We manually specify "s" in codes
        assert "s" not in self.opt.frame_ids
        
        self.num_scales = len(self.opt.scales) # [0, 1, 2, 3]        
        self.num_pose_frames = 2 
        self.use_pose = True

        # Initialize the generator network
        if self.opt.netG_mode == "sharinGAN":
            self.netG = all_networks.define_G(3, 3, 64, 9, 'batch',
                                                    'PReLU', 'ResNet', 'kaiming', 0,
                                                    False, [self.opt.gpu])
            
        elif self.opt.netG_mode == "monodepth2":
            self.netG = networks.netG(
                self.opt.num_layers_G, 
                self.opt.scales, 
                [0])
            
        # Initialize the depth (and pose) task network 
        self.netT = networks.netT(
            self.opt.num_layers_T, 
            self.opt.scales, 
            self.num_pose_frames, 
            self.opt.frame_ids,
            self.use_pose,
            self.opt.predict_right_disp)
        
        self.netG.cuda(self.opt.gpu)
        self.netT.cuda(self.opt.gpu)


        # Training Configuration details
        self.batch_size = self.opt.batch_size
        self.workers = self.opt.num_workers
        self.iteration = None
        
        # Transforms
        joint_transform_list = [transf.RandomImgAugment(no_flip=False, no_rotation=False, no_augment=False, size=(192,640))]
        img_transform_list = [tr.ToTensor(), tr.Normalize([.5, .5, .5], [.5, .5, .5])]
        self.joint_transform = tr.Compose(joint_transform_list)
        self.img_transform = tr.Compose(img_transform_list)

        self.saved_models_dir = 'saved_models'

        # Initialize Data
        if self.opt.val_seq is None:
            self.seqs = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
        else:
            self.seqs = [self.opt.val_seq]
        
        self.real_val_datasets = {}
        self.real_val_loaders = {}
        self.get_validation_data()


    def loop_iter(self, loader):
        while True:
            for data in iter(loader):
                yield data


    def get_validation_data(self):
        for seq in self.seqs:
            self.real_val_datasets[seq] = real_dataset(
                root_dir="data/kitti/kitti_raw",
                height=self.opt.height,
                width=self.opt.width,
                frame_ids=self.opt.frame_ids,
                num_scales=4,
                phase="odom")
    
            self.real_val_loaders[seq] = DataLoader(
                self.real_val_datasets[seq],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers = self.workers,
                pin_memory=True,
                drop_last=False)


    def load_prev_model(self, saved_model):
        # saved_models = glob.glob(os.path.join('/vulcanscratch/koutilya/projects/Domain_Adaptation/Common_Domain_Adaptation-Lighting/saved_models_ablation_studies', 'Depth_Estimator_WI_geom_new_VKitti_bicubic_da-'+str(self.iteration)+'.pth.tar' ))
        if osp.isfile(saved_model):
            model_state = torch.load(saved_model)
            if type(model_state['netG_state_dict']) is tuple:
                assert len(model_state['netG_state_dict']) == 1
                self.netG.load_state_dict(model_state['netG_state_dict'][0])
            else:
                self.netG.load_state_dict(model_state['netG_state_dict'])
            
            if type(model_state['netT_state_dict']) is tuple:
                assert len(model_state['netT_state_dict']) == 1
                self.netT.load_state_dict(model_state['netT_state_dict'][0])
            else:
                self.netT.load_state_dict(model_state['netT_state_dict'])
                
            return True
        return False


    
    def Validate(self, val_iter):
        self.netG.eval()
        self.netT.eval()
        
        saved_model = os.path.join(self.root_dir, self.saved_models_dir, self.opt.exp, 'Depth_Estimator_da-{}.pth.tar'.format(val_iter))
        
        self.load_prev_model(saved_model)
        for seq in self.seqs:
            self.save_pose_2(saved_model, seq, val_iter)
            


    
    def construct_recon_inputs(self):
        """Construct the inputs for netT using reconstructed shared features        
        """
        real_recon_inputs= {}
        
        # Used for multi-scale reconstruction loss
        if self.opt.netG_mode == "sharinGAN":
            real_recon_inputs[("color_aug", 0, 0)] = self.real_recon_imgs[("gen", 0)]
        else:
            for scale in self.opt.scales:
                real_recon_inputs[("color_aug", 0, scale)] = self.real_recon_imgs[("gen", scale)]

        # Used for pose encoder
        for f_i in self.opt.frame_ids:
            if f_i == 0: continue
            _, real_recon_tmp = self.netG(self.real_inputs["color_aug", f_i, 0])
            real_recon_inputs[("color_aug", f_i, 0)] = real_recon_tmp[("gen", 0)]
            

        if self.opt.direct_raw_img:
            for f_i in self.opt.frame_ids: # [0, -1, 1]
                for scale in self.opt.scales:
                    real_recon_inputs[("color", f_i, scale)] = self.real_inputs["color", f_i, scale]
                    
        else:
            for f_i in self.opt.frame_ids: # [0, -1, 1]
                _, real_recon_tmp = self.netG(self.real_inputs["color", f_i, 0])
                for scale in self.opt.scales:
                    real_recon_inputs[("color", f_i, scale)] = real_recon_tmp[("gen", scale)]
            
        real_recon_inputs[("K", 0)] = self.real_inputs[("K", 0)]
        real_recon_inputs[("inv_K", 0)] = self.real_inputs[("inv_K", 0)]

        return real_recon_inputs
    

    def save_pose(self, saved_model, seq, val_iter):
        print("==========================================")
        print("=> saving the relative poses of seq {}".format(seq))
        print("=> using {}".format(saved_model))
        self.netG.eval()
        self.netT.eval()
        
        
        ## NOTE: Read camera intrinsics for 640x192 or 512x256
        fx, baseline, wCalib, hCalib = readCalib("dataset_files/Kitti-Odom/calib", seq, self.opt.width, self.opt.height)
        
        num_samples = len(self.real_val_datasets[seq])
        
        START_IND = 1
        END_IND = 1590

        i_global = START_IND
        rel_poses = {}
        with torch.no_grad():
            for i, data in enumerate(self.real_val_loaders[seq]):
                self.real_inputs = data
                for key, ipt in self.real_inputs.items():
                    self.real_inputs[key] = ipt.cuda(self.opt.gpu, non_blocking=True)
                    
                _, self.real_recon_imgs = self.netG(self.real_inputs["color", 0, 0])
                real_recon_inputs = self.construct_recon_inputs()
                real_outputs = self.netT(real_recon_inputs)
                
                poses = real_outputs[("cam_T_cam", 0, 1)]
                
                # cam_T_cam: from i + START_IND to i + START_IND + 1: (batch, 4, 4)
                for ib in range(poses.shape[0]):
                    rel_poses[(i_global, i_global + 1)] = poses[ib].cpu().numpy()
                    i_global += 1 
        
        gt_poses = []
        with open("/home/szha2609/data/kitti/odometry/dataset/poses/09.txt", "r") as f:
            i_gt = 0
            for line in f.readlines():
                line = line.strip().split()
                line = [float(x) for x in line]
                line.extend([0., 0., 0., 1.])
                pose = np.reshape(np.array(line), (4,4))
                gt_poses.append((i_gt, pose))
                i_gt += 1
        
        pred_poses = {}
        with open("tmp_09.txt", "w") as f:
            for i, gt_p in enumerate(gt_poses):
                assert i == gt_p[0]
                if i in [0, 1]:
                    pose = gt_p[1]
                else:
                    # pose = rel_poses[(i-1, i)] @ pred_poses[i-1][1]
                    # pose = np.linalg.inv(rel_poses[(i-1, i)]) @ pred_poses[i-1][1]
                    pose = pred_poses[i-1][1] @ np.linalg.inv(rel_poses[(i-1, i)])
                    
                pred_poses[i] = (i, pose)
                
                output = pose.flatten()[:12]
                output = [str(x) for x in output]
                output = " ".join(output)
                f.write("{}\n".format(output))



if __name__=='__main__':
    """
    This script is used for checking how to map kitti gt-pose format to our pose prediction
    """
    
    # NOTE: options that need to be made consistent with save_depth.py
    # except for --frame_ids 0 -1 1 by default still
    
    options = SharinOptions()
    opt = options.parse()
    opt.gpu = 0
    solver = Solver(opt)
    solver.Validate(opt.val_iter)