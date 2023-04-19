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

from Dataloaders.Kitti_Odom_dataloader import MonoKittiOdomDataset as odom_dataset
import Dataloaders.transform as transf


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
        self.root_dir = 'PTNet_Baseline/'
        self.opt = opt
        
        self.val_string = 'test'
        if self.opt.val:
            self.val_string = 'val'
        
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
            self.real_val_datasets[seq] = odom_dataset(
                root_dir="data/kitti/odometry/dataset/sequences",
                height=self.opt.height,
                width=self.opt.width,
                seq=seq)
    
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
            
            self.netT.load_state_dict(model_state['netT_state_dict'])
                
            return True
        return False


    def tensor2im(self, depth):
        """Transform normalized depth values back to [0, 80m]
        """
        # (batch, 1, 192, 640) => (batch, 192, 640, 1)
        depth_numpy = depth.cpu().data.float().numpy().transpose(0,2,3,1)
        depth_numpy = (depth_numpy + 1.0) / 2.0 # Unnormalize between 0 and 1
        return depth_numpy * MAX_DEPTH

    
    def Validate(self, val_iter):
        self.netG.eval()
        self.netT.eval()
        
        saved_model = os.path.join(self.root_dir, self.saved_models_dir, self.opt.exp, 'PTNet_baseline-{}_bicubic.pth.tar'.format(val_iter))
        
        self.load_prev_model(saved_model)
        for seq in self.seqs:
            self.save_depth(saved_model, seq, val_iter)
            

    # def get_gt_depth(self, folder, depth_idx):
    #     """
    #     => folder: e.g. "2011_09_28/"2011_09_28_drive_0002_sync"
    #     => depth_idx: e.g. 30
    #     """
    #     depth_file = os.path.join("./data/kitti/kitti_raw/depth_from_velodyne", folder, "{:010d}.png".format(depth_idx))
        
    #     # Sen -> The PNG file is uint16 and depth = float(I) / 256.0 (valid if > 0)
    #     depth = Image.open(depth_file)

    #     # Sen -> Bug? By KITTI depth data format: Should divided by 256.0
    #     # depth = np.array(depth, dtype=np.float32) / 255.0
    #     depth = np.array(depth, dtype=np.float32) / 256.0
    #     return depth      
    

    def save_depth(self, saved_model, seq, val_iter):
        print("==========================================")
        print("=> saving the disparity npy of seq {}".format(seq))
        print("=> using {}".format(saved_model))
        self.netG.eval()
        self.netT.eval()
        self.netT.encoder.eval()
        self.netT.depth.eval()
        
        ## NOTE: Read camera intrinsics for 640x192 or 512x256
        fx, baseline, wCalib, hCalib = readCalib("dataset_files/Kitti-Odom/calib", seq, self.opt.width, self.opt.height)
        
        num_samples = len(self.real_val_datasets[seq])
        
        # NOTE: now use self.opt.post_process to control the final results 
        disparities_left = np.zeros((num_samples, self.opt.height, self.opt.width), dtype=np.float32)
        
        if self.opt.predict_right_disp:
            disparities_right = np.zeros((num_samples, self.opt.height, self.opt.width), dtype=np.float32)
        
        if not self.opt.post_process:
            raise ValueError("Error: Please use --post_process")
        
        # NOTE: now use self.opt.post_process to control the final results 
        # disparities_pp_left = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
        # disparities_pp_right = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)

        with torch.no_grad():
            for i, data in enumerate(self.real_val_loaders[seq]):
                input_color = data[("color", 0, 0)].cuda(self.opt.gpu)
                if self.opt.post_process:
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                
                _, recon_img = self.netG(input_color)   
                input_recon = recon_img[("gen", 0)]             
                outputs = self.netT.depth(self.netT.encoder(input_recon))
                
                # Note: load the left disparities for evaluation
                scale = 0
                if self.opt.predict_right_disp:
                    disp_left = outputs[("disp", scale)][:, 0, :, :].unsqueeze(1)
                    disp_right = outputs[("disp", scale)][:, 1, :, :].unsqueeze(1)
                else:
                    disp_left = outputs[("disp", scale)]
                    
                # NOTE: (batch(*2), 1, height, width), e.g. (16(*2), 1, 192, 640)
                # => pred_disp and _ is calculated using max_depth = 100.0 (monodepth2)
                # => normalized_depth is depth / 80.0 and then Normalize((0.5,), (0.5,))
                pred_disp_left, _, normalized_depth_left = disp_to_depth(disp_left, self.opt.min_depth, self.opt.max_depth)  
                
                if self.opt.predict_right_disp:
                    pred_disp_right, _, normalized_depth_right = disp_to_depth(disp_right, self.opt.min_depth, self.opt.max_depth)  
                
                if self.opt.val_depth_mode == "disp":
                    pred_disp_left = pred_disp_left.cpu()[:, 0].numpy() # (batch(*2), 192, 640)
                    if self.opt.post_process:
                        N = pred_disp_left.shape[0] // 2
                        pred_disp_left = batch_post_process_disparity(pred_disp_left[:N], pred_disp_left[N:, :, ::-1])
                    
                    if self.opt.predict_right_disp:
                        pred_disp_right = pred_disp_right.cpu()[:, 0].numpy() # (batch(*2), 192, 640)
                        if self.opt.post_process:
                            N = pred_disp_right.shape[0] // 2
                            pred_disp_right = batch_post_process_disparity(pred_disp_right[:N], pred_disp_right[N:, :, ::-1])
                    
                    curr_batch_size = pred_disp_left.shape[0]
                    
                elif self.opt.val_depth_mode == "normalized_depth":
                    raise ValueError("Not Supported NOW")
                

                for t_id in range(curr_batch_size):
                    t_id_global = (i*self.batch_size)+t_id
                    
                    # NOTE: Not NOW
                    # pred_depth_left[pred_depth_left < MIN_DEPTH] = MIN_DEPTH
                    # pred_depth_left[pred_depth_left > MAX_DEPTH] = MAX_DEPTH
                    # if self.opt.predict_right_disp:
                    #     pred_depth_right[pred_depth_right < MIN_DEPTH] = MIN_DEPTH
                    #     pred_depth_right[pred_depth_right > MAX_DEPTH] = MAX_DEPTH
                    
                    real_disp_left = baseline * fx * pred_disp_left [t_id]
                    real_disp_right = baseline * fx * pred_disp_right[t_id]
                    
                    disparities_left[t_id_global] = real_disp_left.squeeze()
                    disparities_right[t_id_global] = real_disp_right.squeeze()
                    
                    if t_id_global % 1000 == 0:
                        print("=> Processed {:d} of {:d} images".format(t_id_global, num_samples))
                    
        dst_path = os.path.join(self.root_dir, self.saved_models_dir, self.opt.exp, 'disps', seq)
        if not os.path.isdir(dst_path):
            os.makedirs(dst_path)
        
        if not self.opt.post_process:
            raise ValueError("Error: Please use --post_process")
        
        dst_file_left = os.path.join(dst_path, "disparities_left_pp_{}x{}.npy".format(self.opt.width, self.opt.height))
        if os.path.isfile(dst_file_left):
            os.remove(dst_file_left)
        np.save(dst_file_left, disparities_left)
        
        if self.opt.predict_right_disp:
            dst_file_right = os.path.join(dst_path, "disparities_right_pp_{}x{}.npy".format(self.opt.width, self.opt.height))
            if os.path.isfile(dst_file_right):
                os.remove(dst_file_right)
            np.save(dst_file_right, disparities_right)
        
        print("=> Done!")
        


if __name__=='__main__':
    
    # NOTE: options that need to be made consistent
    #   => --height --width --scales --predict_right_disp
    #   => --num_layers_G --netG_mode
    #   => --num_layers_T --frame_ids 
    #   => --val_iter --exp --post_process
    
    options = SharinOptions()
    opt = options.parse()
    opt.gpu = 0
    solver = Solver(opt)
    solver.Validate(opt.val_iter)