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
from torchvision import transforms 
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from networks import all_networks
from networks import networks
from bilinear_sampler import bilinear_sampler_1d_h
from networks.layers import disp_to_depth 

from Dataloaders.VKitti2_dataloader import VKitti2 as syn_dataset


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
        
        self.scene = "Scene20"
        self.seqs = ["clone", "sunset", "morning", "15-deg-left", "15-deg-right", "rain", "overcast", "fog", "30-deg-left", "30-deg-right"]
        
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
        

        self.saved_models_dir = 'saved_models'
        self.img_files = None
        self.resize = transforms.Resize((self.opt.height, self.opt.width), interpolation=Image.ANTIALIAS)
        self.to_tensor = transforms.ToTensor()


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
        
        saved_model = os.path.join(self.root_dir, self.saved_models_dir, self.opt.exp, 'Depth_Estimator_da-{}.pth.tar'.format(val_iter))
        
        self.load_prev_model(saved_model)
        for seq in self.seqs:
            self.save_depth(saved_model, seq, val_iter)
            

    def read_data_vkitti2(self, seq):
        data_path = "data/virtual-kitti/vKitti2/{}/{}/frames/rgb/Camera_0".format(self.scene, seq)
        self.img_files = glob.glob("{}/*.jpg".format(data_path))
        self.img_files.sort() 
        
    
    def save_depth(self, saved_model, seq, val_iter):
        print("==========================================")
        print("=> saving the disparity npy of {}: {}".format(self.scene, seq))
        print("=> using {}".format(saved_model))
        self.netG.eval()
        self.netT.eval()
        self.netT.encoder.eval()
        self.netT.depth.eval()
        
        ## NOTE: Read camera intrinsics for 640x192 or 512x256
        baseline = 0.532725
        K = np.array([[0.5837429, 0, 0.4995974, 0],
                  [0, 1.9333565, 0.4986667, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    
        focal_baseline = K[0,0] * baseline
        
        self.read_data_vkitti2(seq)
        
        num_samples = len(self.img_files)
                
        if self.opt.save_feat:
            feat_path = os.path.join(self.root_dir, "results_DVSO/vKitti2/results", self.opt.exp, 'feats', self.scene, seq)
            if not os.path.isdir(feat_path):
                os.makedirs(feat_path)
        else:
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
            for i, img_file in enumerate(self.img_files):
                
                # Check whether the order is correct
                img_index = int(img_file.split("/")[-1].split(".")[0].split("_")[-1])
                assert i == img_index
                
                # Read img from jpg
                color = Image.open(img_file).convert("RGB")
                color = self.resize(color)
                color = self.to_tensor(color)
                
                input_color = color.cuda(self.opt.gpu)
                input_color = input_color.unsqueeze(0) # ~ batch_size 1 => (1, 3, 192, 640)
                
                if self.opt.post_process and not self.opt.save_feat:
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                
                _, recon_img = self.netG(input_color)   
                input_recon = recon_img[("gen", 0)]
                
                if self.opt.save_feat:
                    feat = input_recon.cpu()[0].numpy() # (c, h, w)
                    feat = np.rollaxis(feat, 0, 3) # (h, w, c)
                    
                    # [-1, 1] from tanh() to [0, 255]
                    # feat = np.uint8((feat + 1.0) / 2.0 * 255)
                    # NOTE: based on recon_loss => directly compared with color_aug 
                    feat[feat < 0] = 0
                    feat = np.uint8(feat * 255)
                    
                    feat = Image.fromarray(feat)
                    feat.save(os.path.join(feat_path, "{:06d}.jpg".format(i)))
                    if i % 300 == 0:
                        print("=> Processed {:d} of {:d} images".format(i, num_samples))
                else:             
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
                        
                    elif self.opt.val_depth_mode == "normalized_depth":
                        raise ValueError("Not Supported NOW")
                    

                
                    real_disp_left = focal_baseline * pred_disp_left[0]
                    real_disp_right = focal_baseline * pred_disp_right[0]
                    
                    disparities_left[i] = real_disp_left.squeeze()
                    disparities_right[i] = real_disp_right.squeeze()
                    
                    if i % 1000 == 0:
                        print("=> Processed {:d} of {:d} images".format(i, num_samples))
        
        if not self.opt.save_feat:
            dst_path = os.path.join(self.root_dir, "results_DVSO/vKitti2/results", self.opt.exp, 'disps', self.scene, seq)
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