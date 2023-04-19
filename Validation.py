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

from Dataloaders.VKitti2_dataloader import VKitti2 as syn_dataset
from Dataloaders.Kitti_dataloader import MonoKittiDataset as real_dataset
import Dataloaders.transform as transf


MIN_DEPTH = 1e-3
MAX_DEPTH = 80


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
        
        # self.writer = SummaryWriter(os.path.join(self.root_dir,'tensorboard_logs/Vkitti-kitti',self.val_string))
        self.saved_models_dir = 'saved_models'

        # Initialize Data
        # Initialize Data
        self.get_validation_data()


    def loop_iter(self, loader):
        while True:
            for data in iter(loader):
                yield data


    def get_validation_data(self):
        phase = "val" if self.opt.val else "test"
        self.real_val_dataset = real_dataset(
            root_dir="data/kitti/kitti_raw",
            height=self.opt.height,
            width=self.opt.width,
            frame_ids=[0],
            num_scales=4,
            phase=phase)
    
        self.real_val_loader = DataLoader(
            self.real_val_dataset,
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

    def tensor2im(self, depth):
        """Transform normalized depth values back to [0, 80m]
        """
        # (batch, 1, 192, 640) => (batch, 192, 640, 1)
        depth_numpy = depth.cpu().data.float().numpy().transpose(0,2,3,1)
        depth_numpy = (depth_numpy + 1.0) / 2.0 # Unnormalize between 0 and 1
        return depth_numpy * MAX_DEPTH

    
    def Validate(self):
        self.netG.eval()
        self.netT.eval()
        
        saved_models = glob.glob(os.path.join(self.root_dir, self.saved_models_dir, self.opt.exp, 'Depth_Estimator_da*.pth.tar' ))
        
        for saved_model in saved_models:
            self.load_prev_model(saved_model)
            self.Validation(saved_model)
            

    def get_gt_depth(self, folder, depth_idx):
        """
        => folder: e.g. "2011_09_28/"2011_09_28_drive_0002_sync"
        => depth_idx: e.g. 30
        """
        depth_file = os.path.join("./data/kitti/kitti_raw/depth_from_velodyne", folder, "{:010d}.png".format(depth_idx))
        
        # The PNG file is uint16 and depth = float(I) / 256.0 (valid if > 0)
        depth = Image.open(depth_file)

        # By KITTI depth data format: Should divided by 256.0
        # depth = np.array(depth, dtype=np.float32) / 255.0
        depth = np.array(depth, dtype=np.float32) / 256.0
        return depth      
    

    def Validation(self, saved_model):
        phase = "val" if self.opt.val else "test"
        print("==========================================")
        print("=> evaluating ({}): {}".format(phase, saved_model))
        self.netG.eval()
        self.netT.eval()
        self.netT.encoder.eval()
        self.netT.depth.eval()
        
        num_samples = len(self.real_val_dataset)
        abs_rel = np.zeros(num_samples, np.float32)
        sq_rel = np.zeros(num_samples,np.float32)
        rmse = np.zeros(num_samples,np.float32)
        rmse_log = np.zeros(num_samples,np.float32)
        a1 = np.zeros(num_samples,np.float32)
        a2 = np.zeros(num_samples,np.float32)
        a3 = np.zeros(num_samples,np.float32)

        with torch.no_grad():
            for i, data in enumerate(self.real_val_loader):
                input_color = data[("color", 0, 0)].cuda(self.opt.gpu)
                if self.opt.post_process:
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                
                _, recon_img = self.netG(input_color)   
                input_recon = recon_img[("gen", 0)]             
                outputs = self.netT.depth(self.netT.encoder(input_recon))
                
                # Note: load the left disparities for evaluation
                scale = 0
                if self.opt.predict_right_disp:
                    disp = outputs[("disp", scale)][:, 0, :, :].unsqueeze(1)
                else:
                    disp = outputs[("disp", scale)]
                # NOTE: (batch(*2), 1, height, width), e.g. (16(*2), 1, 192, 640)
                # => pred_disp and _ is calculated using max_depth = 100.0 (monodepth2)
                # => normalized_depth is depth / 80.0 and then Normalize((0.5,), (0.5,))
                pred_disp, _, normalized_depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)  
                
                if self.opt.val_depth_mode == "disp":
                    pred_disp = pred_disp.cpu()[:, 0].numpy() # (batch(*2), 192, 640)
                    if self.opt.post_process:
                        N = pred_disp.shape[0] // 2
                        pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
                    
                    curr_batch_size = pred_disp.shape[0]
                    
                    
                elif self.opt.val_depth_mode == "normalized_depth":
                    # 0-80m, (batch, height, width, 1)
                    depth_numpy = self.tensor2im(normalized_depth) 
                    curr_batch_size = pred_disp.shape[0]
                

                for t_id in range(curr_batch_size):
                    t_id_global = (i*self.batch_size)+t_id
                    
                    h, w = self.opt.height, self.opt.width
                    
                    # e.g. 2011_09_28/"2011_09_28_drive_0002_sync 0000000030 l
                    fpath = self.real_val_dataset.filepaths[t_id_global].strip()
                    folder = fpath.split()[0]
                    depth_idx = int(fpath.split()[1])
                    assert fpath.split()[2] == "l"
                    
                    gt_depth = self.get_gt_depth(folder, depth_idx) 
                    gt_height, gt_width = gt_depth.shape[:2]

                    if self.opt.val_depth_mode == "disp":
                        pred_depth = cv2.resize(pred_disp[t_id], (gt_width, gt_height))
                        pred_depth = 1 / pred_depth
                        
                    elif self.opt.val_depth_mode == "normalized_depth":
                        pred_depth = cv2.resize(depth_numpy[t_id], (gt_width, gt_height),interpolation=cv2.INTER_LINEAR)
                    
                    
                    # NOTE: crop used by Garg ECCV16 
                    # => also the "eigen" split's crop in monodepth2
                    mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

                    crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,
                                        0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
                    

                    crop_mask = np.zeros(mask.shape)
                    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                    mask = np.logical_and(mask, crop_mask)

                    pred_depth = pred_depth[mask]
                    gt_depth = gt_depth[mask]
                    
                    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
                    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
                    
                    abs_rel[t_id_global], sq_rel[t_id_global], rmse[t_id_global], rmse_log[t_id_global], a1[t_id_global], a2[t_id_global], a3[t_id_global] = compute_errors(gt_depth,pred_depth)
            
            
            print ('{:>10},{:>10},{:>10},{:>10},{:>10},{:>10},{:>10}'.format('abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3'))
            print ('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
                .format(abs_rel.mean(),sq_rel.mean(),rmse.mean(),rmse_log.mean(),a1.mean(),a2.mean(),a3.mean()))


if __name__=='__main__':
    
    # NOTE: options that need to be made consistent
    #   => --height --width --scales --predict_right_disp
    #   => --num_layers_G --netG_mode
    #   => --num_layers_T --frame_ids 
    
    options = SharinOptions()
    opt = options.parse()
    opt.gpu = 0
    solver = Solver(opt)
    solver.Validate()