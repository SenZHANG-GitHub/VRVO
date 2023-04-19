import os
import glob
import numpy as np
import random
from tqdm import tqdm
import json

import torch
from torch import nn
import torch.optim as Optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as tr
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


from networks import all_networks
from networks import networks

from Dataloaders.VKitti2_dataloader import VKitti2 as syn_dataset
from Dataloaders.Kitti_dataloader import MonoKittiDataset as real_dataset
from Dataloaders.transform import *

class Solver_Gen():
    def __init__(self, options):
        self.root_dir = './Gen_Baseline'
        
        self.opt = options
        
        # Seed
        self.seed = 1729
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        
        self.num_scales = len(self.opt.scales) # [0, 1, 2, 3]
        
        # NOTE: Now frame_ids are only used for temporal consistency
        # NOTE: We manually specify "s" in codes
        assert "s" not in self.opt.frame_ids

        # Initialize networks
        if self.opt.netG_mode == "sharinGAN":
            self.netG = all_networks.define_G(3, 3, 64, 9, 'batch',
                                                    'PReLU', 'ResNet', 'kaiming', 0,
                                                    False, [self.opt.gpu])
            self.netG.cuda(self.opt.gpu)
            netG_params = self.netG.parameters()
            
        elif self.opt.netG_mode == "monodepth2":
            self.netG = networks.netG(
                self.opt.num_layers_G, 
                self.opt.scales, 
                [0])
            self.netG.to_device(self.opt.gpu)
            netG_params = self.netG.get_parameters()
            
        # Initialize Loss
        self.netG_loss_fn = nn.MSELoss()
        self.netG_loss_fn = self.netG_loss_fn.cuda(self.opt.gpu)
        
        # Initialize Optimizers
        self.netG_optimizer = Optim.Adam(netG_params, lr=5e-5, betas=(0.5,0.9))
        
        # Training Configuration details
        if self.opt.distributed:
            self.batch_size = int(self.opt.batch_size / self.opt.ngpus)
            self.workers = int((self.opt.num_workers + self.opt.ngpus -1) / self.opt.ngpus)
            self.netT = DDP(self.netT, device_ids=[self.opt.gpu], find_unused_parameters=True)
        else:
            self.batch_size = self.opt.batch_size
            self.workers = self.opt.num_workers 
            
        self.START_ITER = 0
        
        self.kr = 1
        self.kd = 1 
        
        if self.opt.rank % self.opt.ngpus == 0:
            self.model_path = os.path.join(self.root_dir, "saved_models", self.opt.exp)
            self.log_path = os.path.join(self.root_dir, "tensorboard_logs/vKitti2/", self.opt.exp ,"AE_Baseline/Resnet_NEW")
            
            if not os.path.isdir(self.model_path):
                os.makedirs(self.model_path)
            if not os.path.isdir(self.log_path):
                os.makedirs(self.log_path)
                
            self.writer = SummaryWriter(self.log_path)
            
            print("=> Models are saved to {}\n".format(self.model_path))
            print("=> Tensorboard events are saved to {}\n".format(self.log_path))


        # # Transforms
        # joint_transform_list = [RandomImgAugment(no_flip=False, no_rotation=False, no_augment=False, size=(192,640))]
        # img_transform_list = [tr.ToTensor(), tr.Normalize([.5, .5, .5], [.5, .5, .5])]
        # self.joint_transform = tr.Compose(joint_transform_list)
        # self.img_transform = tr.Compose(img_transform_list)
        # self.depth_transform = tr.Compose([DepthToTensor()])

        # Initialize Data
        self.syn_image, self.syn_label, self.real_image = None, None, None
        self.get_training_data()
        self.get_training_dataloaders()
        
        if self.opt.rank % self.opt.ngpus == 0:
            self.save_opts()
    
    
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        opts_dir = os.path.join(self.log_path, "options")
        if not os.path.exists(opts_dir):
            os.makedirs(opts_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(opts_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)


    def loop_iter(self, loader):
        while True:
            for data in iter(loader):
                yield data


    def get_training_data(self):
        self.syn_dataset = syn_dataset(
            root_dir="data/virtual-kitti",
            height=self.opt.height,
            width=self.opt.width,
            baseline=self.opt.vbaseline,
            frame_ids=self.opt.frame_ids,
            num_scales=4,
            phase="train",
            preload=self.opt.preload_virtual_data)
        
        self.real_dataset = real_dataset(
            root_dir="data/kitti/kitti_raw",
            height=self.opt.height,
            width=self.opt.width,
            frame_ids=self.opt.frame_ids,
            num_scales=4,
            phase="train",
            folder=self.opt.kitti_folder)
        
    
    def get_training_dataloaders(self):
        # if we want to use multi-gpu, we need to combine syn/real_datasets to use train_sampler -> currently disable parallelism
        assert self.opt.distributed is False 
        train_sampler=None 
        
        self.syn_loader = DataLoader(
            self.syn_dataset,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            num_workers = self.workers // 2,
            pin_memory=True,
            drop_last=True)
        
        self.real_loader = DataLoader(
            self.real_dataset,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            num_workers = self.workers // 2,
            pin_memory=True,
            drop_last=True)
        
        self.syn_iter = self.loop_iter(self.syn_loader)
        self.real_iter = self.loop_iter(self.real_loader)


    def load_prev_model(self):
        saved_models = glob.glob(os.path.join(self.root_dir, 'saved_models', 'AE_Resnet_Baseline.pth.tar' ))
        if len(saved_models)>0:
            if self.opt.distributed:
                loc = "cuda:{}".format(self.opt.gpu)
                model_state = torch.load(saved_models[0], map_location=loc)
            else:
                model_state = torch.load(saved_models[0])
                
            self.netG.load_state_dict(model_state['netG_state_dict'])
            
            self.netG_optimizer.load_state_dict(model_state['netG_optimizer']
                                                )
            self.START_ITER = model_state['iteration']+1
            return True
        return False


    def save_model(self, iter_id):
        """ we only save the last model
        """
        torch.save({
                'iteration': iter_id,
                'netG_state_dict': self.netG.state_dict(),
                'netG_optimizer': self.netG_optimizer.state_dict(),
                }, os.path.join(self.model_path, 'AE_Resnet_Baseline.pth.tar'))
        
        
    def get_syn_data(self):
        self.syn_inputs = next(self.syn_iter)
        for key, ipt in self.syn_inputs.items():
            self.syn_inputs[key] = ipt.cuda(self.opt.gpu, non_blocking=True)
        
        
    def get_real_data(self):
        self.real_inputs = next(self.real_iter)
        for key, ipt in self.real_inputs.items():
            self.real_inputs[key] = ipt.cuda(self.opt.gpu, non_blocking=True)
        

    def update_netG(self):
        
        # NOTE: we use "color_aug" as input, thus we should compare with "color_aug" for reconstruction as well
        _, self.real_recon_imgs = self.netG(self.real_inputs["color_aug", 0, 0])
        _, self.syn_recon_imgs = self.netG(self.syn_inputs["color_aug", 0, 0])
        
        
        self.real_recon_losses = self.compute_netG_losses(self.real_inputs, self.real_recon_imgs)
        self.syn_recon_losses = self.compute_netG_losses(self.syn_inputs, self.syn_recon_imgs)
        
        self.netG_loss = self.real_recon_losses["loss"] + self.syn_recon_losses["loss"]
        
        self.netG_optimizer.zero_grad()
        self.netG_loss.backward()
        self.netG_optimizer.step()

    
    def compute_netG_losses(self, inputs, recon_imgs):
        """
        => inputs: ("color_aug", 0, scale) is used as image_input
        => recon_imgs: ("gen", scale) is the reconstructed images
            => for netG_mode == "sharinGAN", we only have scale 0
        """
        assert self.opt.netG_mode in ["monodepth2", "sharinGAN"]
        
        scales = self.opt.scales if self.opt.netG_mode == "monodepth2" else [0]
        
        losses = {}
        loss = 0
        
        for scale in scales:
            recon_loss = self.netG_loss_fn(inputs[("color_aug", 0, scale)], recon_imgs[("gen", scale)])
            losses["{}/recon_loss".format(scale)] = recon_loss / (2 ** scale)
            loss += losses["{}/recon_loss".format(scale)]
        
        loss /= len(scales)
        losses["loss"] = loss
        return losses
        


    def train_iter(self, iter_id):
        
        self.get_syn_data()
        self.get_real_data()
        
        self.update_netG()

        ###################################################
        #### Tensorboard Logging
        ################################################### 
        if self.opt.rank % self.opt.ngpus == 0:                 
            self.writer.add_scalar('Total Generator loss', self.netG_loss, iter_id)
            for key in sorted(list(self.real_recon_losses)):
                self.writer.add_scalar("Real/{}".format(key), self.real_recon_losses[key], iter_id)
            for key in sorted(list(self.syn_recon_losses)):
                self.writer.add_scalar("Syn/{}".format(key), self.syn_recon_losses[key], iter_id)





        
