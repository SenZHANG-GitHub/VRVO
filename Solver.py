
import os
import glob
import numpy as np
import random
import json
from tqdm import tqdm
import imageio
import cv2
from PIL import Image

import torch
from torch import nn
import torch.optim as Optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms as tr
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from networks import all_networks
from networks import networks
from networks.da_net import Discriminator

from networks.layers import Project3D
from networks.layers import BackprojectDepth 
from networks.layers import disp_to_depth 
from networks.layers import SSIM
from networks.layers import get_smooth_loss

from bilinear_sampler import bilinear_sampler_1d_h

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
    def __init__(self, options):
        self.root_dir = '.'
        self.opt = options

        # Seed
        self.seed = 1729 # The famous Hardy-Ramanujan number
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
            self.netG.cuda(self.opt.gpu)
            netG_params = self.netG.parameters()

            self.resize = {}
            for i in range(self.num_scales):
                if i == 0: continue
                s = 2 ** i
                self.resize[i] = tr.Resize((self.opt.height // s, self.opt.width // s))
            
        elif self.opt.netG_mode == "monodepth2":
            self.netG = networks.netG(
                self.opt.num_layers_G, 
                self.opt.scales, 
                [0])
            self.netG.to_device(self.opt.gpu)
            netG_params = self.netG.get_parameters()

        # Initialize the discriminator network
        self.netD = [Discriminator(nout=1, last_layer_activation=False, hidden_in=5120)]
        if self.opt.D_multi_scale:
            for hidden_in in [1536, 1024, 512]:            
                self.netD.append(Discriminator(nout=1, last_layer_activation=False, hidden_in=hidden_in))

        # Initialize the depth (and pose) task network 
        self.netT = networks.netT(
            self.opt.num_layers_T, 
            self.opt.scales, 
            self.num_pose_frames, 
            self.opt.frame_ids,
            self.use_pose,
            self.opt.predict_right_disp)
        netT_params = self.netT.get_parameters()

        self.netG.cuda(self.opt.gpu)
        self.netT.cuda(self.opt.gpu)
        for disc in self.netD:
            disc.cuda(self.opt.gpu)
        
        # Initialize Optimizers
        self.netG_optimizer = Optim.Adam(netG_params, lr=1e-5)
        self.netD_optimizer = []
        for disc in self.netD:
            self.netD_optimizer.append(Optim.Adam(disc.parameters(), lr=1e-5))
        self.netT_optimizer = Optim.Adam(netT_params, lr=1e-5)


        # Training Configuration details
        self.batch_size = self.opt.batch_size
        self.workers = self.opt.num_workers 
        self.total_iterations = self.opt.total_iterations # In paper: 150000 rather than 20000
        self.START_ITER = 0

        self.kr = 1
        self.kd = 1 
        self.kcritic = 5

        # self.netD_loss = self.just_adv_loss + self.gamma*gp
        self.gamma = 10 
        
        # Initialize generator network losses
        self.netG_loss_fn = nn.MSELoss()
        self.netG_loss_fn = self.netG_loss_fn.cuda(self.opt.gpu)
        
        # Initialize task network losses
        self.ssim = SSIM() 
        self.ssim.cuda(self.opt.gpu)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            # NOTE: since backproject_depth and project_3d are initialized using opt.batch_size  => we should drop the last batch!
            self.backproject_depth[scale] = BackprojectDepth(self.batch_size, h, w)
            self.backproject_depth[scale].cuda(self.opt.gpu)

            self.project_3d[scale] = Project3D(self.batch_size, h, w)
            self.project_3d[scale].cuda(self.opt.gpu)
        

        # Transforms
        joint_transform_list = [transf.RandomImgAugment(no_flip=False, no_rotation=False, no_augment=False, size=(192,640))]
        img_transform_list = [tr.ToTensor(), tr.Normalize([.5, .5, .5], [.5, .5, .5])]
        self.joint_transform = tr.Compose(joint_transform_list)
        self.img_transform = tr.Compose(img_transform_list)

        
        self.model_path = os.path.join(self.root_dir, "saved_models", self.opt.exp)
        self.log_path = os.path.join(self.root_dir, "tensorboard_logs/vKitti2-Kitti/train", self.opt.exp)
        for _p in [self.model_path, self.log_path]:
            if not os.path.isdir(_p):
                os.makedirs(_p)
                
        self.writer = SummaryWriter(self.log_path)
        
        # self.saved_models_dir = 'saved_models/{}'.format(self.opt.exp)
        # if not os.path.isdir(self.saved_models_dir):
        #     os.mkdir(self.saved_models_dir)

        # Initialize Data
        self.real_image, self.syn_image, self.syn_label = None, None, None

        self.get_training_data()
        self.get_training_dataloaders() 
        self.get_val_data()
        self.get_val_dataloader() 
        
        self.netD_loss_out, self.netG_loss_out, self.netT_loss_out = {}, {}, {}
    
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
    
    
    def get_val_data(self):
        self.real_val_dataset = real_dataset(
            root_dir="data/kitti/kitti_raw",
            height=self.opt.height,
            width=self.opt.width,
            frame_ids=[0],
            num_scales=4,
            phase="val",
            folder=self.opt.kitti_folder)
    
    
    def get_val_dataloader(self):
        self.real_val_loader = DataLoader(
            self.real_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers = self.workers,
            pin_memory=True,
            drop_last=True)


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


    def load_pretrained_models(self):
        # Load Gen_Baseline pretrained model: e.g. "Gen_Baseline/saved_models/tmp/AE_Resnet_Baseline.pth.tar"
        model_state = torch.load(os.path.join(self.root_dir, self.opt.pretrained_model_G))
        self.netG.load_state_dict(model_state['netG_state_dict'])
        self.netG_optimizer.load_state_dict(model_state['netG_optimizer'])
        
        # Load PTNet_Baseline pretrained model: e.g. "PTNet_Baseline/saved_models/tmp/PTNet_baseline-126999_bicubic.pth.tar"
        model_state = torch.load(os.path.join(self.root_dir, self.opt.pretrained_model_T))
        self.netT.load_state_dict(model_state['netT_state_dict'])
        self.netT_optimizer.load_state_dict(model_state['netT_optimizer'])
            

    def load_prev_model(self):
        # e.g. "saved_models/Depth_Estimator_WI_geom_bicubic_da-9*.pth.tar"
        saved_models = glob.glob(os.path.join(self.root_dir, self.opt.load_joint_model))
        if len(saved_models)>0:
            saved_iters = [int(s.split('-')[-1].split('.')[0]) for s in saved_models]
            recent_id = saved_iters.index(max(saved_iters))
            saved_model = saved_models[recent_id]
            model_state = torch.load(saved_model)
            self.netG.load_state_dict(model_state['netG_state_dict'])
            self.netT.load_state_dict(model_state['netT_state_dict'])

            self.netG_optimizer.load_state_dict(model_state['netG_optimizer'])
            self.netT_optimizer.load_state_dict(model_state['netT_optimizer'])
            
            for i,disc in enumerate(self.netD):
                disc.load_state_dict(model_state['netD'+str(i)+'_state_dict'])
                self.netD_optimizer[i].load_state_dict(model_state['netD'+str(i)+'_optimizer_state_dict'])
            
            self.START_ITER = model_state['iteration']+1
            return True
        return False


    def save_model(self, iter_id):
        final_dict = {}
        final_dict['iteration'] = iter_id,
        final_dict['netG_state_dict'] = self.netG.state_dict(),
        final_dict['netT_state_dict'] = self.netT.state_dict(),
        final_dict['netG_optimizer'] = self.netG_optimizer.state_dict(),
        final_dict['netT_optimizer'] = self.netT_optimizer.state_dict(),
        
        for i,disc in enumerate(self.netD):
            final_dict['netD'+str(i)+'_state_dict'] = disc.state_dict()
            final_dict['netD'+str(i)+'_optimizer_state_dict'] = self.netD_optimizer[i].state_dict() 
        
        torch.save(final_dict, os.path.join(self.model_path, 'Depth_Estimator-da_tmp.pth.tar'))
        
        os.system('mv '+os.path.join(self.model_path, 'Depth_Estimator-da_tmp.pth.tar')+' '+os.path.join(self.model_path, 'Depth_Estimator_da-'+str(iter_id)+'.pth.tar'))
    
    
    def rm_model(self, rm_iter):
        os.system('rm '+ os.path.join(self.model_path, 'Depth_Estimator_da-'+str(rm_iter)+'.pth.tar'))
        

    def get_syn_data(self):
        self.syn_inputs = next(self.syn_iter)
        for key, ipt in self.syn_inputs.items():
            self.syn_inputs[key] = ipt.cuda(self.opt.gpu, non_blocking=True)


    def get_real_data(self):
        self.real_inputs = next(self.real_iter)
        for key, ipt in self.real_inputs.items():
            self.real_inputs[key] = ipt.cuda(self.opt.gpu, non_blocking=True)

        
    def gradient_penalty(self, model, h_s, h_t):
        # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
        batch_size =min(h_s.size(0), h_t.size(0))
        h_s = h_s[:batch_size]
        h_t = h_t[:batch_size]
        size = len(h_s.shape)
        alpha = torch.rand(batch_size)#, 1, 1, 1)
        for ki in range(1,size):
            alpha = alpha.unsqueeze(ki)
        alpha = alpha.expand_as(h_s)
        alpha = alpha.cuda()
        differences = h_t - h_s
        interpolates = h_s + (alpha * differences)
        # interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()
        interpolates = Variable(interpolates.cuda(), requires_grad=True)
        preds = model(interpolates)
        gradients = torch.autograd.grad(preds, interpolates,
                            grad_outputs=torch.ones_like(preds).cuda(),
                            retain_graph=True, create_graph=True)[0]
        gradients = gradients.view(batch_size,-1) 
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1)**2).mean()
        return gradient_penalty

    
    def gradient_x(self,img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx


    def gradient_y(self,img):
        gy = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gy


    # calculate the gradient loss
    def get_smooth_weight(self, depths, Images, num_scales):

        depth_gradient_x = [self.gradient_x(d) for d in depths]
        depth_gradient_y = [self.gradient_y(d) for d in depths]

        Image_gradient_x = [self.gradient_x(img) for img in Images]
        Image_gradient_y = [self.gradient_y(img) for img in Images]

        weight_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in Image_gradient_x]
        weight_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in Image_gradient_y]

        smoothness_x = [depth_gradient_x[i] * weight_x[i] for i in range(num_scales)]
        smoothness_y = [depth_gradient_y[i] * weight_y[i] for i in range(num_scales)]

        loss_x = [torch.mean(torch.abs(smoothness_x[i]))/2**i for i in range(num_scales)]
        loss_y = [torch.mean(torch.abs(smoothness_y[i]))/2**i for i in range(num_scales)]

        return sum(loss_x+loss_y)
    

    def reset_netD_grad(self, i=None):
        if i==None:
            for disc_op in self.netD_optimizer:
                disc_op.zero_grad()
        else:
            raise NotImplementedError()
            for idx, disc_op in enumerate(self.netD):
                if idx==i:
                    continue
                else:
                    disc_op.zero_grad()


    def reset_grad(self, exclude=None):
        if(exclude==None):
            self.netG_optimizer.zero_grad()
            self.reset_netD_grad()
            self.netT_optimizer.zero_grad()
        elif(exclude=='netG'):
            self.reset_netD_grad()
            self.netT_optimizer.zero_grad()
        elif(exclude=='netD'):
            self.netG_optimizer.zero_grad()
            self.netT_optimizer.zero_grad()
        elif(exclude=='netT'):
            self.netG_optimizer.zero_grad()
            self.reset_netD_grad()


    def forward_netD(self, mode='gen'):
        # NOTE: We can also use multi-scales here
        # => or only the last scale for simplicity
        
        # NOTE: netD has severe time consumption!
        # => using D_multi_scale: 50 hr -> 80 hr
        if self.opt.D_multi_scale:
            self.D_real = [self.netD[s](self.real_recon_imgs[("gen", s)]) for s in self.opt.scales]
            self.D_syn = [self.netD[s](self.syn_recon_imgs[("gen", s)]) for s in self.opt.scales]
        else:
            self.D_real = [self.netD[0](self.real_recon_imgs[("gen", 0)])]
            self.D_syn = [self.netD[0](self.syn_recon_imgs[("gen", 0)])]
        

    def loss_from_disc(self, mode='gen'):
        if self.opt.D_multi_scale:
            self.just_adv_loss = 0
            
            for s in self.opt.scales:
                if self.opt.correct_D_loss:
                    self.just_adv_loss += (torch.abs(self.D_syn[s] - self.D_real[s]).mean()) / (2**s)
                else:
                    self.just_adv_loss += (self.D_syn[s].mean() - self.D_real[s].mean()) / (2**s)
                    
            # self.just_adv_loss /= len(self.opt.scales)
            
        else:
            if self.opt.correct_D_loss:
                self.just_adv_loss = torch.abs(self.D_syn[0] - self.D_real[0]).mean()
            else:
                self.just_adv_loss = self.D_syn[0].mean() - self.D_real[0].mean()
                
        if mode == 'disc':
            self.just_adv_loss = -1* self.just_adv_loss
        

    def set_requires_grad(self, model, mode=False):
        for param in model.parameters():
            param.requires_grad = mode 


    def generate_images_pred(self, inputs, outputs, temporal_only=False):
        """Generate the warped (reprojected) color images for a minibatch.
        => temporal_only: only compute temporal consistency
        => Generated images are saved into the `outputs` dictionary.
        => inputs:  ("K", 0), ("inv_K", 0), "stereo_T", "stereo_T_right", "fb",
                    ("color", frame_id, 0), ("color", 0, 0), ("color", "s", 0) for grid_sample()
        => outputs: ("normalized_depth", 0, scale), (normalized_depth_right", 0, scale),
        =>          ("color", frame_id, scale), ("color_identity", frame_id, scale),
        =>          ("color", "right_est", scale),
        =>          ("disp_est", "r_to_l", scale), ("disp_est", "l_to_r", scale),
        =>          ("scaled_disp", "l", scale), ("scaled_disp", "r", scale)
        """
        for scale in self.opt.scales:
            if self.opt.predict_right_disp:
                disp = outputs[("disp", scale)][:, 0, :, :].unsqueeze(1)
                disp_right = outputs[("disp", scale)][:, 1, :, :].unsqueeze(1)
            else:
                disp = outputs[("disp", scale)]
                            
            # NOTE: we upsample each scale to the first scale before computing losses
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)    
            
            # NOTE: here depth is directly 1 / scaled_disp! 
            # and the depth is already rescaled to min_depth and max_depth!
            scaled_disp, depth, normalized_depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("normalized_depth", 0, scale)] = normalized_depth
            
            if not temporal_only and self.opt.predict_right_disp:
                
                disp_right = F.interpolate(
                disp_right, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                
                scaled_disp_right, depth_right, normalized_depth_right = disp_to_depth(disp_right, self.opt.min_depth, self.opt.max_depth)
                
                outputs[("normalized_depth_right", 0, scale)] = normalized_depth_right
            

            # NOTE: NOW we separately compute temporal and stereo geometric reprojection!
            # here is the temporal geometric reprojection
            
            if not temporal_only and self.opt.stereo_mode == "monodepth2":
                iter_ids = self.opt.frame_ids[1:] + ["s"]
            else:
                iter_ids = self.opt.frame_ids[1:]
                
            assert "s" not in self.opt.frame_ids
            
            for i, frame_id in enumerate(iter_ids):
                
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    # predicted from pose decoder
                    T = outputs[("cam_T_cam", 0, frame_id)]
                    
                # NOTE: source_scale is always 0 if we interpolate disp to be also to be 640x192
                cam_points = self.backproject_depth[0](
                    depth, inputs[("inv_K", 0)])
                pix_coords = self.project_3d[0](
                    cam_points, inputs[("K", 0)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                # NOTE: for stereo images, we also use 3D space projection based on K and depth rather than disparities with bilinear_sampler_1d!
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, 0)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, 0)]
            
            # NOTE: The notation difference
            # => outputs[("color", frame_id, scale)]: frame_id: 0/-1/1/"s": the estimated I_0 (I_left) from frame_id
            # => outputs[("color", "right_est", scale)]: the estimated I_right from I_0 (I_left)
            
            if not temporal_only and self.opt.stereo_mode == "monodepth2" and self.opt.predict_right_disp:
                T = inputs["stereo_T_right"]
                cam_points = self.backproject_depth[0](
                    depth_right, inputs[("inv_K", 0)])
                pix_coords = self.project_3d[0](
                    cam_points, inputs[("K", 0)], T)
                
                outputs[("sample", "right_est", scale)] = pix_coords
                outputs[("color", "right_est", scale)] = F.grid_sample(
                    inputs[("color", 0, 0)],
                    outputs[("sample", "right_est", scale)],
                    padding_mode="border")
            
            # NOTE: compute the stereo geometric reprojection based on bilinear_sampler_1d here
            # bilinear_sampler_1d_h only needs normalized disp and will * width inside the function
            # scaled_disp here from the depth inverse to fb * depth inverse
            if not temporal_only:
                for ib in range(scaled_disp.shape[0]):
                    scaled_disp[ib, :, :, :] *= inputs["fb"][ib]
                    if self.opt.predict_right_disp:
                        scaled_disp_right[ib, :, :, :] *= inputs["fb"][ib]
                
            # depth range: (0.1, 100) -> scaled_disp_range: (0.01, 10)
            # fb: 0.58 * 0.54 -> ~0.3, thus scaled_disp.max() is possible to be > 1 at the beginning of training
            # assert scaled_disp.max() <= 1

            # NOTE: for left to right, baseline > 0 while the pixels move left (* -1.0)
            if not temporal_only and self.opt.stereo_mode == "sharinGAN":
                # outputs[("color", "s", scale)] is left_est from right image
                outputs[("color", "s", scale)] = self.generate_image_left(
                        inputs[("color", "s", 0)], scaled_disp)
                
                if self.opt.predict_right_disp:
                    # outputs[("color", "right_est", scale)] is right_est from left image
                    outputs[("color", "right_est", scale)] = self.generate_image_right(
                        inputs[("color", 0, 0)], scaled_disp_right)
            
            if not temporal_only and self.opt.predict_right_disp:
                # NOTE: scaled_disp and scaled_disp_right are real disparities that have * fb 
                # bilinear_sampler_1d_h() will * width to make it pixel-level inside the func
                outputs["disp_est", "r_to_l", scale] = self.generate_image_left(
                    scaled_disp_right, scaled_disp)
                outputs["disp_est", "l_to_r", scale] = self.generate_image_right(
                    scaled_disp, scaled_disp_right)
                
                outputs["scaled_disp", "l", scale] = scaled_disp 
                outputs["scaled_disp", "r", scale] = scaled_disp_right
                

    def construct_recon_inputs(self):
        """Construct the inputs for netT using reconstructed shared features        
        """
        real_recon_inputs, syn_recon_inputs = {}, {}
        
        # Used for multi-scale reconstruction loss
        if self.opt.netG_mode == "sharinGAN":
            real_recon_inputs[("color_aug", 0, 0)] = self.real_recon_imgs[("gen", 0)]
            syn_recon_inputs[("color_aug", 0, 0)] = self.syn_recon_imgs[("gen", 0)]
        else:
            for scale in self.opt.scales:
                real_recon_inputs[("color_aug", 0, scale)] = self.real_recon_imgs[("gen", scale)]
                syn_recon_inputs[("color_aug", 0, scale)] = self.syn_recon_imgs[("gen", scale)]

        # Used for pose encoder
        for f_i in self.opt.frame_ids:
            if f_i == 0: continue
            _, real_recon_tmp = self.netG(self.real_inputs["color_aug", f_i, 0])
            _, syn_recon_tmp = self.netG(self.syn_inputs["color_aug", f_i, 0])
            real_recon_inputs[("color_aug", f_i, 0)] = real_recon_tmp[("gen", 0)]
            syn_recon_inputs[("color_aug", f_i, 0)] = syn_recon_tmp[("gen", 0)]

        if self.opt.direct_raw_img:
            for f_i in self.opt.frame_ids: # [0, -1, 1]
                for scale in self.opt.scales:
                    real_recon_inputs[("color", f_i, scale)] = self.real_inputs["color", f_i, scale]
                    syn_recon_inputs[("color", f_i, scale)] = self.syn_inputs["color", f_i, scale]
            
            syn_recon_inputs[("color", "s", 0)] = self.syn_inputs["color", "s", 0]
            if self.opt.predict_right_disp:
                # Used for computing smoothness loss for predicted right disp
                for scale in self.opt.scales:
                    if scale == 0: continue
                    syn_recon_inputs[("color", "s", scale)] = self.syn_inputs["color", "s", scale]
                    
        else:
            for f_i in self.opt.frame_ids: # [0, -1, 1]
                _, real_recon_tmp = self.netG(self.real_inputs["color", f_i, 0])
                _, syn_recon_tmp = self.netG(self.syn_inputs["color", f_i, 0])
                for scale in self.opt.scales:
                    real_recon_inputs[("color", f_i, scale)] = real_recon_tmp[("gen", scale)]
                    syn_recon_inputs[("color", f_i, scale)] = syn_recon_tmp[("gen", scale)]
            
            
            _, syn_recon_tmp = self.netG(self.syn_inputs["color", "s", 0])
            syn_recon_inputs[("color", "s", 0)] = syn_recon_tmp[("gen", 0)]
            if self.opt.predict_right_disp:
                # Used for computing smoothness loss for predicted right disp
                for scale in self.opt.scales:
                    if scale == 0: continue
                    syn_recon_inputs[("color", "s", scale)] = syn_recon_tmp[("gen", scale)]


        real_recon_inputs[("K", 0)] = self.real_inputs[("K", 0)]
        real_recon_inputs[("inv_K", 0)] = self.real_inputs[("inv_K", 0)]

        syn_recon_inputs[("K", 0)] = self.syn_inputs[("K", 0)]
        syn_recon_inputs[("inv_K", 0)] = self.syn_inputs[("inv_K", 0)]
        syn_recon_inputs["stereo_T"] = self.syn_inputs["stereo_T"]
        syn_recon_inputs["stereo_T_right"] = self.syn_inputs["stereo_T_right"]
        syn_recon_inputs["fb"] = self.syn_inputs["fb"]
        syn_recon_inputs[("depth", "l", 0)] = self.syn_inputs[("depth", "l", 0)]
        syn_recon_inputs[("depth", "r", 0)] = self.syn_inputs[("depth", "r", 0)]
        
        return real_recon_inputs, syn_recon_inputs


    def update_netG(self):
        self.set_requires_grad(self.netT, False)
        for disc in self.netD:
            self.set_requires_grad(disc, False)
        
        _, self.real_recon_imgs = self.netG(self.real_inputs["color_aug", 0, 0])
        _, self.syn_recon_imgs = self.netG(self.syn_inputs["color_aug", 0, 0])

        # NOTE: In PTNet
        # => we use "color_aug" to generate depth and pose
        # => and then we back-sample from "color"
        # NOTE: Thus here
        # => we use reconstructed "color_aug" for depth and pose prediction
        # => and the back-sample from reconstructed "color"
        real_recon_inputs, syn_recon_inputs = self.construct_recon_inputs()
        
        # NOTE: here we use the reconstructed feature map to predict depth!!!
        # NOTE: In netT.forward(), we will feed inputs["color_aug", 0, 0] to encoder 
        real_outputs = self.netT(real_recon_inputs)
        syn_outputs = self.netT(syn_recon_inputs)

        # NOTE: backwarped results will be saved in real/syn_outputs for loss calculation
        self.generate_images_pred(real_recon_inputs, real_outputs, temporal_only=True)
        self.generate_images_pred(syn_recon_inputs, syn_outputs, temporal_only=False)

        # Sen -> Will use real/syn_recon_image to update netD loss (self.just_adv_loss)
        self.forward_netD() 
        self.loss_from_disc()

        real_recon_losses = self.compute_netG_losses(self.real_inputs, self.real_recon_imgs)
        syn_recon_losses = self.compute_netG_losses(self.syn_inputs, self.syn_recon_imgs)

        recon_loss = real_recon_losses["loss"] + syn_recon_losses["loss"]

        # NOTE: task_losses include:
        # => "real": smooth_loss, temporal_consistency_loss
        # => "syn": "real" losses + lr_consistency_loss, stereo_consistency_loss, gt_depth_loss
        real_task_losses = self.compute_netT_losses(real_recon_inputs, real_outputs, temporal_only=True)
        syn_task_losses = self.compute_netT_losses(syn_recon_inputs, syn_outputs, temporal_only=False)
        
        task_loss = real_task_losses["loss"] + syn_task_losses["loss"]        
        
        # Used for tensorboard logging: only output scale 0
        self.netG_loss_out["real/recon_loss"] = real_recon_losses["loss"]
        self.netG_loss_out["syn/recon_loss"] = syn_recon_losses["loss"]
        self.netG_loss_out["real/task_loss"] = real_task_losses["loss"]
        self.netG_loss_out["syn/task_loss"] = syn_task_losses["loss"]
        self.netG_loss_out["just_adv_loss"] = self.just_adv_loss 
        
        # only output scale 0 losses -> Note that total loss /= num_scales to match each scale's loss
        for k, v in real_task_losses.items():
            if "0" in k:
                self.netG_loss_out["real/task_loss/{}".format(k)] = v 
        for k, v in syn_task_losses.items():
            if "0" in k:
                self.netG_loss_out["syn/task_loss/{}".format(k)] = v
        
        self.netG_loss = self.just_adv_loss + task_loss + recon_loss
        
        self.reset_grad()
        self.netG_loss.backward()
        self.reset_grad(exclude='netG')
        self.netG_optimizer.step()

        self.set_requires_grad(self.netT, True)
        for disc in self.netD:
            self.set_requires_grad(disc, True)


    def update_netT(self):

        self.set_requires_grad(self.netG, False)
        for disc in self.netD:
            self.set_requires_grad(disc, False)

        _, self.real_recon_imgs = self.netG(self.real_inputs["color_aug", 0, 0])
        _, self.syn_recon_imgs = self.netG(self.syn_inputs["color_aug", 0, 0])

        # NOTE: In PTNet
        # => we use "color_aug" to generate depth and pose
        # => and then we back-sample from "color"
        # NOTE: Thus here
        # => we use reconstruted "color_aug" for depth and pose prediction
        # => and the back-sample from reconstructed "color"
        real_recon_inputs, syn_recon_inputs = self.construct_recon_inputs()
        
        # NOTE: here we use the reconstructed feature map to predict depth!!!
        # NOTE: In netT.forward(), we will feed inputs["color_aug", 0, 0] to encoder 
        real_outputs = self.netT(real_recon_inputs)
        syn_outputs = self.netT(syn_recon_inputs)

        # NOTE: backwarped results will be saved in real/syn_outputs for loss calculation
        self.generate_images_pred(real_recon_inputs, real_outputs, temporal_only=True)
        self.generate_images_pred(syn_recon_inputs, syn_outputs, temporal_only=False)
        
        # NOTE: task_losses include:
        # => "real": smooth_loss, temporal_consistency_loss
        # => "syn": "real" losses + lr_consistency_loss, stereo_consistency_loss, gt_depth_loss
        real_task_losses = self.compute_netT_losses(real_recon_inputs, real_outputs, temporal_only=True)
        syn_task_losses = self.compute_netT_losses(syn_recon_inputs, syn_outputs, temporal_only=False)

        task_loss = real_task_losses["loss"] + syn_task_losses["loss"] 
        
        # Used for tensorboard logging: only output scale-0
        self.netT_loss_out["real/task_loss"] = real_task_losses["loss"]
        self.netT_loss_out["syn/task_loss"] = syn_task_losses["loss"]
        for k, v in real_task_losses.items():
            if "0" in k:
                self.netT_loss_out["real/task_loss/{}".format(k)] = v 
        for k, v in syn_task_losses.items():
            if "0" in k:
                self.netT_loss_out["syn/task_loss/{}".format(k)] = v
        
        self.netT_loss = task_loss
        
        self.reset_grad()
        self.netT_loss.backward()
        self.reset_grad(exclude='netT')
        self.netT_optimizer.step()

        self.set_requires_grad(self.netG, True)
        for disc in self.netD:
            self.set_requires_grad(disc, True)


    def update_netD(self):
        
        self.set_requires_grad(self.netG, False)

        with torch.no_grad():
            _, self.syn_recon_imgs  = self.netG(self.syn_inputs["color_aug", 0, 0])
            _, self.real_recon_imgs = self.netG(self.real_inputs["color_aug", 0, 0])

        for _ in range(self.kcritic):
            # Will use self.syn_recon_image and self.real_recon_image to calc self.just_adv_loss
            self.forward_netD(mode='disc')
            self.loss_from_disc(mode='disc')
            
            # NOTE: We can also use multi-scales here
            # => Currently only the last scale for simplicity
            if self.opt.D_multi_scale:
                gp = 0
                for s in self.opt.scales:
                    gp += self.gradient_penalty(self.netD[s], self.syn_recon_imgs[("gen", s)], self.real_recon_imgs[("gen", s)]) / (2 ** s)
            
            else:
                gp = self.gradient_penalty(self.netD[0], self.syn_recon_imgs[("gen", 0)], self.real_recon_imgs[("gen", 0)])
            
            # Use for tensorboard logging
            self.netD_loss_out["just_adv_loss"] = self.just_adv_loss 
            self.netD_loss_out["gradient_penalty"] = self.gamma * gp
            
            self.netD_loss = self.just_adv_loss + self.gamma * gp
            self.netD_step()

        self.set_requires_grad(self.netG, True)


    def netD_step(self):
        self.reset_grad()
        self.netD_loss.backward()
        self.reset_grad(exclude='netD')
        for disc_op in self.netD_optimizer:
            disc_op.step()


    def train_iter(self, iter_id):
        self.get_syn_data()
        self.get_real_data()
        ###################################################
        #### Update netD
        ###################################################
        
        self.update_netD()
    
        ###################################################
        #### Update netG
        ###################################################            
        
        for i in range(self.kr):
            self.update_netG()

        ###################################################
        #### Update netT
        ###################################################
        
        self.update_netT()

        ###################################################
        #### Tensorboard Logging
        ###################################################            
        self.writer.add_scalar('1) Total Generator loss', self.netG_loss, iter_id)
        for k, v in self.netG_loss_out.items():
            self.writer.add_scalar('1) Total Generator loss/{}'.format(k), v, iter_id)
            
        self.writer.add_scalar('2) Total Discriminator loss', self.netD_loss, iter_id)
        for k, v in self.netD_loss_out.items():
            self.writer.add_scalar('2) Total Discriminator loss/{}'.format(k), v, iter_id)
        
        self.writer.add_scalar('3) Total Depth Regressor loss', self.netT_loss, iter_id)
        for k, v in self.netT_loss_out.items():
            self.writer.add_scalar('3) Total Depth Regressor loss/{}'.format(k), v, iter_id)

    

    def compute_netT_losses(self, inputs, outputs, temporal_only):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        total_losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            losses = {}
            loss = 0
                        
            temporal_reproj_losses = []
            
            # NOTE: disp here is only used to compute smoothness loss
            if not temporal_only and self.opt.predict_right_disp:
                disp = outputs[("disp", scale)][:, 0, :, :].unsqueeze(1)
                disp_right = outputs[("disp", scale)][:, 1, :, :].unsqueeze(1)
            else:
                disp = outputs[("disp", scale)]
            
            # NOTE: color is used with disp to computing smoothness loss
            color = inputs[("color", 0, scale)]            
            target = inputs[("color", 0, 0)]
            
            if not temporal_only and self.opt.predict_right_disp:
                color_right = inputs[("color", "s", scale)]
                target_right = inputs[("color", "s", 0)]

            if not temporal_only and self.opt.stereo_mode == "monodepth2":
                iter_ids = self.opt.frame_ids[1:] + ["s"]
            else:
                iter_ids = self.opt.frame_ids[1:]
            
            assert "s" not in self.opt.frame_ids
            
            for frame_id in iter_ids:
                pred = outputs[("color", frame_id, scale)]
                temporal_reproj_losses.append(self.compute_reprojection_loss(pred, target))

            temporal_reproj_losses = torch.cat(temporal_reproj_losses, 1)

            # NOTE: automasking from monodepth2
            identity_temporal_reproj_losses = []
            for frame_id in iter_ids:
                pred = inputs[("color", frame_id, 0)]
                identity_temporal_reproj_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_temporal_reproj_losses = torch.cat(identity_temporal_reproj_losses, 1)

            if self.opt.avg_reprojection:
                identity_temporal_reproj_loss = identity_temporal_reproj_losses.mean(1, keepdim=True)
            else:
                # save both images, and do min all at once below
                identity_temporal_reproj_loss = identity_temporal_reproj_losses

            if self.opt.avg_reprojection:
                temporal_reproj_loss = temporal_reproj_losses.mean(1, keepdim=True)
            else:
                temporal_reproj_loss = temporal_reproj_losses

            # NOTE: Apply automasking
            # add random numbers to break ties
            identity_temporal_reproj_loss += torch.randn(
                identity_temporal_reproj_loss.shape).cuda() * 0.00001
            combined = torch.cat((identity_temporal_reproj_loss, temporal_reproj_loss), dim=1)

            # NOTE: here we add the auto-masked geometric loss
            # NOTE: here we only take the min of all temporal frames + stereo frame!
            # NOTE: should stereo frame be considered separatedly??? 
            # NOTE: Updated: now to_optimise is only w.r.t. temporal consistency if stereo_mode == "sharinGAN"
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_temporal_reproj_loss.shape[1] - 1).float()
            
            losses["temporal_reproj_loss"] = to_optimise.mean()
            losses["temporal_reproj_loss"] *= self.opt.temp_reproj_weight
            loss += losses["temporal_reproj_loss"]
            
            # NOTE: Add the right stereo consistency loss while downweighted due to to_optimise is the min of [-1, 1, "s"]
            if not temporal_only and self.opt.stereo_mode == "monodepth2" and self.opt.predict_right_disp:
                pred_right = outputs[("color", "right_est", scale)]
                losses["right_temp_reproj_loss"] = self.compute_reprojection_loss(pred_right, target_right).mean() 
                losses["right_temp_reproj_loss"] *= self.opt.temp_reproj_weight
                losses["right_temp_reproj_loss"] /= 3.0
                loss += losses["right_temp_reproj_loss"]
                
            # NOTE: Updated: add stereo consistency loss from bilinear_sampler_1d and disparity 
            #   => Now add left/right_stereo_reproj_loss even for monodepth2
            if not temporal_only and (self.opt.add_sharinGAN_gc or self.opt.stereo_mode == "sharinGAN"):
                pred = outputs[("color", "s", scale)]
                losses["left_stereo_reproj_loss"] = self.compute_reprojection_loss(pred, target).mean() 
                losses["left_stereo_reproj_loss"] *= self.opt.stereo_gc_weight
                losses["left_stereo_reproj_loss"] /= 2.0
                loss += losses["left_stereo_reproj_loss"]
                
                if self.opt.predict_right_disp:
                    pred_right = outputs[("color", "right_est", scale)]
                    losses["right_stereo_reproj_loss"] = self.compute_reprojection_loss(pred_right, target_right).mean() 
                    losses["right_stereo_reproj_loss"] *= self.opt.stereo_gc_weight
                    losses["right_stereo_reproj_loss"] /= 2.0
                    loss += losses["right_stereo_reproj_loss"]
            
            # NOTE: Add the lr consistency loss from monodepth here 
            if not temporal_only and self.opt.predict_right_disp:
                losses["lr_left_loss"] = torch.abs(outputs[("disp_est", "r_to_l", scale)] - 
                                         outputs[("scaled_disp", "l", scale)]).mean()
                losses["lr_right_loss"] = torch.abs(outputs[("disp_est", "l_to_r", scale)] - 
                                         outputs[("scaled_disp", "r", scale)]).mean()
                
                losses["lr_left_loss"] *= self.opt.lr_loss_weight
                losses["lr_right_loss"] *= self.opt.lr_loss_weight
                
                loss += losses["lr_left_loss"]
                loss += losses["lr_right_loss"]
            
            # NOTE: here we compute the ground-truth depth loss
            # NOTE: the ground-truth depth is normalized by (0.5,) and (0.5,)
            if not temporal_only:
                losses["left_gt_depth_loss"] = torch.abs(outputs[("normalized_depth", 0, scale)] - inputs[("depth", "l", 0)]).mean()
                
                losses["left_gt_depth_loss"] *= self.opt.gt_depth_weight
                loss += losses["left_gt_depth_loss"]
                
                if self.opt.predict_right_disp:
                    losses["right_gt_depth_loss"] = torch.abs(outputs[("normalized_depth_right", 0, scale)] - inputs[("depth", "r", 0)]).mean()
                    
                    losses["right_gt_depth_loss"] *= self.opt.gt_depth_weight
                    loss += losses["right_gt_depth_loss"]
            
            # NOTE: here we add the smoothness loss of the disp
            # NOTE: smooth_loss will be down-weighted by the scale
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            losses["left_smooth_loss"] = get_smooth_loss(norm_disp, color)
            losses["left_smooth_loss"] = losses["left_smooth_loss"] / (2 ** scale)
            losses["left_smooth_loss"] *= self.opt.disparity_smoothness
            loss += losses["left_smooth_loss"]
            
            if not temporal_only and self.opt.predict_right_disp:
                mean_disp_right = disp_right.mean(2, True).mean(3, True)
                norm_disp_right = disp_right / (mean_disp_right + 1e-7)
                losses["right_smooth_loss"] = get_smooth_loss(norm_disp_right, color_right)
                losses["right_smooth_loss"] = losses["right_smooth_loss"] / (2 ** scale)
                losses["right_smooth_loss"] *= self.opt.disparity_smoothness
                loss += losses["right_smooth_loss"]
                
            total_loss += loss
            total_losses["{}/loss".format(scale)] = loss
            for key, val in losses.items():
                total_losses["{}/{}".format(scale, key)] = val

        # NOTE: now total_losses: total_loss /= 4 to match each scale's loss scale 
        total_loss /= self.num_scales
        total_losses["loss"] = total_loss
        return total_losses
    

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
            recon_loss = recon_loss / (2 ** scale)
            recon_loss *= self.opt.recon_loss_weight
            losses["{}/recon_loss".format(scale)] = recon_loss
            loss += recon_loss
        
        loss /= len(scales)
        losses["loss"] = loss
        return losses
    
    
    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)
    
    
    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)
    
    
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    
    def val(self, iter_id):
        """Validation by gt_depth error metrics
        """
        self.netG.eval()
        self.netT.eval()
        self.netT.encoder.eval()
        self.netT.depth.eval()
        for disp in self.netD:
            disp.eval()
            
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
                    t_id_global = (i*self.batch_size) + t_id
     
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

                    # print('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
                    #     .format(t_id, abs_rel[t_id], sq_rel[t_id], rmse[t_id], rmse_log[t_id], a1[t_id], a2[t_id], a3[t_id]))

            print("====================================")
            print("=> validation finished: {} samples".format(num_samples))
            print('{:>10},{:>10},{:>10},{:>10},{:>10},{:>10},{:>10}'.format('abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3'))
            print('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
                .format(abs_rel.mean(),sq_rel.mean(),rmse.mean(),rmse_log.mean(),a1.mean(),a2.mean(),a3.mean()))

        val_loss = abs_rel.mean()
        if self.opt.rank % self.opt.ngpus == 0:
            self.writer.add_scalar("Validation/abs_rel", val_loss, iter_id)
        
        self.netG.train()
        self.netT.train()
        self.netT.encoder.train()
        self.netT.depth.train()
        for disp in self.netD:
            disp.train()
        
        return val_loss
            
            
    def tensor2im(self, depth):
        """Transform normalized depth values back to [0, 80m]
        """
        # (batch, 1, 192, 640) => (batch, 192, 640, 1)
        depth_numpy = depth.cpu().data.float().numpy().transpose(0,2,3,1)
        depth_numpy = (depth_numpy + 1.0) / 2.0 # Unnormalize between 0 and 1
        return depth_numpy * MAX_DEPTH
        
        
    def get_gt_depth(self, folder, depth_idx):
        """
        => folder: e.g. "2011_09_28/"2011_09_28_drive_0002_sync"
        => depth_idx: e.g. 30
        """
        depth_file = os.path.join("./data/kitti/kitti_raw/depth_from_velodyne", folder, "{:010d}.png".format(depth_idx))
        
        # Sen -> The PNG file is uint16 and depth = float(I) / 256.0 (valid if > 0)
        depth = Image.open(depth_file)

        # Sen -> Bug? By KITTI depth data format: Should divided by 256.0
        # depth = np.array(depth, dtype=np.float32) / 255.0
        depth = np.array(depth, dtype=np.float32) / 256.0
        return depth        
        