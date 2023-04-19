
import os
import glob
import numpy as np
import random
from tqdm import tqdm
import imageio
import cv2
import json
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
# from networks.da_net import Discriminator

from networks.layers import Project3D
from networks.layers import BackprojectDepth 
from networks.layers import disp_to_depth 
from networks.layers import SSIM
from networks.layers import get_smooth_loss

from bilinear_sampler import bilinear_sampler_1d_h

# from Dataloaders.VKitti2_dataloader import VKitti2 as syn_dataset
from Dataloaders.Kitti_dataloader import MonoKittiDataset as real_dataset
import Dataloaders.transform as transf


class Solver_Real():
    """Use for ablation study that do not use virtual data
    """
    def __init__(self, options):
        self.root_dir = './Real_Baseline'
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

        # Remove the generator network
        # Remove the discriminator network

        # Initialize the depth (and pose) task network 
        self.netT = networks.netT(
            self.opt.num_layers_T, 
            self.opt.scales, 
            self.num_pose_frames, 
            self.opt.frame_ids,
            self.use_pose,
            self.opt.predict_right_disp)
        netT_params = self.netT.get_parameters()

        self.netT.cuda(self.opt.gpu)
        
        # Initialize Optimizers => NOTE:
        #   => We use the same setting as PTNet_Baseline
        #   => Since Solver.py will also use this setting implicitly
        self.netT_optimizer = Optim.Adam(netT_params, lr=1e-4, betas=(0.95,0.999))
        self.lr_scheduler = Optim.lr_scheduler.StepLR(
            self.netT_optimizer, self.opt.scheduler_step_size, 0.1)


        # Training Configuration details
        self.batch_size = self.opt.batch_size
        self.workers = self.opt.num_workers 
        self.total_iterations = self.opt.total_iterations
        self.START_ITER = 0

        # Initialize task network losses
        self.ssim = SSIM() 
        self.ssim.cuda(self.opt.gpu)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            # NOTE: since backproject_depth and project_3d are initialized using opt.batch_size  => we should drop the last batch
            self.backproject_depth[scale] = BackprojectDepth(self.batch_size, h, w)
            self.backproject_depth[scale].cuda(self.opt.gpu)

            self.project_3d[scale] = Project3D(self.batch_size, h, w)
            self.project_3d[scale].cuda(self.opt.gpu)

        
        self.model_path = os.path.join(self.root_dir, "saved_models", self.opt.exp)
        self.log_path = os.path.join(self.root_dir, "tensorboard_logs/Kitti/train", self.opt.exp)
        for _p in [self.model_path, self.log_path]:
            if not os.path.isdir(_p):
                os.makedirs(_p)
                
        self.writer = SummaryWriter(self.log_path)
        
        # self.saved_models_dir = 'saved_models/{}'.format(self.opt.exp)
        # if not os.path.isdir(self.saved_models_dir):
        #     os.mkdir(self.saved_models_dir)

        # Initialize Data
        self.real_image = None

        self.get_training_data()
        self.get_training_dataloaders() 
        self.get_val_data()
        self.get_val_dataloader() 
        
        self.netT_loss_out = {}

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
            frame_ids=self.opt.frame_ids, # [0]
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
        
        self.real_loader = DataLoader(
            self.real_dataset,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            num_workers = self.workers,
            pin_memory=True,
            drop_last=True)
        
        self.real_iter = self.loop_iter(self.real_loader)


    def save_model(self, iter_id):
        final_dict = {}
        final_dict['iteration'] = iter_id,
        final_dict['netT_state_dict'] = self.netT.state_dict(),
        final_dict['netT_optimizer'] = self.netT_optimizer.state_dict(),
        
        torch.save(final_dict, os.path.join(self.model_path, 'Depth_Estimator-real_tmp.pth.tar'))
        
        os.system('mv '+os.path.join(self.model_path, 'Depth_Estimator-real_tmp.pth.tar')+' '+os.path.join(self.model_path, 'Depth_Estimator_real-'+str(iter_id)+'.pth.tar'))
    
    
    def rm_model(self, rm_iter):
        os.system('rm '+ os.path.join(self.model_path, 'Depth_Estimator_real-'+str(rm_iter)+'.pth.tar'))
        

    def get_real_data(self):
        self.real_inputs = next(self.real_iter)
        for key, ipt in self.real_inputs.items():
            self.real_inputs[key] = ipt.cuda(self.opt.gpu, non_blocking=True)


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
            

            # NOTE: NOW we separately compute temporal and stereo geometric reprojection
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
                

    def update_netT(self):

        self.set_requires_grad(self.netT, True)

        real_outputs = self.netT(self.real_inputs)

        # NOTE: backwarped results will be saved in real/syn_outputs for loss calculation
        self.generate_images_pred(self.real_inputs, real_outputs, temporal_only=True)
        
        # NOTE: task_losses include:
        # => "real": smooth_loss, temporal_consistency_loss
        # => "syn": "real" losses + lr_consistency_loss, stereo_consistency_loss, gt_depth_loss
        real_task_losses = self.compute_netT_losses(self.real_inputs, real_outputs, temporal_only=True)

        task_loss = real_task_losses["loss"] 
        
        # Used for tensorboard logging: only output scale-0
        self.netT_loss_out["real/task_loss"] = real_task_losses["loss"]
        for k, v in real_task_losses.items():
            if "0" in k:
                self.netT_loss_out["real/task_loss/{}".format(k)] = v 
        
        self.netT_loss = task_loss
        
        self.netT_optimizer.zero_grad()
        self.netT_loss.backward()
        self.netT_optimizer.step()



    def train_iter(self, iter_id):
        self.get_real_data()

        ###################################################
        #### Update netT
        ###################################################
        self.update_netT()

        ###################################################
        #### Tensorboard Logging
        ###################################################
        self.writer.add_scalar('Total Depth Regressor loss', self.netT_loss, iter_id)
        for k, v in self.netT_loss_out.items():
            self.writer.add_scalar('Total Depth Regressor loss/{}'.format(k), v, iter_id)

    

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
            # NOTE: here we only take the min of all temporal frames + stereo frame
            # NOTE: now to_optimise is only w.r.t. temporal consistency if stereo_mode == "sharinGAN"
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
        """Validation by task loss!
        Reason for not using abs_rel:
            => Now the depth is not scale-aware 
            => Thus abs_rel is not a proper metric in this case
        """
        self.netT.eval()
        
        with torch.no_grad():
            val_loss = 0.0
            cnt = 0
            for i, real_inputs in enumerate(self.real_val_loader):
                for key, ipt in real_inputs.items():
                    real_inputs[key] = ipt.cuda(self.opt.gpu, non_blocking=True)
                
                real_outputs = self.netT(real_inputs)

                self.generate_images_pred(real_inputs, real_outputs, temporal_only=True)
                
                real_task_losses = self.compute_netT_losses(real_inputs, real_outputs, temporal_only=True)
                val_loss += real_task_losses["loss"]
                cnt = i+1

            val_loss /= (1+i)
            print("=========================")
            print("=> gpu: {}".format(self.opt.gpu))
            print("=> validation finished: {} samples".format(cnt))
            print("=> current val_loss (averaged): {}".format(val_loss))


        if self.opt.rank % self.opt.ngpus == 0:
            self.writer.add_scalar("Validation/val_loss", val_loss, iter_id)
        
        self.netT.train()
        
        return val_loss
            
              
        