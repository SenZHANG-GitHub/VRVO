import os
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
import imageio
import time

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
from networks.RBDN_original import RBDN_network

from Dataloaders.VKitti_dataloader import VKitti as syn_dataset
from Dataloaders.Kitti_dataloader import DepthToTensor, KittiDataset as real_dataset
import Dataloaders.transform as transf


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

        # Initialize networks
        self.netG = all_networks.define_G(3, 3, 64, 9, 'batch',
                                                  'PReLU', 'ResNet', 'kaiming', 0,
                                                  False, [0])

        self.netG.cuda()

        # Training Configuration details
        self.batch_size = 16
        self.iteration = None
        # Transforms
        joint_transform_list = [transf.RandomImgAugment(no_flip=True, no_rotation=True, no_augment=True, size=(192,640)]
        img_transform_list = [tr.ToTensor(), tr.Normalize([.5, .5, .5], [.5, .5, .5])]
        self.joint_transform = tr.Compose(joint_transform_list)
        self.img_transform = tr.Compose(img_transform_list)
        self.depth_transform = tr.Compose([DepthToTensor()])
        
        self.saved_models_dir = 'saved_models'

        # Initialize Data
        self.get_validation_data()

    def loop_iter(self, loader):
        while True:
            for data in iter(loader):
                yield data

    def get_validation_data(self):
        self.syn_val_dataloader = DataLoader(syn_dataset(train=False), batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        self.real_val_dataset = real_dataset(data_file='test.txt',phase='test',img_transform=self.img_transform, joint_transform=self.joint_transform, depth_transform=self.depth_transform)
        self.real_val_dataloader = DataLoader(self.real_val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=4)
        
    def load_prev_model(self):
        saved_models = glob.glob(os.path.join(self.root_dir, self.saved_models_dir, 'Depth_Estimator_da-'+str(self.iteration)+'.pth.tar' ))
        if len(saved_models)>0:
            saved_iters = [int(s.split('-')[2].split('.')[0]) for s in saved_models]
            recent_id = saved_iters.index(max(saved_iters))
            saved_model = saved_models[recent_id]
            model_state = torch.load(saved_model)
            self.netG.load_state_dict(model_state['netG_state_dict'])

            return True
        return False

    def Validate(self):
        self.netG.eval()
        saved_models = glob.glob(os.path.join(self.root_dir, self.saved_models_dir, 'Depth_Estimator_da*.pth.tar' ))
        self.iteration = self.opt.iter
        self.load_prev_model()
        self.Validation()

    def Validation(self):
        if not os.path.exists(os.path.join('Visualization_results/'+'/'+str(self.iteration))):
            os.system('mkdir -p '+os.path.join('Visualization_results/'+'/'+str(self.iteration)))
        with torch.no_grad():
            for i,(data, depth_filenames) in tqdm(enumerate(self.real_val_dataloader)): 
                self.real_val_image = data['left_img']#, data['depth'] # self.real_depth is a numpy array 
                self.real_val_image = Variable(self.real_val_image.cuda())

                _, self.real_translated_image = self.netG(self.real_val_image)
                
                real_val_image_numpy = self.real_val_image.cpu().data.numpy().transpose(0,2,3,1)
                real_translated_image_numpy = self.real_translated_image.cpu().data.numpy().transpose(0,2,3,1)
                real_val_image_numpy = (real_val_image_numpy + 1.0) / 2.0
                real_translated_image_numpy = (real_translated_image_numpy + 1.0) / 2.0
                if i==1:
                    for k in range(self.real_val_image.size(0)):
                        np.save('Visualization_results/'+'/'+str(self.iteration)+'/Real_'+str(16*i+k)+'.png', real_val_image_numpy[k])
                        np.save('Visualization_results/'+'/'+str(self.iteration)+'/Real_translated_'+str(16*i+k)+'.png', real_translated_image_numpy[k])
                    break
            
            for i,(self.syn_val_image, _) in tqdm(enumerate(self.syn_val_dataloader)): 
                self.syn_val_image = Variable(self.syn_val_image.cuda())

                _, self.syn_translated_image = self.netG(self.syn_val_image)
                
                syn_val_image_numpy = self.syn_val_image.cpu().data.numpy().transpose(0,2,3,1)
                syn_translated_image_numpy = self.syn_translated_image.cpu().data.numpy().transpose(0,2,3,1)
                syn_val_image_numpy = (syn_val_image_numpy + 1.0) / 2.0
                syn_translated_image_numpy = (syn_translated_image_numpy + 1.0) / 2.0
                if i==1:
                    for k in range(self.syn_val_image.size(0)):
                        np.save('Visualization_results/'+'/'+str(self.iteration)+'/Syn_'+str(16*i+k)+'.png', syn_val_image_numpy[k])
                        np.save('Visualization_results/'+'/'+str(self.iteration)+'/Syn_translated_'+str(16*i+k)+'.png', syn_translated_image_numpy[k])
                    break
def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', default=999, type=int, help="Indicate what iteration of the saved model to be started with for Validation")
    opt = parser.parse_args()
    return opt

if __name__=='__main__':
    opt = get_params()
    solver = Solver(opt)
    solver.Validate()