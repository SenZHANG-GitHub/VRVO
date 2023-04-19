import os
import torch
import numpy as np
from PIL import Image
from torch.utils import data

from torchvision import transforms 
from torchvision import utils
import random

class MonoKittiOdomDataset(data.Dataset):
    """Only used for generating depth npy files
    => only data[("color", 0, 0)] is required
    """
    def __init__(self, 
                 root_dir, 
                 height,
                 width,
                 seq):

        # e.g. "data/kitti/odometry/dataset/sequences"
        self.root_dir = root_dir
        self.height = height 
        self.width = width 
        self.seq = seq 
        
        self.file = "dataset_files/Kitti-Odom/{}.txt".format(self.seq)
        
        with open(self.file,'r') as f:
            # e.g. "01/image_2/000008.jpg 01/image_3/000008.jpg\n"
            self.filepaths = f.readlines()        
            
        self.to_tensor = transforms.ToTensor()
            
        self.interp = Image.ANTIALIAS
        self.resize = transforms.Resize((self.height, self.width),
                                            interpolation=self.interp)
            
                                    
    def __len__(self):
        return len(self.filepaths)

    
    def get_color(self, image_file):
        color = Image.open(os.path.join(self.root_dir, image_file)).convert("RGB")
        
        return color

    
    def preprocess(self, inputs):
        """Resize colour images to the required scales and to_tensor()
        """
        inputs[("color", 0, 0)] = self.resize(inputs[("color", 0, -1)])
        inputs[("color", 0, 0)] = self.to_tensor(inputs[("color", 0, 0)])
        
            
    def __getitem__(self, index):
        
        inputs = {}
        
        # ["01/image_2/000008.jpg", "01/image_3/000008.jpg"]
        line = self.filepaths[index].strip().split() 
        
        # use the left image as input
        image_file = line[0]

        # NOTE: ("color", frame_id, scale)
        inputs[("color", 0, -1)] = self.get_color(image_file)
        
        self.preprocess(inputs)
        del inputs[("color", 0, -1)]

        return inputs