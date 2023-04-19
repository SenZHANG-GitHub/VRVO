import os
import torch
import numpy as np
from PIL import Image
from torch.utils import data

from torchvision import transforms 
from torchvision import utils
import random

from Dataloaders.Kitti_Odom_dataloader import MonoKittiOdomDataset 


def resize_kitti(seq):
    print("============================")
    print("=> processing {}...".format(seq))
    odom_dataset = MonoKittiOdomDataset(
        "data/kitti/odometry/dataset/sequences",
        192,
        640,
        seq
    )
    outpath = os.path.join("data/kitti/odometry/resized_imgs", seq, "image_2_640x192")
    for f in odom_dataset.filepaths:
        # e.g. "01/image_2/000008.jpg 01/image_3/000008.jpg\n"
        img_file = f.split()[0]
        img_jpg = img_file.split("/")[-1] # "000008.jpg"
        color = Image.open(os.path.join(odom_dataset.root_dir, img_file)).convert("RGB")
        color = odom_dataset.resize(color)
        color.save(os.path.join(outpath, img_jpg))


if __name__ == "__main__":
    for seq in ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]:
        resize_kitti(seq)