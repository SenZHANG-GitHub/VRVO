"""
This is Sen's naive implementation of forward warping
-> The float disparities are simply floored to int
"""


import numpy as np
import pdb
import os
import argparse
import sys
from tqdm import tqdm

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--exp", type=str)
    return parser.parse_args()

def readCalib(seq):
    lines = []
    with open("dataset_files/Kitti-Odom/calib/camera_640x192_{}.txt".format(seq), mode="r") as f:
        lines = f.readlines()
    fx = float(lines[0].strip().split()[0])
    w = int(lines[3].strip().split()[0])
    h = int(lines[3].strip().split()[1])
    bl = float(lines[4].strip())
    return fx, w, h, bl


def warp_right_disp(opt):
    """Warp the left disparity to a warped right disparity using monodepth2 left_disp
    -> opt.disp_path should be data/kitti/odometry/monodepth2_disps
    -> usage: python warp_right_disp.py --seq 09 --disp_path data/kitti/odometry/monodepth2_disps
    """
    
    opt.disp_path = "saved_models/{}/disps".format(opt.exp)
    
    fx, w, h, bl = readCalib(opt.seq)
    disp_left = np.load("{}/{}/disparities_left_pp_640x192.npy".format(opt.disp_path, opt.seq))
    disp_right = np.zeros(disp_left.shape, dtype=disp_left.dtype)

    for idx in tqdm(range(disp_left.shape[0])):
        tmp_left = disp_left[idx] * w

        virtual_right = np.zeros((h, w))

        for iw in range(w):
            for ih in range(h):
                new_w = iw + int(tmp_left[ih, iw])
                if 0 <= new_w < w:
                    if virtual_right[ih, new_w] > 0:
                        # If multiple-pixel collisions happen, we use the closer one to relieve occlusion
                        if abs(tmp_left[ih, iw]) > abs(virtual_right[ih, new_w]):
                            virtual_right[ih, new_w] = tmp_left[ih, iw] 
                    else:
                        virtual_right[ih, new_w] = tmp_left[ih, iw] 
                    # virtual_right_rev[ih, new_w] = tmp_left[ih, iw] * -1

        # By comparing with tmp_right, virtual_right is correct without times -1
        mask = virtual_right > 0
        disp_right[idx] = virtual_right / w
    
    out_path = "{}/{}/disparities_right_pp_warped.npy".format(opt.disp_path, opt.seq)
    print("=> Saving warped results to {}".format(out_path))
    np.save(out_path, disp_right)


if __name__ == "__main__":
    opt = parseArgs()
    warp_right_disp(opt)




