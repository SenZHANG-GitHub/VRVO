import pdb
import os

seqs = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
for seq in seqs:
    os.system("cp ../../../data/kitti/odometry/monodepth_disps/{}/camera_small.txt camera_512x256_{}.txt".format(seq, seq, seq))