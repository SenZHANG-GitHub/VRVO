import pdb
import os
import numpy as np


def process_file(ftype):
    imgs = {}
    scenes = []
    with open("{}_bk.txt".format(ftype), "r") as f:
        for line in f.readlines():
            # e.g. vKitti2/Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00434.jpg
            line = line.strip().split("/")
            scene = "/".join(line[:3]) # e.g. vKitti2/Scene01/15-deg-left
            img = int(line[-1].split(".")[0][4:])
            if scene not in scenes:
                scenes.append(scene)
                imgs[scene] = []
            imgs[scene].append(img)

    with open("{}_syn_vkitti2.txt".format(ftype), "w") as f:
        for scene in scenes:
            seqs = imgs[scene]
            seqs = sorted(seqs, reverse=False)
            # Remove the first and last images for temporal consistency indexing
            seqs = seqs[1:-1]
            for idx in seqs:
                f.write("{} {}\n".format(scene, idx))




if __name__ == "__main__":
    process_file("train")
    process_file("test")


