from glob import glob
import pdb


def get_scenes(phase, scenes):
    with open("bk_new/{}.txt".format(phase), "r") as f:
        for line in f.readlines():
            tmp = line.split()[0]
            if tmp not in scenes:
                scenes.append(tmp)
    return scenes


def write_file(phase, rm_lines):
    out_lines = []
    with open("bk_new/{}.txt".format(phase), "r") as f:
        for line in f.readlines():
            if line in rm_lines:
                continue
            out_lines.append(line)
    
    with open("{}.txt".format(phase), "w") as f:
        for line in out_lines:
            f.write(line)


def main():
    scenes = []
    for phase in ["train", "test", "val"]:
        scenes = get_scenes(phase, scenes)
    root_dir = "../../data/kitti/kitti_raw"
    
    rm_lines = []
    for scene in scenes:
        imgs = glob("{}/{}/image_02/data/*.jpg".format(root_dir, scene))
        imgs = [int(x.split("/")[-1].split(".")[0]) for x in imgs]
        assert min(imgs) == 0
        # remove the first and last frame due to frame_ids[0, -1, 1]
        rm_lines.append("{} {} l\n".format(scene, min(imgs)))
        rm_lines.append("{} {} l\n".format(scene, max(imgs)))

    # Not need to re-write val.txt and test.txt
    for phase in ["train"]:
        write_file(phase, rm_lines)
    

if __name__ == "__main__":
    main()
    
    