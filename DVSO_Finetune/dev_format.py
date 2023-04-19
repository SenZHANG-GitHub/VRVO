"""
This script is used for transforming DSO results (with ids rather than timestamps) to KITTI format

=> For evo:
    => Results are saved in aligned_seq_suf/
    => No id remained, but each line of dso and kitti results are matched

=> For Zhan:
    => Results are saved in zhan_seq_suf/seq.txt
    => Also save info_seq.txt 
        => start {}\n end {}\n percent {}%\n

"""
from __future__ import absolute_import, division, print_function
import pdb
import os
import argparse
from shutil import rmtree
import numpy as np 
from transforms3d import euler 
from transforms3d import quaternions as quat

parser = argparse.ArgumentParser(description="Formatting DSO results for KITTI")
parser.add_argument("--seq", type=str, required=True, help="e.g. 09")
parser.add_argument("--result_path", type=str, required=True, help="e.g. tmp") 
parser.add_argument("--param", type=str, required=True, help="e.g. tmp")

def arr2str(arr):
    """
    Transfer an np.array to a string separated by space
    """
    out = " ".join([str(x) for x in arr])
    return out 


def dso_to_kitti(seq, result_path, param):
    """
    (1) Read in dso results and kitti gt results
    (2) Output aligned dso_results and kitti_results 
    """
    fpath = "{}/{}/result.txt".format(result_path, param)

    ## Processing dso results for current experiment
    #   => key: id, value: kitti_pose
    #   => kitti_pose: np.array(12): row-aligned T matrix
    dso_results = dict()
    dso_ids = []
    with open(fpath, "r") as f:
        for line in f.readlines():
            line = line.split()
            i = int(line[0])
            pose = [float(x) for x in line[1:]]
            trans = np.array(pose[:3]).reshape((3,1))
            qx, qy, qz, qw = pose[3:]
            rot = quat.quat2mat([qw, qx, qy, qz])
            T = np.hstack([rot, trans]).flatten()
            dso_ids.append(i)
            dso_results[i] = T
    # pdb.set_trace()
    
    ## Fetching ground-truth poses from kitti dataset
    #   => key: id, value: kitti_pose
    #   => kitti_pose: np.array(12): row-aligned T matrix
    gt_results = dict()
    gt_ids = []
    gpath = "../data/kitti/odometry/dataset/poses/{}.txt".format(seq)
    with open(gpath, "r") as g:
        for i, line in enumerate(g.readlines()):
            T = [float(x) for x in line.split()]
            gt_ids.append(i)
            gt_results[i] = T

    ## Check whether there is dso_id that does not appear in gt_ids
    for i in dso_ids:
        if i not in gt_ids:
            raise ValueError("There exists dso_id ({}) not in gt_ids!".format(i))

    ## Processing ids to be included 
    # => Based on whether the ids are continuous in dso_ids
    id_seqs = []
    id_seq = [dso_ids[0]]
    for i, dso_id in enumerate(dso_ids):
        if i == 0: 
            continue
        
        if dso_ids[i] == dso_ids[i-1] + 1:
            id_seq.append(dso_id)
        else:
            id_seqs.append(id_seq)
            id_seq = [dso_id]

    id_seqs.append(id_seq)
    
    # print("================================")
    # print("=> seq: {}, exp: {}, param: {} finished".format(seq, exp, param))
    # print("=> number of records in dso_results: {}".format(len(dso_ids)))
    # print("=> number of records in gt_results: {}".format(len(gt_ids)))
    # print("    => The starting index: {}".format(gt_ids[0]))
    # print("    => The ending index: {}".format(gt_ids[-1]))
    # print("=> number of continuous id sequences in dso_results: {}".format(len(id_seqs)))
    # for i, s in enumerate(id_seqs):
    #     print("    => length of sub_id_seq {}: {}".format(i, len(s)))

    ## The length of id_seqs should be 2 or 1
    # => The first id_seq is [0]
    # => The second id_seq should middle indices without beginning and ending of all images
    # => If the length of id_seqs is 1: No beginning images are skipped
    if len(id_seqs) not in [1, 2]:
        raise ValueError("Something wrong: More continuous subsequenes appear!")
    if len(id_seqs) == 2:
        assert len(id_seqs[0]) == 1 
        assert id_seqs[0][0] == 0
    id_eval = id_seqs[-1]
    # print("=> sub_id_seq to be evaluated {}: {}".format(len(id_seqs) - 1, len(id_eval)))
    # print("    => The starting index: {}".format(id_eval[0]))
    # print("    => The ending index: {}".format(id_eval[-1]))
    # print("    => The percentage of ids included: {:.2f}%".format(100 * len(id_eval)/len(gt_ids)))

    # ## Save dso/kitti_results to aligned_XX/
    # # Write results files for evo
    # # => dso_09.txt and kitti_09.txt
    # opath = "results/{}/aligned_{}_{}".format(seq, seq, suf)
    # if not os.path.isdir(opath):
    #     os.mkdir(opath)

    # with open("{}/dso_{}.txt".format(opath, seq), "w") as df:
    #     with open("{}/kitti_{}.txt".format(opath, seq), "w") as kf:
    #         for i in id_eval:
    #             df.write("{}\n".format(arr2str(dso_results[i])))
    #             kf.write("{}\n".format(arr2str(gt_results[i])))

    # Write results files for Huangying-Zhan/kitti-odom-eval
    opath = "{}/{}/zhan_{}".format(result_path, param, seq)

    if not os.path.isdir(opath):
        os.mkdir(opath)
    with open("{}/{}.txt".format(opath, seq), "w") as zf:
        for i in id_eval:
            zf.write("{} {}\n".format(i, arr2str(dso_results[i])))

    # with open("{}/info_{}.txt".format(opath, seq), "w") as zf:
    #     zf.write("start {}\n".format(id_eval[0]))
    #     zf.write("end {}\n".format(id_eval[-1]))
    #     zf.write("percent {}\n".format(100 * len(id_eval)/len(gt_ids)))

    
if __name__ == "__main__":
    """
    usage: python dev_format.py --seq 09/10 --result_path xxx --param xxx 
    """
    opt = parser.parse_args()
    # tmp_path = opt.result_path.split("/")
    # assert tmp_path[0] == "DVSO_Finetune"
    # tmp_path = "/".join(tmp_path[1:])
    dso_to_kitti(opt.seq, opt.result_path, opt.param)


