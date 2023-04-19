# VRVO

This is the official PyTorch implementation for **[Towards Scale Consistent Monocular Visual Odometry by Learning from the Virtual World], ICRA 2022**

If you find this work useful in your research, please consider citing our paper:

```
@inproceedings{zhang2022towards,
  title={Towards scale consistent monocular visual odometry by learning from the virtual world},
  author={Zhang, Sen and Zhang, Jing and Tao, Dacheng},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)},
  pages={5601--5607},
  year={2022},
  organization={IEEE}
}
```

* This project is developed under python 3.6 and pytorch 1.3.1

* **Our re-implemented DVSO** (C++) is released at https://github.com/SenZHANG-GitHub/DVSO 
    * You will need this repo in **stage 4: DVSO_Finetune**
    * Yang, Nan, et al. "Deep virtual stereo odometry: Leveraging deep depth prediction for monocular direct sparse odometry." Proceedings of the European conference on computer vision (ECCV). 2018.


## Usage
+ **Stage 1: PTNet_Baseline pretraining (main_PTNet.py)**
    + --num_layers_T --stereo_mode --vbaseline --predict_right_disp
+ **Stage 2: Gen_Baseline pretraining (main_DVSO.py)**
    + --num_layers_G --netG_mode
+ **Stage 3: Domain adaptation joint training (main.py)**
    + --num_layers_G --num_layers_T
    + --kitti_folder matters!
        + "Kitti" will use Eigen split (Deprecated!)
        + "Kitti-Zhan" will use seq 00-08 for training/val and 09-10 for test (Default now!)
+ **Stage 4: DVSO_Finetune (main_DVSO.py)**
    + --dvso_epochs --dvso_train_seqs --dvso_test_seqs --dvso_resume_exp --dvso_resume_iter --dvso_home_path
    + --dvso_param (wStereoPosFlag wCorrectedFlag wGradFlag wStereo scaleENergyLeftTHR scaleWJI2SumTHR warpright checkWarpValid maskWarpGrad) 
    + If we only want to finetune netT while fixing netG and netD
        + --dvso_netT_only
        + If we further want to use only real dataset during dvso finetuning and --dvso_netT_only
            + --dvso_real_only
    + If we want to use dvso's sparse depth maps for supervision
        + --use_dvso_depth --dvso_depth_weight
    + If we want to maintain the intermediate disparities rather than cleaning them to save space
        + --dvso_maintain_disp
    + If we only want to test the model (--dvso_resume_exp --dvso_resume_iter) and then quit
        + --dvso_test_only
    + If we want to use L2 loss for dvso axisangle and translation vec6 poses
        + --use_pose_loss --rot_weight --trans_weight
    + If we want to use dvso poses for temporal photometric consistency loss
        + --use_dvso_photo --dvso_photo_weight


## Acknowledgment
This repo is built upon the excellent works of [SharinGAN](https://github.com/koutilya-pnvr/SharinGAN), [monodepth]https://github.com/mrharicot/monodepth), and [monodepth2](https://github.com/nianticlabs/monodepth2). The borrowed codes are licensed under the original license respectively.



