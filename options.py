import os 
import argparse 

class SharinOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="sharinDVSO options")

        # PATHS
        # self.parser.add_argument("--data_path",
        #                          type=str,
        #                          help="path to the training data",
        #                          default=os.path.join(file_dir, "kitti_data"))
        # self.parser.add_argument("--log_dir",
        #                          type=str,
        #                          help="log directory",
        #                          default=os.path.join(os.path.expanduser("~"), "tmp"))
        
        
        # TRAINING options
        self.parser.add_argument("--exp",
                                 type=str,
                                 help="the name of the folder to save the model in")
        # self.parser.add_argument("--model_name",
        #                          type=str,
        #                          help="the name of the folder to save the model in",
        #                          required=True,
        #                          choices=["Gen_Baseline", "PTNet_Baseline", "Joint"])
        self.parser.add_argument("--num_layers_T",
                                 type=int,
                                 help="number of resnet layers for task net",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--num_layers_G",
                                 type=int,
                                 help="number of resnet layers for generator net",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        
        self.parser.add_argument("--stereo_mode",
                                 type=str,
                                 default="none",
                                 choices=["none", "monodepth2", "sharinGAN"])
        
        self.parser.add_argument("--netG_mode",
                                 type=str,
                                 default="monodepth2",
                                 choices=["monodepth2", "sharinGAN"])

        self.parser.add_argument("--pretrained_model_G",
                                 type=str,
                                 default=None,
                                 help="path of pretrained generator model")

        self.parser.add_argument("--pretrained_model_T",
                                 type=str,
                                 default=None,
                                 help="path of pretrained task model")
        
        self.parser.add_argument("--resume",
                                 default=None,
                                 help="The resumed joint model to load")
        
        
        # NOTE: OPTIONS for loss weights
        self.parser.add_argument("--temp_reproj_weight", 
                                 type=float,
                                 default=1.0)
        
        self.parser.add_argument("--stereo_gc_weight",
                                 type=float,
                                 default=1.0)
        
        self.parser.add_argument("--recon_loss_weight",
                                 type=float,
                                 default=10,
                                 help="for both real and syn")
        
        self.parser.add_argument("--gt_depth_weight", 
                                 type=float,
                                 default=1.0,
                                 help="loss weight for ground-truth depth")

        self.parser.add_argument("--lr_loss_weight",
                                 type=float,
                                 help="used only when --predict_right_disp",
                                 default=1.0)
        
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-1)
        
        self.parser.add_argument("--add_sharinGAN_gc",
                                 help="if set will disable sharinGAN gc for monodepth2 loss style",
                                 action="store_true")

        # NOTE: PARAMS for joint model training
        self.parser.add_argument("--correct_D_loss",
                                 help="if set will use torch.abs() for D_syn - D_real",
                                 action="store_true")
        
        self.parser.add_argument("--D_multi_scale",
                                 help="if set will use multi scale losses for D",
                                 action="store_true")
                                 
        
        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size: (1) 16 for PTNet_Baseline (2) 6 for Gen_Baseline (3) 2 for Joint",
                                 required=True)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        self.parser.add_argument("--top_k_val",
                                 type=int,
                                 help="number of models to save based on val_loss",
                                 default=3)
        self.parser.add_argument("--total_iterations",
                                 type=int,
                                 help="total_iterations",
                                 default=200000)
        self.parser.add_argument("--ngpus",
                                 type=int,
                                 help="number of gpus to use",
                                 default=1)
        
        self.parser.add_argument("--vbaseline",
                                 type=float,
                                 default=0.532725,
                                 help="baseline of virtual kitti")
        
        self.parser.add_argument("--predict_right_disp",
                                 help="if set predict the right disparity and use lr consistency loss",
                                 action="store_true")
        
        self.parser.add_argument("--direct_raw_img",
                                 help="if set will use raw img for direct photometric loss, otherwise use the shared feat",
                                 action="store_true")
        
        
        
        
        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=8)
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--hpc",
                                 help="disable tqdm in HPC",
                                 action="store_true")
        self.parser.add_argument("--preload_virtual_data",
                                 help="preload depth and image data of virtual dataset into memory",
                                 action="store_true")
        self.parser.add_argument('--dist-url', 
                                 default='env://', 
                                 type=str, 
                                 help='url used to set up distributed training')
        self.parser.add_argument('--dist-backend', 
                                 default='nccl', 
                                 type=str,
                                 help='distributed backend')
        
        

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])
        
        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)
        
        # EVALUATION options
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")
        self.parser.add_argument("--val",
                                 help="if set will use val.txt otherwise test.txt",
                                 action="store_true")
        self.parser.add_argument("--val_average_disp",
                                 help="if set will average disp prediction from all scales",
                                 action="store_true")
        self.parser.add_argument("--val_depth_mode",
                                 type=str,
                                 default="disp",
                                 choices=["disp", "normalized_depth"])
        self.parser.add_argument("--val_iter", 
                                  default=None,
                                  type=int)
        self.parser.add_argument("--val_seq", 
                                  default=None,
                                  type=str)
        
        self.parser.add_argument("--save_feat",
                                 help="only used in save_depth.py, will only save features",
                                 action="store_true")
        
        
        # DVSO FINETUNE options
        self.parser.add_argument("--dvso_epochs",
                                 type=int,
                                 default=5,
                                 help="dvso will be run at the beginning of each epoch")
        self.parser.add_argument("--dvso_train_seqs",
                                 type=str,
                                 nargs="+",
                                 default=["00", "01", "02", "03", "04", "05", "06", "07", "08"],
                                 help="e.g. 00 01 02")
        self.parser.add_argument("--dvso_test_seqs",
                                 type=str,
                                 nargs="+",
                                 default=["09", "10"],
                                 help="e.g. 09 10")
        self.parser.add_argument("--dvso_resume_exp",
                                 type=str,
                                 default=None,
                                 help="domain adaptation experiment name to resume, e.g. local-0614a")
        self.parser.add_argument("--dvso_resume_iter",
                                 type=int, 
                                 default=None, 
                                 help="the iteration model of the resumed, e.g. 125999")
        self.parser.add_argument("--dvso_param",
                                 type=str, 
                                 default="Before_Ori_Hit_1_2_2_1_1_1", 
                                 help="wStereoPosFlag wCorrectedFlag wGradFlag wStereo scaleENergyLeftTHR scaleWJI2SumTHR warpright checkWarpValid maskWarpGrad")
        self.parser.add_argument("--dvso_home_path",
                                 type=str,
                                 default="/home/szha2609",
                                 help="used in DVSO_Finetune/run_dvso.sh")

        self.parser.add_argument("--use_dvso_depth",
                                 help="use dvso sparse depth for supervision",
                                 action="store_true")
        self.parser.add_argument("--dvso_depth_weight",
                                 type=float,
                                 default=1.0,
                                 help="loss weight for dvso depth")

        self.parser.add_argument("--dvso_netT_only",
                                 help="only finetune netT during ",
                                 action="store_true")
        self.parser.add_argument("--dvso_real_only",
                                 help="only use real dataset during dvso finetuning. Only take effect for --dvso_netT_only",
                                 action="store_true")

        self.parser.add_argument("--dvso_maintain_disp",
                                 help="if set, will maintain the intermediate disps to save space, otherwise will clean them",
                                 action="store_true")
        self.parser.add_argument("--dvso_disp_exist",
                                 help="if set, will use the previously maintained disp rather than recomputing them",
                                 action="store_true")
        self.parser.add_argument("--dvso_test_only",
                                 help="only test the loaded model and then quit",
                                 action="store_true")

        self.parser.add_argument("--use_pose_loss",
                                 help="use L2 loss for dvso axisangle and translation vec6 poses",
                                 action="store_true")
        self.parser.add_argument("--rot_weight",
                                 type=float,
                                 default=100.,
                                 help="weight for rotation loss")
        self.parser.add_argument("--trans_weight",
                                 type=float,
                                 default=1.,
                                 help="weight for translation loss")
        
        self.parser.add_argument("--use_dvso_photo",
                                 help="use dvso poses for temporal photometric consistency loss",
                                 action="store_true")
        self.parser.add_argument("--dvso_photo_weight",
                                 type=float,
                                 default=1.)

        self.parser.add_argument("--dvso_netT_lr",
                                 type=float,
                                 default=1e-5)

        # CORRECT TRAIN/VAL/TEST split
        self.parser.add_argument("--kitti_folder",
                                 type=str,
                                 default="Kitti-Zhan",
                                 help="the folder name under dataset_files")

        # FOR TESTING BEST DVSO FINETUNE MODELS
        self.parser.add_argument("--dvso_test_best",
                                 help="only test the loaded model, --dvso_test_exp, --dvso_test_iter, --dvso_test_T, and use --dvso_test_seqs",
                                 action="store_true")
        self.parser.add_argument("--dvso_test_T",
                                 help="test pretrain-T models, where netT use raw images rather than netG results",
                                 action="store_true")
        self.parser.add_argument("--dvso_test_exp",
                                 type=str,
                                 default=None)
        self.parser.add_argument("--dvso_test_iter",
                                 type=int,
                                 default=None)
        self.parser.add_argument("--dvso_test_run",
                                 type=int,
                                 default=5,
                                 help="run 5 times of DVSO and take the median")

        
        
    def parse(self):
        self.options = self.parser.parse_args()
        
        return self.options
        
        
        