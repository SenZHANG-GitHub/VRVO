import pdb
import networks 
import torch
import torch.nn as nn
from networks.layers import transformation_from_parameters


class netT(nn.Module):
    """Pytorch module for the depth (/ pose) prediction network
    """
    def __init__(self, num_layers, scales, num_pose_frames, frame_ids, use_pose, predict_right_disp):
        super(netT, self).__init__()
        
        self.frame_ids = frame_ids
        self.num_pose_frames = num_pose_frames
        self.predict_right_disp = predict_right_disp

        # pretrained=True (on ImageNet)
        self.encoder = networks.ResnetEncoder(
            num_layers, True) 
        
        # if predict_right_disp is True, predict both left/right disps
        num_output_channels = 2 if self.predict_right_disp else 1
        self.depth = networks.DepthDecoder(
            self.encoder.num_ch_enc, scales, num_output_channels)
        
        self.use_pose = use_pose
        if self.use_pose:
            self.pose_encoder = networks.ResnetEncoder(
                num_layers,
                True,
                num_input_images=num_pose_frames)
            
            self.pose = networks.PoseDecoder(
                self.pose_encoder.num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)
            
    
    def to_device(self, device):
        """Put the models to gpu
        """
        self.encoder.cuda(device)
        self.depth.cuda(device)
        if self.use_pose:
            self.pose_encoder.cuda(device)
            self.pose.cuda(device)
    
    
    def get_parameters(self):
        """Get the parameters to train
        """
        parameters_to_train = []
        parameters_to_train += list(self.encoder.parameters())
        parameters_to_train += list(self.depth.parameters())
        if self.use_pose:
            parameters_to_train += list(self.pose_encoder.parameters())
            parameters_to_train += list(self.pose.parameters())
        return parameters_to_train
    
    
    def forward(self, inputs, just_depth=False):
        """Follow monodepth2 practice while adding monodepth's right disp prediction
        => inputs["color_aug", 0, 0] is used for encoder and depth
        => [inputs["color_aug", f_i, 0] for f_i in self.frame_ids] is used for pose
        => output: ("disp", scale) for scale in [0, 1, 2, 3], ("cam_T_cam", 0, f_i)
        """
        # NOTE: we only feed the image with frame_id 0 through the depth encoder
        if just_depth:
            # under this setting: inputs is just the image
            features = self.encoder(inputs)
            outputs = self.depth(features)
            return outputs
            
        features = self.encoder(inputs["color_aug", 0, 0])
        outputs = self.depth(features)
        if self.use_pose:
            outputs.update(self.predict_poses(inputs, features))
        return outputs
    
    
    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        # self.num_pose_frames = 2 if opt.pose_model_input == "paired"
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_ids}

            for f_i in self.frame_ids[1:]:
                assert f_i != "s"
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.pose_encoder(torch.cat(pose_inputs, 1))]
                
                
                # NOTE: (batch, num_frames_to_predict_for=2, 1, 3) for axis and trans
                # => axis[:, 1] and trans[:, 1] are actually not used. Redundency here
                axisangle, translation = self.pose(pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                # axisangle[:, 0] -> (batch, 1, 3)
                # translation[:, ] -> (batch, 1, 3)
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        else:
            raise NotImplementedError()
        
        return outputs
    
    
    
    
class netG(nn.Module):
    """Pytorch module for the image auto-encoder (shared space generator)
    """
    def __init__(self, num_layers, scales, frame_ids):
        super(netG, self).__init__()
        
        self.frame_ids = frame_ids

        # pretrained=True (on ImageNet)
        self.encoder = networks.ResnetEncoder(
            num_layers, True) 
        
        num_output_channels = 3 
        
        # NOTE: GenDecoder last layer: Tanh(Conv3x3()) -> outputs[("gen", scale)]
        self.decoder = networks.GenDecoder(
            self.encoder.num_ch_enc, scales, num_output_channels)
        
    
    def to_device(self, device):
        """Put the models to gpu
        """
        self.encoder.cuda(device)
        self.decoder.cuda(device)
    
    
    def get_parameters(self):
        """Get the parameters to train
        """
        parameters_to_train = []
        parameters_to_train += list(self.encoder.parameters())
        parameters_to_train += list(self.decoder.parameters())
        return parameters_to_train
    
    
    def forward(self, input_img):
        """Follow monodepth2 practice while adding monodepth's right disp prediction
        => outputs: ("gen", scale) for scale in [0, 1, 2, 3]
        """
        # NOTE: we only feed the image with frame_id 0 through the depth encoder
        features = self.encoder(input_img)
        outputs = self.decoder(features)
        return features, outputs
    
    
    
