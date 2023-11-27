import torch
from torch import nn
from torch.nn import functional as F
from utils.mano import MANO
from .common_utils import rot6d2mat, batch_rodrigues, mat2aa

mano = MANO()




class mano_regHead(nn.Module):
    def __init__(self, mano_layer=mano.layer, feature_size=1024, mano_neurons=[1024, 512]):
        super(mano_regHead, self).__init__()

        # 6D representation of rotation matrix
        self.pose6d_size = 16 * 6
        self.mano_pose_size = 16 * 3

        # Base Regression Layers
        mano_base_neurons = [feature_size] + mano_neurons
        base_layers = []
        for layer_idx, (inp_neurons, out_neurons) in enumerate(
                zip(mano_base_neurons[:-1], mano_base_neurons[1:])):
            base_layers.append(nn.Linear(inp_neurons, out_neurons))
            base_layers.append(nn.LeakyReLU(inplace=True))
        self.mano_base_layer = nn.Sequential(*base_layers)
        # Pose layers
        self.pose_reg = nn.Linear(mano_base_neurons[-1], self.pose6d_size)
        # Shape layers
        self.shape_reg = nn.Linear(mano_base_neurons[-1], 10)

        self.mano_layer = mano_layer

    def forward(self, features, gt_mano_params=None):
        mano_features = self.mano_base_layer(features)
        pred_mano_pose_6d = self.pose_reg(mano_features)
        
        pred_mano_pose_rotmat = rot6d2mat(pred_mano_pose_6d.view(-1, 6)).view(-1, 16, 3, 3).contiguous()
        pred_mano_shape = self.shape_reg(mano_features)
        pred_mano_pose = mat2aa(pred_mano_pose_rotmat.view(-1, 3, 3)).contiguous().view(-1, self.mano_pose_size)
        pred_verts, pred_joints = self.mano_layer(th_pose_coeffs=pred_mano_pose, th_betas=pred_mano_shape)

        pred_verts /= 1000
        pred_joints /= 1000

        pred_mano_results = {
            "verts3d": pred_verts,
            "joints3d": pred_joints,
            "mano_shape": pred_mano_shape,
            "mano_pose": pred_mano_pose_rotmat,
            "mano_pose_aa": pred_mano_pose}

        if gt_mano_params is not None:
            gt_mano_shape = gt_mano_params[:, self.mano_pose_size:]
            gt_mano_pose = gt_mano_params[:, :self.mano_pose_size].contiguous()
            gt_mano_pose_rotmat = batch_rodrigues(gt_mano_pose.view(-1, 3)).view(-1, 16, 3, 3)
            gt_verts, gt_joints = self.mano_layer(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)

            gt_verts /= 1000
            gt_joints /= 1000

            gt_mano_results = {
                "verts3d": gt_verts,
                "joints3d": gt_joints,
                "mano_shape": gt_mano_shape,
                "mano_pose": gt_mano_pose_rotmat}
        else:
            gt_mano_results = None

        return pred_mano_results, gt_mano_results
