import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from config import cfg

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from utils.mano import MANO
from .common_utils import rot6d2mat, batch_rodrigues, mat2aa
from .pnblock import TransitionDown, TransformerBlock, ElementwiseMLP
mano = MANO()

def pc_normalize(pc):
    centroid = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=2, keepdim=True)))
    pc = pc / m
    return pc

@torch.jit.script
def sample_normal_jit(mu, sigma):
    rho = mu.mul(0).normal_()
    z = rho.mul_(sigma).add_(mu)
    return z, rho

class Normal:
    def __init__(self, mu, log_sigma, sigma=None):
        self.mu = mu
        self.log_sigma = log_sigma
        self.sigma = torch.exp(log_sigma) if sigma is None else sigma

    def sample(self, t=1.):
        return sample_normal_jit(self.mu, self.sigma * t)

    def sample_given_rho(self, rho):
        return rho * self.sigma + self.mu

    def mean(self):
        return self.mu

    def log_p(self, samples):
        normalized_samples = (samples - self.mu) / self.sigma
        log_p = - 0.5 * normalized_samples * normalized_samples - 0.5 * np.log(2 * np.pi) - self.log_sigma
        return log_p


class PointNetPlusPlusEncoder(nn.Module):
    """
        PointNet++-style encoder.

        Attributes:
            npoints_per_layer [int]: cardinality of point cloud for each layer
            nneighbor int: number of neighbors for set abstraction
            d_transformer int: internal dimensions

        Default values:
            npoints_per_layer: [778, 389, 256]
            nneighbor: 16
            d_transformer: 512
            nfinal_transformers: 2
        """
    def __init__(self, npoints_per_layer=[778, 512, 389, 256], nneighbor=16, d_transformer=128, nfinal_transformers=1):
        super(PointNetPlusPlusEncoder, self).__init__()
        self.fc_begin = nn.Sequential(
            nn.Linear(3, d_transformer),
            nn.ReLU(),
            nn.Linear(d_transformer, d_transformer)
        )
        self.downsampling = nn.ModuleList()
        self.elementwise = nn.ModuleList()

        for i in range(len(npoints_per_layer) - 1):
            old_npoints = npoints_per_layer[i]
            new_npoints = npoints_per_layer[i + 1]
            self.downsampling.append(
                TransitionDown(new_npoints, min(nneighbor, old_npoints), d_transformer)
            )
            self.elementwise.append(ElementwiseMLP(d_transformer))

        # full self attention layers
        self.final_transformers = nn.ModuleList()
        self.final_elementwise = nn.ModuleList()

        for i in range(nfinal_transformers):
            self.final_transformers.append(
                TransformerBlock(d_transformer, -1, group_all=True)
            )
        for i in range(nfinal_transformers):
            self.final_elementwise.append(
                ElementwiseMLP(dim=d_transformer)
            )
        self.fc2_mean = nn.Linear(d_transformer, d_transformer)
        self.fc2_logvar = nn.Linear(d_transformer, d_transformer)


    def forward(self, x):
        """
        :param x [B x n x 3] (or [B x n x 3]): input point cloud
        :return:
                 mean [B x npoints_per_layer[-1] x d_transformer] like [B, 256, 512]
                 logvar [B x npoints_per_layer[-1] x d_transformer] like [B, 256, 512]
        """
        x = x.contiguous()
        feats = self.fc_begin(x)
        feats = feats.contiguous()
        for i in range(len(self.downsampling)):
            x, feats = self.downsampling[i](x, feats)
            feats = self.elementwise[i](feats)

        for i, att_block in enumerate(self.final_transformers):
            feats = att_block(x, feats)
            feats = self.final_elementwise[i](feats) #local feats

        #global_feats = feats.max(dim=1)[0] #[B, 512]
        #global_feats = global_feats.repeat(1, 256, 1)
        #feats = torch.cat((feats, global_feats), 2)
        mean = self.fc2_mean(feats)
        logvar = self.fc2_logvar(feats) #[B, 256, 512]

        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        self.mano_layer = mano.layer
        self.pose6d_size = 16 * 6
        self.mano_pose_size = 16 * 3

        # Define the rest of the decoder, with output size matching the MANO parameters
        self.fc1 = nn.Linear(latent_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        # self.fc6 = nn.Linear(128, 48 + 3 + 10)  # 45 for hand pose (15 joints * 3), 3 for hand translation
        # Pose layers
        self.pose_reg = nn.Linear(128, self.pose6d_size)
        # Shape layers
        self.shape_reg = nn.Linear(128, 10)

    def forward(self, z):
        # 我们真的需要这么多bn吗？
        global_feats = z.max(dim=1)[0] #[B, 512]
        x = F.leaky_relu(self.bn1(self.fc1(global_feats)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.2)
        pred_mano_pose_6d = self.pose_reg(x)
        # desired pred_mano_pose_6d: [B, 96]
        pred_mano_pose_rotmat = rot6d2mat(pred_mano_pose_6d.view(-1, 6)).view(-1, 16, 3, 3).contiguous()
        pred_mano_shape = self.shape_reg(x)
        pred_mano_pose = mat2aa(pred_mano_pose_rotmat.view(-1, 3, 3)).contiguous().view(-1, self.mano_pose_size)
        pred_verts, pred_joints = self.mano_layer(th_pose_coeffs=pred_mano_pose, th_betas=pred_mano_shape)

        pred_verts /= 1000
        pred_joints /= 1000

        pred_mano_results = {
            "verts3d": pred_verts,
            "joints3d": pred_joints,
            "mano_shape": pred_mano_shape,
            "mano_pose": pred_mano_pose_rotmat}


        return pred_mano_results


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.encoder = PointNetPlusPlusEncoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar, trans = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, trans



