'''
AIR-Nets
Author: Simon Giebenhain
Code: https://github.com/SimonGiebenhain/AIR-Nets
'''

import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from time import time
import torch.nn.functional as F
import os
import math
import nets.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils as pointnet2_utils

def square_distance(src, dst):
    """
    Code from: https://github.com/qq456cvb/Point-Transformers/blob/master/pointnet_util.py

    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def index_points(points, idx):
    """
    Code from: https://github.com/qq456cvb/Point-Transformers/blob/master/pointnet_util.py
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

class PointNetSetAbstraction(nn.Module):
    """
    Set Abstraction Module, as used in PointNet++
    Uses FPS for downsampling, kNN groupings and maxpooling to abstract the group/neighborhood

    Attributes:
        npoint (int): Output cardinality
        nneigh (int): Size of local grouings/neighborhoods
        in_channel (int): input dimensionality
        dim (int): internal and output dimensionality
    """
    def __init__(self, npoint, nneigh, in_channel, dim):
        super(PointNetSetAbstraction, self).__init__()

        self.npoint = npoint
        self.nneigh = nneigh
        self.fc1 = nn.Linear(in_channel, dim)
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)

        self.bn = nn.BatchNorm1d(dim)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """

        with torch.no_grad():
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint).long()

        new_xyz = index_points(xyz, fps_idx)
        points = self.fc1(points)
        points_ori = index_points(points, fps_idx)

        points = points.permute(0, 2, 1)
        points = points + F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(points))))))
        points = points.permute(0, 2, 1)

        with torch.no_grad():
            dists = square_distance(new_xyz, xyz)  # B x npoint x N
            idx = dists.argsort()[:, :, :self.nneigh]  # B x npoint x K


        grouped_points = index_points(points, idx)


        new_points = points_ori + torch.max(grouped_points, 2)[0]
        new_points = self.bn(new_points.permute(0, 2, 1)).permute(0, 2, 1)
        return new_xyz, new_points

class TransitionDown(nn.Module):
    """
        Attributes:
            npoint: desired number of points for outpout point cloud
            nneigh: size of neighborhood
            dim: number of dimensions of input and interal dimensions
        """
    def __init__(self, npoint, nneighbor, dim) -> None:
        super().__init__()
        self.sa = PointNetSetAbstraction(npoint, nneighbor, dim, dim)

    def forward(self, xyz, feats):
        """
        Executes the downsampling (set abstraction)
        :param xyz: positions of points
        :param feats: features of points
        :return: downsampled version, tuple of (xyz_new, feats_new)
        """
        ret = self.sa(xyz, feats)
        return ret


class TransformerBlock(nn.Module):
    """
    Module for local and global vector self attention, as proposed in the Point Transformer paper.

    Attributes:
        d_model (int): number of input, output and internal dimensions
        k (int): number of points among which local attention is calculated
        pos_only (bool): When set to True only positional features are used
        group_all (bool): When true full instead of local attention is calculated
    """
    def __init__(self, d_model, k, pos_only=False, group_all=False) -> None:
        super().__init__()

        self.pos_only = pos_only

        self.bn = nn.BatchNorm1d(d_model)

        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        self.group_all = group_all

    def forward(self, xyz, feats=None):
        """
        :param xyz [b x n x 3]: positions in point cloud
        :param feats [b x n x d]: features in point cloud
        :return:
            new_features [b x n x d]:
        """

        with torch.no_grad():
            # full attention
            if self.group_all:
                b, n, _ = xyz.shape
                knn_idx = torch.arange(n, device=xyz.device).unsqueeze(0).unsqueeze(1).repeat(b, n, 1)
            # local attention using KNN
            else:
                dists = square_distance(xyz, xyz)
                knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k

        knn_xyz = index_points(xyz, knn_idx)

        if not self.pos_only:
            ori_feats = feats
            x = feats

            q_attn = self.w_qs(x)
            k_attn = index_points(self.w_ks(x), knn_idx)
            v_attn = index_points(self.w_vs(x), knn_idx)

        pos_encode = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x d

        if not self.pos_only:
            attn = self.fc_gamma(q_attn[:, :, None] - k_attn + pos_encode)
        else:
            attn = self.fc_gamma(pos_encode)


        attn = functional.softmax(attn, dim=-2)  # b x n x k x d
        if not self.pos_only:
            res = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode)
        else:
            res = torch.einsum('bmnf,bmnf->bmf', attn, pos_encode)



        if not self.pos_only:
            res = res + ori_feats
        res = self.bn(res.permute(0, 2, 1)).permute(0, 2, 1)

        return res


class ElementwiseMLP(nn.Module):
    """
    Simple MLP, consisting of two linear layers, a skip connection and batch norm.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        """
        :param x: [B x n x d]
        :return: [B x n x d]
        """
        x = x.permute(0, 2, 1)
        return self.bn(x + F.relu((self.conv2(F.relu((self.conv1(x))))))).permute(0, 2, 1)



