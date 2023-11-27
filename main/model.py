import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.backbone import FPN
from nets.transformer import Transformer
from nets.regressor import Regressor
from nets.pnvae import PointNetPlusPlusEncoder, Decoder, Normal
from utils.mano import MANO
from nets.common_utils import get_groundtruth
from config import cfg
import sys
sys.path.append("..")
from models.new_diffusion import *
import math

def sample_joint_features(img_feat, joint_xy):
    height, width = img_feat.shape[2:]
    x = 2 * joint_xy[:,:,0] - 1
    y = 2 * joint_xy[:,:,1] - 1
    grid = torch.stack((x,y),2)[:,:,None,:]
    img_feat = F.grid_sample(img_feat, grid, align_corners=True)[:,:,:,0] # batch_size, channel_dim, joint_num
    img_feat = img_feat.permute(0,2,1).contiguous() # batch_size, joint_num, channel_dim
    return img_feat


class Model(nn.Module):
    def __init__(self, backbone, FIT, SET, regressor, enc, dec, diffusion):
        super(Model, self).__init__()
        self.backbone = backbone
        self.FIT = FIT
        self.SET = SET
        self.regressor = regressor
        self.enc = enc
        self.dec = dec
        self.diffusion = diffusion
        if cfg.stage == 'vae':
            self.trainable_modules = [self.enc, self.dec]
        elif cfg.stage == 'diffusion':
            #self.trainable_modules = [self.diffusion]
            self.trainable_modules = [self.SET, self.regressor, self.diffusion]
    
    def forward(self, inputs, targets, meta_info, mode):

        if mode == 'train':
            gt_mano_params = torch.cat([targets['mano_pose'], targets['mano_shape']], dim=1)
            gt_mano_params = gt_mano_params.cuda()
            gt_mano_results = get_groundtruth(gt_mano_params)
            gt_mano_results = gt_mano_results
        else:
            gt_mano_params = None
            gt_mano_results = None

        if cfg.stage == 'diffusion':
            p_feats, s_feats = self.backbone(inputs['img'])  # primary, secondary feats
            feats = self.FIT(s_feats, p_feats)
            feats = self.SET(feats, feats)
            condition, _, preds_joints_img, heatmap  = self.regressor(feats, gt_mano_params)
            if mode == 'train':
                joints = gt_mano_results['joints3d']
            else:
                joints = condition['joints3d']
            #feats = heatmap
            if mode == 'train':
                feats = sample_joint_features(feats, targets['joints_img'])
            else:
                feats = sample_joint_features(feats, preds_joints_img[0])

            heatmap = heatmap.reshape([heatmap.shape[0],256,-1])
            heatmap = heatmap.permute(0,2,1).contiguous()
            feats = torch.cat([feats, heatmap], dim=1)
            #feats = feats.reshape([feats.shape[0],256,-1])
            #feats = feats.transpose(1,2)

        # elif cfg.stage == 'vae':

       
        if mode == 'train':
            # x = torch.cat((gt_mano_results['verts3d']), dim=1)
            mu, logvar = self.enc(gt_mano_results['verts3d'])
            dist = Normal(mu=mu, log_sigma=logvar)  # (B, F)
            feature = dist.sample()[0]
            # std = torch.exp(0.5 * logvar)
            # eps = torch.randn_like(std)
            # feature = eps * std + mu
            # loss functions
            loss = {}
            if cfg.stage == 'vae':
                pred_mano_results = self.dec(feature)
                loss['mano_verts'] = cfg.lambda_mano_verts * F.mse_loss(pred_mano_results['verts3d'], gt_mano_results['verts3d'])
                loss['mano_joints'] = cfg.lambda_mano_joints * F.mse_loss(pred_mano_results['joints3d'], gt_mano_results['joints3d'])
                loss['mano_pose'] = cfg.lambda_mano_pose * F.mse_loss(pred_mano_results['mano_pose'], gt_mano_results['mano_pose'])
                loss['mano_shape'] = cfg.lambda_mano_shape * F.mse_loss(pred_mano_results['mano_shape'], gt_mano_results['mano_shape'])
                #todo: 调整这个0.5
                loss['kl'] = 0.5 * torch.mean(torch.exp(logvar) + mu ** 2 - 1.0 - logvar)
                #loss['joints_img'] = cfg.lambda_joints_img * F.mse_loss(preds_joints_img[0], targets['joints_img'])
            elif cfg.stage == 'diffusion':
                ####################训练请注释掉中间这几行######################
                #batch = feats.shape[0]
                #dim = (batch, 256, 128)
                #input_feature = torch.randn(dim).cuda()
                #latent_feature = self.diffusion(input_feature, feats, joints, is_train=False)
                #pred_mano_results = self.dec(latent_feature)
                ####################训练请注释掉中间这几行######################
                loss['diffusion'], pred_feature = self.diffusion(feature, feats, joints, is_train=True)
                pred_mano_results = self.dec(pred_feature)
                loss['mano_verts'] = cfg.lambda_mano_verts * F.mse_loss(pred_mano_results['verts3d'], gt_mano_results['verts3d'])
                loss['mano_joints'] = cfg.lambda_mano_joints * F.mse_loss(pred_mano_results['joints3d'], gt_mano_results['joints3d'])
                loss['mano_pose'] = cfg.lambda_mano_pose * F.mse_loss(pred_mano_results['mano_pose'], gt_mano_results['mano_pose'])
                loss['mano_shape'] = cfg.lambda_mano_shape * F.mse_loss(pred_mano_results['mano_shape'], gt_mano_results['mano_shape'])
                ###########loss for handoccnet############
                loss['handocc_verts'] = cfg.lambda_mano_verts * F.mse_loss(condition['verts3d'], gt_mano_results['verts3d'])
                loss['handocc_joints'] = cfg.lambda_mano_joints * F.mse_loss(condition['joints3d'], gt_mano_results['joints3d'])
                loss['handocc_pose'] = cfg.lambda_mano_pose * F.mse_loss(condition['mano_pose'], gt_mano_results['mano_pose'])
                loss['handocc_shape'] = cfg.lambda_mano_shape * F.mse_loss(condition['mano_shape'], gt_mano_results['mano_shape'])
                loss['joints_img'] = cfg.lambda_joints_img * F.mse_loss(preds_joints_img[0], targets['joints_img'])

            out = {}
            out['joints_coord_cam'] = pred_mano_results['joints3d']
            out['mesh_coord_cam'] = pred_mano_results['verts3d']
            return loss
            #return out

        else:
            # test output
            if cfg.stage == 'diffusion':
                batch = feats.shape[0]
                dim = (batch, 256, 128)
                input_feature = torch.randn(dim).cuda()
                latent_feature = self.diffusion(input_feature, feats, joints, is_train=False)
                pred_mano_results = self.dec(latent_feature)

            out = {}
            out['joints_coord_cam'] = pred_mano_results['joints3d']
            out['mesh_coord_cam'] = pred_mano_results['verts3d']
            return out

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

def init_weights(m):
    print(type(m))
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        #nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        #nn.init.constant_(m.bias,0)

def set_parameter_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = requires_grad

def get_model(mode):

    ###以下这一组模型的参数永远都不更新
    backbone = FPN(pretrained=True)
    FIT = Transformer(injection=True) # feature injecting transformer
    SET = Transformer(injection=False) # self enhancing transformer
    regressor = Regressor()
    set_parameter_requires_grad(backbone, requires_grad=False)
    set_parameter_requires_grad(FIT, requires_grad=False)
    set_parameter_requires_grad(SET, requires_grad=True)
    set_parameter_requires_grad(regressor, requires_grad=True)
    #####以下这一组模型的参数在训练vae时更新，在训练diffusion时不更新
    enc = PointNetPlusPlusEncoder()
    dec = Decoder(latent_dim=128)

    if mode == 'train':
        # FIT.apply(init_weights)
        # SET.apply(init_weights)
        # regressor.apply(init_weights)
        backbone = load_ckpt(cfg.ckpt, backbone, 'backbone')
        FIT = load_ckpt(cfg.ckpt, FIT, 'FIT')
        SET = load_ckpt(cfg.ckpt, SET, 'SET')
        regressor = load_ckpt(cfg.ckpt, regressor, 'regressor')
        if cfg.stage == 'vae':
            enc.apply(init_weights)
            dec.apply(init_weights)
            diffusion = torch.nn.Identity()
        elif cfg.stage == 'diffusion':
            #todo: freeze the weights of vae,
            enc = load_ckpt(cfg.ckpt, enc, 'enc')
            dec = load_ckpt(cfg.ckpt, dec, 'dec')
            set_parameter_requires_grad(enc, requires_grad=False)
            set_parameter_requires_grad(dec, requires_grad=False)
            #todo: init weights for diffusion
            diffusion = D3DP()
            #diffusion = DDPM()
            pass
    else:
        diffusion = D3DP()
        #diffusion = DDPM()
        
    model = Model(backbone, FIT, SET, regressor, enc, dec, diffusion)
    
    return model

def load_ckpt(path, model, string):
    ckpt_path = path
    ckpt = torch.load(ckpt_path)
    m_ckpt = model.state_dict()
    # module_name = str(model)
    module_name = string
    for k in m_ckpt.keys():
        if 'module.'+module_name+'.'+k not in ckpt['network'].keys():
            print('ERROR')
            print(k)
        else:
            new_key = 'module.'+module_name+'.'+k
            state_dict = {k: ckpt['network'][new_key]}
            m_ckpt.update(state_dict)

    model.load_state_dict(m_ckpt)

    return model

