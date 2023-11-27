# adopt from:
# - VQVAE: https://github.com/nadavbh12/VQ-VAE
# - Encoder: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py

from __future__ import print_function

import torch
import torch.utils.data
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from einops import rearrange

from .vqvae_modules import DGCNN_encoder, Decoder
from .quantizer import VectorQuantizer


def init_weights(net, init_type='normal', gain=0.01):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

    # propagate to children
    for m in net.children():
        m.apply(init_func)


class VQVAE(nn.Module):
    def __init__(self,
                 n_embed=512,
                 embed_dim=4,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super(VQVAE, self).__init__()

        self.n_embed = n_embed
        self.embed_dim = embed_dim

        self.encoder = DGCNN_encoder(k=20, emb_dims=n_embe)
        self.decoder = Decoder(latent_dim=n_embed)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=1.0,
                                        remap=remap, sane_index_shape=sane_index_shape, legacy=False)
        ########看一下下面这两行的维度
        #self.quant_conv = torch.nn.Conv2d(3, embed_dim, 1)
        #self.post_quant_conv = torch.nn.Conv2d(embed_dim, 3, 1)
        #todo: change first dimension
        # quantize后的输出：[bs, 512]
        #self.quant_conv = torch.nn.Conv1d(3, embed_dim, 1)
        #self.post_quant_conv = torch.nn.Conv1d(embed_dim, 3, 1)
        self.quant_conv = torch.nn.Linear(1, self.embed_dim)
        self.post_quant_conv = torch.nn.Linear(self.embed_dim, 1)

        init_weights(self.encoder, 'normal', 0.02)
        init_weights(self.decoder, 'normal', 0.02)
        init_weights(self.quant_conv, 'normal', 0.02)
        init_weights(self.post_quant_conv, 'normal', 0.02)

    def encode(self, x):
        h = self.encoder(x)    # (bs,512)
        # todo: 给h增加一维度，上一行后x为[bs,latent_dim],下面之前应该变为[bs, latent_dim, 1]
        h = h.unsqueeze(-1)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, is_voxel=False)
        return quant, emb_loss, info

    def encode_no_quant(self, x):
        h = self.encoder(x)
        h = h.unsqueeze(-1)
        h = self.quant_conv(h)
        # quant, emb_loss, info = self.quantize(h, is_voxel=True)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        quant = quant.squeeze(-1)
        dec = self.decoder(quant)
        return dec

    def decode_no_quant(self, h):
        # also go through quantization layer
        quant, emb_loss, info = self.quantize(h, is_voxel=True)
        quant = self.post_quant_conv(quant)
        quant = quant.squeeze(-1)
        dec = self.decoder(quant)
        return dec

    def forward(self, input, verbose=False, forward_no_quant=False, encode_only=False):

        if forward_no_quant:
            # for diffusion model's training
            z = self.encode_no_quant(input)
            if encode_only:
                return z

            dec = self.decode_no_quant(z)
            return dec, z

        quant, diff, info = self.encode(input)
        dec = self.decode(quant)

        if verbose:
            return dec, quant, diff, info
        else:
            return dec, diff