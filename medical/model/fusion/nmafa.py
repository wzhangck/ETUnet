import torch.nn as nn
from einops import rearrange
from .multi_spatial_att import MultiAttentionFusion

class ESCA(nn.Module):
    def __init__(self, model_num,
                 in_channels,
                 hidden_size,
                 img_size,
                 mlp_size=256,
                 self_num_layer=4,
                 window_size=(4, 4, 4),
                 ):

        '''
        model_num=model_num,# 4
        in_channels=fea[4],# 128
        hidden_size=fea[4],# 128
        img_size=new_image_size,# [8,8,8]
        mlp_size=2*fea[4],# 256
        self_num_layer=self_num_layer,# 2
        window_size=window_size,# (4, 4, 4)
        token_mixer_size=token_mixer_size,# 32
        token_learner=token_learner# True
        '''

        super().__init__()
        self.img_size = img_size
        self.hidden_size = hidden_size

        self.multi_att = MultiAttentionFusion(in_channels=model_num*in_channels,
                                              hidden_size=hidden_size,
                                              img_size=img_size,
                                              mlp_size=mlp_size,
                                              num_layers=self_num_layer,
                                              window_size=window_size,
                                              )

    def forward(self, x):
        # x: (batch, model_num, hidden_size, d, w, h)

        q = rearrange(x, "b m f d w h -> b (m f) d w h")

        q = self.multi_att(q)

        return q
