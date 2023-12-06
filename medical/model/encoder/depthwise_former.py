from typing import Sequence
import torch
import torch.nn as nn
from medical.model.fusion.layers import get_config
from medical.model.fusion.skip_connection import CrossAttn


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_size, padding_size):
        # (1,16,3,1,1)
        super(Convolution, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride_size, padding_size),
            LayerNormChannel(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv_1(x)
        return x


class TwoConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_size, padding_size):
        # (1,16,3,1,1)
        super(TwoConv, self).__init__()
        self.conv_1 = Convolution(in_channels, out_channels, kernel_size, stride_size, padding_size)
        self.conv_2 = Convolution(out_channels, out_channels, kernel_size, stride_size, padding_size)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
            self,
            in_chns: int,
            cat_chns: int,
            out_chns: int,
            img_size: tuple,
            token_mixer_size:int,
            pool_size=(2, 2, 2),
    ):

        super().__init__()

        up_chns = out_chns

        self.upsample = torch.nn.ConvTranspose3d(in_chns, up_chns, kernel_size=pool_size, stride=pool_size, padding=0)

        self.convs = TwoConv(cat_chns + up_chns, out_chns, 3, 1, 1)

        if img_size[0] == 128:
            self.cross_attn = None
        else:
            self.cross_attn = CrossAttn(in_channels=out_chns, hidden_size=out_chns, img_size=img_size,
                                        token_mixer=token_mixer_size)

    def forward(self, x: torch.Tensor, x_e: torch.Tensor):

        x_0 = self.upsample(x)

        if self.cross_attn is not None:
            x_0 = self.cross_attn(x_0, x_e) + x_0

        x = self.convs(torch.cat([x_0, x_e], dim=1))

        return x


class MlpChannel(nn.Module):
    """
    Implementation of MLP with 1*1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Conv3d(config.hidden_size, config.mlp_dim, 1)
        self.act = nn.GELU() # 非线性 GELU
        self.fc2 = nn.Conv3d(config.mlp_dim, config.hidden_size, 1)
        self.drop1 = nn.Dropout(config.dropout_rate)
        self.drop2 = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.drop2(x)
        return x

# LayerNorm3D
class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W, D]
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config
        in_channels = config.in_channels
        patch_size = config.patch_size

        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                          out_channels=config.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size,
                                          )

        self.norm = LayerNormChannel(num_channels=config.hidden_size)

    def forward(self, x):

        x = self.patch_embeddings(x)

        x = self.norm(x)

        return x


class DWBlock(nn.Module):

    def __init__(self, config, ratio = 2):
        super().__init__()
        self.dim = config.hidden_size
        self.window_size = 7
        self.out_dim = config.hidden_size*ratio

        self.conv0 = nn.Conv3d(self.dim, self.out_dim, kernel_size=1, stride=1, bias=False) # 升维

        self.ln1 = LayerNormChannel(self.out_dim)
        self.relu1 = nn.GELU() 

        self.conv1 = nn.Conv3d(self.out_dim, self.out_dim, kernel_size=self.window_size, stride=1, padding=3,
                               groups=self.out_dim)

        self.gn1 = nn.GroupNorm(self.out_dim, self.out_dim) 

        self.relu2 = nn.GELU()

        self.conv2 = nn.Conv3d(self.out_dim, self.dim, kernel_size=1, stride=1, bias=False) # 逐点卷积

        self.ln2 = LayerNormChannel(self.dim)

    def forward(self, x):

        x = self.conv0(x)
        x = self.ln1(x)
        x = self.relu1(x)

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu2(x)

        x = self.conv2(x)
        # x = self.ln2(x)

        return x


class DWConv(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNormChannel(config.hidden_size, eps=1e-6) # 3D LayerNorm
        self.ffn_norm = LayerNormChannel(config.hidden_size, eps=1e-6)
        self.ffn = MlpChannel(config)
        self.attn = DWBlock(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x) + x
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x

class DepthWiseformer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 img_size,
                 patch_size,
                 mlp_size,
                 num_layers,
                 pool_ratio,
                 ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.config = get_config(in_channels=in_channels,
                                 hidden_size=out_channels,
                                 patch_size=patch_size,
                                 mlp_dim=mlp_size,
                                 img_size=img_size,
                                 pool_ratio=pool_ratio,
                                 )

        self.block_list = nn.ModuleList(
            [
                DWConv(self.config) for i in range(num_layers)
            ]
        )

        self.embeddings = Embeddings(self.config)

    def forward(self, x, out_hidden=False):

        x = self.embeddings(x)

        hidden_state = []

        for l in self.block_list:
            x = l(x) + x
            hidden_state.append(x)
        if out_hidden:  # False
            return x, hidden_state
        return x


class DepthWiseFormerEncoder(nn.Module):
    def __init__(
            self,
            img_size,
            in_channels,  # 1
            features: Sequence[int],
            pool_size,
    ):
        '''
        img_size=image_size_s[1:],# [(64,64,64),(32,32,32),(16,16,16),(8,8,8)]
        fea=fea, # (16, 16, 32, 64, 128, 16)
        pool_size=pool_size # ((2,2,2), (2,2,2), (2,2,2), (2,2,2))
        '''
        super().__init__()

        fea = features
        self.drop = nn.Dropout()
        self.in_channels = in_channels
        self.features = features
        self.img_size = img_size

        pool_ratios = [[12, 16, 20, 24], [6, 8, 10, 12], [3, 4, 5, 6], [1, 2, 3, 4]]

        self.conv_0 = TwoConv(in_channels, features[0], 3, 1, 1)

        self.down_1 = DepthWiseformer(in_channels=fea[0], out_channels=fea[1], img_size=img_size[0],
                                      patch_size=pool_size[0], mlp_size=fea[1] * 2,
                                      pool_ratio=pool_ratios[0],
                                      num_layers=2,
                                      )

        self.down_2 = DepthWiseformer(in_channels=fea[1], out_channels=fea[2], img_size=img_size[1],
                                      patch_size=pool_size[1], mlp_size=fea[2] * 2,
                                      pool_ratio=pool_ratios[1],
                                      num_layers=2,
                                      )

        self.down_3 = DepthWiseformer(in_channels=fea[2], out_channels=fea[3], img_size=img_size[2],
                                      patch_size=pool_size[2], mlp_size=fea[3] * 2,
                                      pool_ratio=pool_ratios[2],
                                      num_layers=2,
                                      )

        self.down_4 = DepthWiseformer(in_channels=fea[3], out_channels=fea[4], img_size=img_size[3],
                                      patch_size=pool_size[3], mlp_size=fea[4] * 2,
                                      pool_ratio=pool_ratios[3],
                                      num_layers=2,
                                      )

    def forward(self, x: torch.Tensor):

        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        return x4, x3, x2, x1, x0


class DWFormerEncoder(nn.Module):
    def __init__(self, model_num,
                 img_size,
                 fea,
                 pool_size,
                 ):
        '''
        model_num=model_num,# 4
        img_size=image_size_s[1:],# [(64,64,64),(32,32,32),(16,16,16),(8,8,8)]
        fea=fea, # (16, 16, 32, 64, 128, 16)
        pool_size=pool_size # ((2,2,2), (2,2,2), (2,2,2), (2,2,2))
        '''

        super().__init__()
        self.model_num = model_num
        self.encoders = nn.ModuleList([])
        for i in range(model_num):
            encoder = DepthWiseFormerEncoder(
                img_size=img_size,
                in_channels=1,
                pool_size=pool_size,
                features=fea,
            )

            self.encoders.append(encoder)

    def forward(self, x):
        encoder_out = []
        x = x.unsqueeze(dim=2) 

        for i in range(self.model_num):  # 4
            encoder_out.append(self.encoders[i](x[:, i]))

        return encoder_out



