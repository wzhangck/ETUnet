from torch import nn, einsum
import torch
from .layers import Attention, PostionEmbedding, get_config, Mlp
from einops.layers.torch import Rearrange
from einops import rearrange, repeat


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, types=0, is_window=False):
        super(Embeddings, self).__init__()
        self.is_window = is_window
        self.types = types
        self.config = config
        in_channels = config.in_channels
        patch_size = config.patch_size

        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                          out_channels=config.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size
                                          )

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):

        x = self.patch_embeddings(x)

        return x


class WinAttn_SepViT(nn.Module):
    def __init__(self, config, dropout=0.1, window_size=2):
        '''
        SepViT:<https://arxiv.org/abs/2203.15380>
        '''
        super().__init__()

        self.heads = config.num_heads
        self.dim = config.hidden_size
        self.dim_head = int(self.dim / self.heads)
        self.scale = self.dim_head ** -0.5
        self.window_size = window_size

        self.inner_dim = self.dim_head * self.heads

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_qkv = nn.Conv1d(self.dim, self.inner_dim * 3, 1, bias=False)

        self.window_tokens = nn.Parameter(torch.randn(self.dim))

        self.window_tokens_to_qk = nn.Sequential(
            nn.LayerNorm(self.dim_head),
            nn.GELU(),
            Rearrange('b h n c -> b (h c) n'),
            nn.Conv1d(self.inner_dim, self.inner_dim * 2, 1),
            Rearrange('b (h c) n -> b h n c', h=self.heads),
        )

        # window attention
        self.window_attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Conv3d(self.inner_dim, self.dim, 1),
            nn.Dropout(dropout)
        )

        self.attention_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.attention_norm_cross = nn.LayerNorm(self.dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(self.dim, eps=1e-6)

    def forward(self, x):
        # x:b,c,d,w,h
        batch, depth, height, width, heads, wsz = x.shape[0], *x.shape[-3:], self.heads, self.window_size

        x = rearrange(x, 'b c (d w1) (h w2) (w w3) -> (b d h w) c (w1 w2 w3)', w1=wsz, w2=wsz,
                      w3=wsz)

        w = repeat(self.window_tokens, 'c -> b c 1', b=x.shape[0])

        x = torch.cat((w, x), dim=-1)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h=heads), (q, k, v))

        q = q * self.scale

        dots = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)

        window_tokens, windowed_fmaps = out[:, :, 0], out[:, :, 1:]

        window_tokens = rearrange(window_tokens, '(b z x y) h d -> b h (z x y) d', x=height // wsz, y=width // wsz,
                                  z=depth // wsz)

        windowed_fmaps = rearrange(windowed_fmaps, '(b z x y) h n d -> b h (z x y) n d', x=height // wsz,
                                   y=width // wsz, z=depth // wsz)

        w_q, w_k = self.window_tokens_to_qk(window_tokens).chunk(2, dim=-1)

        w_q = w_q * self.scale

        w_dots = einsum('b h i d, b h j d -> b h i j', w_q, w_k)

        w_attn = self.window_attend(w_dots)

        aggregated_windowed_fmap = einsum('b h i j, b h j w d -> b h i w d', w_attn, windowed_fmaps)

        fmap = rearrange(aggregated_windowed_fmap, 'b h (z x y) (w1 w2 w3) d -> b (h d) (z w1) (x w2) (y w3)',
                         x=height // wsz, y=width // wsz, z=depth // wsz, w1=wsz, w2=wsz, w3=wsz)

        fmap = rearrange(fmap, 'b c d h w-> b (d h w) c')

        return fmap


class SpatialAttn(nn.Module):
    def __init__(self, config, is_position=True):  # True
        super().__init__()

        self.config = config
        self.is_position = is_position  # True
        self.d_attention = Attention(config)
        self.h_attention = Attention(config)
        self.w_attention = Attention(config)

        self.window_attention = WinAttn_SepViT(config)

        d, w, h = config.img_size[0], config.img_size[1], config.img_size[2]

        self.dAtttn_position_shape = (w * h, d, config.hidden_size)
        self.wAtttn_position_shape = (h * d, w, config.hidden_size)
        self.hAtttn_position_shape = (w * d, h, config.hidden_size)

        if is_position:
            self.dAttn_position = PostionEmbedding(config, img_size=self.dAtttn_position_shape, types=1)
            self.hAttn_position = PostionEmbedding(config, img_size=self.hAtttn_position_shape, types=1)
            self.wAttn_position = PostionEmbedding(config, img_size=self.wAtttn_position_shape, types=1)
            self.window_position = PostionEmbedding(config,img_size=None, types=2)

    def forward(self, x):

        batch_size, hidden_size, D, W, H = x.shape

        x_1 = rearrange(x, "b c d w h -> (b w h) d c")
        x_2 = rearrange(x, "b c d w h -> (b h d) w c")
        x_3 = rearrange(x, "b c d w h -> (b w d) h c")

        if self.is_position:
            x_1 = self.dAttn_position(x_1)
            x_2 = self.wAttn_position(x_2)
            x_3 = self.hAttn_position(x_3)
            x_4 = self.window_position(x)

        x_1 = self.d_attention(x_1)

        x_2 = self.w_attention(x_2)

        x_3 = self.h_attention(x_3)

        x_4 = self.window_attention(x_4)

        x_1 = rearrange(x_1, "(b w h) d c -> b (d w h) c", d=D, w=W, h=H)  # [B,N,C]

        x_2 = rearrange(x_2, "(b w d) h c -> b (d w h) c", d=D, w=W, h=H)  # [B,N,C]

        x_3 = rearrange(x_3, "(b h d) w c -> b (d w h) c", d=D, w=W, h=H)  # [B,N,C]

        out = x_1 + x_2 + x_3 + x_4

        return out


class ChannelAttn(nn.Module):
    def __init__(self, config, k_size=3):
        super(ChannelAttn, self).__init__()

        self.dim = config.hidden_size  # 输入通道数
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)

        y = self.sigmoid(y)

        return x * y.expand_as(x)


class MultiAttBlock(nn.Module):
    def __init__(self, config, is_position=False):
        super(MultiAttBlock, self).__init__()
        self.config = config
        self.input_shape = config.img_size
        self.hidden_size = config.hidden_size

        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)

        self.spatial_attn = SpatialAttn(config)
        self.channel_attn = ChannelAttn(config)

    def forward(self, x):

        batch_size, hidden_size, D, W, H = x.shape  # B,C,D,W,H

        x = rearrange(x, "b c d w h -> b (d w h) c")

        h = x

        x = self.attention_norm(x)

        x = rearrange(x, "b (d w h) c -> b c d w h", d=D, w=W, h=H)

        x_spa = self.spatial_attn(x)

        x_cha = self.channel_attn(x)

        x_cha = rearrange(x_cha, "b c d w h -> b (d w h) c")

        x = x_spa + x_cha + h

        h = x

        x = self.ffn_norm(x)

        x = self.ffn(x)

        x = x + h

        x = x.transpose(-1, -2)

        out_size = (self.input_shape[0] // self.config.patch_size[0],
                    self.input_shape[1] // self.config.patch_size[1],
                    self.input_shape[2] // self.config.patch_size[2],)

        x = x.view((batch_size, self.config.hidden_size, out_size[0], out_size[1], out_size[2])).contiguous()

        return x


class MultiAttentionFusion(nn.Module):
    def __init__(self, in_channels,
                 hidden_size,
                 img_size,
                 num_heads=8,
                 mlp_size=256,
                 num_layers=4,
                 window_size=(4, 4, 4),
                 out_hidden=False):
        '''
        in_channels=model_num*in_channels,# 4*128
        hidden_size=hidden_size,# 128
        img_size=img_size,# # [8,8,8]
        mlp_size=mlp_size,# 256
        num_layers=self_num_layer,# 2
        window_size=window_size# (4, 4, 4)
        '''

        super().__init__()

        self.config = get_config(in_channels=in_channels,
                                 hidden_size=hidden_size,
                                 window_size=window_size,
                                 img_size=img_size,
                                 mlp_dim=mlp_size,
                                 num_heads=num_heads,  # 8
                                 patch_size=(1, 1, 1)
                                 )

        self.block_list = nn.ModuleList(
            [
                MultiAttBlock(self.config, is_position=True)
                for i in range(num_layers)
            ])

        self.embeddings = Embeddings(self.config)
        self.out_hidden = out_hidden

    def forward(self, x):  # [1, 4*128, 8, 8, 8]

        x = self.embeddings(x)

        hidden_states = []
        for l in self.block_list:
            x = l(x)
            if self.out_hidden:
                hidden_states.append(x)

        if self.out_hidden:
            return x, hidden_states

        return x
