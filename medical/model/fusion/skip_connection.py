import math
from torch import nn
import torch
from einops import rearrange
from medical.model.fusion.layers import get_config, Mlp
# from medical.model.encoder.depthwise_former import LayerNormChannel

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
        img_size = config.img_size  # [8,8,8]
        in_channels = config.in_channels  # 128
        patch_size = config.patch_size  # [1,1,1]

        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])

        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                          out_channels=config.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size
                                          )

        self.norm = LayerNormChannel(num_channels=config.hidden_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))  # [1,n_patches,128]

    def forward(self, x):

        x = self.norm(self.patch_embeddings(x))

        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings

        return embeddings  # [B,N,C]


class AttentionCrossModal(nn.Module):
    def __init__(self, config):
        super(AttentionCrossModal, self).__init__()

        self.num_attention_heads = config.num_heads  # 8
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, kv):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(kv)
        mixed_value_layer = self.value(kv)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output


class CrossAttBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size  # 128
        self.config = config

        self.attention_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.attention_norm_cross = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = AttentionCrossModal(config)

    def forward(self, q, kv):

        h = q
        attn = self.attn(q, kv)
        x = attn + h  # [B,N,C]

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x) + h

        return x


class TokenLearner(nn.Module):
    def __init__(self, in_channels, S):
        super().__init__()
        self.token_conv = nn.Conv3d(in_channels=in_channels, out_channels=S, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        selected = self.token_conv(x)

        selected = rearrange(selected, "b s d w h -> b s (d w h) 1")  # [B,S,N,1]

        selected = torch.sigmoid(selected)

        x = rearrange(x, "b c d w h -> b 1 (d w h) c")  # [B,1,N,C]

        out = (x * selected).mean(dim=2)
        return out



class CrossAttn(nn.Module):
    def __init__(self, in_channels,
                 hidden_size,
                 img_size,
                 mlp_size=256,
                 token_mixer=128,  # token Learner N
                 token_learner=True):
        '''
        model_num=model_num,# 4
        in_channels=in_channels, # 128
        hidden_size=hidden_size, # 128
        img_size=img_size, # [8,8,8]
        mlp_size=mlp_size, # 256
        token_mixer_size=token_mixer_size,# 128
        token_learner=token_learner # True
        '''

        super().__init__()

        self.in_channels = in_channels
        self.hidden_size = hidden_size

        patch_size = (1, 1, 1)

        self.config = get_config(in_channels=in_channels,
                                 hidden_size=hidden_size,
                                 patch_size=patch_size,
                                 img_size=img_size,
                                 mlp_dim=mlp_size,
                                 )

        self.img_size = img_size
        self.q_embeddings = Embeddings(self.config)
        self.kv_embeddings = Embeddings(self.config)

        self.token_learner = token_learner  # True

        if self.token_learner:
            self.token_mixer = TokenLearner(in_channels=in_channels, S=token_mixer)

        self.cross_attn = CrossAttBlock(config=self.config)

    def forward(self, q, kv):

        q = self.q_embeddings(q)

        kv = self.kv_embeddings(kv)

        if self.token_learner:  # True
            kv = rearrange(kv, "b (d w h) c -> b c d w h", d=self.img_size[0], w=self.img_size[1],
                           h=self.img_size[2])

            kv = self.token_mixer(kv)  # [B,S,C]

        batch_size = kv.shape[0]

        cross_out = self.cross_attn(q, kv)

        cross_out = cross_out.transpose(-1, -2)

        cross_out = cross_out.view((batch_size, self.hidden_size, self.img_size[0], self.img_size[1], self.img_size[2]))

        return cross_out
