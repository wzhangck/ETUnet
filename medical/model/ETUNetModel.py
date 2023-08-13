import torch.nn as nn
from encoder.depthwise_former import DWFormerEncoder, Convolution,  UpCat
from fusion.nmafa import ESCA
import torch
from einops import rearrange


class ETUNet(nn.Module):
    def __init__(self, model_num,
                 out_channels,
                 image_size,
                 fea=(8, 8, 32, 64, 128, 8),
                 window_size=(2, 2, 2),
                 pool_size=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                 self_num_layer=4,
                 token_mixer_size=128,
                 multi_fusion=True,
            ):
        super().__init__()

        self.out_channels = out_channels
        self.model_num = model_num
        self.pool_size = pool_size

        pool_size_all = [1, 1, 1]
        image_size_s = [image_size]

        for p in pool_size:
            pool_size_all = [pool_size_all[i] * p[i] for i in range(len(p))]  # [2,2,2]，[4,4,4]，[8,8,8]，[16,16,16]

            image_size_s.append((image_size_s[-1][0] // p[0], image_size_s[-1][1] // p[1], image_size_s[-1][2] // p[2]))

        # new_image_size——[8,8,8]
        new_image_size = [image_size[i] // pool_size_all[i] for i in range(3)]

        self.encoder = DWFormerEncoder(model_num=model_num,  # 4
                                       img_size=image_size_s[1:],
                                       fea=fea,  # (16, 16, 32, 64, 128, 16)
                                       pool_size=pool_size,
                                       )

        self.fusion = ESCA(model_num=model_num,
                           in_channels=fea[4],
                           hidden_size=fea[4],
                           img_size=new_image_size,
                           mlp_size=2 * fea[4],
                           self_num_layer=self_num_layer,
                           window_size=window_size,
                        )

        self.fusion_conv_5 = Convolution(model_num * fea[4], fea[4], 3, 1, 1)
        self.fusion_conv_1 = Convolution(model_num * fea[0], fea[0], 3, 1, 1)
        self.fusion_conv_2 = Convolution(model_num * fea[1], fea[1], 3, 1, 1)
        self.fusion_conv_3 = Convolution(model_num * fea[2], fea[2], 3, 1, 1)
        self.fusion_conv_4 = Convolution(model_num * fea[3], fea[3], 3, 1, 1)

        self.upcat_4 = UpCat(fea[4], fea[3], fea[3], pool_size=pool_size[3],
                             img_size=(16, 16, 16),token_mixer_size=token_mixer_size)
        self.upcat_3 = UpCat(fea[3], fea[2], fea[2], pool_size=pool_size[2],
                             img_size=(32, 32, 32),token_mixer_size=token_mixer_size)
        self.upcat_2 = UpCat(fea[2], fea[1], fea[1], pool_size=pool_size[1],
                             img_size=(64, 64, 64),token_mixer_size=token_mixer_size)
        self.upcat_1 = UpCat(fea[1], fea[0], fea[5], pool_size=pool_size[0],
                             img_size=(128, 128, 128),token_mixer_size=token_mixer_size)

        self.final_conv = nn.Conv3d(fea[5], out_channels, 1, 1)

        self.multiFusion = multi_fusion

        self.fea = fea

        self.encoder_4 = Convolution(in_channels=self.fea[3] + self.fea[2] + self.fea[1] + self.fea[0],
                                     out_channels=self.fea[3], kernel_size=1, stride_size=1,
                                     padding_size=0)

        self.encoder_3 = Convolution(in_channels=self.fea[2] + self.fea[1] + self.fea[0],
                                     out_channels=self.fea[2], kernel_size=1, stride_size=1,
                                     padding_size=0)

        self.encoder_2 = Convolution(in_channels=self.fea[1] + self.fea[0],
                                     out_channels=self.fea[1], kernel_size=1, stride_size=1,
                                     padding_size=0)

    def forward(self, x):
        assert x.shape[1] == self.model_num  # 4

        encoder_x = self.encoder(x)

        encoder_1 = torch.stack([encoder_x[i][4] for i in range(self.model_num)], dim=1)

        encoder_2 = torch.stack([encoder_x[i][3] for i in range(self.model_num)], dim=1)

        encoder_3 = torch.stack([encoder_x[i][2] for i in range(self.model_num)], dim=1)

        encoder_4 = torch.stack([encoder_x[i][1] for i in range(self.model_num)], dim=1)

        encoder_5 = torch.stack([encoder_x[i][0] for i in range(self.model_num)], dim=1)

        # ESCA
        fusion_out = self.fusion(encoder_5)
        encoder_5 = rearrange(encoder_5, "b n c d w h -> b (n c) d w h")

        fusion_out_cnn = self.fusion_conv_5(encoder_5)

        fusion_out = fusion_out + fusion_out_cnn

        encoder_1 = rearrange(encoder_1, "b n c d w h -> b (n c) d w h")
        encoder_2 = rearrange(encoder_2, "b n c d w h -> b (n c) d w h")
        encoder_3 = rearrange(encoder_3, "b n c d w h -> b (n c) d w h")
        encoder_4 = rearrange(encoder_4, "b n c d w h -> b (n c) d w h")

        encoder_1_cnn = self.fusion_conv_1(encoder_1)
        encoder_2_cnn = self.fusion_conv_2(encoder_2)
        encoder_3_cnn = self.fusion_conv_3(encoder_3)
        encoder_4_cnn = self.fusion_conv_4(encoder_4)

        if self.multiFusion:
            encoder_1_down4_cnn = nn.AvgPool3d(kernel_size=8)(encoder_1_cnn)
            encoder_1_down3_cnn = nn.AvgPool3d(kernel_size=4)(encoder_1_cnn)
            encoder_1_down2_cnn = nn.AvgPool3d(kernel_size=2)(encoder_1_cnn)

            encoder_2_down4_cnn = nn.AvgPool3d(kernel_size=4)(encoder_2_cnn)
            encoder_2_down3_cnn = nn.AvgPool3d(kernel_size=2)(encoder_2_cnn)

            encoder_3_down4_cnn = nn.AvgPool3d(kernel_size=2)(encoder_3_cnn)

            encoder_4_cnn = torch.cat([encoder_1_down4_cnn, encoder_2_down4_cnn, encoder_3_down4_cnn, encoder_4_cnn],
                                      dim=1)

            encoder_3_cnn = torch.cat([encoder_1_down3_cnn, encoder_2_down3_cnn, encoder_3_cnn], dim=1)
            encoder_2_cnn = torch.cat([encoder_1_down2_cnn, encoder_2_cnn], dim=1)

            encoder_4_cnn = self.encoder_4(encoder_4_cnn)

            encoder_3_cnn = self.encoder_3(encoder_3_cnn)

            encoder_2_cnn = self.encoder_2(encoder_2_cnn)

        u4 = self.upcat_4(fusion_out, encoder_4_cnn)
        u3 = self.upcat_3(u4, encoder_3_cnn)
        u2 = self.upcat_2(u3, encoder_2_cnn)
        u1 = self.upcat_1(u2, encoder_1_cnn)

        logits = self.final_conv(u1)

        return logits

if __name__ == '__main__':
    model = ETUNet(4, 3, (128, 128, 128))
    data = torch.randn([1, 4, 128, 128, 128])

    from thop import profile

    flops, params = profile(model, inputs=(data,))

    print(flops / 1e9, params / 1e6)
