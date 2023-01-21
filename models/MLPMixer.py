import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class PatchEmbeddings(nn.Module):

    def __init__(
        self,
        patch_size: int,
        hidden_dim: int,
        channels: int
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=hidden_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            Rearrange("b c h w -> b (h w) c")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class GlobalAveragePooling(nn.Module):

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


class Classifier(nn.Module):

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Linear(input_dim, num_classes)
        nn.init.zeros_(self.model.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MLPBlock(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MixerBlock(nn.Module):

    def __init__(
        self,
        num_patches: int,
        num_channels: int,
        tokens_hidden_dim: int,
        channels_hidden_dim: int
    ):
        super().__init__()
        self.token_mixing_img = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b p c -> b c p"),
            MLPBlock(num_patches, tokens_hidden_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing_img = nn.Sequential(
            nn.LayerNorm(num_channels),
            MLPBlock(num_channels, channels_hidden_dim)
        )
        self.token_mixing_sv = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b p c -> b c p"),
            MLPBlock(num_patches, tokens_hidden_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing_sv = nn.Sequential(
            nn.LayerNorm(num_channels),
            MLPBlock(num_channels, channels_hidden_dim)
        )

    def forward(self, img: torch.Tensor, sv: torch.Tensor) -> torch.Tensor:
        x_token_img = img + self.token_mixing_img(img)
        x_img = x_token_img + self.channel_mixing_img(x_token_img)

        x_token_sv = sv + self.token_mixing_sv(sv)
        x_sv = x_token_sv + self.channel_mixing_sv(x_token_sv)
        return x_img, x_sv


class MLPMixer(nn.Module):

    def __init__(
        self,
        num_classes: int,
        image_size: int = 256,
        channels: int = 3,
        patch_size: int = 32,
        num_layers: int = 8,
        hidden_dim: int = 512,
        tokens_hidden_dim: int = 256,
        channels_hidden_dim: int = 2048
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.embed = PatchEmbeddings(patch_size, hidden_dim, channels)
        layers = [
            MixerBlock(
                num_patches=num_patches,
                num_channels=hidden_dim,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim
            )
            for _ in range(num_layers)
        ]
        self.layers = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = GlobalAveragePooling(dim=1)
        self.classifier = Classifier(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.embed(x)           # [b, p, c]
        # print(x.shape)
        x = self.layers(x)          # [b, p, c]
        # print(x.shape)
        x = self.norm(x)            # [b, p, c]
        # print(x.shape)
        x = self.pool(x)            # [b, c]
        # print(x.shape)
        x = self.classifier(x)      # [b, num_classes]
        return x

class MixerBlock_fuse(nn.Module):

    def __init__(
        self,
        num_patches: int,
        num_channels: int,
        tokens_hidden_dim: int,
        channels_hidden_dim: int
    ):
        super().__init__()
        self.token_mixing_img = nn.Sequential(
            nn.LayerNorm(num_channels*2),
            Rearrange("b p c -> b c p"),
            MLPBlock(num_patches, tokens_hidden_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing_img = nn.Sequential(
            nn.LayerNorm(num_channels*2),
            MLPBlock(num_channels*2, channels_hidden_dim)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # print(img.shape)
        # print(self.token_mixing_img(img).shape)
        x_token_img = img + self.token_mixing_img(img)
        # print('x_token_img.shape, self.token_mixing_img(img).shape, self.channel_mixing_img(x_token_img).shape', x_token_img.shape, self.token_mixing_img(img).shape, self.channel_mixing_img(x_token_img).shape)
        x_img = x_token_img + self.channel_mixing_img(x_token_img)
        # print(x_img.shape)
        return x_img

class MixerBlock_late_fuse(nn.Module):

    def __init__(
        self,
        num_patches: int,
        num_channels: int,
        tokens_hidden_dim: int,
        channels_hidden_dim: int
    ):
        super().__init__()
        self.token_mixing_img = nn.Sequential(
            nn.LayerNorm(num_channels*4),
            Rearrange("b p c -> b c p"),
            MLPBlock(num_patches, tokens_hidden_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing_img = nn.Sequential(
            nn.LayerNorm(num_channels*4),
            MLPBlock(num_channels*4, channels_hidden_dim)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # print(img.shape)
        # print(self.token_mixing_img(img).shape)
        x_token_img = img + self.token_mixing_img(img)
        # print('x_token_img.shape, self.token_mixing_img(img).shape, self.channel_mixing_img(x_token_img).shape', x_token_img.shape, self.token_mixing_img(img).shape, self.channel_mixing_img(x_token_img).shape)
        x_img = x_token_img + self.channel_mixing_img(x_token_img)
        # print(x_img.shape)
        return x_img

class MMF_MLPMixer(nn.Module):

    def __init__(
        self,
        num_classes: int,
        image_size: int = 256,
        channels: int = 256,
        patch_size: int = 16,
        num_layers: int = 4,
        hidden_dim: int = 256,
        tokens_hidden_dim: int = 128,
        channels_hidden_dim: int = 1024
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.embed_img = PatchEmbeddings(patch_size, hidden_dim, channels)
        self.embed_sv = PatchEmbeddings(patch_size, hidden_dim, channels)
        # self.embed_fuse = PatchEmbeddings(patch_size, hidden_dim, channels*2)
        self.Mixerlayer1 = MixerBlock(
                num_patches=num_patches,
                num_channels=hidden_dim,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim
            )
        self.Mixerlayer2 = MixerBlock(
                num_patches=num_patches,
                num_channels=hidden_dim,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim
            )
        self.Mixerlayer3 = MixerBlock(
            num_patches=num_patches,
            num_channels=hidden_dim,
            tokens_hidden_dim=tokens_hidden_dim,
            channels_hidden_dim=channels_hidden_dim
        )
        self.Mixerlayer4 = MixerBlock(
                num_patches=num_patches,
                num_channels=hidden_dim,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim
            )

        self.Mixerlayer_fuse_early = MixerBlock_fuse(
            num_patches=num_patches,
            num_channels=hidden_dim,
            tokens_hidden_dim=tokens_hidden_dim,
            channels_hidden_dim=channels_hidden_dim
        )
        # self.Mixerlayer_fuse_mid = MixerBlock_fuse(
        #         num_patches=num_patches,
        #         num_channels=hidden_dim,
        #         tokens_hidden_dim=tokens_hidden_dim,
        #         channels_hidden_dim=channels_hidden_dim
        #     )
        self.Mixerlayer_fuse_late = MixerBlock_fuse(
                num_patches=num_patches,
                num_channels=hidden_dim,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim
            )
        # self.MLP_early = MLPBlockFuse(hidden_dim*2, hidden_dim*2)
        # self.MLP_late = MLPBlockFuse(hidden_dim*2, hidden_dim*2)

        # self.layers = nn.Sequential(*layers)
        # self.layers_sv = nn.Sequential(*layers)
        # self.norm_img = nn.LayerNorm(hidden_dim)
        # self.norm_sv = nn.LayerNorm(hidden_dim)
        # self.norm_early = nn.LayerNorm(hidden_dim*2)
        # # self.norm_mid = nn.LayerNorm(hidden_dim*2)
        # self.norm_late = nn.LayerNorm(hidden_dim*2)
        self.norm = nn.LayerNorm(hidden_dim*6)

        self.pool = GlobalAveragePooling(dim=1)
        # self.pool_early = nn.AdaptiveAvgPool1d(256)
        # self.pool_late = nn.AdaptiveAvgPool1d(256)
        self.classifier = Classifier(hidden_dim*6, num_classes)
        # self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=120, nhead=4)
        # self.TransformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=4), num_layers=1)
        # src = torch.rand(10, 15, 120)
        # out = encoder_layer(src)

    def forward(self, img: torch.Tensor, sv: torch.Tensor) -> torch.Tensor:
        b, c, h, w = img.shape
        # x = torch.cat([img, sv], 1)
        # print(x_fuse.shape)
        # x_fuse = self.embed_fuse(x_fuse)
        # x_img = self.ChannelAttentionModule(x_img)
        x_img = self.embed_img(img)           # [b, p, c]
        x_sv = self.embed_sv(sv)           # [b, p, c]
        # print(x_img.shape, x_sv.shape)
        # x = torch.cat([x_img, x_sv], 1)
        # print(x.shape)
        x_img_1, x_sv_1 = self.Mixerlayer1(x_img, x_sv)          # [b, p, c]
        # print(x_img_1.shape, x_sv_1.shape)

        # x_fuse_early = self.MLP_early(x_fuse_early)
        x_fuse_early = torch.cat([x_img_1, x_sv_1], 2)
        # print(x_img_1.shape, x_sv_1.shape, x_fuse_early.shape)
        # print('x_fuse_early.shape:', x_fuse_early.shape)
        x_fuse_early = self.Mixerlayer_fuse_early(x_fuse_early)
        # print('x_fuse_early.shape:', x_fuse_early.shape)

        x_sv_1 = x_img_1 + x_sv_1
        x_img_2, x_sv_2 = self.Mixerlayer2(x_img_1, x_sv_1)
        x_img = x_img + x_img_2
        x_sv = x_sv + x_sv_2 + x_img_2

        x_img_3, x_sv_3 = self.Mixerlayer3(x_img, x_sv)

        # x_fuse_mid = torch.cat([x_img_3, x_sv_3], 2)
        # # print(x_img_1.shape, x_sv_1.shape, x_fuse_early.shape)
        # x_fuse_mid = self.Mixerlayer_fuse_mid(x_fuse_mid)

        x_sv_3 = x_img_3 + x_sv_3
        x_img_4, x_sv_4 = self.Mixerlayer4(x_img_3, x_sv_3)


        x_fuse_late = torch.cat([x_img_4, x_sv_4], 2)

        # print('x_fuse_late.shape:', x_fuse_late.shape)
        x_fuse_late = self.Mixerlayer_fuse_late(x_fuse_late)
        # print('x_fuse_late.shape:', x_fuse_late.shape)


        x_img = x_img + x_img_4
        x_sv = x_sv + x_sv_4

        x = torch.cat([x_img, x_sv, x_fuse_early, x_fuse_late], 2)
        # x = torch.cat([x_img, x_sv], 2)
        # print(x.shape)
        x = self.norm(x)
        x = self.pool(x)            # [b, c]

        x = self.classifier(x)      # [b, num_classes]
        return x

class MMF_MLPMixer_old(nn.Module):

    def __init__(
        self,
        num_classes: int,
        image_size: int = 256,
        channels: int = 256,
        patch_size: int = 16,
        num_layers: int = 4,
        hidden_dim: int = 256,
        tokens_hidden_dim: int = 128,
        channels_hidden_dim: int = 1024
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.embed_img = PatchEmbeddings(patch_size, hidden_dim, channels)
        self.embed_sv = PatchEmbeddings(patch_size, hidden_dim, channels)
        # self.embed_fuse = PatchEmbeddings(patch_size, hidden_dim, channels*2)
        self.Mixerlayer1 = MixerBlock(
                num_patches=num_patches,
                num_channels=hidden_dim,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim
            )
        self.Mixerlayer2 = MixerBlock(
                num_patches=num_patches,
                num_channels=hidden_dim,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim
            )
        self.Mixerlayer3 = MixerBlock(
            num_patches=num_patches,
            num_channels=hidden_dim,
            tokens_hidden_dim=tokens_hidden_dim,
            channels_hidden_dim=channels_hidden_dim
        )
        self.Mixerlayer4 = MixerBlock(
                num_patches=num_patches,
                num_channels=hidden_dim,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim
            )
        # self.Mixerlayer5 = MixerBlock(
        #     num_patches=num_patches,
        #     num_channels=hidden_dim,
        #     tokens_hidden_dim=tokens_hidden_dim,
        #     channels_hidden_dim=channels_hidden_dim
        # )
        # self.Mixerlayer6 = MixerBlock(
        #         num_patches=num_patches,
        #         num_channels=hidden_dim,
        #         tokens_hidden_dim=tokens_hidden_dim,
        #         channels_hidden_dim=channels_hidden_dim
        #     )

        # self.layers = nn.Sequential(*layers)
        # self.layers_sv = nn.Sequential(*layers)
        self.norm_img = nn.LayerNorm(hidden_dim)
        self.norm_sv = nn.LayerNorm(hidden_dim)
        self.pool = GlobalAveragePooling(dim=1)
        self.classifier = Classifier(hidden_dim*2, num_classes)
        # self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=120, nhead=4)
        # self.TransformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=120, nhead=4), num_layers=4)
        # src = torch.rand(10, 15, 120)
        # out = encoder_layer(src)

    def forward(self, img: torch.Tensor, sv: torch.Tensor) -> torch.Tensor:
        b, c, h, w = img.shape
        # x = torch.cat([img, sv], 1)
        # print(x_fuse.shape)
        # x_fuse = self.embed_fuse(x_fuse)
        x_img = self.embed_img(img)           # [b, p, c]
        x_sv = self.embed_sv(sv)           # [b, p, c]
        # print(x_img.shape, x_sv.shape, x_fuse.shape)
        # x = torch.cat([x_img, x_sv], 1)
        # print(x.shape)
        x_img_1, x_sv_1 = self.Mixerlayer1(x_img, x_sv)          # [b, p, c]
        x_sv_1 = x_img_1 + x_sv_1
        x_img_2, x_sv_2 = self.Mixerlayer2(x_img_1, x_sv_1)
        x_img = x_img + x_img_2
        x_sv = x_sv + x_sv_2 + x_img_2

        x_img_3, x_sv_3 = self.Mixerlayer3(x_img, x_sv)
        x_sv_3 = x_img_3 + x_sv_3
        x_img_4, x_sv_4 = self.Mixerlayer4(x_img_3, x_sv_3)

        x_img = x_img + x_img_4
        x_sv = x_sv + x_sv_4
        # x_img_5, x_sv_5 = self.Mixerlayer5(x_img, x_sv)
        # x_img = x_img + x_img_5
        # x_sv = x_sv + x_sv_5 + x_img
        # x_img_6, x_sv_6 = self.Mixerlayer6(x_img_5, x_sv_5)
        # x_img = x_img + x_img_6
        # x_sv = x_sv + x_sv_6
        # print(x_img.shape, x_sv.shape)
        # x_sv_token, x_sv = self.layers_sv(x_sv)          # [b, p, c]
        # print(x.shape)
        x_img = self.norm_img(x_img)            # [b, p, c]
        x_sv = self.norm_sv(x_sv)            # [b, p, c]
        # print(x_img.shape, x_sv.shape)
        x = torch.cat([x_img, x_sv], 2)
        x = self.pool(x)            # [b, c]
        # print(x.shape)
        x = self.classifier(x)      # [b, num_classes]
        return x

# class MMF_MLPMixer(nn.Module):
#
#     def __init__(
#         self,
#         num_classes: int,
#         image_size: int = 256,
#         channels: int = 256,
#         patch_size: int = 16,
#         num_layers: int = 4,
#         hidden_dim: int = 256,
#         tokens_hidden_dim: int = 128,
#         channels_hidden_dim: int = 1024
#     ):
#         super().__init__()
#         num_patches = (image_size // patch_size) ** 2
#         self.embed_img = PatchEmbeddings(patch_size, hidden_dim, channels)
#         self.embed_sv = PatchEmbeddings(patch_size, hidden_dim, channels)
#         self.token_mixing_merge = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             Rearrange("b p c -> b c p"),
#             MLPBlock(num_patches, tokens_hidden_dim),
#             Rearrange("b c p -> b p c")
#         )
#         self.channel_mixing_merge = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             MLPBlock(hidden_dim, channels_hidden_dim)
#         )
#         # self.embed_fuse = PatchEmbeddings(patch_size, hidden_dim, channels*2)
#         self.Mixerlayer1 = MixerBlock(
#                 num_patches=num_patches,
#                 num_channels=hidden_dim,
#                 tokens_hidden_dim=tokens_hidden_dim,
#                 channels_hidden_dim=channels_hidden_dim
#             )
#         self.Mixerlayer2 = MixerBlock(
#                 num_patches=num_patches,
#                 num_channels=hidden_dim,
#                 tokens_hidden_dim=tokens_hidden_dim,
#                 channels_hidden_dim=channels_hidden_dim
#             )
#         self.Mixerlayer3 = MixerBlock(
#             num_patches=num_patches,
#             num_channels=hidden_dim,
#             tokens_hidden_dim=tokens_hidden_dim,
#             channels_hidden_dim=channels_hidden_dim
#         )
#         self.Mixerlayer4 = MixerBlock(
#                 num_patches=num_patches,
#                 num_channels=hidden_dim,
#                 tokens_hidden_dim=tokens_hidden_dim,
#                 channels_hidden_dim=channels_hidden_dim
#             )
#         # self.Mixerlayer5 = MixerBlock(
#         #     num_patches=num_patches,
#         #     num_channels=hidden_dim,
#         #     tokens_hidden_dim=tokens_hidden_dim,
#         #     channels_hidden_dim=channels_hidden_dim
#         # )
#         # self.Mixerlayer6 = MixerBlock(
#         #         num_patches=num_patches,
#         #         num_channels=hidden_dim,
#         #         tokens_hidden_dim=tokens_hidden_dim,
#         #         channels_hidden_dim=channels_hidden_dim
#         #     )
#
#         # self.layers = nn.Sequential(*layers)
#         # self.layers_sv = nn.Sequential(*layers)
#         self.norm_img = nn.LayerNorm(hidden_dim)
#         self.norm_sv = nn.LayerNorm(hidden_dim)
#         self.pool = GlobalAveragePooling(dim=1)
#         self.classifier = Classifier(hidden_dim*2, num_classes)
#
#     def forward(self, img: torch.Tensor, sv: torch.Tensor) -> torch.Tensor:
#         b, c, h, w = img.shape
#         # x = torch.cat([img, sv], 1)
#         # print(x_fuse.shape)
#         # x_fuse = self.embed_fuse(x_fuse)
#         x_img = self.embed_img(img)           # [b, p, c]
#         x_sv = self.embed_sv(sv)           # [b, p, c]
#         # print(x_img.shape, x_sv.shape, x_fuse.shape)
#         # x = torch.cat([x_img, x_sv], 1)
#         # print(x.shape)
#         x_img_1, x_sv_1 = self.Mixerlayer1(x_img, x_sv)          # [b, p, c]
#         x_sv_1 = x_img_1 + x_sv_1
#         x_img_2, x_sv_2 = self.Mixerlayer2(x_img_1, x_sv_1)
#         x_img = x_img + x_img_2
#         x_sv = x_sv + x_sv_2 + x_img_2
#
#         x_img_3, x_sv_3 = self.Mixerlayer3(x_img, x_sv)
#         x_sv_3 = x_img_3 + x_sv_3
#         x_img_4, x_sv_4 = self.Mixerlayer4(x_img_3, x_sv_3)
#
#         x_img = x_img + x_img_4
#         x_sv = x_sv + x_sv_4
#
#         x_img = x_img + self.token_mixing_merge(x_img)
#         x_img = x_img + self.channel_mixing_merge(x_img)
#
#         x_sv = x_sv + self.token_mixing_merge(x_sv)
#         x_sv = x_sv + self.channel_mixing_merge(x_sv)
#
#         # x_img_5, x_sv_5 = self.Mixerlayer5(x_img, x_sv)
#         # x_img = x_img + x_img_5
#         # x_sv = x_sv + x_sv_5 + x_img
#         # x_img_6, x_sv_6 = self.Mixerlayer6(x_img_5, x_sv_5)
#         # x_img = x_img + x_img_6
#         # x_sv = x_sv + x_sv_6
#         # print(x_img.shape, x_sv.shape)
#         # x_sv_token, x_sv = self.layers_sv(x_sv)          # [b, p, c]
#         # print(x.shape)
#         x_img = self.norm_img(x_img)            # [b, p, c]
#         x_sv = self.norm_sv(x_sv)            # [b, p, c]
#         # print(x_img.shape, x_sv.shape)
#         x = torch.cat([x_img, x_sv], 2)
#         x = self.pool(x)            # [b, c]
#         # print(x.shape)
#         x = self.classifier(x)      # [b, num_classes]
#         return x

def mlp_mixer_s16(num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=16, num_layers=4, hidden_dim=256,
                  tokens_hidden_dim=128, channels_hidden_dim=1024)
    return MLPMixer(num_classes, image_size, channels, **params)

def mlp_mixer_s32(num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=32, num_layers=8, hidden_dim=512,
                  tokens_hidden_dim=256, channels_hidden_dim=2048)
    return MLPMixer(num_classes, image_size, channels, **params)

def mlp_mixer_b16(num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=16, num_layers=12, hidden_dim=768,
                  tokens_hidden_dim=384, channels_hidden_dim=3072)
    return MLPMixer(num_classes, image_size, channels, **params)

def mlp_mixer_b32(num_classes: int, image_size: int = 224, channels: int = 3):
    params = dict(patch_size=32, num_layers=12, hidden_dim=768,
                  tokens_hidden_dim=384, channels_hidden_dim=3072)
    return MLPMixer(num_classes, image_size, channels, **params)


if __name__ == "__main__":
    v = mlp_mixer_s16(num_classes=2, image_size=64)

    img = torch.randn(1, 3, 64, 64)

    preds = v(img) # (1, 1000)
    print(preds.shape)
