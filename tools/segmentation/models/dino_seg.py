"""
dino_seg.py — DINOv2 / DINOv3 ViT 백본 + 작은 분할 디코더.

DINO 는 특징만 뽑는 백본이라 분할 헤드가 따로 없다. 마지막 4개 블록의 특징을 합친 뒤
두 번 키우고, 입력 크기로 맞춰 픽셀별 logits (N, num_classes, H, W) 을 낸다.
출력이 raw 텐서이므로 is_tr=False 로 기존 학습/평가/추론과 그대로 호환된다.

백본 사전학습 가중치는 timm 이 내려받는다. in_chans != 3 이면 timm 이 첫 층 가중치를 맞춰 준다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 표시명 → timm 모델 이름
DINO_BACKBONES = {
    "DINOv2-S": "vit_small_patch14_dinov2.lvd142m",
    "DINOv2-B": "vit_base_patch14_dinov2.lvd142m",
    "DINOv3-S": "vit_small_patch16_dinov3.lvd1689m",
    "DINOv3-B": "vit_base_patch16_dinov3.lvd1689m",
}


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DinoSeg(nn.Module):
    def __init__(self, model_name, in_channels, num_classes, pretrained=False):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            DINO_BACKBONES[model_name], pretrained=pretrained, in_chans=in_channels,
            num_classes=0, dynamic_img_size=True,   # 입력 크기가 달라도 위치 정보를 맞춰 준다
        )
        dim = self.backbone.embed_dim
        self.proj = nn.ModuleList([nn.Conv2d(dim, 128, 1) for _ in range(4)])
        self.fuse = _conv_block(128 * 4, 256)
        self.up1 = _conv_block(256, 128)
        self.up2 = _conv_block(128, 64)
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # indices=4: 마지막 4개 블록, NCHW: (N, 차원, H/패치, W/패치) 모양으로 받는다
        feats = self.backbone.forward_intermediates(
            x, indices=4, intermediates_only=True, output_fmt="NCHW", norm=True)
        y = self.fuse(torch.cat([p(f) for p, f in zip(self.proj, feats)], dim=1))
        y = self.up1(F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False))
        y = self.up2(F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False))
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(y)


def build_dino_seg(model_name, in_channels, num_classes, pretrained=False):
    return DinoSeg(model_name, in_channels, num_classes, pretrained=pretrained)
