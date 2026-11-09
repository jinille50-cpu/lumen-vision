"""
hyperspectral.py — 초분광(HSI) 분광-공간 특화 세그멘테이션 모델

설계 목표
---------
- **가변 밴드 수**(30~100 등 PCA/차원축소 후 임의 채널)에 그대로 동작
- 분광(spectral) + 공간(spatial) 정보를 함께 학습
- 출력은 raw logits (N, num_classes, H, W) → 기존 학습/평가/추론 파이프라인과 100% 호환
  (transformers 계열의 out.logits 가 아니므로 is_tr=False 로 취급)

"통합형(hybrid)" 의도
---------------------
밴드 의존적인 입력단(spectral stem)과 전이 가능한 본체(2D U-Net backbone)를 분리했습니다.
- 데이터 A(예: 60밴드)로 학습한 .pth 를, 데이터 B(예: 100밴드)에 load_pretrained 로
  불러오면 backbone 가중치는 그대로 전이되고 밴드 수가 다른 stem 만 새로 초기화됩니다.
  (strict=False 로드 → shape 안 맞는 stem 키는 자동 skip)
- 추후 외부 사전학습 가중치(DOFA/SpectralGPT 등에서 추출한 backbone)가 생기면
  동일 메커니즘으로 붙일 수 있습니다.

제공 모델
---------
- SpectralSpatialUNet : 3D conv 분광-공간 stem + 2D U-Net  (분광 특화 정석, 다소 무거움)
- SpectralAttnUNet    : 분광 attention(SE) + 1x1 분광압축 + 2D U-Net  (경량, 큰 이미지 권장)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 모든 모델 공통 패딩 배수 (2D U-Net 4단계 다운샘플 → 16 배수 필요)
PAD_MULTIPLE = 16


# ─────────────────────────────────────────────────────────────
#  공통 빌딩 블록
# ─────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    """표준 4단계 2D U-Net. (N, c_in, H, W) → (N, num_classes, H, W)

    이 부분이 '전이 가능한 backbone' 입니다. 밴드 수와 무관하게 구조가 고정되므로
    서로 다른 데이터셋 간 가중치 전이가 가능합니다.
    """

    def __init__(self, c_in, num_classes, base=32):
        super().__init__()
        self.enc1 = DoubleConv(c_in, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class SpectralAttention(nn.Module):
    """SE 방식 분광 attention — 정보량 많은 밴드에 가중치 부여 (channel-wise)."""

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        hidden = max(4, in_channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class Spectral3DStem(nn.Module):
    """3D conv 분광-공간 front-end. 밴드 축을 depth 로 보고 3D conv 적용 후
    분광 차원을 풀링해 고정 임베딩으로 압축한다.

    (N, C_bands, H, W) → (N, embed, H, W)

    메모리 주의: 중간 텐서가 (N, ch, C_bands, H, W) 5D 라 큰 이미지에서는 무겁다.
    첫 conv 에서 밴드 축 stride=2 로 한 번 줄여 비용을 완화했다.
    """

    def __init__(self, embed=32, spec_kernel=7):
        super().__init__()
        pad = spec_kernel // 2
        self.conv1 = nn.Conv3d(1, 4, kernel_size=(spec_kernel, 3, 3),
                               stride=(2, 1, 1), padding=(pad, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(4)
        self.conv2 = nn.Conv3d(4, 8, kernel_size=(spec_kernel, 3, 3),
                               stride=(1, 1, 1), padding=(pad, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(8)
        self.spectral_pool = nn.AdaptiveAvgPool3d((1, None, None))  # 밴드축 → 1
        self.proj = nn.Conv2d(8, embed, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):                 # (N, C, H, W)
        x = x.unsqueeze(1)                # (N, 1, C, H, W)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.spectral_pool(x)         # (N, 8, 1, H, W)
        x = x.squeeze(2)                  # (N, 8, H, W)
        x = self.act(self.proj(x))        # (N, embed, H, W)
        return x


# ─────────────────────────────────────────────────────────────
#  전이 로드 믹스인
# ─────────────────────────────────────────────────────────────
class _PretrainMixin:
    def load_pretrained(self, path, map_location="cpu"):
        """다른 데이터셋에서 학습한 .pth 를 전이 로드.
        밴드 수가 달라 shape 이 안 맞는 stem 키는 strict=False 로 자동 skip 된다.
        반환: (missing_keys, unexpected_keys)
        """
        state = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # shape 안 맞는 키는 미리 걸러서 경고 없이 skip
        own = self.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in own and own[k].shape == v.shape}
        return self.load_state_dict(filtered, strict=False)


# ─────────────────────────────────────────────────────────────
#  최종 모델
# ─────────────────────────────────────────────────────────────
class SpectralSpatialUNet(nn.Module, _PretrainMixin):
    """3D 분광-공간 stem + 2D U-Net. 분광 특화 정석 모델."""

    PAD_MULTIPLE = PAD_MULTIPLE

    def __init__(self, in_channels, num_classes, embed=32, base=32):
        super().__init__()
        self.stem = Spectral3DStem(embed=embed)
        self.backbone = UNet2D(embed, num_classes, base=base)

    def forward(self, x):
        return self.backbone(self.stem(x))


class SpectralAttnUNet(nn.Module, _PretrainMixin):
    """분광 attention + 1x1 분광압축 + 2D U-Net. 경량 — 큰 이미지에 권장."""

    PAD_MULTIPLE = PAD_MULTIPLE

    def __init__(self, in_channels, num_classes, embed=32, base=32):
        super().__init__()
        self.spectral_attn = SpectralAttention(in_channels)
        self.spectral_reduce = nn.Conv2d(in_channels, embed, 1)  # 밴드 의존 stem
        self.backbone = UNet2D(embed, num_classes, base=base)

    def forward(self, x):
        x = self.spectral_attn(x)
        x = self.spectral_reduce(x)
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────
#  레지스트리 — model_builder 가 이 표를 참조해 생성
# ─────────────────────────────────────────────────────────────
HSI_MODELS = {
    "HSI-SpectralUNet (3D)": SpectralSpatialUNet,
    "HSI-SpectralAttnUNet":  SpectralAttnUNet,
}


def build_hsi_model(model_name, in_channels, num_classes):
    """이름으로 HSI 모델 인스턴스 생성. 등록 안 된 이름이면 None."""
    cls = HSI_MODELS.get(model_name)
    if cls is None:
        return None
    return cls(in_channels=in_channels, num_classes=num_classes)
