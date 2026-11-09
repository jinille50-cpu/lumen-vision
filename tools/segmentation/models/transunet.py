"""
transunet.py — TransUNet (R50+ViT-B/16 하이브리드 인코더 + CUP 디코더)

왜 별도 파일인가
----------------
TransUNet은 ViT를 품고 있어 Transformer 특유의 "데이터 굶주림(data hunger)"을
일부 상속합니다. 44장 같은 소규모 데이터에서 scratch 로 돌리면 무너지기 쉽고,
**ImageNet-21k 사전학습 가중치를 제대로 로드해야** 성능이 나옵니다.

그래서 원조 TransUNet과 동일한 백본인
    timm 'vit_base_r50_s16_224.orig_in21k'  (R50 + ViT-B/16, ImageNet-21k 사전학습)
을 그대로 인코더로 사용합니다. timm이 사전학습 다운로드/로드를 처리하고,
in_chans != 3 (다채널/HSI) 도 사전학습 conv 가중치를 자동 적응시킵니다.

구조
----
- 하이브리드 인코더(timm):
    stem.conv+norm → 64ch @ 1/2   (pool 이전, 최고해상도 skip)
    stages[0]      → 256ch @ 1/4  (skip)
    stages[1]      → 512ch @ 1/8  (skip)
    stages[2]      → 1024ch @ 1/16 → patch_embed.proj → ViT 12-layer transformer
- CUP(Cascaded UPsampler) 디코더:
    768 → 512 (conv_more, 1/16)
    → dec1(+skip 512) 1/8, 256
    → dec2(+skip 256) 1/4, 128
    → dec3(+skip 64)  1/2, 64
    → dec4(no skip)   1/1, 16
    → seg head (1x1) → num_classes

출력은 raw logits (N, num_classes, H, W) 이므로 기존 학습/평가/추론 파이프라인과
100% 호환됩니다 (transformers 계열 out.logits 아님 → is_tr=False).

가변 입력 크기: timm dynamic_img_size=True 로 위치 임베딩을 자동 보간하므로
16의 배수 임의 크기에서 동작합니다 (pad_utils 가 16 배수 패딩 담당).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 하이브리드 백본이 1/16 까지 다운샘플 → 16 배수 패딩 필요
PAD_MULTIPLE = 16

# 원조 TransUNet 백본 = R50 + ViT-B/16, ImageNet-21k 사전학습
_TIMM_BACKBONE = "vit_base_r50_s16_224.orig_in21k"


class DecoderBlock(nn.Module):
    """업샘플 → (skip concat) → DoubleConv. 원조 CUP 의 디코더 블록."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        if skip is not None:
            # skip 의 실제 해상도에 맞춰 업샘플 (same-padding 오프바이원 방지)
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2,
                              mode="bilinear", align_corners=False)
        return self.conv2(self.conv1(x))


class TransUNet(nn.Module):
    """R50+ViT-B/16 하이브리드 인코더 + CUP 디코더.

    in_channels : 입력 채널 수 (3 이 아니면 timm 이 사전학습 stem conv 를 자동 적응)
    num_classes : 세그멘테이션 클래스 수
    pretrained  : True 면 ImageNet-21k 사전학습 로드, False 면 scratch
    """

    PAD_MULTIPLE = PAD_MULTIPLE

    def __init__(self, in_channels, num_classes, pretrained=True):
        super().__init__()
        import timm

        self.vit = timm.create_model(
            _TIMM_BACKBONE,
            pretrained=pretrained,
            num_classes=0,            # 분류 헤드 제거
            in_chans=in_channels,     # 다채널 자동 적응
            dynamic_img_size=True,    # 가변 입력 → pos-embed 자동 보간
        )
        self.embed_dim = self.vit.embed_dim          # 768

        # ── ResNet stage 출력을 skip 으로 캡처 (forward hook) ──────
        self._feats = {}
        bb = self.vit.patch_embed.backbone
        bb.stem.norm.register_forward_hook(self._save("s2"))   # 64ch  @1/2
        bb.stages[0].register_forward_hook(self._save("s4"))   # 256ch @1/4
        bb.stages[1].register_forward_hook(self._save("s8"))   # 512ch @1/8
        bb.stages[2].register_forward_hook(self._save("s16"))  # 1024  @1/16

        # ── CUP 디코더 ────────────────────────────────────────────
        self.conv_more = nn.Sequential(
            nn.Conv2d(self.embed_dim, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.dec1 = DecoderBlock(512, 512, 256)   # 1/16 → 1/8
        self.dec2 = DecoderBlock(256, 256, 128)   # 1/8  → 1/4
        self.dec3 = DecoderBlock(128, 64,  64)    # 1/4  → 1/2
        self.dec4 = DecoderBlock(64,  0,   16)    # 1/2  → 1/1
        self.seg_head = nn.Conv2d(16, num_classes, 1)

    def _save(self, key):
        def hook(_module, _inp, out):
            self._feats[key] = out
        return hook

    def forward(self, x):
        B = x.shape[0]
        self._feats.clear()

        # ViT 인코더 통과 → (B, N, 768). 동시에 hook 이 skip 캡처.
        tokens = self.vit.forward_features(x)

        # 1/16 특징맵의 실제 grid 로 토큰 → 2D 복원 (prefix=cls token 등 제거)
        gh, gw = self._feats["s16"].shape[-2:]
        n_prefix = tokens.shape[1] - gh * gw
        patch = tokens[:, n_prefix:, :]                       # (B, gh*gw, 768)
        feat = patch.transpose(1, 2).reshape(B, self.embed_dim, gh, gw)

        d = self.conv_more(feat)
        d = self.dec1(d, self._feats["s8"])
        d = self.dec2(d, self._feats["s4"])
        d = self.dec3(d, self._feats["s2"])
        d = self.dec4(d, None)
        return self.seg_head(d)


def build_transunet(in_channels, num_classes, pretrained=True):
    """model_builder 진입점. (model 반환, is_tr 은 호출측에서 False 로 취급)"""
    return TransUNet(in_channels=in_channels, num_classes=num_classes,
                     pretrained=pretrained)
