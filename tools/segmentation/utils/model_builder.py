"""
model_builder.py — 모델 생성 로직 단일 진입점

기존에는 동일한 모델 빌드 코드가 train_tab / eval_tab / infer_tab 세 곳에
중복돼 있었다. 새 모델을 추가할 때마다 세 곳을 모두 고쳐야 했고, 한 곳만
빠뜨리면 학습은 되는데 평가/추론에서 깨지는 문제가 있었다.

이 모듈의 build_model() 하나로 통합하고, 세 탭이 모두 이를 호출한다.

build_model(model_name, encoder, in_channels, num_classes) -> (model, is_tr)
    is_tr : True 면 transformers 계열(출력이 out.logits), False 면 raw 텐서 출력.
            train_tab._get_logits 가 이 플래그로 출력 포맷을 분기한다.

load_weights(model, path, device, log=None) -> model
    저장된 state_dict 를 안전하게 로드(strict=False). eval/infer 공용.
"""

import os


# SMP CNN 계열 표시명 → smp 클래스명
SMP_MAP = {
    "UNet": "Unet", "UNet++": "UnetPlusPlus", "MAnet": "MAnet",
    "DeepLabV3": "DeepLabV3", "DeepLabV3+": "DeepLabV3Plus",
    "FPN": "FPN", "PSPNet": "PSPNet", "PAN": "PAN", "Linknet": "Linknet",
}

# ─────────────────────────────────────────────────────────────
#  전체 모델 목록 — 단일 소스(single source of truth)
#  eval_tab / infer_tab 의 모델 콤보박스가 이 목록을 그대로 import 한다.
#  새 모델을 추가하면 여기 한 곳만 고치면 세 탭 목차가 함께 갱신된다.
#  (과거엔 각 탭이 목록을 하드코딩해 HSI attention 모델이 평가/추론 목차에서
#   누락되는 문제가 있었다.)
# ─────────────────────────────────────────────────────────────
ALL_MODELS = [
    # SMP CNN 계열
    "UNet", "UNet++", "MAnet", "FPN", "PSPNet", "PAN", "Linknet",
    "DeepLabV3", "DeepLabV3+",
    # Transformer 계열
    "SegFormer-B0", "SegFormer-B2", "SegFormer-B4",
    "TransUNet", "MobileViT", "Mask2Former",
    # 초분광(HSI) 특화 — 분광 attention 포함
    "HSI-SpectralUNet (3D)", "HSI-SpectralAttnUNet",
    # DINO 백본 + 분할 디코더
    "DINOv2-S", "DINOv2-B", "DINOv3-S", "DINOv3-B",
]


SEGFORMER_CONFIGS = {
    "B0": {"depths": [2, 2, 2, 2], "hidden_sizes": [32, 64, 160, 256]},
    "B2": {"depths": [3, 4, 6, 3], "hidden_sizes": [64, 128, 320, 512]},
    "B4": {"depths": [3, 8, 27, 3], "hidden_sizes": [64, 128, 320, 512]},
}

# nvidia 사전학습(ADE/MiT) 체크포인트의 decoder_hidden_size.
#   from_pretrained 로 만든 구조는 이 값을 쓴다(B0=256, B1~B5=768).
#   반면 SegformerConfig 기본값은 항상 256 → scratch 로 학습한 모델은 256.
#   ⇒ 학습 때 사전학습을 썼는지에 따라 헤드 크기가 달라지므로, eval/infer 도
#      동일한 pretrained 값을 받아 같은 구조를 재현해야 .pth 가 그대로 로드된다.
_SEGFORMER_PT_DECODER = {"B0": 256, "B2": 768, "B4": 768}

_ENCODER_FALLBACK = {
    "기본값 (resnet34)": "resnet34",
    "기본값 (모델 내장)": "resnet34",
    "": "resnet34",
}


def _replace_first_conv(model, target_in_channels):
    """3채널 입력층(Conv2d 또는 Linear patch embedding)을 자동 탐색해 교체.
    transformers 계열(MobileViT/Mask2Former 등)의 RGB 가정 입력단을
    다채널로 바꾸는 용도."""
    import torch
    import torch.nn as nn
    if target_in_channels == 3:
        return model

    # ① Conv2d 우선 (대부분의 CNN/Swin 백본)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            orig = getattr(parent, parts[-1])
            new = nn.Conv2d(
                target_in_channels, orig.out_channels,
                kernel_size=orig.kernel_size, stride=orig.stride,
                padding=orig.padding, bias=(orig.bias is not None),
            )
            # RGB 가중치 채널 평균으로 초기화 (멀티스펙트럴 전이학습 표준 방식)
            # 3.0/C 배율로 채널 수 증가에 따른 출력 스케일 보정
            with torch.no_grad():
                w = orig.weight.data.mean(dim=1, keepdim=True)   # [out,1,k,k]
                new.weight.copy_(w.repeat(1, target_in_channels, 1, 1)
                                 * (3.0 / target_in_channels))
                if orig.bias is not None:
                    new.bias.copy_(orig.bias.data)
            setattr(parent, parts[-1], new)
            return model

    # ② Linear patch embedding (DPT/BEiT/ViT 계열)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "patch" in name.lower():
            if module.in_features % 3 == 0:
                patch_pix = module.in_features // 3
                new_in = patch_pix * target_in_channels
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                orig = getattr(parent, parts[-1])
                new = nn.Linear(new_in, orig.out_features,
                                bias=(orig.bias is not None))
                # patch embedding 도 동일하게 채널 평균으로 초기화
                # [out, patch_pix*3] → [out, patch_pix, 3] → 채널 평균 → C채널 복제
                with torch.no_grad():
                    w = orig.weight.data.view(orig.out_features, patch_pix, 3)
                    w = w.mean(dim=2, keepdim=True)              # [out,patch_pix,1]
                    new.weight.copy_(
                        (w.repeat(1, 1, target_in_channels)
                         * (3.0 / target_in_channels)).reshape(orig.out_features, new_in)
                    )
                    if orig.bias is not None:
                        new.bias.copy_(orig.bias.data)
                setattr(parent, parts[-1], new)
                return model

    print(f"[Warning] 3채널 입력층을 찾지 못했습니다. "
          f"{target_in_channels}채널 입력에서 동작 안 할 수 있습니다.")
    return model


def is_hsi_model(model_name):
    """초분광 특화 모델인지 여부."""
    try:
        from tools.segmentation.models.hyperspectral import HSI_MODELS
        return model_name in HSI_MODELS
    except Exception:
        return False


def infer_segformer_decoder_hidden(state):
    """SegFormer 체크포인트 state_dict 에서 decoder_hidden_size 를 역추정.

    저장된 .pth 의 decode_head 텐서 크기가 곧 학습 당시 decoder_hidden_size 다.
    (batch_norm.weight → [decoder_hidden], classifier.weight → [C, decoder_hidden, 1, 1])
    config 에 pretrained 정보가 없어도 이 값으로 구조를 정확히 재현할 수 있다.
    SegFormer 가 아니거나 판단 불가하면 None 반환."""
    if not isinstance(state, dict):
        return None
    for key in ("decode_head.batch_norm.weight",
                "decode_head.linear_fuse.weight",
                "decode_head.classifier.weight"):
        t = state.get(key)
        if t is None:
            continue
        try:
            return int(t.shape[1] if key.endswith("classifier.weight") else t.shape[0])
        except Exception:
            continue
    return None


def build_model(model_name, encoder, in_channels, num_classes, log=None,
                pretrained=None, load_pretrained_weights=True,
                decoder_hidden_size=None):
    """모델 이름으로 (model, is_tr) 생성. 세 탭 공용 단일 진입점.

    pretrained : 전이학습 가중치 선택 문자열(model_tab.PRETRAINED_WEIGHTS 값).
        SegFormer의 경우
          - "…(Full ADE)"  → ADE20K로 finetune된 SegFormer 전체(백본+디코드헤드)를
             from_pretrained로 로드 → 정확한 사전학습 성능.
          - "…(ImageNet…)" → MiT 인코더(백본)만 ImageNet 사전학습, 헤드는 랜덤.
          - 그 외/None     → scratch(config로 생성).
    load_pretrained_weights : True(학습 탭)면 사전학습 가중치를 실제로 내려받아
        로드한다. False(eval/infer 탭)면 다운로드 없이 동일한 '구조'만 재현한다.
        eval/infer 는 .pth 가중치를 덮어쓰므로 가중치 다운로드는 불필요하지만,
        구조(특히 SegFormer decoder_hidden_size)는 학습 때와 동일해야 하므로
        pretrained 값은 반드시 그대로 전달해야 한다."""

    def _log(msg):
        if log:
            log(msg)

    # ── 초분광 특화 모델 (raw 텐서 출력 → is_tr=False) ────────────
    from tools.segmentation.models.hyperspectral import build_hsi_model
    hsi = build_hsi_model(model_name, in_channels, num_classes)
    if hsi is not None:
        _log(f"  초분광 특화 모델 생성: {model_name} "
             f"(in_ch={in_channels}, classes={num_classes})")
        return hsi, False

    # ── DINO 백본 (raw 텐서 출력 → is_tr=False) ──────────────────
    if model_name.startswith("DINO"):
        from tools.segmentation.models.dino_seg import build_dino_seg
        use_pt = load_pretrained_weights and "사전학습" in (pretrained or "")
        return build_dino_seg(model_name, in_channels, num_classes, pretrained=use_pt), False

    import segmentation_models_pytorch as smp
    from transformers import (
        SegformerForSemanticSegmentation, SegformerConfig,
        MobileViTForSemanticSegmentation, MobileViTConfig,
        Mask2FormerForUniversalSegmentation, Mask2FormerConfig,
    )

    encoder = _ENCODER_FALLBACK.get(encoder, encoder)

    # ── SMP CNN 계열 ─────────────────────────────────────────────
    if model_name in SMP_MAP:
        model = getattr(smp, SMP_MAP[model_name])(
            encoder_name=encoder, encoder_weights=None,
            in_channels=in_channels, classes=num_classes,
        )
        return model, False

    # ── TransUNet (R50+ViT-B/16 하이브리드, raw 텐서 출력 → is_tr=False) ─
    if model_name == "TransUNet":
        from tools.segmentation.models.transunet import build_transunet
        pt = pretrained or ""
        use_pt = ("21k" in pt) or ("ImageNet" in pt)
        _log(f"  TransUNet 생성 (in_ch={in_channels}, classes={num_classes}, "
             f"{'ImageNet-21k 사전학습' if use_pt else 'scratch'})")
        if not use_pt:
            _log("  [주의] TransUNet 은 ViT 를 품고 있어 소규모 데이터에서 "
                 "scratch 로는 무너지기 쉽습니다 → ImageNet-21k 권장")
        return build_transunet(in_channels, num_classes, pretrained=use_pt), False

    # ── SegFormer ────────────────────────────────────────────────
    if "SegFormer" in model_name:
        v  = model_name.split("-")[-1]          # "B0"/"B2"/"B4"
        pt = pretrained or ""
        is_ade   = "ADE" in pt
        is_imnet = ("ImageNet" in pt) or ("mit-" in pt)

        # decoder_hidden_size 결정 우선순위:
        #   ① 호출자가 명시(체크포인트에서 역추정한 값) → 가장 신뢰
        #   ② 사전학습(ADE/ImageNet)이면 nvidia 체크포인트 값
        #   ③ 그 외 scratch 기본값(256)
        # ①은 config 에 pretrained 정보가 없는 옛 .pth 도 정확히 재현하기 위함.
        if decoder_hidden_size:
            decoder_hidden = decoder_hidden_size
        elif is_ade or is_imnet:
            decoder_hidden = _SEGFORMER_PT_DECODER.get(v, 768)
        else:
            decoder_hidden = 256

        # 학습 탭에서만 실제 사전학습 가중치를 내려받아 로드한다.
        # eval/infer 는 .pth 로 덮어쓰므로 구조만 아래 config 로 재현(다운로드 X).
        if load_pretrained_weights and (is_ade or is_imnet):
            # ① Full ADE — 백본+디코드헤드 전체 로드 / ② ImageNet — MiT 백본만
            if is_ade:
                repo = f"nvidia/segformer-{v.lower()}-finetuned-ade-512-512"
                _log(f"  SegFormer Full ADE 사전학습 로드: {repo} "
                     f"(num_labels={num_classes}, ignore_mismatched_sizes=True)")
            else:
                repo = f"nvidia/mit-{v.lower()}"
                _log(f"  SegFormer ImageNet 백본 사전학습 로드: {repo} "
                     f"(헤드 랜덤 초기화)")
            model = SegformerForSemanticSegmentation.from_pretrained(
                repo, num_labels=num_classes, ignore_mismatched_sizes=True)
            if in_channels != 3:
                _log(f"  입력 {in_channels}채널 → 패치 임베딩 conv 교체(재초기화)")
                model = _replace_first_conv(model, in_channels)
            return model, True

        # ③ scratch, 또는 eval/infer(사전학습 구조를 다운로드 없이 재현)
        vcfg = SEGFORMER_CONFIGS.get(v, SEGFORMER_CONFIGS["B2"])
        if (is_ade or is_imnet):
            _log(f"  SegFormer 구조 재현(사전학습 {pt}) "
                 f"decoder_hidden_size={decoder_hidden} — 가중치는 .pth 로 로드")
        cfg = SegformerConfig(num_channels=in_channels, num_labels=num_classes,
                              depths=vcfg["depths"], hidden_sizes=vcfg["hidden_sizes"],
                              decoder_hidden_size=decoder_hidden)
        return SegformerForSemanticSegmentation(cfg), True

    # ── MobileViT ────────────────────────────────────────────────
    if model_name == "MobileViT":
        model = MobileViTForSemanticSegmentation(
            MobileViTConfig(num_channels=in_channels, num_labels=num_classes))
        model = _replace_first_conv(model, in_channels)
        return model, True

    # ── Mask2Former ──────────────────────────────────────────────
    if model_name == "Mask2Former":
        model = Mask2FormerForUniversalSegmentation(
            Mask2FormerConfig(num_labels=num_classes))
        model = _replace_first_conv(model, in_channels)
        return model, True

    # ── fallback ─────────────────────────────────────────────────
    _log(f"  [경고] 알 수 없는 모델 '{model_name}' → UNet(resnet34)로 대체")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=in_channels, classes=num_classes)
    return model, False


def load_weights(model, model_path, device, log=None):
    """저장된 .pth 가중치를 모델에 로드(strict=False). eval/infer 공용."""
    import torch

    def _log(msg):
        if log:
            log(msg)

    _log(f"  가중치 로드 중: {os.path.basename(model_path)}")
    try:
        state = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            _log(f"  [경고] 없는 키 {len(missing)}개 (정상 범위면 무시)")
        if unexpected:
            _log(f"  [경고] 예상 외 키 {len(unexpected)}개")
    except Exception as e:
        raise RuntimeError(
            f"가중치 로드 실패: {e}\n"
            f"→ 모델 구조와 .pth 파일이 일치하는지 확인하세요")
    return model
