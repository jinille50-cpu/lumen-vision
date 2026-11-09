"""
pad_utils.py — 모델 입력 크기 제약을 자동으로 처리하는 공통 유틸리티

학습(train) 때 패딩한 크기 정보를 _config.json 에 저장하고,
평가(eval) / 추론(infer) 때 동일하게 패딩 → 결과 언패딩하여
원본 해상도를 그대로 유지합니다.

사용 예:
    from tools.segmentation.utils.pad_utils import get_pad_multiple, pad_np, pad_tensor, unpad_tensor

    # numpy 단계 (데이터셋 로딩 시)
    padded_img, orig_h, orig_w = pad_np(img_arr, multiple=32)
    padded_lbl, _, _           = pad_np(lbl_arr, multiple=32, is_label=True)

    # tensor 단계 (추론 / 평가 시)
    t_pad, orig_h, orig_w = pad_tensor(t, multiple=32)
    out_crop              = unpad_tensor(out, orig_h, orig_w)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
#  모델별 필요 배수 자동 결정
# ─────────────────────────────────────────────────────────────
_PAD_MAP = {
    # SMP 계열 — encoder stride 32
    "UNet":       32,
    "UNet++":     32,
    "MAnet":      32,
    "DeepLabV3":  32,
    "DeepLabV3+": 32,
    "FPN":        32,
    "PSPNet":     32,
    "PAN":        32,
    "Linknet":    32,
    # Transformer 계열 — 내부 patch/stride
    "SegFormer-B0": 4,
    "SegFormer-B2": 4,
    "SegFormer-B4": 4,
    "MobileViT":    32,
    "Mask2Former":  32,
    # DINO — 패치 크기의 배수
    "DINOv2-S": 14,
    "DINOv2-B": 14,
    "DINOv3-S": 16,
    "DINOv3-B": 16,
}
_DEFAULT_MULTIPLE = 32


def get_pad_multiple(model_name: str) -> int:
    """모델 이름으로 필요한 패딩 배수를 반환합니다."""
    return _PAD_MAP.get(model_name, _DEFAULT_MULTIPLE)


# ─────────────────────────────────────────────────────────────
#  numpy 패딩 (데이터셋 로딩 단계)
# ─────────────────────────────────────────────────────────────
def common_pad_size(arrs, multiple = 32) -> tuple:
    """
    크기가 제각각인 배열들을 한 텐서로 묶기 위한 공통 크기.
    (최대 H, 최대 W)를 multiple 배수로 올림해서 반환합니다.
    """
    if multiple is None or multiple <= 0:
        multiple = _DEFAULT_MULTIPLE
    max_h = max(a.shape[0] for a in arrs)
    max_w = max(a.shape[1] for a in arrs)
    return (-(-max_h // multiple) * multiple, -(-max_w // multiple) * multiple)


def pad_np(arr: np.ndarray, multiple = 32,
           is_label: bool = False, target_hw: tuple = None) -> tuple:
    """
    numpy 배열을 multiple 배수 크기로 패딩합니다.

    Args:
        arr:       HW 또는 HWC 형태의 numpy 배열
        multiple:  패딩 기준 배수 (기본 32, None이면 32 사용)
        is_label:  True이면 constant(0) 패딩, False이면 reflect 패딩
        target_hw: 지정하면 (H, W) 크기로 패딩 (common_pad_size 결과 사용)

    Returns:
        (padded_arr, orig_h, orig_w)
    """
    if multiple is None or multiple <= 0:
        multiple = _DEFAULT_MULTIPLE
    h, w = arr.shape[:2]
    if target_hw is not None:
        pad_h, pad_w = target_hw[0] - h, target_hw[1] - w
    else:
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

    if pad_h == 0 and pad_w == 0:
        return arr, h, w

    if is_label:
        # 라벨: 0으로 채움 (클래스 값 보존)
        if arr.ndim == 3:
            padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)),
                            mode='constant', constant_values=0)
        else:
            padded = np.pad(arr, ((0, pad_h), (0, pad_w)),
                            mode='constant', constant_values=0)
    else:
        # 이미지: reflect 패딩 (경계 아티팩트 최소화)
        if arr.ndim == 3:
            padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='reflect')
    return padded, h, w


# ─────────────────────────────────────────────────────────────
#  tensor 패딩 / 언패딩 (추론·평가 단계)
# ─────────────────────────────────────────────────────────────
def pad_tensor(x, multiple = 32) -> tuple:
    """
    torch Tensor (N,C,H,W) 또는 (C,H,W) 를 multiple 배수로 패딩합니다.

    Returns:
        (padded_tensor, orig_h, orig_w)
    """
    import torch.nn.functional as F
    if multiple is None or multiple <= 0:
        multiple = _DEFAULT_MULTIPLE
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h == 0 and pad_w == 0:
        return x, h, w

    # F.pad 순서: (left, right, top, bottom)
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_pad, h, w


def unpad_tensor(x, orig_h: int, orig_w: int):
    """패딩된 tensor를 원본 크기로 크롭합니다."""
    return x[..., :orig_h, :orig_w]


def unpad_np(arr: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """패딩된 numpy 배열을 원본 크기로 크롭합니다."""
    return arr[..., :orig_h, :orig_w]


# ─────────────────────────────────────────────────────────────
#  입력 스케일 (정규화) — 학습 시 _config.json 에 저장, 평가/추론에서 동일 적용
# ─────────────────────────────────────────────────────────────
def input_scale_for(arr: np.ndarray) -> float:
    """이미지 dtype 기준 정규화 분모 (uint8 → 255, uint16 → 65535, 그 외 1)"""
    if arr.dtype == np.uint8:
        return 255.0
    if arr.dtype == np.uint16:
        return 65535.0
    return 1.0


def read_input_scale(model_path: str) -> float:
    """모델 옆 _config.json(ONNX 는 메타데이터) 의 input_scale. 없으면 1.0 (정규화 도입 전 모델 호환)"""
    try:
        from tools.segmentation.utils.config_path import load_config   # .onnx 면 메타데이터 먼저
        return float(load_config(model_path).get("input_scale", 1.0))
    except Exception:
        return 1.0


def select_channels(arr: np.ndarray, channels) -> np.ndarray:
    """(H, W, C) 이미지에서 선택한 채널 번호만 남긴다. channels가 None/빈 목록이거나 2차원이면 그대로."""
    if channels and arr.ndim == 3:
        return arr[..., list(channels)]   # 마지막 축(채널)에서 번호 목록으로 뽑기 (MATLAB의 A(:,:,idx))
    return arr
