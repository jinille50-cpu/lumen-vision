"""Albumentations 증강: 프리셋, 항목 정의, 파이프라인 만들기.
이미지와 마스크를 같이 변환한다. 입력 채널이 3이 아니면(다파장) 색 변환은 쓰지 않는다."""
import numpy as np

# (키, 표시 이름, 수치 이름, 색 변환 여부, 최소, 최대, 간격)
ITEMS = [
    ("HFlip",   "좌우 뒤집기", "p",      False, 0.0, 1.0,   0.1),
    ("VFlip",   "상하 뒤집기", "p",      False, 0.0, 1.0,   0.1),
    ("Rotate",  "회전",        "도",     False, 0.0, 180.0, 5.0),
    ("Scale",   "스케일",      "±비율",  False, 0.0, 0.9,   0.05),
    ("Crop",    "크롭",        "최소 면적", False, 0.1, 1.0, 0.05),
    ("Jitter",  "밝기/대비/채도", "강도", True,  0.0, 1.0,   0.05),
    ("Blur",    "블러",        "σ",      True,  0.1, 5.0,   0.1),
    ("Gray",    "흑백 변환",   "p",      True,  0.0, 1.0,   0.1),
]
ITEM_KEYS = [it[0] for it in ITEMS]
COLOR_KEYS = [it[0] for it in ITEMS if it[3]]

# 프리셋: 키 → 값 (여기에 없는 항목은 끔)
PRESETS = {
    "없음": {},
    "약":   {"HFlip": 0.5, "Rotate": 10.0, "Scale": 0.1, "Jitter": 0.1},
    "보통": {"HFlip": 0.5, "VFlip": 0.5, "Rotate": 30.0, "Scale": 0.2, "Crop": 0.85, "Jitter": 0.2},
    "강":   {"HFlip": 0.5, "VFlip": 0.5, "Rotate": 45.0, "Scale": 0.3, "Crop": 0.7,
             "Jitter": 0.3, "Blur": 1.0, "Gray": 0.1},
}
# 항목을 켰을 때 쓰는 기본 수치
DEFAULTS = {"HFlip": 0.5, "VFlip": 0.5, "Rotate": 30.0, "Scale": 0.2, "Crop": 0.85,
            "Jitter": 0.2, "Blur": 1.0, "Gray": 0.1}


def build_pipeline(aug, hw, n_ch, seed=None):
    """aug = {키: {"on": bool, "value": float}}. 켜진 항목이 없으면 None."""
    import albumentations as A
    import cv2
    color_ok = (n_ch == 3)
    t = []
    for key in ITEM_KEYS:
        c = aug.get(key) or {}
        if not c.get("on"):
            continue
        v = float(c.get("value", DEFAULTS[key]))
        if key in COLOR_KEYS and not color_ok:
            continue
        if key == "HFlip":
            t.append(A.HorizontalFlip(p=v))
        elif key == "VFlip":
            t.append(A.VerticalFlip(p=v))
        elif key == "Rotate":
            # 빈 곳은 이미지 0, 마스크 0(배경)으로 채움
            t.append(A.Rotate(limit=v, p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0))
        elif key == "Scale":
            t.append(A.Affine(scale=(1 - v, 1 + v), p=0.5, fill=0, fill_mask=0))
        elif key == "Crop":
            h, w = hw
            t.append(A.RandomResizedCrop(size=(h, w), scale=(v, 1.0), ratio=(w / h, w / h), p=0.5))
        elif key == "Jitter":
            t.append(A.ColorJitter(brightness=v, contrast=v, saturation=v, hue=0.0, p=0.5))
        elif key == "Blur":
            t.append(A.GaussianBlur(sigma_limit=(0.1, v), p=0.5))
        elif key == "Gray":
            t.append(A.ToGray(p=v))
    if not t:
        return None
    return A.Compose(t, seed=seed)


def apply_one(pipe, img_hwc, mask_hw):
    """이미지 (H,W,C) float32, 마스크 (H,W) → 변환된 (이미지, 마스크)"""
    out = pipe(image=np.ascontiguousarray(img_hwc), mask=np.ascontiguousarray(mask_hw))
    return out["image"], out["mask"]


def apply_batch(pipe, data, labels):
    """학습 배치 (B,C,H,W) 텐서, (B,H,W) 텐서에 샘플마다 따로 증강을 적용"""
    import torch
    x = data.permute(0, 2, 3, 1).numpy()
    y = labels.numpy().astype(np.uint8)   # 클래스 255개 이하 가정 (cv2 보간 때문에 uint8)
    xs, ys = [], []
    for i in range(len(x)):
        a, b = apply_one(pipe, x[i], y[i])
        xs.append(a); ys.append(b)
    data = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous()
    labels = torch.from_numpy(np.stack(ys)).to(labels.dtype)
    return data, labels
