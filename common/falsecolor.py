"""다파장 데이터(밴드가 여러 개인 큐브)를 밴드 3개로 RGB false color 그림으로 바꾼다.
밴드마다 따로 퍼센타일 하한~상한을 0~255로 늘린다 (밝은 점 몇 개 때문에 전체가 어두워지지 않게).
Object Detection(B-053)과 Labeler(B-015)가 같이 쓴다. numpy만 쓴다."""
import numpy as np

DEFAULT_LO = 1.0    # 하한 퍼센타일 (%)
DEFAULT_HI = 99.0   # 상한 퍼센타일 (%)


def stretch(plane, lo=DEFAULT_LO, hi=DEFAULT_HI):
    """2차원 배열 하나를 퍼센타일 lo~hi 범위로 늘려 uint8로. 값이 모두 같으면 0"""
    p = np.asarray(plane, dtype=np.float32)
    a, b = np.percentile(p, [lo, hi])   # percentile: 아래에서 lo%, hi% 위치의 값
    if b <= a:
        return np.zeros(p.shape, np.uint8)
    return (np.clip((p - a) / (b - a), 0, 1) * 255).round().astype(np.uint8)


def false_color(cube, bands, lo=DEFAULT_LO, hi=DEFAULT_HI):
    """cube: (H, W, 밴드 수) 배열, bands: R, G, B 자리에 넣을 밴드 번호 3개 → (H, W, 3) uint8 RGB"""
    if len(bands) != 3:
        raise ValueError(f"false color에는 밴드 3개가 필요합니다 (받은 수: {len(bands)})")
    return np.stack([stretch(cube[..., int(b)], lo, hi) for b in bands], axis=-1)
