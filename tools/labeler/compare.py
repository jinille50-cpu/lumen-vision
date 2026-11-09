"""SAM 3 결과 조각의 그림자 의심 판정 (결과 목록의 '그림자?' 표시, 일괄 SAM 3의 '그림자 의심 조각 빼기')."""
import numpy as np

SHADOW_SHARE = 0.5   # 조각 픽셀 중 이 비율 이상이 '어둡고 색 없음'이면 그림자 의심


def shadow_map(rgb):
    """그림자 후보 픽셀: 밝기 V(최댓값) 150 미만이고 채도 S=(최대-최소)/최대*255 가 60 미만 (OpenCV HSV와 같은 척도).
    rgb는 (H, W, 3) uint8"""
    mx = rgb.max(axis=2).astype(np.float32)
    mn = rgb.min(axis=2).astype(np.float32)
    sat = (mx - mn) / np.maximum(mx, 1) * 255
    return (mx < 150) & (sat < 60)


def is_shadow(mask, shadow):
    n = int(mask.sum())
    return n > 0 and (mask & shadow).sum() / n >= SHADOW_SHARE
