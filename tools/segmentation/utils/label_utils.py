"""
label_utils.py — 라벨(마스크) 이미지 로드 공통 유틸리티

학습 / 평가 / 추론에서 라벨을 동일한 규칙으로 읽기 위해 사용합니다.
  - 컬러(3채널)로 저장된 마스크 → 첫 채널만 사용
  - 0/255 이진 마스크 → 0/1 클래스 인덱스로 변환
    (CrossEntropy 등은 클래스 인덱스 0~C-1 을 기대하므로 255 가 있으면 에러 발생)
"""

import numpy as np
from PIL import Image


def load_label(path):
    lb = np.array(Image.open(path))
    if lb.ndim == 3:
        lb = lb[..., 0]
    vals = np.unique(lb)
    if vals.max() == 255 and set(vals.tolist()) <= {0, 255}:
        lb = (lb > 0).astype(np.uint8)
    return lb


def read_array(path):
    """png/jpg/tif/npy/mat 파일 하나를 numpy 배열로 읽는다 (라벨 클래스, 입력 채널 감지용)"""
    ext = path.lower().rsplit(".", 1)[-1]
    if ext == "npy":
        return np.load(path)
    if ext == "mat":
        import scipy.io as sio
        return list(sio.loadmat(path).values())[-1]  # 마지막 변수를 데이터로 본다
    return np.array(Image.open(path))


def scan_classes(paths, max_files=300):
    """라벨 파일들의 고유값별 픽셀 수를 센다. 파일이 많으면 max_files개만 고르게 뽑는다.
    반환: (클래스 ID별 픽셀 수 dict, 실제로 읽은 파일 수)"""
    if len(paths) > max_files:
        step = len(paths) / max_files
        paths = [paths[int(i * step)] for i in range(max_files)]
    counts = {}
    for p in paths:
        lb = read_array(p)
        if lb.ndim == 3:
            lb = lb[..., 0]
        vals = np.unique(lb)
        if vals.max() == 255 and set(vals.tolist()) <= {0, 255}:
            lb = (lb > 0).astype(np.uint8)  # load_label과 같은 0/255 → 0/1 규칙
        v, c = np.unique(lb, return_counts=True)
        for a, b in zip(v.tolist(), c.tolist()):
            counts[a] = counts.get(a, 0) + b
    return counts, len(paths)


def suggest_weights(ratios, cap=10.0):
    """자동 가중치: 빈도 역수의 제곱근을 배경(클래스 0) = 1.0 기준으로 정규화, 상한 cap.
    배경이 없으면(비율 0) 가장 흔한 클래스를 기준으로 한다."""
    pmax = max(ratios)
    if pmax <= 0:
        return [1.0] * len(ratios)
    ref = ratios[0] if ratios[0] > 0 else pmax
    return [min(cap, (ref / r) ** 0.5) if r > 0 else 1.0 for r in ratios]
