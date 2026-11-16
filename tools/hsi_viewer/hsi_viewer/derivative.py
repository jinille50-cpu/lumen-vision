"""Savitzky-Golay 1차/2차 미분을 큐브 전체 픽셀의 스펙트럼 축에 한 번에 적용 (B-099).

SG 필터는 선형이라 스펙트럼 y (밴드 B개)에 대해 결과 = A @ y 로 쓸 수 있다(A: B×B 행렬).
A 는 scipy savgol_filter 를 단위 행렬에 적용해 만든다 → 가장자리 처리(mode="interp")까지
scipy 와 똑같다. 그다음은 (B, B) @ (B, 픽셀) 행렬 곱 한 번이라 CPU/GPU 모두 빠르다.
x 간격은 밴드 1칸 = 1 (파장 단위가 아니라 밴드 번호 기준).
"""
from __future__ import annotations

import numpy as np

from .baseline import run_blocks


def check_params(n_bands: int, order: int, window: int, poly: int) -> str:
    """설정이 맞으면 "", 틀리면 이유 글자."""
    if order not in (1, 2):
        return "미분 차수는 1 또는 2"
    if window % 2 == 0 or window < 3:
        return "윈도우는 3 이상 홀수"
    if window > n_bands:
        return f"윈도우({window})가 밴드 수({n_bands})보다 큽니다"
    if poly < order:
        return f"다항 차수는 미분 차수({order}) 이상이어야 합니다"
    if poly >= window:
        return "다항 차수는 윈도우보다 작아야 합니다"
    return ""


def sg_matrix(n_bands: int, order: int, window: int, poly: int) -> np.ndarray:
    """(B, B) 행렬 A: A @ y == savgol_filter(y, window, poly, deriv=order)."""
    from scipy.signal import savgol_filter

    # 단위 행렬의 j번째 열(밴드 j만 1)에 필터를 적용한 결과가 A의 j번째 열
    return savgol_filter(np.eye(n_bands), window, poly, deriv=order, axis=0)


def derivative_cube(data: np.ndarray, order: int = 1, window: int = 11, poly: int = 3,
                    device: str = "auto", progress=None) -> tuple[np.ndarray, str]:
    """(H, W, B) 큐브의 모든 픽셀 스펙트럼에 SG 미분을 적용해 (H, W, B) float32 로 돌려준다.

    반환: (결과, 실제 쓴 장치 이름)
    """
    err = check_params(data.shape[2], order, window, poly)
    if err:
        raise ValueError(err)
    A = sg_matrix(data.shape[2], order, window, poly)

    def make_fn(xp, dev):
        if xp is np:
            return lambda Y: A @ Y
        At = xp.tensor(A, dtype=xp.float64, device=dev)
        return lambda Y: At @ Y

    return run_blocks(data, make_fn, device, progress)
