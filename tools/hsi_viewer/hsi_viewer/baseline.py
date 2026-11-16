"""AsLS(비대칭 최소제곱) 베이스라인 제거를 큐브 전체 픽셀에 한 번에 적용 (B-097, B-099).

픽셀마다 (W + lam·DᵀD) z = W y 를 푼다. D는 2차 차분 행렬(표준 AsLS, Eilers & Boelens).
이 행렬은 띠 행렬(대각선 주변 2칸만 값이 있음)이라, 밴드 방향으로만 반복문을 돌고
픽셀 방향은 배열 한 줄 연산으로 동시에 계산한다(벡터화). 같은 코드를
numpy(CPU)와 torch(GPU)에 그대로 쓴다: 두 모듈 모두 sqrt, where, empty_like 이름이 같다.

run_blocks 는 AsLS와 미분(derivative.py)이 같이 쓰는 "픽셀 블록 단위 CPU/GPU 실행기".
"""
from __future__ import annotations

import numpy as np


def penalty_diagonals(n_bands: int, lam: float) -> list[list[float]]:
    """lam·DᵀD (D = 2차 차분)의 대각선들 [주대각, 1칸 위, 2칸 위] — 각 대각선은 파이썬 float 목록."""
    D = np.diff(np.eye(n_bands), n=2, axis=0)   # (B-2, B) 2차 차분 행렬
    H = lam * (D.T @ D)
    return [np.diagonal(H, k).tolist() for k in range(3)]


def _solve(xp, a0, h, b):
    """5중 대각 대칭 행렬 A의 Cholesky 분해(A = L·Lᵀ)로 A x = b 를 푼다.

    a0: (B, N) 주대각(픽셀마다 다름), h: penalty_diagonals 의 1칸·2칸 위 대각(픽셀 공통),
    b: (B, N). 반환 (B, N).
    """
    B = a0.shape[0]
    h1, h2 = h[1], h[2]
    c0 = xp.empty_like(a0)        # L의 주대각
    c1 = xp.zeros_like(a0)        # L[i, i-1]
    c2 = xp.zeros_like(a0)        # L[i, i-2]
    for i in range(B):
        s = a0[i]
        if i >= 2:
            c2[i] = h2[i - 2] / c0[i - 2]
            s = s - c2[i] * c2[i]
        if i >= 1:
            t = h1[i - 1]
            if i >= 2:
                t = t - c2[i] * c1[i - 1]
            c1[i] = t / c0[i - 1]
            s = s - c1[i] * c1[i]
        c0[i] = xp.sqrt(s)
    v = b.clone() if xp.__name__ == "torch" else b.copy()
    for i in range(B):                      # 앞으로: L y = b
        t = v[i]
        if i >= 1:
            t = t - c1[i] * v[i - 1]
        if i >= 2:
            t = t - c2[i] * v[i - 2]
        v[i] = t / c0[i]
    for i in range(B - 1, -1, -1):          # 뒤로: Lᵀ x = y
        t = v[i]
        if i + 1 < B:
            t = t - c1[i + 1] * v[i + 1]
        if i + 2 < B:
            t = t - c2[i + 2] * v[i + 2]
        v[i] = t / c0[i]
    return v


def _asls_block(xp, Y, d0, h, p, n_iter):
    """Y: (B, N) float64, d0: (B, 1) lam·DᵀD 주대각(Y와 같은 장치). 반환: 베이스라인 z (B, N)."""
    w = xp.ones_like(Y)
    lo, hi = xp.full_like(Y, p), xp.full_like(Y, 1.0 - p)   # 가중치 두 값 (torch에서도 float64 유지)
    z = Y
    for _ in range(n_iter):
        z = _solve(xp, w + d0, h, w * Y)
        # 신호가 베이스라인보다 위면 작은 가중치 p, 아래면 1-p
        w = xp.where(Y > z, lo, hi)
    return z


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def run_blocks(data: np.ndarray, make_fn, device: str = "auto", progress=None,
               gpu_elems: int = 16_000_000, cpu_elems: int = 4_000_000) -> tuple[np.ndarray, str]:
    """(H, W, B) 큐브를 픽셀 블록으로 나눠 fn 을 적용하고 (H, W, B) float32 로 돌려준다.

    make_fn(xp, dev) → fn(Y): Y 는 (B, 픽셀) float64 배열(xp = numpy 또는 torch, dev 장치), 같은 모양을 돌려준다.
    device: "auto"(CUDA 있으면 GPU), "cpu", "cuda". 블록 크기 = 원소 수 gpu_elems / cpu_elems 기준.
    progress(done, total): 블록 하나 끝날 때마다 호출. 반환: (결과, 실제 쓴 장치 이름)

    GPU 메모리가 모자라면(B-100) 블록 픽셀 수를 반씩 줄여 다시 하고, min_gpu_pixels 까지 줄여도
    모자라면 남은 픽셀을 CPU로 계산한다. 이 경우 장치 이름에 그 내용을 적어 돌려준다.
    """
    h_, w_, B = data.shape
    if device == "auto":
        device = "cuda" if cuda_available() else "cpu"
    flat = data.reshape(-1, B)
    n = flat.shape[0]
    out = np.empty((n, B), np.float32)
    s = 0            # 다음에 계산할 픽셀 위치
    name = "CPU"
    if device == "cuda":
        import torch

        min_gpu_pixels = 64
        # 한 블록 = 픽셀 몇 개 × 밴드 전체. 정밀도 때문에 float64로 계산 (lam이 크면 float32 분해가 불안정)
        step = max(1, gpu_elems // B)
        step0, fn = step, None
        while s < n:
            oom = False
            try:
                if fn is None:
                    fn = make_fn(torch, "cuda")
                Y = torch.from_numpy(np.ascontiguousarray(flat[s:s + step].T, dtype=np.float64)).to("cuda")
                out[s:s + step] = fn(Y).T.cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                oom = True
            Y = None   # GPU 메모리를 놓아 준다
            if oom:
                # except 블록 밖에서 비워야 오류에 묶인 중간 텐서까지 풀린다
                torch.cuda.empty_cache()
                if fn is None or step <= min_gpu_pixels:
                    break                                   # 더 줄일 수 없음 → 남은 픽셀은 CPU
                step = max(min_gpu_pixels, step // 2)
                continue                                    # 같은 위치부터 작은 블록으로 다시
            s = min(s + step, n)
            if progress:
                progress(s, n)
        name = "GPU (" + torch.cuda.get_device_name(0) + ")"
        if s < n:
            name += f" → 메모리 부족으로 {n - s}픽셀은 CPU"
        elif step < step0:
            name += f", 메모리 부족으로 블록 {step0}→{step}픽셀"
        fn = None
        torch.cuda.empty_cache()
    if s < n:
        step = max(1, cpu_elems // B)
        fn = make_fn(np, "cpu")
        while s < n:
            Y = np.ascontiguousarray(flat[s:s + step].T, dtype=np.float64)   # (B, 픽셀) — 밴드 한 줄이 연속 메모리
            out[s:s + step] = fn(Y).T
            s = min(s + step, n)
            if progress:
                progress(s, n)
    return out.reshape(h_, w_, B), name


def asls_cube(data: np.ndarray, lam: float = 1e5, p: float = 1e-3, n_iter: int = 10,
              device: str = "auto", progress=None) -> tuple[np.ndarray, str]:
    """(H, W, B) 큐브의 모든 픽셀에서 AsLS 베이스라인(2차 차분 평활 벌점)을 빼서 (H, W, B) float32로 돌려준다.

    device: "auto"(CUDA 있으면 GPU), "cpu", "cuda". 반환: (결과 = 원 신호 − 베이스라인, 실제 쓴 장치 이름)
    """
    B = data.shape[2]
    if B <= 3:
        raise ValueError(f"밴드 수({B})가 너무 적습니다")
    h = penalty_diagonals(B, lam)

    def make_fn(xp, dev):
        if xp is np:
            d0 = np.asarray(h[0], np.float64)[:, None]
        else:
            d0 = xp.tensor(h[0], dtype=xp.float64, device=dev)[:, None]
        return lambda Y: Y - _asls_block(xp, Y, d0, h, p, n_iter)

    return run_blocks(data, make_fn, device, progress)
