"""Numerical helpers: false-colour composites, spectra extraction, normalisation, PCA."""
from __future__ import annotations

import numpy as np


def snv(x: np.ndarray) -> np.ndarray:
    """Standard Normal Variate along the last axis."""
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sd + 1e-9)


def _subsample(band: np.ndarray, max_samples: int = 250_000) -> np.ndarray:
    """Flat subsample of a band for fast percentile estimation."""
    n = band.size
    if n <= max_samples:
        return band.ravel()
    step = int(np.ceil(n / max_samples))
    return band.ravel()[::step]


class CompositeCache:
    """Caches per-band percentile levels so that re-renders are cheap."""

    def __init__(self):
        self._levels: dict[tuple, tuple[float, float]] = {}
        self._minmax: dict[int, tuple[float, float]] = {}
        self._data_id = None

    def reset(self, data_id):
        if data_id != self._data_id:
            self._levels.clear()
            self._minmax.clear()
            self._data_id = data_id

    def levels(self, data: np.ndarray, band: int, low: float, high: float) -> tuple[float, float]:
        self.reset(id(data))
        key = (band, round(low, 3), round(high, 3))
        v = self._levels.get(key)
        if v is None:
            s = _subsample(data[:, :, band])
            lo, hi = np.percentile(s, (low, high))
            v = (float(lo), float(hi))
            self._levels[key] = v
        return v

    def minmax(self, data: np.ndarray, band: int) -> tuple[float, float]:
        self.reset(id(data))
        v = self._minmax.get(band)
        if v is None:
            b = data[:, :, band]
            v = (float(b.min()), float(b.max()))
            self._minmax[band] = v
        return v


def false_color(
    data: np.ndarray,
    bands: tuple[int, int, int],
    low: float,
    high: float,
    invert: bool,
    cache: CompositeCache | None = None,
) -> np.ndarray:
    """Return a (H, W, 3) uint8 percentile-stretched composite."""
    out = np.empty(data.shape[:2] + (3,), dtype=np.uint8)
    for i, b in enumerate(bands):
        ch = data[:, :, b]
        if cache is not None:
            lo, hi = cache.levels(data, b, low, high)
        else:
            lo, hi = np.percentile(_subsample(ch), (low, high))
        scale = 255.0 / (hi - lo + 1e-8)
        tmp = (ch - lo) * scale
        np.clip(tmp, 0, 255, out=tmp)
        if invert:
            tmp = 255.0 - tmp
        out[:, :, i] = tmp.astype(np.uint8)
    return out


def single_band(data: np.ndarray, band: int, low: float, high: float, cache: CompositeCache | None = None) -> np.ndarray:
    """Return (H, W) float32 band and its display levels."""
    ch = data[:, :, band]
    if cache is not None:
        lo, hi = cache.levels(data, band, low, high)
    else:
        lo, hi = np.percentile(_subsample(ch), (low, high))
    return ch, (lo, hi)


def pca_composite(data: np.ndarray, n_components: int = 3, max_samples: int = 200_000) -> np.ndarray:
    """(H, W, 3) float32 PCA score image (first three components)."""
    h, w, b = data.shape
    flat = data.reshape(-1, b)
    sub = _subsample_rows(flat, max_samples)
    mean = sub.mean(axis=0)
    cov = np.cov((sub - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:n_components]
    comps = evecs[:, order]
    scores = (flat - mean) @ comps
    if scores.shape[1] < 3:
        scores = np.pad(scores, ((0, 0), (0, 3 - scores.shape[1])))
    return scores.reshape(h, w, 3).astype(np.float32)


def _subsample_rows(flat: np.ndarray, max_rows: int) -> np.ndarray:
    if flat.shape[0] <= max_rows:
        return flat
    step = int(np.ceil(flat.shape[0] / max_rows))
    return flat[::step]


def binned_spectrum(data: np.ndarray, x: int, y: int, bin_size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Mean spectrum of a bin_size x bin_size window centred at (x, y)."""
    half = bin_size // 2
    y0, y1 = max(0, y - half), min(data.shape[0], y + half + 1)
    x0, x1 = max(0, x - half), min(data.shape[1], x + half + 1)
    region = data[y0:y1, x0:x1, :]
    spec = region.reshape(-1, data.shape[2]).mean(axis=0)
    return spec, (x0, y0, x1, y1)


def rect_spectrum(data: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std spectrum of a rectangle."""
    x0, x1 = sorted((max(0, x0), min(data.shape[1], x1)))
    y0, y1 = sorted((max(0, y0), min(data.shape[0], y1)))
    if x1 <= x0 or y1 <= y0:
        z = np.zeros(data.shape[2], np.float32)
        return z, z
    region = data[y0:y1, x0:x1, :].reshape(-1, data.shape[2])
    return region.mean(axis=0), region.std(axis=0)


def spectral_angle(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.arccos(np.clip(a @ b / denom, -1.0, 1.0)))


BAND_MATH_PRESETS = {
    "정규화 차 (NDI)": "(b{i} - b{j}) / (b{i} + b{j} + 1e-6)",
    "비율": "b{i} / (b{j} + 1e-6)",
    "차": "b{i} - b{j}",
    "합": "b{i} + b{j}",
    "전체 평균": "mean",
    "전체 표준편차": "std",
}


def band_math(data: np.ndarray, expr: str) -> np.ndarray:
    """b0..bN 변수와 몇 가지 numpy 함수만 허용하는 밴드 연산. 결과는 (H, W) float32."""
    import re

    expr = expr.strip()
    if not expr:
        raise ValueError("식이 비어 있습니다")
    if expr == "mean":
        return data.mean(axis=2).astype(np.float32)
    if expr == "std":
        return data.std(axis=2).astype(np.float32)
    if not re.fullmatch(r"[0-9a-zA-Z_+\-*/().,\s<>=!&|^%]+", expr):
        raise ValueError("허용되지 않는 문자가 있습니다")
    names = {}
    for m in set(re.findall(r"\bb(\d+)\b", expr)):
        i = int(m)
        if i >= data.shape[2]:
            raise ValueError(f"b{i}: 밴드 번호 범위 초과 (0~{data.shape[2] - 1})")
        names[f"b{i}"] = data[:, :, i]
    safe = {k: getattr(np, k) for k in ("sqrt", "log", "log10", "exp", "abs", "clip", "minimum", "maximum", "where", "sin", "cos", "tanh")}
    safe.update(names)
    try:
        out = eval(expr, {"__builtins__": {}}, safe)  # noqa: S307 - 허용 문자/이름을 위에서 제한
    except ZeroDivisionError:
        raise ValueError("0으로 나눔")
    except NameError as e:
        raise ValueError(f"알 수 없는 이름: {e}")
    out = np.asarray(out, dtype=np.float32)
    if out.shape != data.shape[:2]:
        out = np.broadcast_to(out, data.shape[:2]).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
