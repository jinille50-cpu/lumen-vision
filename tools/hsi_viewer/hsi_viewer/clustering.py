"""Spectral feature extraction per segmented piece and clustering into K groups."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .processing import snv
from .sam_engine import Piece

METHODS = ("KMeans", "Agglomerative (Ward)", "Gaussian Mixture", "Spectral", "By prompt (no clustering)")


@dataclass
class ClusterResult:
    k: int
    method: str
    labels: np.ndarray                      # (N_pieces,) cluster index per piece
    features: np.ndarray                    # (N_pieces, B) per-piece feature spectra
    cluster_map: np.ndarray                 # (H, W) int32, -1 = background
    means: np.ndarray                       # (k, B) pixel-level mean spectrum (normalised space)
    stds: np.ndarray                        # (k, B)
    counts_px: np.ndarray                   # (k,)
    counts_pieces: np.ndarray               # (k,)
    normalised: bool
    bg_norm: bool
    inertia: float | None = None
    silhouette: float | None = None
    extra: dict = field(default_factory=dict)
    # 표시용 통계 (클러스터링 정규화와 무관): {"raw": (feats, means, stds), "snv": (...)}
    display: dict = field(default_factory=dict)


def normalise_pixels(px: np.ndarray, background: np.ndarray | None, use_snv: bool) -> np.ndarray:
    px = np.asarray(px, np.float32)
    if background is not None:
        px = px / (background + 1e-9)
    if use_snv:
        px = snv(px)
    return px


def estimate_background(cube: np.ndarray, label_map: np.ndarray, max_samples: int = 200_000) -> np.ndarray:
    """Median spectrum of all non-object pixels (sub-sampled for speed)."""
    bg_mask = label_map == 0
    n = int(bg_mask.sum())
    if n == 0:
        return np.ones(cube.shape[2], np.float32)
    px = cube[bg_mask]
    if n > max_samples:
        px = px[:: int(np.ceil(n / max_samples))]
    return np.median(px, axis=0).astype(np.float32)


def display_stats(cube: np.ndarray, pieces: list[Piece], cluster_map: np.ndarray, k: int,
                  stat: str = "median") -> dict:
    """그래프/내보내기용 통계. 'raw' = 원 세기, 'snv' = 픽셀별 SNV (배경 정규화 없음).
    클러스터링에 쓴 특징(배경 정규화+SNV)은 거의 평평한 스펙트럼의 미세한 차이를 크게 증폭하므로
    사람이 보는 그래프는 포인트 스펙트럼과 같은 공간으로 그린다."""
    b = cube.shape[2]
    out = {}
    for key, use_snv in (("raw", False), ("snv", True)):
        feats = np.zeros((len(pieces), b), np.float32)
        for i, q in enumerate(pieces):
            px = normalise_pixels(cube[q.mask], None, use_snv)
            if px.shape[0]:
                feats[i] = px.mean(axis=0) if stat == "mean" else np.median(px, axis=0)
        means = np.zeros((k, b), np.float32)
        stds = np.zeros((k, b), np.float32)
        for c in range(k):
            m = cluster_map == c
            if m.any():
                px = normalise_pixels(cube[m], None, use_snv)
                means[c], stds[c] = px.mean(axis=0), px.std(axis=0)
        out[key] = (feats, means, stds)
    return out


def piece_features(
    cube: np.ndarray,
    pieces: list[Piece],
    label_map: np.ndarray,
    use_snv: bool = True,
    bg_norm: bool = True,
    stat: str = "median",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Returns (features (N, B), background (B,) or None)."""
    background = estimate_background(cube, label_map) if bg_norm else None
    feats = []
    for q in pieces:
        px = normalise_pixels(cube[q.mask], background, use_snv)
        if stat == "mean":
            feats.append(px.mean(axis=0))
        else:
            feats.append(np.median(px, axis=0))
    if not feats:
        return np.zeros((0, cube.shape[2]), np.float32), background
    return np.vstack(feats).astype(np.float32), background


def run_clustering(features: np.ndarray, k: int, method: str, random_state: int = 0,
                   pieces: list[Piece] | None = None) -> tuple[np.ndarray, dict]:
    n = features.shape[0]
    info: dict = {}
    if n == 0:
        return np.zeros(0, np.int32), info
    if method.startswith("By prompt"):
        if pieces is None:
            raise ValueError("pieces required for prompt grouping")
        idx = sorted({q.prompt_idx for q in pieces})
        remap = {p: i for i, p in enumerate(idx)}
        labels = np.array([remap[q.prompt_idx] for q in pieces], np.int32)
        info["k_effective"] = len(idx)
        return labels, info

    k = int(max(1, min(k, n)))
    info["k_effective"] = k
    if k == 1:
        return np.zeros(n, np.int32), info

    if method.startswith("KMeans"):
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(features)
        labels = km.labels_.astype(np.int32)
        info["inertia"] = float(km.inertia_)
    elif method.startswith("Agglomerative"):
        from sklearn.cluster import AgglomerativeClustering

        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(features).astype(np.int32)
    elif method.startswith("Gaussian"):
        from sklearn.mixture import GaussianMixture

        gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=random_state, n_init=3)
        labels = gm.fit_predict(features).astype(np.int32)
    elif method.startswith("Spectral"):
        from sklearn.cluster import SpectralClustering

        nn = max(2, min(10, n - 1))
        labels = SpectralClustering(
            n_clusters=k, affinity="nearest_neighbors", n_neighbors=nn, random_state=random_state, assign_labels="discretize"
        ).fit_predict(features).astype(np.int32)
    else:
        raise ValueError(f"Unknown method {method}")

    # Relabel clusters by descending size for stable colours
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts, kind="stable")
    remap = np.empty(k, np.int32)
    remap[order] = np.arange(k)
    labels = remap[labels]

    if 1 < k < n:
        try:
            from sklearn.metrics import silhouette_score

            info["silhouette"] = float(silhouette_score(features, labels))
        except Exception:
            pass
    return labels, info


def build_result(
    cube: np.ndarray,
    pieces: list[Piece],
    label_map: np.ndarray,
    labels: np.ndarray,
    k: int,
    method: str,
    feats: np.ndarray,
    use_snv: bool,
    bg_norm: bool,
    background: np.ndarray | None,
    inertia: float | None = None,
    silhouette: float | None = None,
    extra: dict | None = None,
) -> ClusterResult:
    """조각별 라벨이 정해진 뒤 클러스터 맵과 픽셀 통계를 만든다 (수동 편집 후 재계산에도 사용)."""
    labels = np.asarray(labels, np.int32)
    k = int(max(k, (labels.max() + 1) if labels.size else 1))
    for q, c in zip(pieces, labels):
        q.cluster = int(c)
    cluster_map = np.full(label_map.shape, -1, np.int32)
    for q in pieces:
        cluster_map[q.mask] = q.cluster
    b = cube.shape[2]
    means = np.zeros((k, b), np.float32)
    stds = np.zeros((k, b), np.float32)
    counts_px = np.zeros(k, np.int64)
    for c in range(k):
        m = cluster_map == c
        n = int(m.sum())
        counts_px[c] = n
        if n:
            px = normalise_pixels(cube[m], background, use_snv)
            means[c] = px.mean(axis=0)
            stds[c] = px.std(axis=0)
    counts_pieces = np.bincount(labels, minlength=k).astype(np.int64) if labels.size else np.zeros(k, np.int64)
    ex = {"background": background}
    if extra:
        ex.update(extra)
    return ClusterResult(k=k, method=method, labels=labels, features=feats, cluster_map=cluster_map, means=means,
                         stds=stds, counts_px=counts_px, counts_pieces=counts_pieces, normalised=use_snv,
                         bg_norm=bg_norm, inertia=inertia, silhouette=silhouette, extra=ex,
                         display=display_stats(cube, pieces, cluster_map, k, ex.get("stat", "median")))


def cluster_pieces(
    cube: np.ndarray,
    pieces: list[Piece],
    label_map: np.ndarray,
    k: int,
    method: str = "KMeans",
    use_snv: bool = True,
    bg_norm: bool = True,
    stat: str = "median",
    random_state: int = 0,
) -> ClusterResult:
    feats, background = piece_features(cube, pieces, label_map, use_snv, bg_norm, stat)
    labels, info = run_clustering(feats, k, method, random_state, pieces)
    k_eff = int(info.get("k_effective", k))
    return build_result(cube, pieces, label_map, labels, k_eff, method, feats, use_snv, bg_norm, background,
                        info.get("inertia"), info.get("silhouette"), {"stat": stat})


def relabel(cube: np.ndarray, result: ClusterResult, pieces: list[Piece], label_map: np.ndarray) -> ClusterResult:
    """pieces[i].cluster 를 수동으로 바꾼 뒤 통계를 다시 계산."""
    labels = np.array([max(0, q.cluster) for q in pieces], np.int32)
    k = int(labels.max() + 1) if labels.size else 1
    feats = result.features if result.features.shape[0] == len(pieces) else piece_features(
        cube, pieces, label_map, result.normalised, result.bg_norm)[0]
    sil = None
    if 1 < k < len(labels) and len(set(labels.tolist())) > 1:
        try:
            from sklearn.metrics import silhouette_score

            sil = float(silhouette_score(feats, labels))
        except Exception:
            pass
    method = result.method if result.method.endswith("(수정)") else result.method + " (수정)"
    return build_result(cube, pieces, label_map, labels, k, method, feats, result.normalised,
                        result.bg_norm, result.extra.get("background"), None, sil,
                        {"edited": True, "stat": result.extra.get("stat", "median")})


def suggest_k(features: np.ndarray, k_max: int = 10, random_state: int = 0) -> list[tuple[int, float, float]]:
    """K=2..k_max 에 대해 (k, 실루엣, inertia). 조각이 3개 미만이면 빈 리스트."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = features.shape[0]
    out = []
    for k in range(2, min(k_max, n - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(features)
        try:
            sil = float(silhouette_score(features, km.labels_))
        except Exception:
            sil = float("nan")
        out.append((k, sil, float(km.inertia_)))
    return out


def cluster_pixels(
    cube: np.ndarray,
    pieces: list[Piece],
    label_map: np.ndarray,
    k: int,
    use_snv: bool = True,
    bg_norm: bool = True,
    random_state: int = 0,
    max_fit_samples: int = 100_000,
) -> ClusterResult:
    """Per-pixel MiniBatchKMeans inside object masks; pieces get their majority cluster."""
    from sklearn.cluster import MiniBatchKMeans

    mask = label_map > 0
    background = estimate_background(cube, label_map) if bg_norm else None
    px = normalise_pixels(cube[mask], background, use_snv)
    k = int(max(1, k))
    cluster_map = np.full(label_map.shape, -1, np.int32)
    inertia = None
    if px.shape[0]:
        fit = px if px.shape[0] <= max_fit_samples else px[:: int(np.ceil(px.shape[0] / max_fit_samples))]
        k = min(k, fit.shape[0])
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=3, batch_size=4096).fit(fit)
        labels_px = km.predict(px).astype(np.int32)
        counts = np.bincount(labels_px, minlength=k)
        order = np.argsort(-counts, kind="stable")
        remap = np.empty(k, np.int32)
        remap[order] = np.arange(k)
        cluster_map[mask] = remap[labels_px]
        inertia = float(km.inertia_)

    return pixel_result(cube, pieces, label_map, cluster_map, k, "Pixel KMeans", use_snv, bg_norm, background,
                        inertia=inertia)


def pixel_result(
    cube: np.ndarray,
    pieces: list[Piece],
    label_map: np.ndarray,
    cluster_map: np.ndarray,
    k: int,
    method: str,
    use_snv: bool,
    bg_norm: bool,
    background: np.ndarray | None,
    inertia: float | None = None,
    extra: dict | None = None,
) -> ClusterResult:
    """픽셀 단위 결과(cluster_map)로 통계를 만든다. 객체의 cluster 는 안에서 가장 많은 클래스."""
    k = int(max(1, k))
    feats, _ = piece_features(cube, pieces, label_map, use_snv, bg_norm, "median")
    labels = np.zeros(len(pieces), np.int32)
    for i, q in enumerate(pieces):
        vals = cluster_map[q.mask]
        vals = vals[vals >= 0]
        q.cluster = int(np.bincount(vals, minlength=k).argmax()) if vals.size else 0
        labels[i] = q.cluster

    b = cube.shape[2]
    means = np.zeros((k, b), np.float32)
    stds = np.zeros((k, b), np.float32)
    counts_px = np.zeros(k, np.int64)
    for c in range(k):
        m = cluster_map == c
        n = int(m.sum())
        counts_px[c] = n
        if n:
            p = normalise_pixels(cube[m], background, use_snv)
            means[c], stds[c] = p.mean(axis=0), p.std(axis=0)
    ex = {"background": background}
    if extra:
        ex.update(extra)
    ex["pixel_level"] = True
    return ClusterResult(
        k=k,
        method=method,
        labels=labels,
        features=feats,
        cluster_map=cluster_map,
        means=means,
        stds=stds,
        counts_px=counts_px,
        counts_pieces=np.bincount(labels, minlength=k).astype(np.int64) if len(labels) else np.zeros(k, np.int64),
        normalised=use_snv,
        bg_norm=bg_norm,
        inertia=inertia,
        extra=ex,
        display=display_stats(cube, pieces, cluster_map, k),
    )


def restrict_pixel_result(cube: np.ndarray, result: ClusterResult, pieces: list[Piece],
                          label_map: np.ndarray) -> ClusterResult:
    """객체를 지운 뒤: 남은 객체 밖의 픽셀은 분류에서 빼고 통계만 다시 계산 (재분류는 하지 않음)."""
    cmap = result.cluster_map.copy()
    cmap[label_map == 0] = -1
    return pixel_result(cube, pieces, label_map, cmap, result.k, result.method, result.normalised, result.bg_norm,
                        result.extra.get("background"), result.inertia,
                        {k: v for k, v in result.extra.items() if k not in ("background", "pixel_level")})


def classify_pixels_by_roi(
    cube: np.ndarray,
    pieces: list[Piece],
    label_map: np.ndarray,
    rois: list[tuple[tuple[int, int, int, int], str]],
    use_snv: bool = True,
    bg_norm: bool = False,
    max_train_per_roi: int = 20_000,
) -> ClusterResult:
    """ROI = 클래스 예시. 객체 마스크 안의 모든 픽셀을 ROI 클래스 중 하나로 분류한다.

    rois: [((x0, y0, x1, y1), 클래스 이름), ...]. 같은 이름의 ROI는 한 클래스로 합친다.
    학습 픽셀은 ROI 중 객체 마스크 안쪽만 쓴다 (ROI가 객체 밖에만 있으면 ROI 전체).
    분류기: shrinkage LDA. 클래스 안 분산까지 고려해서, 평균 스펙트럼 하나와 비교하는 것보다
    모양이 비슷한 스펙트럼을 더 잘 가른다.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    names: list[str] = []
    for _, name in rois:
        if name not in names:
            names.append(name)
    if len(names) < 2:
        raise ValueError("클래스가 2개 이상 필요합니다 (서로 다른 이름의 ROI 2개 이상)")
    h, w = label_map.shape
    mask = label_map > 0
    background = estimate_background(cube, label_map) if bg_norm else None
    # ROI 하나하나를 내부 하위 클래스로 학습하고 예측 뒤 이름으로 합친다.
    # 같은 이름 안에 스펙트럼이 서로 다른 예시가 섞여도 각각을 따로 모델링하고,
    # 사전확률은 클래스(이름)마다 같게 둬서 작은 ROI가 큰 ROI에 묻히지 않게 한다.
    xs, ys, sub_cls = [], [], []
    rng = np.random.default_rng(0)
    for (x0, y0, x1, y1), name in rois:
        sel = np.zeros((h, w), bool)
        sel[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
        inside = sel & mask
        if inside.any():
            sel = inside
        px = cube[sel]
        if px.shape[0] < 2:
            raise ValueError(f"'{name}' ROI ({x0},{y0})의 픽셀이 너무 적습니다 ({px.shape[0]} px)")
        if px.shape[0] > max_train_per_roi:
            px = px[rng.choice(px.shape[0], max_train_per_roi, replace=False)]
        xs.append(normalise_pixels(px, background, use_snv))
        ys.append(np.full(px.shape[0], len(sub_cls), np.int32))
        sub_cls.append(names.index(name))
    sub_cls = np.asarray(sub_cls, np.int32)
    per_class = np.bincount(sub_cls, minlength=len(names)).astype(np.float64)
    priors = 1.0 / (len(names) * per_class[sub_cls])
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors / priors.sum())
    lda.fit(np.vstack(xs), np.concatenate(ys))
    train_px = np.bincount(sub_cls, weights=[y.size for y in ys], minlength=len(names)).astype(int).tolist()
    cluster_map = np.full((h, w), -1, np.int32)
    if mask.any():
        cluster_map[mask] = sub_cls[lda.predict(normalise_pixels(cube[mask], background, use_snv))]
    return pixel_result(cube, pieces, label_map, cluster_map, len(names), "ROI 분류 (LDA)", use_snv, bg_norm,
                        background, extra={"names": names, "train_px": train_px})
