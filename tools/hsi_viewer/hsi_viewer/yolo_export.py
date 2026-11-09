"""탐지 결과(마스크)를 YOLO 세그멘테이션 라벨(폴리곤)로 내보내기."""
from __future__ import annotations

import os

import numpy as np


def mask_to_polygons(mask: np.ndarray, min_points: int = 3, epsilon_frac: float = 0.002) -> list[np.ndarray]:
    """(H, W) bool 마스크 -> [(N, 2) int xy 폴리곤, ...] (외곽선, 큰 것부터)."""
    import cv2

    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if len(c) < min_points:
            continue
        eps = epsilon_frac * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, eps, True)
        if len(c) >= min_points:
            polys.append(c.reshape(-1, 2))
    return polys


def export_yolo(
    out_dir: str,
    stem: str,
    rgb: np.ndarray,
    pieces,
    class_of,                       # callable(piece) -> str | None  (None = 제외)
    class_names: list[str],
    largest_only: bool = True,
    save_image: bool = True,
) -> tuple[int, str]:
    """images/<stem>.png, labels/<stem>.txt, classes.txt, data.yaml 작성. (라벨 수, 라벨 경로) 반환."""
    from PIL import Image

    h, w = rgb.shape[:2]
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    if save_image:
        Image.fromarray(rgb).save(os.path.join(img_dir, f"{stem}.png"))

    lines = []
    for q in pieces:
        cname = class_of(q)
        if cname is None or cname not in class_names:
            continue
        cid = class_names.index(cname)
        polys = mask_to_polygons(q.mask)
        if largest_only:
            polys = polys[:1]
        for poly in polys:
            coords = []
            for x, y in poly:
                coords.append(f"{min(max(x / w, 0.0), 1.0):.6f}")
                coords.append(f"{min(max(y / h, 0.0), 1.0):.6f}")
            lines.append(f"{cid} " + " ".join(coords))
    lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(class_names) + "\n")
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\ntrain: images\nval: images\nnames:\n")
        for i, n in enumerate(class_names):
            f.write(f"  {i}: {n}\n")
    return len(lines), lbl_path
