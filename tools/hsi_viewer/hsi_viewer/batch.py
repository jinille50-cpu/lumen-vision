"""폴더 단위 배치 처리: 파일마다 로드 → False color → SAM3 탐지 → 콜백(클러스터링/저장/레퍼런스 생성)."""
from __future__ import annotations

import os
import time
import traceback

import numpy as np
from PySide6 import QtCore

from .io_utils import SUPPORTED_EXTS, load_cube
from .processing import CompositeCache, false_color
from .sam_engine import SamEngine, SegmentationResult


def list_cubes(folder: str, exts: tuple[str, ...] = (".tif", ".tiff", ".npy", ".mat", ".h5", ".hdf5", ".hdr")) -> list[str]:
    out = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and name.lower().endswith(tuple(e for e in exts if e in SUPPORTED_EXTS)):
            out.append(p)
    return out


def render_overlay_png(rgb: np.ndarray, label_map: np.ndarray, color_of, alpha: float = 0.55,
                       texts: list[tuple[float, float, str]] | None = None) -> "PIL.Image.Image":
    """캔버스 없이 numpy/PIL 로 오버레이 이미지를 만든다. color_of(piece_id) -> (r,g,b) 또는 None."""
    from PIL import Image, ImageDraw

    out = rgb.astype(np.float32).copy()
    ids = np.unique(label_map)
    for pid in ids:
        if pid <= 0:
            continue
        col = color_of(int(pid))
        if col is None:
            continue
        m = label_map == pid
        out[m] = out[m] * (1 - alpha) + np.asarray(col, np.float32) * alpha
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if texts:
        d = ImageDraw.Draw(img)
        for x, y, t in texts:
            w = d.textlength(t) if hasattr(d, "textlength") else 7 * len(t)
            d.rectangle((x - w / 2 - 2, y - 7, x + w / 2 + 2, y + 7), fill=(0, 0, 0, 160))
            d.text((x - w / 2, y - 6), t, fill=(255, 255, 255))
    return img


class BatchRunner(QtCore.QObject):
    """SamEngine 을 공유해 파일 목록을 순서대로 처리한다. 콜백은 메인 스레드에서 실행."""

    progress = QtCore.Signal(int, int, str)      # done, total, 현재 파일
    log = QtCore.Signal(str)
    finished = QtCore.Signal(bool)               # True = 정상 완료, False = 취소/오류 중단

    def __init__(self, engine: SamEngine, files: list[str], display: dict, rgb_bands: tuple, sam: dict,
                 on_file, parent=None):
        """on_file(path, cube, rgb, seg: SegmentationResult) -> str (로그 한 줄)"""
        super().__init__(parent)
        self.engine = engine
        self.files = list(files)
        self.display = display
        self.rgb_bands = tuple(rgb_bands)
        self.sam = sam
        self.on_file = on_file
        self.idx = 0
        self.cancelled = False
        self.running = False
        self._cache = CompositeCache()
        self._current = None
        self.t0 = 0.0

    def start(self):
        if self.running:
            return
        self.running = True
        self.cancelled = False
        self.idx = 0
        self.t0 = time.time()
        self.engine.finished.connect(self._on_seg)
        self.engine.failed.connect(self._on_fail)
        self._next()

    def cancel(self):
        self.cancelled = True

    def _done(self, ok: bool):
        try:
            self.engine.finished.disconnect(self._on_seg)
            self.engine.failed.disconnect(self._on_fail)
        except Exception:
            pass
        self.running = False
        self.finished.emit(ok)

    def _next(self):
        if self.cancelled:
            self.log.emit("취소됨")
            self._done(False)
            return
        if self.idx >= len(self.files):
            self.log.emit(f"완료: {len(self.files)}개 파일, {time.time() - self.t0:.1f}s")
            self._done(True)
            return
        path = self.files[self.idx]
        self.progress.emit(self.idx, len(self.files), os.path.basename(path))
        try:
            cube = load_cube(path)
            n = cube.n_bands
            bands = tuple(min(b, n - 1) for b in self.rgb_bands)
            rgb = false_color(cube.data, bands, self.display["low"], self.display["high"], self.display["invert"], self._cache)
            self._current = (path, cube, rgb)
        except Exception as e:  # noqa: BLE001
            self.log.emit(f"✗ {os.path.basename(path)}: 로드 실패 - {e}")
            self.idx += 1
            QtCore.QTimer.singleShot(0, self._next)
            return
        ok = self.engine.segment(
            rgb=rgb, key=("batch", path), prompts=self.sam["prompts"], bboxes=None, inside_only=False,
            conf=self.sam["conf"], nms_iou=self.sam["nms_iou"], min_area=self.sam["min_area"],
            overlap_iou=self.sam["overlap_iou"], small_first=self.sam.get("small_first", False),
        )
        if not ok:
            self.log.emit("SAM3 엔진이 사용 중입니다. 잠시 후 다시 시도하세요.")
            self._done(False)

    def _on_seg(self, seg: SegmentationResult):
        path, cube, rgb = self._current
        try:
            msg = self.on_file(path, cube, rgb, seg)
            self.log.emit(f"✓ {os.path.basename(path)}: {msg}")
        except Exception as e:  # noqa: BLE001
            self.log.emit(f"✗ {os.path.basename(path)}: {e}\n{traceback.format_exc(limit=2)}")
        self.idx += 1
        QtCore.QTimer.singleShot(0, self._next)

    def _on_fail(self, msg: str):
        path = self._current[0] if self._current else "?"
        self.log.emit(f"✗ {os.path.basename(path)}: {msg.splitlines()[0]}")
        self.idx += 1
        QtCore.QTimer.singleShot(0, self._next)
