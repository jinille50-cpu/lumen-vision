"""SAM3 engine facade: a worker *process* (see sam_server.py) driven from a QThread.

The QThread only does blocking pipe I/O (which releases the GIL), so the GUI stays
responsive even while the 3.4 GB model is being loaded in the child process.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
from PySide6 import QtCore

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sam3.pt")


@dataclass
class Piece:
    id: int                       # 1-based
    mask: np.ndarray              # (H, W) bool - disjoint from other pieces
    bbox: tuple[int, int, int, int]
    score: float
    prompt: str                   # which text prompt produced it
    prompt_idx: int
    area: int = 0
    centroid: tuple[float, float] = (0.0, 0.0)
    cluster: int = -1

    def recompute(self):
        self.area = int(self.mask.sum())
        if self.area:
            ys, xs = np.nonzero(self.mask)
            self.centroid = (float(xs.mean()), float(ys.mean()))


@dataclass
class SegmentationResult:
    pieces: list[Piece]
    label_map: np.ndarray         # (H, W) int32, 0 = background, i = piece id
    prompts: list[str]
    elapsed: float = 0.0
    raw_count: int = 0
    params: dict = field(default_factory=dict)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter / (np.logical_or(a, b).sum() + 1e-9))


def postprocess(raw: dict, prompts: list[str], min_area: int, overlap: float, max_pieces: int = 500,
                inside_boxes: list | None = None, small_first: bool = False) -> SegmentationResult:
    """Unpack the server reply into disjoint, filtered Piece objects + a label map.

    small_first: 겹치는 픽셀을 누가 가질지. False(기본) = 점수 높은 마스크가 먼저 가져감.
    True = 작은 마스크가 가져감 → 큰 객체 안에 있는 부분(뚜껑 등)이 따로 남고 큰 객체는 그만큼 빠진다.
    """
    h, w = raw["shape"]
    n = len(raw["scores"])
    label_map = np.zeros((h, w), np.int32)
    pieces: list[Piece] = []
    if n:
        masks = np.unpackbits(raw["masks"], axis=1, count=h * w).reshape(n, h, w).astype(bool)
        scores, cls, boxes, names = raw["scores"], raw["cls"], raw["boxes"], raw["names"]
        if list(names) == ["visual"]:
            names = ["박스 예시"]
        kept: list[Piece] = []
        for i in np.argsort(-scores):
            m = masks[i]
            if m.sum() < min_area:
                continue
            if inside_boxes:
                bx = boxes[i]
                cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
                if not any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in inside_boxes):
                    continue
            if any(mask_iou(m, q.mask) >= overlap for q in kept):
                continue
            ci = int(cls[i])
            kept.append(Piece(id=0, mask=m, bbox=tuple(int(v) for v in boxes[i]), score=float(scores[i]),
                              prompt=str(names[ci]) if ci < len(names) else str(ci), prompt_idx=ci))
            if len(kept) >= max_pieces:
                break
        if small_first:
            # 작은 마스크가 겹치는 픽셀을 가짐: 큰 것부터 칠하고 작은 것을 나중에 덮는다
            owner = np.zeros((h, w), np.int32)
            for j in sorted(range(len(kept)), key=lambda j: -int(kept[j].mask.sum())):
                owner[kept[j].mask] = j + 1
            for j, q in enumerate(kept):     # ids stay in score order
                q.mask = owner == j + 1
        for q in kept:                       # make disjoint, high score first (기본)
            q.mask &= label_map == 0
            q.recompute()
            if q.area >= min_area:
                q.id = len(pieces) + 1
                label_map[q.mask] = q.id
                pieces.append(q)
    return SegmentationResult(pieces=pieces, label_map=label_map, prompts=prompts,
                              elapsed=float(raw.get("elapsed", 0.0)), raw_count=n)


class _PipeThread(QtCore.QThread):
    """Blocks on the pipe and re-emits server events as Qt signals."""

    event = QtCore.Signal(object)

    def __init__(self, conn, parent=None):
        super().__init__(parent)
        self.conn = conn
        self._stop = False

    def run(self):
        while not self._stop:
            try:
                if self.conn.poll(0.2):
                    self.event.emit(self.conn.recv())
            except (EOFError, OSError):
                self.event.emit({"ev": "died"})
                break
            except Exception as e:  # noqa: BLE001
                self.event.emit({"ev": "error", "error": f"{e}\n{traceback.format_exc()}"})

    def stop(self):
        self._stop = True


class SamEngine(QtCore.QObject):
    status = QtCore.Signal(str)
    progress = QtCore.Signal(int)
    model_ready = QtCore.Signal(bool)
    finished = QtCore.Signal(object)       # SegmentationResult
    failed = QtCore.Signal(str)

    def __init__(self, model_path: str = DEFAULT_MODEL, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.busy = False
        self.ready = False
        self.proc: mp.Process | None = None
        self.conn = None
        self.reader: _PipeThread | None = None
        self._job: dict | None = None
        self._t0 = 0.0

    # ------------------------------------------------------------------ process mgmt
    @property
    def model_exists(self) -> bool:
        return os.path.isfile(self.model_path)

    def _ensure_process(self) -> bool:
        if self.proc is not None and self.proc.is_alive():
            return True
        try:
            from . import sam_server

            ctx = mp.get_context("spawn")
            parent_conn, child_conn = ctx.Pipe()
            self.proc = ctx.Process(target=sam_server._entry, args=(child_conn, self.model_path),
                                    name="SAM3Server", daemon=True)
            self.proc.start()
            child_conn.close()
            self.conn = parent_conn
            self.reader = _PipeThread(parent_conn, self)
            self.reader.event.connect(self._on_event)
            self.reader.start()
            self.ready = False
            return True
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"SAM3 프로세스를 시작하지 못했습니다: {e}\n{traceback.format_exc()}")
            return False

    def shutdown(self):
        try:
            if self.conn is not None:
                self.conn.send({"cmd": "quit"})
        except Exception:
            pass
        if self.reader is not None:
            self.reader.stop()
            self.reader.wait(1000)
        if self.proc is not None and self.proc.is_alive():
            self.proc.join(2.0)
            if self.proc.is_alive():
                self.proc.terminate()

    # ------------------------------------------------------------------ API
    def set_model_path(self, path: str) -> bool:
        """sam3.pt 위치를 바꾼다. 이미 다른 모델을 올린 프로세스는 닫고 다음에 다시 로딩. 작업 중이면 거절."""
        if path == self.model_path:
            return True
        if self.busy:
            return False
        self.shutdown()
        self.proc = self.conn = self.reader = None
        self.ready = False
        self.model_path = path
        return True

    def load_model(self):
        if self.busy or not self._ensure_process():
            return
        if self.ready:
            self.model_ready.emit(True)
            return
        self.busy = True
        self.progress.emit(-1)
        self.conn.send({"cmd": "load", "model": self.model_path})

    def segment(self, **job) -> bool:
        """job: rgb, key, prompts, conf, nms_iou, min_area, overlap_iou (점/박스 지정이면 points 또는 box)"""
        if self.busy:
            self.status.emit("SAM3 작업 중입니다…")
            return False
        if not self._ensure_process():
            return False
        self._job = job
        self.busy = True
        self._t0 = time.time()
        self.progress.emit(-1)
        if not self.ready:
            self.conn.send({"cmd": "load", "model": self.model_path})
            return True
        self._send_segment()
        return True

    def _send_segment(self):
        job = self._job
        self.conn.send({
            "cmd": "segment",
            "rgb": np.ascontiguousarray(job["rgb"]),
            "key": job.get("key"),
            "prompts": list(job.get("prompts") or []),
            "bboxes": [list(map(float, b)) for b in (job.get("bboxes") or [])],
            "points": job.get("points"),
            "point_labels": job.get("point_labels"),
            "box": job.get("box"),
            "conf": float(job.get("conf", 0.2)),
            "nms_iou": float(job.get("nms_iou", 0.7)),
        })

    # ------------------------------------------------------------------ events
    def _on_event(self, msg: dict):
        ev = msg.get("ev")
        if ev == "status":
            self.status.emit(msg.get("text", ""))
        elif ev == "ready":
            ok = bool(msg.get("ok"))
            self.ready = ok
            if not ok:
                self.busy = False
                self._job = None
                self.progress.emit(100)
                self.model_ready.emit(False)
                self.failed.emit("SAM3 모델 로딩 실패: " + str(msg.get("error")))
                return
            self.model_ready.emit(True)
            if self._job is not None:
                self._send_segment()
            else:
                self.busy = False
                self.progress.emit(100)
        elif ev == "result":
            job = self._job or {}
            self._job = None
            self.busy = False
            self.progress.emit(100)
            try:
                res = postprocess(msg, list(job.get("prompts", [])), int(job.get("min_area", 30)),
                                  float(job.get("overlap_iou", 0.7)),
                                  inside_boxes=job.get("bboxes") if job.get("inside_only") else None,
                                  small_first=bool(job.get("small_first", False)))
                res.elapsed = time.time() - self._t0
                res.params = {k: v for k, v in job.items() if k != "rgb"}
                self.status.emit(f"SAM3: 객체 {len(res.pieces)}개 ({res.elapsed:.2f}s)")
                self.finished.emit(res)
            except Exception as e:  # noqa: BLE001
                self.failed.emit(f"SAM3 결과 처리 실패: {e}\n{traceback.format_exc()}")
        elif ev == "error":
            self._job = None
            self.busy = False
            self.progress.emit(100)
            self.failed.emit("SAM3 오류: " + str(msg.get("error")))
        elif ev == "died":
            self.busy = False
            self.ready = False
            self._job = None
            self.progress.emit(100)
            self.failed.emit("SAM3 프로세스가 종료되었습니다. 다시 실행하면 재시작합니다.")
