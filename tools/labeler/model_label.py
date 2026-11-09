"""내 모델로 1차 라벨링: Segmentation(.pt/.onnx, _config.json 또는 ONNX 메타데이터) 또는 객체 인식(YOLO/RT-DETR .pt/.onnx).
모델 확인(probe)과 추론은 백그라운드 스레드에서 한다 (torch, ultralytics는 여기서 처음 불러온다).
결과를 파일로 쓰는 일과 auto_labels.json 기록은 app.py가 한다."""
import os
import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from tools.labeler import cube
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QPlainTextEdit, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

SKIP, ADD = -1, -2   # 클래스 매칭 칸의 특별 값: 넣지 않음, 새 클래스로 추가


# ── 모델 확인 ───────────────────────────────────────────────
def probe(path):
    """모델 종류와 클래스 이름을 알아낸다.
    돌려주는 값: {"kind": "seg"|"det", "names": [...], "desc": 한 줄 설명, "cfg": seg config, "task": det의 task}"""
    from tools.segmentation.utils.config_path import load_config
    cfg = load_config(path)
    if cfg and cfg.get("model_name"):   # Segmentation 도구가 저장한 config
        n = int(cfg.get("num_classes", 2))
        names = list(cfg.get("class_names") or [])[:n]
        names += [f"class{i}" for i in range(len(names), n)]
        desc = (f"Segmentation · {cfg['model_name']}" + (f"({cfg.get('encoder')})" if cfg.get("encoder") else "")
                + f" · 클래스 {n} · 입력 채널 {cfg.get('in_channels', 3)}" + (" · ONNX" if path.lower().endswith(".onnx") else ""))
        return {"kind": "seg", "names": names, "desc": desc, "cfg": cfg}
    from tools.yolo.yolo_module_wrap import load_detector   # YOLO 또는 RT-DETR
    model = load_detector(path)
    names = model.names
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)   # {번호: 이름} → 목록
    task = getattr(model, "task", "detect")
    desc = f"객체 인식 · {type(model).__name__} · {task} · 클래스 {len(names)}"
    return {"kind": "det", "names": names, "desc": desc, "task": task}


class ProbeThread(QThread):
    done = Signal(object)   # dict (그대로 넘기려고 object)
    failed = Signal(str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            self.done.emit(probe(self.path))
        except Exception as e:  # noqa: BLE001  -- 어떤 실패든 이유를 보여준다
            self.failed.emit(f"모델을 열지 못했습니다: {e}")


# ── 저장 ────────────────────────────────────────────────────
def write_mask_array(path, arr):
    """(H, W) uint8 배열(클래스 번호+1, 0은 배경)을 마스크 png로"""
    from PySide6.QtGui import QImage
    arr = np.ascontiguousarray(arr, np.uint8)
    h, w = arr.shape
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not QImage(arr.data, w, h, w, QImage.Format_Grayscale8).save(path):
        raise OSError(f"마스크를 쓸 수 없음: {path}")


def write_box_txt(path, boxes, w, h):
    """boxes: [(클래스, x1, y1, x2, y2)] 픽셀 → YOLO txt (정규화 중심/폭/높이)"""
    lines = [f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
             for c, x1, y1, x2, y2 in boxes]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


# ── 추론 ────────────────────────────────────────────────────
def _read_image(path, cfg):
    """Segmentation 입력: (H, W, C) 또는 (H, W) float32, 학습 때 고른 채널만. 큐브는 원래 값 그대로(밴드 전부)"""
    from PIL import Image
    from tools.segmentation.utils.pad_utils import select_channels
    if cube.is_cube_file(path):
        arr = cube.load(path).data   # (H, W, 밴드) float32
    else:
        arr = np.array(Image.open(path))
    in_ch = int(cfg.get("in_channels", 3))
    if arr.ndim == 3 and arr.shape[2] == 4 and in_ch == 3:
        arr = arr[..., :3]   # PNG 투명도(알파) 채널은 뺀다
    arr = select_channels(arr, cfg.get("channels"))
    if arr.ndim == 2 and in_ch == 3:
        arr = np.stack([arr] * 3, axis=-1)   # 흑백 이미지를 3채널 모델에 넣을 때
    return arr.astype(np.float32)


class InferThread(QThread):
    result = Signal(int, object)   # 이미지 번호, 결과 dict
    log = Signal(str)
    done = Signal(str)             # 끝난 이유

    def __init__(self, path, info, items, conf, as_mask, view, parent=None):
        """items: [(이미지 번호, 경로)], as_mask: 객체 인식 모델이 마스크도 내면(YOLO-seg) 마스크로 받을지,
        view: 큐브를 객체 인식 모델에 넣을 때 쓸 false color 보기 설정 (모델 config에 설정이 있으면 그것을 쓴다)"""
        super().__init__(parent)
        self.path, self.info, self.items, self.conf, self.as_mask = path, info, items, conf, as_mask
        self.view = view
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            if self.info["kind"] == "seg":
                self._run_seg()
            else:
                self._run_det()
            self.done.emit("멈춤" if self._stop else "끝")
        except Exception as e:  # noqa: BLE001
            import traceback
            self.log.emit(traceback.format_exc())
            self.done.emit(f"오류로 멈춤: {e}")

    def _run_det(self):
        from tools.yolo.yolo_module_wrap import load_detector
        from tools.yolo.work_state import find_config, load_config
        self.log.emit("모델 불러오는 중…")
        model = load_detector(self.path)
        ycfg = load_config(find_config(self.path)) or {}
        view = dict(self.view or {})
        ch, fc = ycfg.get("channel"), ycfg.get("false_color")
        if isinstance(ch, list) and len(ch) == 3 and isinstance(fc, dict):   # Object Detection에서 false color로 학습한 모델
            view = {"bands": ch, "lo": fc.get("lo", 1.0), "hi": fc.get("hi", 99.0), "single": None}
            self.log.emit(f"큐브는 모델 설정대로 false color: 밴드 {ch}, 범위 {view['lo']}~{view['hi']}%")
        for i, p in self.items:
            if self._stop:
                return
            src = p
            if cube.is_cube_file(p):   # 큐브는 RGB로 바꿔서 넣는다 (ultralytics는 numpy면 BGR로 받는다)
                try:
                    cb = cube.load(p)
                    src = np.ascontiguousarray(cube.view_rgb(cb.data, cube.fit_view(view, cb.n_bands))[..., ::-1])
                except Exception as e:  # noqa: BLE001
                    self.result.emit(i, {"error": f"큐브를 읽지 못함: {e}"})
                    continue
            r = model.predict(source=src, conf=self.conf, verbose=False, retina_masks=self.as_mask)[0]
            boxes = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                boxes = [(int(c), float(s), *map(float, b)) for c, s, b in
                         zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), xyxy)]
            masks = None
            if self.as_mask and getattr(r, "masks", None) is not None:
                masks = r.masks.data.cpu().numpy() > 0.5   # retina_masks=True면 원본 크기
            self.result.emit(i, {"boxes": boxes, "masks": masks, "shape": r.orig_shape})

    def _run_seg(self):
        import torch
        import torch.nn.functional as F
        from tools.segmentation.utils.pad_utils import get_pad_multiple, pad_tensor, unpad_tensor
        cfg = self.info["cfg"]
        name = cfg["model_name"]
        n_cls = int(cfg.get("num_classes", 2))
        in_ch = len(cfg["channels"]) if cfg.get("channels") else int(cfg.get("in_channels", 3))   # 고른 채널만 넣는다
        in_scale = float(cfg.get("input_scale", 1.0))
        multiple = get_pad_multiple(name) if name.startswith("DINO") else max(int(cfg.get("pad_multiple") or 32), 32)
        self.log.emit("모델 불러오는 중…")
        if self.path.lower().endswith(".onnx"):
            from tools.segmentation.utils.quantize_onnx import OnnxModel
            device = torch.device("cpu")
            model, is_tr = OnnxModel(self.path, providers=["CPUExecutionProvider"], threads=0), False
        else:
            from tools.segmentation.utils.model_builder import build_model, load_weights
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, is_tr = build_model(name, cfg.get("encoder", "resnet34"), int(cfg.get("in_channels", 3)), n_cls,
                                       pretrained=cfg.get("pretrained"), load_pretrained_weights=False)
            model = load_weights(model, self.path, device, log=self.log.emit).to(device).eval()
        self.log.emit(f"장치: {device}")
        with torch.no_grad():
            for i, p in self.items:
                if self._stop:
                    return
                try:
                    raw = _read_image(p, cfg)
                except Exception as e:  # noqa: BLE001
                    self.result.emit(i, {"error": f"이미지를 읽지 못함: {e}"})
                    continue
                n_ch = raw.shape[2] if raw.ndim == 3 else 1
                if n_ch != in_ch:   # 모델과 채널 수가 다른 이미지는 이 장만 실패로 두고 계속한다
                    self.result.emit(i, {"error": f"채널 수가 다름 (모델 {in_ch}, 이미지 {n_ch})"})
                    continue
                t = torch.from_numpy(raw) / in_scale
                t = t.permute(2, 0, 1) if t.ndim == 3 else t.unsqueeze(0)   # (H, W, C) → (C, H, W)
                t = t.unsqueeze(0).to(device)
                t_pad, h, w = pad_tensor(t, multiple=multiple)
                try:
                    out = model(t_pad)
                except Exception as e:  # noqa: BLE001  -- 크기 등 이 이미지만의 문제면 다음 장으로
                    self.result.emit(i, {"error": f"추론 실패: {str(e).splitlines()[0][:150]}"})
                    continue
                logits = unpad_tensor(out.logits if is_tr else out, h, w)
                if logits.shape[-2:] != (h, w):
                    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
                prob = torch.softmax(logits, 1)[0]
                conf, label = prob.max(0)   # 픽셀마다 가장 높은 확률과 그 클래스
                label = label.cpu().numpy().astype(np.uint8)
                conf = conf.cpu().numpy()
                cls_conf = {int(c): float(conf[label == c].mean()) for c in np.unique(label)}   # 클래스별 평균 확률
                self.result.emit(i, {"label": label, "cls_conf": cls_conf, "shape": (h, w)})


# ── 창 ─────────────────────────────────────────────────────
class ModelLabelDialog(QDialog):
    start_requested = Signal(dict)   # {"path", "info", "map": [모델 클래스별 Labeler 번호 또는 SKIP/ADD], "conf", "skip"}
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("모델로 1차 라벨링")
        self.setModal(True)   # 도는 동안 본 창의 라벨을 고치지 못하게
        self.resize(660, 620)
        self.info = None
        self.classes = []
        self._probe = None
        v = QVBoxLayout(self)
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("학습한 모델 (.pt, .onnx)")
        self.path_edit.editingFinished.connect(lambda: self.set_model(self.path_edit.text().strip()))
        b = QPushButton("열기")
        b.clicked.connect(self._browse)
        row.addWidget(self.path_edit, 1)
        row.addWidget(b)
        v.addLayout(row)
        self.desc = QLabel("")
        self.desc.setObjectName("Hint")
        self.desc.setWordWrap(True)
        v.addWidget(self.desc)
        v.addWidget(QLabel("모델 클래스 → Labeler 클래스"))
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["모델 클래스", "Labeler 클래스"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        v.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("점수 하한 (객체 인식)"))
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.05, 0.95)
        self.conf.setSingleStep(0.05)
        self.conf.setDecimals(2)
        self.conf.setValue(0.25)
        row.addWidget(self.conf)
        row.addStretch(1)
        v.addLayout(row)
        self.skip = QCheckBox("라벨이 이미 있는 이미지는 건너뛰기")
        self.skip.setChecked(True)
        v.addWidget(self.skip)
        self.target = QLabel("")
        self.target.setObjectName("Hint")
        v.addWidget(self.target)
        self.bar = QProgressBar()
        v.addWidget(self.bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        v.addWidget(self.log, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.start_btn = QPushButton("시작")
        self.start_btn.setObjectName("Primary")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("멈춤")
        self.stop_btn.clicked.connect(self.stop_requested)
        self.close_btn = QPushButton("닫기")
        self.close_btn.clicked.connect(self.hide)
        for b in (self.start_btn, self.stop_btn, self.close_btn):
            row.addWidget(b)
        v.addLayout(row)
        self.set_running(False)

    def setup(self, classes, mode_seg, target_text, model_path):
        self.classes = list(classes)
        self.mode_seg = mode_seg
        self.target.setText(target_text)
        self.bar.setValue(0)
        self.log.clear()
        if model_path and model_path != self.path_edit.text():
            self.set_model(model_path)
        elif self.info:
            self._fill_map()   # Labeler 클래스 목록이 바뀌었을 수 있으므로 다시 채운다
        self.set_running(False)

    def _browse(self):
        start = os.path.dirname(self.path_edit.text()) if self.path_edit.text() else ""
        path, _ = QFileDialog.getOpenFileName(self, "모델 파일", start, "모델 (*.pt *.pth *.onnx)")
        if path:
            self.set_model(path)

    def set_model(self, path):
        if not path or self._probe is not None:
            return
        self.path_edit.setText(path)
        self.info = None
        self.table.setRowCount(0)
        if not os.path.isfile(path):
            self.desc.setText("모델 파일이 없습니다")
            self.set_running(False)
            return
        self.desc.setText("모델 확인 중… (처음엔 torch/ultralytics를 불러와 몇 초 걸림)")
        self._probe = ProbeThread(path, self)
        self._probe.done.connect(self._probed)
        self._probe.failed.connect(lambda m: self.desc.setText(m))
        self._probe.finished.connect(self._probe_end)
        self.set_running(False)
        self._probe.start()

    def _probe_end(self):
        self._probe = None
        self.set_running(False)

    def _probed(self, info):
        self.info = info
        self.desc.setText(info["desc"])
        self._fill_map()

    def _fill_map(self):
        names = self.info["names"]
        self.table.setRowCount(len(names))
        low = [c.lower() for c in self.classes]
        for r, n in enumerate(names):
            self.table.setItem(r, 0, QTableWidgetItem(f"{r}  {n}"))
            cb = QComboBox()
            cb.addItem("넣지 않음", SKIP)
            for i, c in enumerate(self.classes):
                cb.addItem(f"{i + 1 if self.mode_seg else i}  {c}", i)   # Segmentation 모드는 배경 0, 클래스 1부터
            cb.addItem(f"새 클래스로 추가 ({n})", ADD)
            if str(n).lower() in low:
                cb.setCurrentIndex(low.index(str(n).lower()) + 1)   # 이름이 같은 클래스
            elif self.info["kind"] == "seg" and r == 0:
                cb.setCurrentIndex(0)   # Segmentation 0번은 배경
            else:
                cb.setCurrentIndex(cb.count() - 1)
            self.table.setCellWidget(r, 1, cb)

    def set_running(self, on):
        ready = self.info is not None and self._probe is None
        self.start_btn.setEnabled(not on and ready)
        self.stop_btn.setEnabled(on)
        self.close_btn.setEnabled(not on)
        for w in (self.table, self.conf, self.skip, self.path_edit):
            w.setEnabled(not on)

    def progress(self, i, n, line):
        self.bar.setMaximum(max(n, 1))
        self.bar.setValue(i)
        if line:
            self.log.appendPlainText(line)

    def reject(self):   # Esc/창 닫기: 도는 중에는 닫지 않는다
        if self.close_btn.isEnabled():
            super().reject()

    def _start(self):
        mapping = [self.table.cellWidget(r, 1).currentData() for r in range(self.table.rowCount())]
        self.start_requested.emit({"path": self.path_edit.text(), "info": self.info, "map": mapping,
                                   "conf": self.conf.value(), "skip": self.skip.isChecked(), "t0": time.time()})
