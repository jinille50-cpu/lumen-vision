import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QLineEdit, QScrollArea,
    QComboBox, QSlider, QListWidget, QListWidgetItem, QProgressBar,
    QTextEdit, QSplitter, QSpinBox, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage

# "tiff" 선택 시 .tif, "jpg" 선택 시 .jpeg 파일도 함께 매칭됨
IMG_EXTS = ["png", "jpg", "bmp", "tiff"]


def _ext_suffixes(ext):
    """확장자 선택값 → 실제 매칭할 접미사 튜플 (tiff는 .tif, jpg는 .jpeg 포함)"""
    e = ext.lower()
    if e == "tiff":
        return (".tiff", ".tif")
    if e == "jpg":
        return (".jpg", ".jpeg")
    return (f".{e}",)


# ═══════════════════════════════════════════════════════════════
#  추론 Worker — YOLO.predict()
# ═══════════════════════════════════════════════════════════════
from tools.yolo.work_state import pick_open, pick_dir, load_config


class InferWorker(QThread):
    progress_update = Signal(int, int)
    result_ready    = Signal(int, object, str)  # idx, RGB np(plotted), filename
    log_msg         = Signal(str)
    finished        = Signal(str)               # out_dir

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            import ultralytics  # noqa: F401  설치 여부만 확인 (모델은 load_detector가 연다)
            import torch
            import numpy as np
            from PIL import Image
        except ImportError as e:
            self.log_msg.emit(f"[오류] 필수 패키지 미설치: {e}")
            self.finished.emit(""); return

        from tools.yolo.yolo_module_wrap import extract_channel, false_color_image, load_detector
        from tools.yolo.work_state import find_config, load_config

        cfg = self.config
        out_dir = cfg.get("out_dir", "./YOLO_infer")
        channel = cfg.get("channel")
        # 학습 때 false color(밴드 3개 → RGB)로 학습한 모델이면 TIFF 입력도 같은 밴드, 같은 하한/상한으로 바꾼다
        mcfg = load_config(find_config(cfg["model_path"])) or {}
        fc, fc_bands = mcfg.get("false_color"), mcfg.get("channel")
        if not (fc and isinstance(fc_bands, list) and len(fc_bands) == 3):
            fc = None
        os.makedirs(out_dir, exist_ok=True)
        try:
            device = 0 if torch.cuda.is_available() else "cpu"
            self.log_msg.emit(f"[추론 시작] 디바이스: {'GPU' if device == 0 else 'CPU'}")
            model = load_detector(cfg["model_path"])   # YOLO 또는 RT-DETR (.pt/.onnx)

            files = cfg["img_files"]
            total = len(files)
            self.log_msg.emit(f"  대상 {total}장  |  conf={cfg['conf']}  iou={cfg['iou']}"
                              + (f"  |  채널={channel}" if channel is not None and not fc else ""))
            if fc:
                self.log_msg.emit(f"  모델 config에 따라 false color 적용: 밴드 {fc_bands}, 하한 {fc['lo']}% / 상한 {fc['hi']}%")

            for idx, ip in enumerate(files):
                if self._stop:
                    self.log_msg.emit("[중단]"); break
                src = ip
                if fc and ip.lower().endswith((".tiff", ".tif")):
                    # [..., ::-1]: RGB → BGR (Ultralytics는 numpy 입력을 cv2처럼 BGR로 본다, 학습 때 PNG를 읽은 것과 같게)
                    src = false_color_image(ip, fc_bands, fc)[..., ::-1].copy()
                elif channel is not None and ip.lower().endswith((".tiff", ".tif")):
                    # 지정 채널만 추출해 3채널 ndarray로 추론
                    ch = extract_channel(ip, channel)
                    src = np.stack([ch] * 3, axis=-1)
                results = model.predict(
                    src, conf=cfg["conf"], iou=cfg["iou"],
                    imgsz=cfg.get("imgsz", 640), device=device, verbose=False,
                )
                res = results[0]
                # 박스 그려진 이미지 (BGR ndarray) → RGB
                plotted = res.plot()           # numpy BGR
                rgb = plotted[:, :, ::-1].copy()
                fname = os.path.splitext(os.path.basename(ip))[0]
                Image.fromarray(rgb).save(os.path.join(out_dir, f"{fname}_pred.png"))

                n_det = len(res.boxes) if res.boxes is not None else 0
                self.log_msg.emit(f"  [{idx+1}/{total}] {os.path.basename(ip)} — 탐지 {n_det}개")
                self.progress_update.emit(idx + 1, total)
                self.result_ready.emit(idx, rgb, os.path.basename(ip))

            self.log_msg.emit(f"추론 완료 → {out_dir}")
            self.finished.emit(out_dir)
        except Exception as e:
            import traceback
            self.log_msg.emit(f"[오류] {e}\n{traceback.format_exc()}")
            self.finished.emit("")


# ═══════════════════════════════════════════════════════════════
#  InferTab UI
# ═══════════════════════════════════════════════════════════════
class InferTab(QWidget):
    step_done = Signal()   # 단계 표시줄에 '끝남'을 알리는 신호

    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state
        self.worker = None
        self.preview_imgs = []   # [(rgb_np, fname), ...]
        self.preview_idx = 0
        self.ws = shared_state["work"]
        self._build()
        self.model_path.textChanged.connect(lambda t: self.ws.set_model(t))
        self.ws.changed.connect(self._on_work_changed)
        self._on_work_changed("model_path"); self._on_work_changed("config_path"); self._on_work_changed("img_dir")

    def _on_work_changed(self, key):
        """작업 상태가 바뀌면 칸을 따라 바꾼다 (모델, 새 config의 imgsz, 비어 있는 입력 폴더는 데이터 폴더로)"""
        if key == "model_path" and self.model_path.text() != self.ws.get(key):
            self.model_path.setText(self.ws.get(key))
        elif key == "config_path":
            cfg = load_config(self.ws.get(key)) or {}
            if cfg.get("imgsz"):
                self.imgsz.setValue(int(cfg["imgsz"]))
        elif key == "img_dir" and not self.input_dir.text().strip():
            self.input_dir.setText(self.ws.get(key))

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)

        # ── 왼쪽 ─────────────────────────────────────────────────
        left_w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); scroll.setWidget(inner)
        lv = QVBoxLayout(inner); lv.setSpacing(12); lv.setContentsMargins(16, 16, 16, 16)

        mg = QGroupBox("모델")
        m = QGridLayout(mg)
        m.addWidget(QLabel("모델 (.pt/.onnx):"), 0, 0)
        self.model_path = QLineEdit(); self.model_path.setPlaceholderText("학습 후 자동 반영 또는 직접 선택")
        m.addWidget(self.model_path, 0, 1)
        b = QPushButton("찾기"); b.setObjectName("secondary")
        b.clicked.connect(self._browse_model); m.addWidget(b, 0, 2)
        btn_exp = QPushButton("내보낸 모델 열기"); btn_exp.setObjectName("secondary")
        btn_exp.setToolTip("최종 모델 내보내기로 만든 export 폴더를 고르면 모델, config, imgsz를 자동으로 불러옵니다")
        btn_exp.clicked.connect(self._browse_export)
        m.addWidget(btn_exp, 1, 1)
        m.setColumnStretch(1, 1)
        lv.addWidget(mg)

        ig = QGroupBox("입력")
        g = QGridLayout(ig)
        g.addWidget(QLabel("이미지 폴더:"), 0, 0)
        self.input_dir = QLineEdit(); self.input_dir.setPlaceholderText("추론할 이미지 폴더")
        g.addWidget(self.input_dir, 0, 1)
        b = QPushButton("찾기"); b.setObjectName("secondary")
        b.clicked.connect(lambda: self._browse_dir(self.input_dir)); g.addWidget(b, 0, 2)
        g.addWidget(QLabel("확장자:"), 1, 0)
        self.ext_combo = QComboBox(); self.ext_combo.addItems(IMG_EXTS)
        self.ext_combo.currentTextChanged.connect(self._on_ext_changed)
        g.addWidget(self.ext_combo, 1, 1)
        self.ch_check = QCheckBox("TIFF 특정 채널만 사용")
        self.ch_check.toggled.connect(lambda on: self.ch_spin.setEnabled(on))
        g.addWidget(self.ch_check, 2, 0)
        ch_h = QHBoxLayout()
        ch_h.addWidget(QLabel("채널 번호:"))
        self.ch_spin = QSpinBox(); self.ch_spin.setRange(0, 63); self.ch_spin.setEnabled(False)
        ch_h.addWidget(self.ch_spin); ch_h.addStretch()
        g.addLayout(ch_h, 2, 1)
        scan_btn = QPushButton("파일 스캔"); scan_btn.setObjectName("secondary")
        scan_btn.clicked.connect(self._scan_input)
        g.addWidget(scan_btn, 3, 0, 1, 3)
        self.input_list = QListWidget(); self.input_list.setMaximumHeight(120)
        g.addWidget(self.input_list, 4, 0, 1, 3)
        g.setColumnStretch(1, 1)
        lv.addWidget(ig)

        og = QGroupBox("추론 옵션")
        o = QGridLayout(og)
        o.addWidget(QLabel("Confidence:"), 0, 0)
        self.conf_slider = QSlider(Qt.Horizontal); self.conf_slider.setRange(1, 99); self.conf_slider.setValue(25)
        self.conf_lbl = QLabel("0.25")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_lbl.setText(f"{v/100:.2f}"))
        ch = QHBoxLayout(); ch.addWidget(self.conf_slider); ch.addWidget(self.conf_lbl)
        o.addLayout(ch, 0, 1)
        o.addWidget(QLabel("IoU (NMS):"), 1, 0)
        self.iou_slider = QSlider(Qt.Horizontal); self.iou_slider.setRange(1, 99); self.iou_slider.setValue(45)
        self.iou_lbl = QLabel("0.45")
        self.iou_slider.valueChanged.connect(lambda v: self.iou_lbl.setText(f"{v/100:.2f}"))
        ih = QHBoxLayout(); ih.addWidget(self.iou_slider); ih.addWidget(self.iou_lbl)
        o.addLayout(ih, 1, 1)
        o.addWidget(QLabel("Image Size:"), 2, 0)
        self.imgsz = QSpinBox(); self.imgsz.setRange(64, 2048); self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)
        o.addWidget(self.imgsz, 2, 1)
        o.setColumnStretch(1, 1)
        lv.addWidget(og)

        sg = QGroupBox("출력")
        s = QGridLayout(sg)
        s.addWidget(QLabel("저장 폴더:"), 0, 0)
        self.output_dir = QLineEdit(); self.output_dir.setText("./YOLO_infer")
        s.addWidget(self.output_dir, 0, 1)
        b = QPushButton("찾기"); b.setObjectName("secondary")
        b.clicked.connect(lambda: self._browse_dir(self.output_dir)); s.addWidget(b, 0, 2)
        s.setColumnStretch(1, 1)
        lv.addWidget(sg)

        bh = QHBoxLayout()
        self.run_btn = QPushButton("추론 시작"); self.run_btn.clicked.connect(self._run_infer)
        self.stop_btn = QPushButton("중단"); self.stop_btn.setObjectName("secondary")
        self.stop_btn.clicked.connect(self._stop_infer); self.stop_btn.setEnabled(False)
        bh.addWidget(self.run_btn); bh.addWidget(self.stop_btn)
        lv.addLayout(bh)
        self.progress = QProgressBar(); self.progress.setValue(0)
        lv.addWidget(self.progress)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(110)
        lv.addWidget(self.log_box)
        lv.addStretch()

        lw = QVBoxLayout(left_w); lw.setContentsMargins(0, 0, 0, 0); lw.addWidget(scroll)
        splitter.addWidget(left_w)

        # ── 오른쪽: 미리보기 ─────────────────────────────────────
        right_w = QWidget()
        rv = QVBoxLayout(right_w); rv.setContentsMargins(16, 16, 16, 16); rv.setSpacing(6)
        prev_l = QLabel("결과 미리보기"); prev_l.setObjectName("section_title")
        rv.addWidget(prev_l)

        self.preview_scroll = QScrollArea(); self.preview_scroll.setWidgetResizable(True)
        self.preview_label = QLabel("추론 실행 후 박스가 그려진 결과가 여기 표시됩니다")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setObjectName("hint")
        self.preview_scroll.setWidget(self.preview_label)
        rv.addWidget(self.preview_scroll, 1)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("이전"); self.prev_btn.setObjectName("secondary")
        self.next_btn = QPushButton("다음"); self.next_btn.setObjectName("secondary")
        self.img_idx_lbl = QLabel("0 / 0")
        self.prev_btn.clicked.connect(self._prev); self.next_btn.clicked.connect(self._next)
        nav.addWidget(self.prev_btn); nav.addWidget(self.img_idx_lbl); nav.addWidget(self.next_btn)
        rv.addLayout(nav)

        splitter.addWidget(right_w)
        splitter.setSizes([420, 580])
        outer.addWidget(splitter)

        self._on_ext_changed(self.ext_combo.currentText())

    # ── helpers ──────────────────────────────────────────────────
    def _on_ext_changed(self, ext):
        """채널 선택은 tiff에서만 의미 있음"""
        is_tiff = ext.lower() == "tiff"
        self.ch_check.setEnabled(is_tiff)
        self.ch_spin.setEnabled(is_tiff and self.ch_check.isChecked())
        if not is_tiff:
            self.ch_check.setChecked(False)

    def _get_channel(self):
        if self.ext_combo.currentText().lower() == "tiff" and self.ch_check.isChecked():
            return self.ch_spin.value()
        return None

    def _browse_model(self):
        path = pick_open(self, "모델 파일", "모델 (*.pt *.pth *.onnx);;All (*)", self.model_path.text(), self.state)
        if path: self.model_path.setText(path)

    def _browse_export(self):
        folder = pick_dir(self, self.model_path.text(), self.state)
        if folder:
            self.open_exported(folder)

    def open_exported(self, folder):
        """최종 모델 내보내기 폴더를 불러온다: 모델 형식 자동 판별, config(imgsz)와 클래스 이름 로드. 성공하면 True"""
        from tools.yolo.export_bundle import find_exported_model
        path, fmt = find_exported_model(folder)
        if not path:
            self.log_box.append(f"[오류] 내보낸 모델 파일(.pt/.onnx)이 없습니다: {folder}")
            return False
        self.log_box.append(f"내보낸 모델 열기: {os.path.basename(path)} ({('PyTorch', 'ONNX', 'ONNX INT8')[fmt]})")
        self.model_path.setText(path)   # work.set_model이 같은 이름 config를 찾아 imgsz를 채운다
        cfg = load_config(self.ws.get("config_path")) or {}
        if cfg.get("class_names"):
            self.state["class_names"] = list(cfg["class_names"])
            self.log_box.append(f"  클래스 {len(cfg['class_names'])}개, imgsz={cfg.get('imgsz', '-')}")
        return True

    def _browse_dir(self, edit):
        path = pick_dir(self, edit.text(), self.state)
        if path: edit.setText(path)

    def _scan_input(self):
        self.input_list.clear()
        path = self.input_dir.text().strip()
        suffixes = _ext_suffixes(self.ext_combo.currentText())
        if not path or not os.path.isdir(path): return
        files = sorted([f for f in os.listdir(path) if f.lower().endswith(suffixes)])
        for f in files:
            self.input_list.addItem(QListWidgetItem(f))
        self.log_box.append(f"스캔: {len(files)}개 파일")
        # tiff면 첫 이미지의 채널 수 로그
        if self.ext_combo.currentText().lower() == "tiff" and files:
            from tools.yolo.yolo_module_wrap import get_channel_count
            n_ch = get_channel_count(os.path.join(path, files[0]))
            self.ch_spin.setMaximum(max(0, n_ch - 1))
            self.log_box.append(f"TIFF 채널 수: {n_ch} ({files[0]} 기준)")

    def _run_infer(self):
        mp = self.model_path.text().strip()
        img = self.input_dir.text().strip()
        if not mp or not os.path.isfile(mp):
            self.log_box.append("[오류] 모델 파일(.pt/.onnx)을 선택하세요"); return
        if not img or not os.path.isdir(img):
            self.log_box.append("[오류] 이미지 폴더를 선택하세요"); return

        suffixes = _ext_suffixes(self.ext_combo.currentText())
        if self.input_list.count() > 0:
            files = [os.path.join(img, self.input_list.item(i).text())
                     for i in range(self.input_list.count())]
        else:
            files = sorted([os.path.join(img, f) for f in os.listdir(img)
                            if f.lower().endswith(suffixes)])
        if not files:
            self.log_box.append("[오류] 추론할 이미지가 없습니다"); return

        self.preview_imgs.clear(); self.preview_idx = 0
        self.log_box.clear(); self.progress.setValue(0)
        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)

        config = {
            "model_path": mp,
            "img_files": files,
            "conf": self.conf_slider.value() / 100.0,
            "iou": self.iou_slider.value() / 100.0,
            "imgsz": self.imgsz.value(),
            "channel": self._get_channel(),
            "out_dir": self.output_dir.text().strip() or "./YOLO_infer",
        }
        self.worker = InferWorker(config)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.result_ready.connect(self._on_result)
        self.worker.log_msg.connect(self.log_box.append)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _stop_infer(self):
        if self.worker: self.worker.stop()

    def _on_progress(self, cur, total):
        self.progress.setRange(0, total); self.progress.setValue(cur)

    def _on_result(self, idx, rgb, fname):
        self.preview_imgs.append((rgb, fname))
        if idx == 0:
            self.preview_idx = 0; self._show_preview()
        self.img_idx_lbl.setText(f"{self.preview_idx+1} / {len(self.preview_imgs)}")

    def _on_finished(self, out_dir):
        self.run_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.step_done.emit()
        total = len(self.preview_imgs)
        self.img_idx_lbl.setText(f"{min(self.preview_idx+1, total)} / {total}")

    def _show_preview(self):
        if not self.preview_imgs:
            return
        rgb, fname = self.preview_imgs[self.preview_idx]
        h, w = rgb.shape[:2]
        # numpy RGB → QImage (연속 메모리 보장)
        import numpy as np
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(pix)
        self.preview_label.resize(pix.size())

    def _prev(self):
        if self.preview_idx > 0:
            self.preview_idx -= 1; self._show_preview()
            self.img_idx_lbl.setText(f"{self.preview_idx+1} / {len(self.preview_imgs)}")

    def _next(self):
        if self.preview_idx < len(self.preview_imgs) - 1:
            self.preview_idx += 1; self._show_preview()
            self.img_idx_lbl.setText(f"{self.preview_idx+1} / {len(self.preview_imgs)}")
