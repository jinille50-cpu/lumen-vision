import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QLineEdit, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QTextEdit, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap


# ═══════════════════════════════════════════════════════════════
#  평가 Worker — YOLO.val()
# ═══════════════════════════════════════════════════════════════
from tools.yolo.work_state import pick_open, load_config


def results_dir(model_path):
    """체크포인트 폴더 안 Results/. 모델이 <체크포인트>/[full|foldN]/weights/ 안에 있으면 체크포인트 폴더까지 올라간다."""
    import re
    d = os.path.dirname(os.path.abspath(model_path))
    if os.path.basename(d) == "weights":
        d = os.path.dirname(d)
    if os.path.basename(d) == "full" or re.fullmatch(r"fold\d+", os.path.basename(d)):
        d = os.path.dirname(d)
    return os.path.join(d, "Results")


def save_results(cfg, model, metrics, results):
    """클래스별 지표(P, R, AP50, AP50-95), 혼동 행렬 그림, 요약 json을 Results/에 저장하고 폴더를 돌려준다."""
    import json
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")   # 화면 없이 그림 파일만 만든다
    from matplotlib.figure import Figure
    from common import theme
    from common.palette import class_colors

    out = results_dir(cfg["model_path"])
    os.makedirs(out, exist_ok=True)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    nc = len(names)
    colors = class_colors(nc + 1)[1:]   # YOLO에는 배경 클래스가 없어 0번(흰색)을 뺀다

    per_class = {}
    box = metrics.box
    for k, ci in enumerate(box.ap_class_index):   # 라벨이 있는 클래스만 들어 있다
        p, r, ap50, ap = box.class_result(k)
        per_class[names[int(ci)]] = {"Precision": float(p), "Recall": float(r),
                                     "AP50": float(ap50), "AP50-95": float(ap)}

    cm = np.asarray(metrics.confusion_matrix.matrix, dtype=float)   # (nc+1)x(nc+1), 마지막은 배경
    labels = [names[i] for i in range(nc)] + ["background"]
    summary = {
        "model_path": cfg["model_path"], "data_yaml": cfg["data_yaml"],
        "split": cfg.get("split", "val"), "imgsz": cfg.get("imgsz", 640),
        "class_names": [names[i] for i in range(nc)],
        "mAP50": results["mAP50"], "mAP50-95": results["mAP50-95"],
        "Precision": results["Precision"], "Recall": results["Recall"],
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},   # 행: 예측, 열: 정답 (Ultralytics 방식)
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 클래스별 지표 막대 그림 (클래스 색은 common 팔레트)
    if per_class:
        fig = Figure(figsize=(8, 4), facecolor=theme.BG0)
        ax = fig.add_subplot(1, 1, 1)
        keys = ["Precision", "Recall", "AP50", "AP50-95"]
        cnames = list(per_class)
        width = 0.8 / len(cnames)
        for j, cn in enumerate(cnames):
            ci = [names[i] for i in range(nc)].index(cn)
            ax.bar([x + j * width for x in range(len(keys))], [per_class[cn][k] for k in keys],
                   width, label=cn, color=colors[ci])
        ax.set_xticks([x + 0.4 - width / 2 for x in range(len(keys))]); ax.set_xticklabels(keys)
        ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.set_title("Per-class metrics")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "per_class_metrics.png"), dpi=150)

    # 혼동 행렬 (개수 + 정답 열 기준 비율)
    norm = cm / (cm.sum(axis=0, keepdims=True) + 1e-9)
    fig = Figure(figsize=(11, 4.5), facecolor=theme.BG0)
    n = len(labels)
    for k, (data, title, fmt) in enumerate(((cm, "Confusion Matrix (Count)", "{:.0f}"),
                                            (norm, "Confusion Matrix (Normalized)", "{:.2f}"))):
        ax = fig.add_subplot(1, 2, k + 1)
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=max(data.max(), 1e-9))
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
        for t, tl in enumerate(ax.get_xticklabels()):   # 글자 색을 클래스 색으로 (배경은 회색)
            tl.set_color(colors[t] if t < nc else theme.MUTED)
        for t, tl in enumerate(ax.get_yticklabels()):
            tl.set_color(colors[t] if t < nc else theme.MUTED)
        ax.set_title(title, fontsize=10); ax.set_xlabel("True"); ax.set_ylabel("Predicted")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=7,
                        color=theme.BG0 if data[i, j] > data.max() * 0.6 else theme.TEXT)   # 진한 칸은 흰 글자
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    return out


class EvalWorker(QThread):
    result_ready = Signal(dict)     # 지표 dict
    log_msg      = Signal(str)
    finished     = Signal(str)      # save_dir

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            import ultralytics  # noqa: F401  설치 여부만 확인 (모델은 load_detector가 연다)
            import torch
            from tools.yolo.yolo_module_wrap import load_detector
        except ImportError as e:
            self.log_msg.emit(f"[오류] ultralytics 미설치: {e}")
            self.finished.emit(""); return

        cfg = self.config
        try:
            self.log_msg.emit(f"[평가 시작] 모델: {os.path.basename(cfg['model_path'])}")
            self.log_msg.emit(f"  data.yaml: {cfg['data_yaml']}")
            device = 0 if torch.cuda.is_available() else "cpu"
            model = load_detector(cfg["model_path"])   # YOLO 또는 RT-DETR

            # ── 모델 입력 채널 ↔ data.yaml channels 일치 검증 ──────────
            # (TIFF 채널 추출 학습 모델을 다른 채널 수의 yaml로 평가하면
            #  Ultralytics 내부에서 채널 불일치로 실패 → 사전에 명확히 안내)
            yaml_ch = 3
            try:
                import yaml as _yaml
                with open(cfg["data_yaml"], "r", encoding="utf-8") as f:
                    data_cfg = _yaml.safe_load(f) or {}
                yaml_ch = int(data_cfg.get("channels", 3))
            except Exception:
                data_cfg = {}
            model_ch = None
            try:
                model_ch = next(m.in_channels for m in model.model.modules()
                                if isinstance(m, torch.nn.Conv2d))
            except Exception:
                pass
            if model_ch is not None:
                self.log_msg.emit(f"  입력 채널 — 모델: {model_ch} / data.yaml: {yaml_ch}")
                if model_ch != yaml_ch:
                    self.log_msg.emit(
                        f"[오류] 모델 입력 채널({model_ch})과 data.yaml의 채널 수({yaml_ch})가 다릅니다.\n"
                        f"  → 이 모델을 학습할 때 생성된 data.yaml을 선택하세요.\n"
                        f"     (채널 추출 학습 시: 체크포인트 폴더의 data.yaml 또는 foldN/data.yaml —\n"
                        f"      추출된 dataset 폴더와 'channels: {model_ch}' 키를 포함한 파일)")
                    self.finished.emit(""); return

            metrics = model.val(
                data=cfg["data_yaml"],
                imgsz=cfg.get("imgsz", 640),
                batch=cfg.get("batch", 16),
                device=device,
                split=cfg.get("split", "val"),
                plots=True,
                # 그림(PR 곡선 등)을 실행 폴더의 runs/ 대신 체크포인트 폴더 Results/val/ 에 (같은 이름 그림과 섞이지 않게 하위 폴더)
                project=results_dir(cfg["model_path"]), name="val", exist_ok=True,
            )
            # 지표 추출
            box = metrics.box
            results = {
                "mAP50":     float(box.map50),
                "mAP50-95":  float(box.map),
                "Precision": float(box.mp),
                "Recall":    float(box.mr),
            }
            # per-class mAP50
            per_class = {}
            try:
                names = model.names
                for i, ap in enumerate(box.ap50):
                    cname = names.get(i, f"class{i}") if isinstance(names, dict) else str(i)
                    per_class[cname] = float(ap)
            except Exception:
                pass
            results["per_class"] = per_class

            save_dir = str(getattr(metrics, "save_dir", "") or "")
            try:
                out = save_results(cfg, model, metrics, results)
                self.log_msg.emit(f"Results 저장: {out}")
            except Exception as e:
                self.log_msg.emit(f"[Results 저장 오류] {e}")
            self.log_msg.emit("평가 완료")
            self.result_ready.emit(results)
            self.finished.emit(save_dir)
        except Exception as e:
            import traceback
            self.log_msg.emit(f"[오류] {e}\n{traceback.format_exc()}")
            self.finished.emit("")


# ═══════════════════════════════════════════════════════════════
#  EvalTab UI
# ═══════════════════════════════════════════════════════════════
class EvalTab(QWidget):
    step_done = Signal()   # 단계 표시줄에 '끝남'을 알리는 신호

    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state
        self.worker = None
        self.ws = shared_state["work"]
        self._build()
        self.model_path.textChanged.connect(lambda t: self.ws.set_model(t))
        self.yaml_path.textChanged.connect(lambda t: self.ws.set("data_yaml", t))
        self.ws.changed.connect(self._on_work_changed)
        self._on_work_changed("model_path"); self._on_work_changed("config_path"); self._on_work_changed("data_yaml")

    def set_data_yaml(self, path):
        self.ws.set("data_yaml", path)

    def _on_work_changed(self, key):
        """작업 상태가 바뀌면 칸을 따라 바꾼다 (다른 탭에서 바꾼 모델/yaml, 새 config의 imgsz)"""
        if key == "model_path" and self.model_path.text() != self.ws.get(key):
            self.model_path.setText(self.ws.get(key))
        elif key == "data_yaml":
            if self.yaml_path.text() != self.ws.get(key):
                self.yaml_path.setText(self.ws.get(key))
            self._show_val_dir(self.ws.get(key))
        elif key == "config_path":
            cfg = load_config(self.ws.get(key)) or {}
            if cfg.get("imgsz"):
                self.imgsz.setValue(int(cfg["imgsz"]))
            y = cfg.get("data_yaml", "")
            if y and os.path.isfile(y):
                self.ws.set("data_yaml", y)

    def _show_val_dir(self, yaml_file):
        """data.yaml의 val 폴더(학습 때 쓴 Validation 이미지)를 보여준다 — 폴더로 분리였으면 데이터 탭 Val 폴더와 같다"""
        val = ""
        try:
            import yaml
            with open(yaml_file, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
            val = str(d.get("val", ""))
            if val and not os.path.isabs(val):
                val = os.path.normpath(os.path.join(str(d.get("path", "") or os.path.dirname(yaml_file)), val))
        except Exception:
            pass
        self.val_info.setText(val or "—")

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)

        # ── 왼쪽 ─────────────────────────────────────────────────
        left_w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); scroll.setWidget(inner)
        lv = QVBoxLayout(inner); lv.setSpacing(10); lv.setContentsMargins(14, 14, 14, 14)

        mg = QGroupBox("모델 / 데이터")
        m = QGridLayout(mg)
        m.addWidget(QLabel("모델 (.pt):"), 0, 0)
        self.model_path = QLineEdit(); self.model_path.setPlaceholderText("학습 후 자동 반영 또는 직접 선택")
        m.addWidget(self.model_path, 0, 1)
        b = QPushButton("찾기"); b.setObjectName("secondary")
        b.clicked.connect(self._browse_model); m.addWidget(b, 0, 2)

        m.addWidget(QLabel("data.yaml:"), 1, 0)
        self.yaml_path = QLineEdit(); self.yaml_path.setPlaceholderText("학습 시 생성된 data.yaml")
        m.addWidget(self.yaml_path, 1, 1)
        b = QPushButton("찾기"); b.setObjectName("secondary")
        b.clicked.connect(self._browse_yaml); m.addWidget(b, 1, 2)

        m.addWidget(QLabel("Val 폴더:"), 4, 0)
        self.val_info = QLabel("—"); self.val_info.setObjectName("hint")
        self.val_info.setWordWrap(True)
        m.addWidget(self.val_info, 4, 1, 1, 2)

        m.addWidget(QLabel("Split:"), 2, 0)
        self.split_combo = QComboBox(); self.split_combo.addItems(["val", "train", "test"])
        m.addWidget(self.split_combo, 2, 1)

        m.addWidget(QLabel("Image Size:"), 3, 0)
        from PySide6.QtWidgets import QSpinBox
        self.imgsz = QSpinBox(); self.imgsz.setRange(64, 2048); self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)
        m.addWidget(self.imgsz, 3, 1)
        m.setColumnStretch(1, 1)
        lv.addWidget(mg)

        self.run_btn = QPushButton("평가 시작")
        self.run_btn.clicked.connect(self._run_eval)
        lv.addWidget(self.run_btn)

        res_l = QLabel("평가 결과"); res_l.setObjectName("section_title")
        lv.addWidget(res_l)
        self.result_table = QTableWidget(0, 2)
        self.result_table.setHorizontalHeaderLabels(["지표", "값"])
        hh = self.result_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.result_table.verticalHeader().setVisible(False)
        lv.addWidget(self.result_table)

        self.log_box = QTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(120)
        lv.addWidget(self.log_box)
        lv.addStretch()

        lw = QVBoxLayout(left_w); lw.setContentsMargins(0, 0, 0, 0); lw.addWidget(scroll)
        splitter.addWidget(left_w)

        # ── 오른쪽: Ultralytics figure 표시 ───────────────────────
        right_w = QWidget()
        rv = QVBoxLayout(right_w); rv.setContentsMargins(14, 14, 14, 14); rv.setSpacing(8)
        fig_l = QLabel("결과 그래프  (Ultralytics 생성)"); fig_l.setObjectName("section_title")
        rv.addWidget(fig_l)

        fh = QHBoxLayout()
        fh.addWidget(QLabel("그래프 선택:"))
        self.fig_combo = QComboBox()
        self.fig_combo.currentTextChanged.connect(self._show_figure)
        fh.addWidget(self.fig_combo, 1)
        rv.addLayout(fh)

        self.fig_scroll = QScrollArea(); self.fig_scroll.setWidgetResizable(True)
        self.fig_label = QLabel("평가 실행 후 결과 그래프가 여기 표시됩니다")
        self.fig_label.setAlignment(Qt.AlignCenter)
        self.fig_label.setObjectName("hint")
        self.fig_scroll.setWidget(self.fig_label)
        # 스크롤 영역(스플리터 드래그 포함) 크기 변화 시 그래프 재스케일
        self.fig_scroll.viewport().installEventFilter(self)
        rv.addWidget(self.fig_scroll, 1)

        splitter.addWidget(right_w)
        splitter.setSizes([420, 580])
        outer.addWidget(splitter)

    # ── helpers ──────────────────────────────────────────────────
    def _browse_model(self):
        path = pick_open(self, "모델 파일", "PyTorch (*.pt *.pth);;All (*)", self.model_path.text(), self.state)
        if path: self.model_path.setText(path)

    def _browse_yaml(self):
        path = pick_open(self, "data.yaml", "YAML (*.yaml *.yml);;All (*)", self.yaml_path.text(), self.state)
        if path: self.yaml_path.setText(path)

    def _run_eval(self):
        mp = self.model_path.text().strip()
        yp = self.yaml_path.text().strip()
        if not mp or not os.path.isfile(mp):
            self.log_box.append("[오류] 모델 파일(.pt)을 선택하세요"); return
        if not yp or not os.path.isfile(yp):
            self.log_box.append("[오류] data.yaml을 선택하세요"); return

        self.run_btn.setEnabled(False); self.log_box.clear()
        self.result_table.setRowCount(0)
        config = {
            "model_path": mp, "data_yaml": yp,
            "split": self.split_combo.currentText(),
            "imgsz": self.imgsz.value(),
        }
        self.worker = EvalWorker(config)
        self.worker.result_ready.connect(self._on_result)
        self.worker.log_msg.connect(self.log_box.append)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_result(self, results):
        rows = [
            ("mAP50",     results.get("mAP50")),
            ("mAP50-95",  results.get("mAP50-95")),
            ("Precision", results.get("Precision")),
            ("Recall",    results.get("Recall")),
        ]
        for name, val in rows:
            r = self.result_table.rowCount()
            self.result_table.insertRow(r)
            self.result_table.setItem(r, 0, QTableWidgetItem(name))
            self.result_table.setItem(r, 1, QTableWidgetItem(f"{val:.4f}" if val is not None else "—"))
        # per-class
        for cname, ap in results.get("per_class", {}).items():
            r = self.result_table.rowCount()
            self.result_table.insertRow(r)
            self.result_table.setItem(r, 0, QTableWidgetItem(f"AP50 [{cname}]"))
            self.result_table.setItem(r, 1, QTableWidgetItem(f"{ap:.4f}"))

    def _on_finished(self, save_dir):
        self.run_btn.setEnabled(True)
        self.log_box.append("평가 완료")
        self.step_done.emit()
        if save_dir and os.path.isdir(save_dir):
            self.log_box.append(f"결과 폴더: {save_dir}")
            self._load_figures(save_dir)

    def _load_figures(self, save_dir):
        """결과 폴더에서 png 그래프 목록 채우기"""
        self._fig_dir = save_dir
        pngs = sorted([f for f in os.listdir(save_dir) if f.lower().endswith(".png")])
        self.fig_combo.blockSignals(True)
        self.fig_combo.clear()
        self.fig_combo.addItems(pngs)
        self.fig_combo.blockSignals(False)
        if pngs:
            # F1 curve 우선 표시 (없으면 confusion matrix → 첫 파일)
            preferred = next((p for p in pngs if "f1" in p.lower()),
                             next((p for p in pngs if "confusion" in p.lower()), pngs[0]))
            self.fig_combo.setCurrentText(preferred)
            self._show_figure(preferred)

    def _show_figure(self, name):
        if not name or not hasattr(self, "_fig_dir"):
            return
        path = os.path.join(self._fig_dir, name)
        if not os.path.isfile(path):
            return
        pix = QPixmap(path)
        if not pix.isNull():
            self._orig_pix = pix
            self._rescale_figure()
        else:
            self._orig_pix = None
            self.fig_label.setText(f"이미지를 열 수 없음: {name}")

    def _rescale_figure(self):
        """원본 픽스맵을 스크롤 영역 너비에 맞춰 비율 유지하며 표시"""
        pix = getattr(self, "_orig_pix", None)
        if pix is None or pix.isNull():
            return
        avail = self.fig_scroll.viewport().width() - 2
        if avail <= 0:
            avail = pix.width()
        # 원본보다 크게 확대하지는 않음
        target_w = min(avail, pix.width())
        scaled = pix.scaledToWidth(target_w, Qt.SmoothTransformation)
        self.fig_label.setPixmap(scaled)
        self.fig_label.resize(scaled.size())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale_figure()

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if obj is self.fig_scroll.viewport() and event.type() == QEvent.Resize:
            self._rescale_figure()
        return super().eventFilter(obj, event)
