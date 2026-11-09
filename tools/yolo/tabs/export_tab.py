"""내보내기 탭: 최종 모델을 export/<모델>_<날짜-시각>/ 폴더로 모은다 (PyTorch / ONNX / ONNX INT8)."""
import os

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSpinBox, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

from tools.yolo.work_state import load_config, pick_open


class ExportWorker(QThread):
    log_msg = Signal(str)
    done    = Signal(str)     # 만들어진 폴더 경로 (실패하면 빈 글자)

    def __init__(self, model_path, fmt, data_yaml, imgsz):
        super().__init__()
        self.args = (model_path, fmt, data_yaml, imgsz)

    def run(self):
        out = ""
        try:
            from tools.yolo.export_bundle import export_bundle
            model_path, fmt, data_yaml, imgsz = self.args
            out = export_bundle(model_path, fmt, data_yaml, imgsz, log=self.log_msg.emit)
        except Exception as e:
            self.log_msg.emit(f"[오류] 내보내기 실패: {type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}")
        self.done.emit(out)


class ExportTab(QWidget):
    step_done = Signal()   # 단계 표시줄에 '끝남'을 알리는 신호

    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state
        self.ws = shared_state["work"]
        self.worker = None
        self._build()
        self.model_path.textChanged.connect(lambda t: self.ws.set_model(t))
        self.yaml_path.textChanged.connect(lambda t: self.ws.set("data_yaml", t))
        self.yaml_path.textChanged.connect(lambda _: self._refresh_formats())
        self.ws.changed.connect(self._on_work_changed)
        for k in ("model_path", "config_path", "data_yaml"):
            self._on_work_changed(k)
        self._refresh_formats()

    def _on_work_changed(self, key):
        """작업 상태가 바뀌면 칸을 따라 바꾼다 (모델, data.yaml, config의 imgsz)"""
        if key == "model_path" and self.model_path.text() != self.ws.get(key):
            self.model_path.setText(self.ws.get(key))
        elif key == "data_yaml" and self.yaml_path.text() != self.ws.get(key):
            self.yaml_path.setText(self.ws.get(key))
        elif key == "config_path":
            cfg = load_config(self.ws.get(key)) or {}
            if cfg.get("imgsz"):
                self.imgsz.setValue(int(cfg["imgsz"]))

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)

        # ── 왼쪽: 입력 ───────────────────────────────────────────
        left_w = QWidget()
        lv = QVBoxLayout(left_w); lv.setSpacing(12); lv.setContentsMargins(16, 16, 16, 16)

        mg = QGroupBox("모델")
        m = QGridLayout(mg)
        m.addWidget(QLabel("모델 (.pt):"), 0, 0)
        self.model_path = QLineEdit(); self.model_path.setPlaceholderText("학습 후 자동 반영 또는 직접 선택")
        m.addWidget(self.model_path, 0, 1)
        b = QPushButton("찾기"); b.setObjectName("secondary"); b.clicked.connect(self._browse_model)
        m.addWidget(b, 0, 2)
        m.addWidget(QLabel("입력 크기 (imgsz):"), 1, 0)
        self.imgsz = QSpinBox(); self.imgsz.setRange(64, 2048); self.imgsz.setSingleStep(32); self.imgsz.setValue(640)
        m.addWidget(self.imgsz, 1, 1)
        m.setColumnStretch(1, 1)
        lv.addWidget(mg)

        dg = QGroupBox("INT8 캘리브레이션")
        d = QGridLayout(dg)
        d.addWidget(QLabel("data.yaml:"), 0, 0)
        self.yaml_path = QLineEdit(); self.yaml_path.setPlaceholderText("ONNX INT8에만 필요 (학습 이미지로 양자화 범위 계산)")
        d.addWidget(self.yaml_path, 0, 1)
        b = QPushButton("찾기"); b.setObjectName("secondary"); b.clicked.connect(self._browse_yaml)
        d.addWidget(b, 0, 2)
        d.setColumnStretch(1, 1)
        lv.addWidget(dg)

        fg = QGroupBox("이 PC에서 내보낼 수 있는 형식")
        fv = QVBoxLayout(fg)
        self.fmt_info = QLabel(); self.fmt_info.setObjectName("hint"); self.fmt_info.setWordWrap(True)
        fv.addWidget(self.fmt_info)
        lv.addWidget(fg)
        lv.addStretch()
        splitter.addWidget(left_w)

        # ── 오른쪽: 로그 + 내보내기 ──────────────────────────────
        right_w = QWidget()
        rv = QVBoxLayout(right_w); rv.setContentsMargins(16, 16, 16, 16); rv.setSpacing(6)
        title = QLabel("내보내기 로그"); title.setObjectName("section_title")
        rv.addWidget(title)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        rv.addWidget(self.log_box, 1)
        ex = QHBoxLayout()
        ex.addStretch()
        ex.addWidget(QLabel("형식:"))
        self.export_fmt = QComboBox(); self.export_fmt.addItems(["PyTorch", "ONNX", "ONNX INT8"])
        ex.addWidget(self.export_fmt)
        self.export_btn = QPushButton("최종 모델 내보내기"); self.export_btn.clicked.connect(self._export_final)
        ex.addWidget(self.export_btn)
        rv.addLayout(ex)
        splitter.addWidget(right_w)
        splitter.setSizes([420, 580])
        outer.addWidget(splitter)

    # ── helpers ──────────────────────────────────────────────────
    def _refresh_formats(self):
        """형식별 가능 여부를 보여주고, 불가능한 형식은 콤보에서 선택 못 하게 한다"""
        from tools.yolo.export_bundle import available_formats
        avail = available_formats(self.yaml_path.text().strip())
        lines = []
        model = self.export_fmt.model()
        for i in range(self.export_fmt.count()):
            f = self.export_fmt.itemText(i)
            why = avail.get(f, "")
            lines.append(f"{f}: 가능" if not why else f"{f}: 불가 — {why}")
            if isinstance(model, QStandardItemModel):
                model.item(i).setEnabled(not why)
            if why and self.export_fmt.currentIndex() == i:
                self.export_fmt.setCurrentIndex(0)   # 불가능한 형식이 골라져 있으면 PyTorch로
        self.fmt_info.setText("\n".join(lines))

    def _browse_model(self):
        path = pick_open(self, "모델 파일", "PyTorch (*.pt);;All (*)", self.model_path.text(), self.state)
        if path: self.model_path.setText(path)

    def _browse_yaml(self):
        path = pick_open(self, "data.yaml", "YAML (*.yaml *.yml);;All (*)", self.yaml_path.text(), self.state)
        if path: self.yaml_path.setText(path)

    def _export_final(self):
        model = self.model_path.text().strip()
        if not model or not os.path.isfile(model) or not model.lower().endswith(".pt"):
            self.log_box.append("[오류] 내보낼 모델(.pt)이 없습니다. 모델을 먼저 고르세요 (ONNX 파일은 다시 내보낼 수 없음)"); return
        fmt = self.export_fmt.currentText()
        self.export_btn.setEnabled(False)
        self.log_box.append(f"[최종 모델 내보내기] {fmt}  {model}")
        self.worker = ExportWorker(model, fmt, self.yaml_path.text().strip(), self.imgsz.value())
        self.worker.log_msg.connect(self.log_box.append)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _on_done(self, out):
        self.export_btn.setEnabled(True)
        if out:
            self.step_done.emit()
