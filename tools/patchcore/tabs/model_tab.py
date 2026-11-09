"""모델 탭: PatchCore 백본, 사용할 층, coreset 비율, 입력 크기."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget,
)

from common import settings
from tools.patchcore import core
from tools.patchcore.ui_common import TOOL, primary_button, two_pane

LAYERS = ["layer1", "layer2", "layer3", "layer4"]


class ModelTab(QWidget):
    step_done = Signal()

    def __init__(self, work):
        super().__init__()
        self.work = work
        self._build()
        self._load_last()

    def _build(self):
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 8, 0)
        g = QGroupBox("PatchCore")
        f = QFormLayout(g)
        self.backbone = QComboBox()
        self.backbone.addItems(core.BACKBONES)   # 첫 항목 wide_resnet50_2가 기본값
        f.addRow("백본", self.backbone)

        lw = QWidget()
        lh = QHBoxLayout(lw)
        lh.setContentsMargins(0, 0, 0, 0)
        self.layer_checks = []
        for name in LAYERS:
            c = QCheckBox(name)
            c.setChecked(name in ("layer2", "layer3"))
            lh.addWidget(c)
            self.layer_checks.append(c)
        f.addRow("사용할 층", lw)

        self.coreset = QDoubleSpinBox()
        self.coreset.setRange(0.01, 1.0)
        self.coreset.setSingleStep(0.05)
        self.coreset.setDecimals(2)
        self.coreset.setValue(0.1)
        f.addRow("coreset 비율", self.coreset)

        sw = QWidget()
        sh = QHBoxLayout(sw)
        sh.setContentsMargins(0, 0, 0, 0)
        self.size_h = QSpinBox()
        self.size_w = QSpinBox()
        for s in (self.size_h, self.size_w):
            s.setRange(32, 2048)
            s.setSingleStep(32)
            s.setValue(256)
        sh.addWidget(self.size_h)
        sh.addWidget(QLabel("x"))
        sh.addWidget(self.size_w)
        sh.addStretch(1)
        f.addRow("입력 크기 (높이 x 너비)", sw)
        lv.addWidget(g)
        lv.addStretch(1)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 0, 0, 0)
        info = QLabel(
            "백본: 사전학습 CNN. 정상 이미지의 중간층 특징을 메모리 뱅크에 모아 두고, 검사 이미지의 특징과 거리를 비교해 이상 점수를 만든다.\n\n"
            "사용할 층: 얕은 층(layer1~2)은 작은 결함, 깊은 층(layer3~4)은 큰 구조에 민감하다. 기본은 layer2 + layer3.\n\n"
            "coreset 비율: 메모리 뱅크에 남길 특징의 비율. 줄이면 빨라지고 정확도가 떨어질 수 있다.\n\n"
            "입력 크기: 이미지를 이 크기로 줄여 넣는다. 작은 결함이 보이지 않으면 키운다.")
        info.setObjectName("Hint")
        info.setWordWrap(True)
        rv.addWidget(info)
        rv.addStretch(1)

        self.save_btn = primary_button("저장하고 다음")
        self.save_btn.clicked.connect(self._save)
        two_pane(self, left, right, self.save_btn)

    def params(self):
        return {"backbone": self.backbone.currentText(),
                "layers": [c.text() for c in self.layer_checks if c.isChecked()],
                "coreset": self.coreset.value(),
                "size": (self.size_h.value(), self.size_w.value())}

    def _save(self):
        if not self.params()["layers"]:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "모델", "사용할 층을 하나 이상 고르세요.")
            return
        p = self.params()
        self.work.set("size", p["size"])
        settings.set_value(TOOL, "backbone", p["backbone"])
        settings.set_value(TOOL, "layers", ",".join(p["layers"]))
        settings.set_value(TOOL, "coreset", p["coreset"])
        settings.set_value(TOOL, "size", f"{p['size'][0]},{p['size'][1]}")
        self.step_done.emit()

    def _load_last(self):
        bb = settings.get_value(TOOL, "backbone")
        if bb and self.backbone.findText(str(bb)) >= 0:
            self.backbone.setCurrentText(str(bb))
        layers = settings.get_value(TOOL, "layers")
        if layers:
            for c in self.layer_checks:
                c.setChecked(c.text() in str(layers).split(","))
        co = settings.get_value(TOOL, "coreset")
        if co:
            self.coreset.setValue(float(co))
        size = settings.get_value(TOOL, "size")
        if size:
            h, w = str(size).split(",")
            self.size_h.setValue(int(h))
            self.size_w.setValue(int(w))
