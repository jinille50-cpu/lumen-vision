"""SAM 3 일괄 라벨링: 폴더 전체를 글자 프롬프트로 자동 라벨링하는 창과 저장 도우미.
실제 SAM 호출 순서(한 장 끝나면 다음 장)는 app.py가 맡는다. 자동 라벨 기록은 <폴더>/auto_labels.json (검수 B-011에서 사용)."""
import json
import os
import time

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QHBoxLayout, QHeaderView, QLabel,
    QPlainTextEdit, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

AUTO_FILE = "auto_labels.json"


def load_auto(folder):
    """{이미지 파일 이름: 기록}. 없거나 깨졌으면 빈 dict"""
    try:
        with open(os.path.join(folder, AUTO_FILE), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_auto(folder, auto):
    with open(os.path.join(folder, AUTO_FILE), "w", encoding="utf-8") as f:
        json.dump(auto, f, ensure_ascii=False, indent=1)


def read_rgb(path):
    """이미지 파일을 (H, W, 3) uint8 RGB 배열로. 못 읽으면 None"""
    img = QImage(path)
    if img.isNull():
        return None
    img = img.convertToFormat(QImage.Format_RGB888)
    buf = np.frombuffer(img.constBits(), np.uint8, img.height() * img.bytesPerLine())
    return buf.reshape(img.height(), img.bytesPerLine())[:, :img.width() * 3].reshape(img.height(), img.width(), 3).copy()


def write_mask(path, pieces, classes):
    """점수 낮은 것부터 칠해 겹치는 곳은 점수 높은 결과가 갖게 한 뒤 마스크 png(클래스 번호+1)로 저장"""
    h, w = pieces[0].mask.shape
    arr = np.zeros((h, w), np.uint8)
    for q, c in sorted(zip(pieces, classes), key=lambda t: t[0].score):
        arr[q.mask] = min(c + 1, 255)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not QImage(arr.data, w, h, w, QImage.Format_Grayscale8).save(path):
        raise OSError(f"마스크를 쓸 수 없음: {path}")


def write_boxes(path, pieces, classes, w, h):
    lines = [f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
             for q, c in zip(pieces, classes) for x1, y1, x2, y2 in [q.bbox]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def record(pieces, classes, shadows, mode):
    """auto_labels.json 한 장 기록. 검수에서 점수 낮은 것부터 볼 수 있게 점수를 남긴다"""
    scores = [round(q.score, 3) for q in pieces]
    return {"status": "auto", "mode": mode, "time": time.strftime("%Y-%m-%d %H:%M"),
            "n": len(pieces), "min_score": min(scores) if scores else None,
            "scores": scores, "classes": list(classes), "prompts": [q.prompt for q in pieces], "shadow_removed": shadows}


class BatchDialog(QDialog):
    start_requested = Signal(dict)   # 설정 {"words": {단어: 클래스}, "conf", "skip", "no_shadow"}
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM 3 일괄 라벨링")
        self.setModal(True)   # 도는 동안 본 창의 라벨을 고치지 못하게 (같은 파일을 덮어쓸 수 있으므로)
        self.resize(620, 560)
        v = QVBoxLayout(self)
        self.info = QLabel("")
        self.info.setObjectName("Hint")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        v.addWidget(QLabel("단어마다 넣을 클래스 (단어는 SAM 3 보조의 프롬프트 칸에서 가져옴)"))
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["단어", "클래스"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        v.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("점수 하한"))
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.05, 0.95)
        self.conf.setSingleStep(0.05)
        self.conf.setDecimals(2)
        row.addWidget(self.conf)
        row.addStretch(1)
        v.addLayout(row)
        self.skip = QCheckBox("라벨이 이미 있는 이미지는 건너뛰기")
        self.skip.setChecked(True)
        self.no_shadow = QCheckBox("그림자 의심 조각 빼기")
        self.no_shadow.setChecked(False)   # 기본은 그림자 의심 조각도 모두 라벨로 넣는다
        v.addWidget(self.skip)
        v.addWidget(self.no_shadow)
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

    def setup(self, info, words, classes, cur_cls, conf, first=0):   # first: 화면 클래스 번호 시작(Segmentation 1)
        self.info.setText(info)
        self.table.setRowCount(len(words))
        for r, wd in enumerate(words):
            self.table.setItem(r, 0, QTableWidgetItem(wd))
            cb = QComboBox()
            cb.addItems([f"{i + first}  {n}" for i, n in enumerate(classes)])
            cb.setCurrentIndex(cur_cls)
            self.table.setCellWidget(r, 1, cb)
        self.conf.setValue(conf)
        self.bar.setValue(0)
        self.log.clear()
        self.start_btn.setEnabled(bool(words))

    def set_running(self, on):
        self.start_btn.setEnabled(not on and self.table.rowCount() > 0)
        self.stop_btn.setEnabled(on)
        self.close_btn.setEnabled(not on)
        self.table.setEnabled(not on)
        for w in (self.conf, self.skip, self.no_shadow):
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
        words = {self.table.item(r, 0).text(): self.table.cellWidget(r, 1).currentIndex()
                 for r in range(self.table.rowCount())}
        self.start_requested.emit({"words": words, "conf": self.conf.value(),
                                   "skip": self.skip.isChecked(), "no_shadow": self.no_shadow.isChecked()})
