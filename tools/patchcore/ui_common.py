"""PatchCore 화면들이 같이 쓰는 것: 작업 상태, 폴더/파일 선택창, 백그라운드 실행, 이미지 표시."""
import os
import traceback

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFileDialog, QLabel, QSizePolicy

from common import settings

TOOL = "patchcore"   # common 설정에 저장할 때 쓰는 이름


class WorkState(QObject):
    """현재 작업: 모델(ckpt), 정상 폴더, 테스트 폴더, 입력 크기, 임계값. 한 탭에서 바꾸면 다른 탭이 따라 바뀐다."""
    changed = Signal(str)

    def __init__(self):
        super().__init__()
        self._v = {"ckpt": "", "normal_dir": "", "test_dir": "", "size": (256, 256), "threshold": 0.5}

    def get(self, key):
        return self._v[key]

    def set(self, key, value):
        if self._v[key] == value:
            return
        self._v[key] = value
        self.changed.emit(key)


def _existing_dir(path):
    path = (path or "").strip()
    while path:
        if os.path.isdir(path):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return ""


def start_dir(current=""):
    """칸의 경로 → 마지막으로 연 폴더 순으로 선택창 시작 위치를 정한다."""
    return _existing_dir(current) or _existing_dir(settings.get_value(TOOL, "last_dir", ""))


def pick_dir(parent, current="", title="폴더 선택"):
    path = QFileDialog.getExistingDirectory(parent, title, start_dir(current))
    if path:
        settings.set_value(TOOL, "last_dir", path)
    return path


def pick_open(parent, title, filt, current=""):
    path, _ = QFileDialog.getOpenFileName(parent, title, start_dir(current), filt)
    if path:
        settings.set_value(TOOL, "last_dir", os.path.dirname(path))
    return path


class Worker(QThread):
    """함수를 다른 스레드에서 실행한다. fn(log) 형태로 부르고, 결과는 done, 오류는 failed로 알린다."""
    log = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn(self.log.emit))
        except Exception as e:   # 어떤 오류든 화면에 이유를 보여주려고 모두 잡는다
            self.log.emit(traceback.format_exc())
            self.failed.emit(f"{type(e).__name__}: {e}")


def to_pixmap(rgb):
    """HxWx3 uint8 numpy 배열 -> QPixmap"""
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())


class ImageView(QLabel):
    """창 크기에 맞춰 비율을 유지하며 그림을 보여주는 칸"""

    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setObjectName("Muted")
        self._pm = None

    def set_pixmap(self, pm):
        self._pm = pm
        self._fit()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def _fit(self):
        if self._pm is not None:
            self.setPixmap(self._pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


def two_pane(owner, left, right, button=None):
    """탭 공통 틀: 왼쪽 설정(너비 고정 느낌), 오른쪽 미리보기. button이 있으면 오른쪽 아래에 하늘색 채운 버튼."""
    from PySide6.QtWidgets import QHBoxLayout, QSplitter, QVBoxLayout, QWidget
    outer = QVBoxLayout(owner)
    outer.setContentsMargins(16, 16, 16, 16)
    sp = QSplitter(Qt.Horizontal)
    sp.addWidget(left)
    sp.addWidget(right)
    sp.setStretchFactor(0, 0)
    sp.setStretchFactor(1, 1)
    sp.setSizes([420, 700])
    outer.addWidget(sp, 1)
    if button is not None:
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button)
        outer.addLayout(row)


def primary_button(text):
    from PySide6.QtWidgets import QPushButton
    b = QPushButton(text)
    b.setObjectName("Primary")
    return b
