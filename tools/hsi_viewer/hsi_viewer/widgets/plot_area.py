"""우측 상단 그래프 영역: 헤더의 전환 버튼으로 포인트 / 객체 / 클러스터 스펙트럼을 번갈아 표시."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .common import SegmentedControl
from .spectrum_view import ClusterSpectrumView, PieceSpectrumView, SpectrumView


class PlotArea(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.spectrum = SpectrumView()
        self.piece = PieceSpectrumView()
        self.cluster = ClusterSpectrumView()
        # 세 뷰의 자체 헤더 대신 공용 헤더 사용
        for v in (self.spectrum, self.piece, self.cluster):
            v.layout().itemAt(0).layout().setContentsMargins(12, 2, 12, 0)
            v.title.hide()

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(12, 8, 12, 0)
        self.switch = SegmentedControl(["포인트 스펙트럼", "객체 스펙트럼", "클러스터 스펙트럼"], 0)
        self.switch.setFixedWidth(420)
        self.switch.setEnabledIndex(1, False)
        self.switch.setEnabledIndex(2, False)
        header.addWidget(self.switch)
        header.addStretch(1)
        lay.addLayout(header)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.spectrum)
        self.stack.addWidget(self.piece)
        self.stack.addWidget(self.cluster)
        lay.addWidget(self.stack, 1)
        self.switch.changed.connect(self.stack.setCurrentIndex)

    def _show(self, idx: int, available: bool, switch_to: bool):
        self.switch.setEnabledIndex(idx, available)
        if available and switch_to:
            self.switch.setCurrent(idx)
            self.stack.setCurrentIndex(idx)
        elif not available and self.stack.currentIndex() == idx:
            self.switch.setCurrent(0)
            self.stack.setCurrentIndex(0)

    def show_pieces(self, available: bool, switch_to: bool = False):
        self._show(1, available, switch_to)

    def show_cluster(self, available: bool, switch_to: bool = False):
        self._show(2, available, switch_to)
