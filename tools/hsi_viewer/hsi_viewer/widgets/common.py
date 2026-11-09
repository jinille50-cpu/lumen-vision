"""Small reusable widgets: cards, labelled sliders, segmented controls, colour chips."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .. import theme


class Card(QtWidgets.QFrame):
    """Rounded panel with a title row, optional subtitle and collapsible body."""

    def __init__(self, title: str, subtitle: str = "", collapsible: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumWidth(0)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 14)
        outer.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(6)
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("CardTitle")
        self.title_label.setVisible(not collapsible)
        header.addWidget(self.title_label, 1 if not collapsible else 0)
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setVisible(collapsible)
        self.toggle.setObjectName("CardToggle")
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(QtCore.Qt.DownArrow)
        self.toggle.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.toggle.setMinimumWidth(40)
        self.toggle.setCursor(QtCore.Qt.PointingHandCursor)
        header.addWidget(self.toggle, 1)
        self.header_extra = QtWidgets.QHBoxLayout()
        self.header_extra.setSpacing(6)
        header.addLayout(self.header_extra)
        outer.addLayout(header)

        self.subtitle = QtWidgets.QLabel(subtitle)
        self.subtitle.setObjectName("CardSubtitle")
        self.subtitle.setWordWrap(True)
        self.subtitle.setVisible(bool(subtitle))
        outer.addWidget(self.subtitle)

        self.body = QtWidgets.QWidget()
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 2, 0, 0)
        self.body_layout.setSpacing(8)
        outer.addWidget(self.body, 1)

        if collapsible:
            self.toggle.toggled.connect(self._on_toggle)
        else:
            self.toggle.setCheckable(False)
            self.toggle.setArrowType(QtCore.Qt.NoArrow)

    def _on_toggle(self, checked: bool):
        self.body.setVisible(checked)
        self.subtitle.setVisible(checked and bool(self.subtitle.text()))
        self.toggle.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)

    def add(self, widget_or_layout):
        if isinstance(widget_or_layout, QtWidgets.QLayout):
            self.body_layout.addLayout(widget_or_layout)
        else:
            self.body_layout.addWidget(widget_or_layout)
        return widget_or_layout


def section_label(text: str) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text.upper())
    lab.setObjectName("SectionLabel")
    return lab


def hint(text: str) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setObjectName("Hint")
    lab.setWordWrap(True)
    return lab


def form_row(label: str, widget: QtWidgets.QWidget, label_width: int = 86) -> QtWidgets.QHBoxLayout:
    row = QtWidgets.QHBoxLayout()
    row.setSpacing(8)
    lab = QtWidgets.QLabel(label)
    lab.setObjectName("Muted")
    lab.setFixedWidth(label_width)
    row.addWidget(lab)
    row.addWidget(widget, 1)
    return row


class LabeledSlider(QtWidgets.QWidget):
    """Horizontal slider + numeric box, sharing one value (float with `decimals`)."""

    valueChanged = QtCore.Signal(float)

    def __init__(self, label: str, vmin: float, vmax: float, value: float, step: float = 1.0,
                 decimals: int = 0, suffix: str = "", label_width: int = 86, parent=None):
        super().__init__(parent)
        self.decimals = decimals
        self.scale = 10 ** decimals
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.label = QtWidgets.QLabel(label)
        self.label.setObjectName("Muted")
        self.label.setFixedWidth(label_width)
        lay.addWidget(self.label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(int(round(vmin * self.scale)), int(round(vmax * self.scale)))
        self.slider.setSingleStep(max(1, int(round(step * self.scale))))
        self.slider.setPageStep(max(1, int(round(step * self.scale * 10))))
        lay.addWidget(self.slider, 1)
        if decimals == 0:
            self.box = QtWidgets.QSpinBox()
            self.box.setRange(int(vmin), int(vmax))
            self.box.setSingleStep(int(step))
        else:
            self.box = QtWidgets.QDoubleSpinBox()
            self.box.setRange(vmin, vmax)
            self.box.setDecimals(decimals)
            self.box.setSingleStep(step)
        self.box.setSuffix(suffix)
        self.box.setFixedWidth(72)
        self.box.setAlignment(QtCore.Qt.AlignRight)
        self.box.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        lay.addWidget(self.box)
        self._block = False
        self.slider.valueChanged.connect(self._from_slider)
        self.box.valueChanged.connect(self._from_box)
        self.setValue(value)

    def setRange(self, vmin: float, vmax: float):
        self._block = True
        self.slider.setRange(int(round(vmin * self.scale)), int(round(vmax * self.scale)))
        if self.decimals == 0:
            self.box.setRange(int(vmin), int(vmax))
        else:
            self.box.setRange(vmin, vmax)
        self._block = False

    def value(self) -> float:
        return float(self.box.value())

    def setValue(self, v: float):
        self._block = True
        self.box.setValue(v if self.decimals else int(round(v)))
        self.slider.setValue(int(round(v * self.scale)))
        self._block = False

    def _from_slider(self, iv: int):
        if self._block:
            return
        self._block = True
        v = iv / self.scale
        self.box.setValue(v if self.decimals else int(round(v)))
        self._block = False
        self.valueChanged.emit(float(v))

    def _from_box(self, v):
        if self._block:
            return
        self._block = True
        self.slider.setValue(int(round(float(v) * self.scale)))
        self._block = False
        self.valueChanged.emit(float(v))


class SegmentedControl(QtWidgets.QWidget):
    """A row of mutually exclusive buttons."""

    changed = QtCore.Signal(int)

    def __init__(self, labels: list[str], current: int = 0, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: list[QtWidgets.QPushButton] = []
        for i, text in enumerate(labels):
            b = QtWidgets.QPushButton(text)
            b.setCheckable(True)
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.setObjectName("SegL" if i == 0 else "SegR" if i == len(labels) - 1 else "Seg")
            b.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            self.group.addButton(b, i)
            lay.addWidget(b)
            self.buttons.append(b)
        self.buttons[current].setChecked(True)
        self.group.idClicked.connect(self.changed)

    def current(self) -> int:
        return self.group.checkedId()

    def setCurrent(self, i: int):
        self.buttons[i].setChecked(True)

    def setEnabledIndex(self, i: int, enabled: bool):
        self.buttons[i].setEnabled(enabled)


class ColorChip(QtWidgets.QLabel):
    def __init__(self, color: str, size: int = 12, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setColor(color)

    def setColor(self, color: str):
        self.setStyleSheet(f"background:{color}; border-radius:{self.width() // 2}px; border:1px solid #00000055;")


def make_button(text: str, kind: str = "", tooltip: str = "", icon: QtGui.QIcon | None = None) -> QtWidgets.QPushButton:
    b = QtWidgets.QPushButton(text)
    if kind:
        b.setObjectName(kind)
    if tooltip:
        b.setToolTip(tooltip)
    if icon is not None:
        b.setIcon(icon)
    b.setCursor(QtCore.Qt.PointingHandCursor)
    return b


def color_icon(color: str, size: int = 10) -> QtGui.QIcon:
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setBrush(QtGui.QColor(color))
    p.setPen(QtCore.Qt.NoPen)
    p.drawEllipse(0, 0, size, size)
    p.end()
    return QtGui.QIcon(pm)


def hsep() -> QtWidgets.QFrame:
    f = QtWidgets.QFrame()
    f.setFrameShape(QtWidgets.QFrame.HLine)
    f.setStyleSheet(f"color:{theme.BORDER}; background:{theme.BORDER}; max-height:1px; border:none;")
    return f
