"""Spectrum plots: multi-point spectra with draggable R/G/B band cursors; cluster mean +/- std."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import theme


def _style_plot(pw: pg.PlotWidget, xlabel: str, ylabel: str):
    pi = pw.getPlotItem()
    pi.showGrid(x=True, y=True, alpha=0.18)
    pi.getAxis("bottom").setPen(pg.mkPen(theme.BORDER))
    pi.getAxis("left").setPen(pg.mkPen(theme.BORDER))
    pi.getAxis("bottom").setTextPen(pg.mkPen(theme.MUTED))
    pi.getAxis("left").setTextPen(pg.mkPen(theme.MUTED))
    pi.setLabel("bottom", xlabel, color=theme.MUTED)
    pi.setLabel("left", ylabel, color=theme.MUTED)
    pi.getViewBox().setBackgroundColor(theme.BG0)
    pi.layout.setContentsMargins(6, 10, 14, 6)
    pw.setBackground(theme.BG1)
    pi.setMenuEnabled(False)
    pi.hideButtons()
    # 그래프는 고정: 마우스 휠/드래그로 확대·이동하지 않음 (범위는 데이터에서 직접 계산)
    pi.setMouseEnabled(x=False, y=False)
    pi.disableAutoRange()


def _fit_y(pi: pg.PlotItem, arrays, pad: float = 0.08):
    """주어진 배열들의 최소/최대로 Y 범위를 한 번만 고정 (마우스 미리보기는 포함하지 않음)."""
    vals = [np.asarray(a, np.float32) for a in arrays if a is not None and np.size(a)]
    if not vals:
        return
    allv = np.concatenate([v.ravel() for v in vals])
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        return
    lo, hi = float(allv.min()), float(allv.max())
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    span = hi - lo
    pi.setYRange(lo - span * pad, hi + span * pad, padding=0)


class SpectrumView(QtWidgets.QWidget):
    """Point/ROI spectra + three draggable band cursors."""

    cursorMoved = QtCore.Signal(int, int)        # channel idx (0,1,2), band index
    cursorMoveFinished = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(12, 8, 12, 0)
        self.title = QtWidgets.QLabel("스펙트럼")
        self.title.setObjectName("CardTitle")
        top.addWidget(self.title)
        self.info = QtWidgets.QLabel("")
        self.info.setObjectName("Hint")
        top.addWidget(self.info, 1, QtCore.Qt.AlignRight)
        lay.addLayout(top)

        self.pw = pg.PlotWidget()
        _style_plot(self.pw, "밴드", "세기")
        lay.addWidget(self.pw, 1)
        self.pi = self.pw.getPlotItem()
        self.legend = self.pi.addLegend(offset=(-8, -8), labelTextColor=theme.TEXT, brush=pg.mkBrush(theme.BG1 + "cc"),
                                        pen=pg.mkPen(theme.BORDER))
        self.legend.setZValue(30)
        self.legend.setVisible(False)

        self._x = np.arange(1, dtype=np.float32)
        self._curves: list[pg.PlotDataItem] = []
        self._fills: list[pg.FillBetweenItem] = []
        self._autoscale = True

        self.hover_curve = pg.PlotDataItem(pen=pg.mkPen(theme.MUTED, width=1.2, style=QtCore.Qt.DotLine))
        self.hover_curve.setZValue(5)
        self.pi.addItem(self.hover_curve, ignoreBounds=True)

        self.cursors: list[pg.InfiniteLine] = []
        for i, (c, lab) in enumerate(((theme.CURSOR_R, "R"), (theme.CURSOR_G, "G"), (theme.CURSOR_B, "B"))):
            line = pg.InfiniteLine(pos=i, angle=90, movable=True, pen=pg.mkPen(c, width=2),
                                   hoverPen=pg.mkPen(c, width=3), label=lab,
                                   labelOpts={"position": 0.95, "color": c, "fill": pg.mkBrush(theme.BG1 + "cc"),
                                              "movable": False})
            line.setZValue(10)
            line.sigDragged.connect(lambda ln, i=i: self._cursor_dragged(i, ln))
            line.sigPositionChangeFinished.connect(lambda ln: self.cursorMoveFinished.emit())
            self.pi.addItem(line, ignoreBounds=True)
            self.cursors.append(line)
        self._cursor_block = False

    # ------------------------------------------------------------------ API
    def set_axis(self, x: np.ndarray, label: str):
        self._x = np.asarray(x, dtype=np.float32)
        self.pi.setLabel("bottom", label, color=theme.MUTED)
        n = len(self._x)
        if n > 1:
            for ln in self.cursors:
                ln.setBounds((float(self._x.min()), float(self._x.max())))
        if n:
            self.pi.setXRange(float(self._x.min()), float(self._x.max()), padding=0.02)

    def set_ylabel(self, label: str):
        self.pi.setLabel("left", label, color=theme.MUTED)

    def set_autoscale(self, on: bool):
        self._autoscale = on

    def set_spectra(self, entries: list[dict]):
        """entries: dict(label, color, y, std=None, width=2, style='solid'|'dash')."""
        for c in self._curves:
            self.pi.removeItem(c)
        for f in self._fills:
            self.pi.removeItem(f)
        self._curves.clear()
        self._fills.clear()
        self.legend.clear()
        for e in entries:
            y = np.asarray(e["y"], np.float32)
            x = self._x if len(self._x) == len(y) else np.arange(len(y), dtype=np.float32)
            style = QtCore.Qt.DashLine if e.get("style") == "dash" else QtCore.Qt.SolidLine
            pen = pg.mkPen(e["color"], width=e.get("width", 2), style=style)
            curve = pg.PlotDataItem(x, y, pen=pen, name=e.get("label"), antialias=True)
            curve.setZValue(8)
            self.pi.addItem(curve)
            self._curves.append(curve)
            sd = e.get("std")
            if sd is not None:
                sd = np.asarray(sd, np.float32)
                hi = pg.PlotDataItem(x, y + sd)
                lo = pg.PlotDataItem(x, y - sd)
                col = QtGui.QColor(e["color"])
                col.setAlpha(45)
                fill = pg.FillBetweenItem(hi, lo, brush=pg.mkBrush(col))
                fill.setZValue(6)
                self.pi.addItem(fill)
                self._fills.append(fill)
        if self._autoscale:
            arrays = []
            for e in entries:
                y = np.asarray(e["y"], np.float32)
                arrays.append(y)
                if e.get("std") is not None:
                    sd = np.asarray(e["std"], np.float32)
                    arrays += [y + sd, y - sd]
            _fit_y(self.pi, arrays)
        self.legend.setVisible(bool(entries))
        self.info.setText(f"{len(entries)}개 트레이스" if entries else "")

    def set_hover(self, y: np.ndarray | None):
        if y is None:
            self.hover_curve.setData([], [])
            return
        x = self._x if len(self._x) == len(y) else np.arange(len(y), dtype=np.float32)
        self.hover_curve.setData(x, np.asarray(y, np.float32))

    def set_cursors(self, bands: tuple[int, int, int]):
        self._cursor_block = True
        n = len(self._x)
        for ln, b in zip(self.cursors, bands):
            b = int(np.clip(b, 0, max(0, n - 1)))
            ln.setPos(float(self._x[b]) if n else b)
        self._cursor_block = False

    def set_cursors_visible(self, on: bool):
        for ln in self.cursors:
            ln.setVisible(on)

    def band_from_x(self, xv: float) -> int:
        if len(self._x) == 0:
            return 0
        return int(np.argmin(np.abs(self._x - xv)))

    def _cursor_dragged(self, i: int, line: pg.InfiniteLine):
        if self._cursor_block:
            return
        b = self.band_from_x(float(line.value()))
        # snap to band position
        self._cursor_block = True
        line.setPos(float(self._x[b]))
        self._cursor_block = False
        self.cursorMoved.emit(i, b)


class ClusterSpectrumView(QtWidgets.QWidget):
    """Cluster mean +/- std spectra, with optional per-piece traces."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(12, 8, 12, 0)
        self.title = QtWidgets.QLabel("클러스터 스펙트럼")
        self.title.setObjectName("CardTitle")
        top.addWidget(self.title)
        self.info = QtWidgets.QLabel("")
        self.info.setObjectName("Hint")
        top.addWidget(self.info, 1, QtCore.Qt.AlignRight)
        self.chk_pieces = QtWidgets.QCheckBox("객체별")
        self.chk_pieces.setChecked(True)
        self.chk_pieces.setToolTip("각 객체의 대표 스펙트럼을 가는 선으로 표시")
        top.addWidget(self.chk_pieces)
        self.chk_std = QtWidgets.QCheckBox("±1σ")
        self.chk_std.setChecked(True)
        top.addWidget(self.chk_std)
        lay.addLayout(top)

        self.pw = pg.PlotWidget()
        _style_plot(self.pw, "밴드", "세기")
        lay.addWidget(self.pw, 1)
        self.pi = self.pw.getPlotItem()
        self.legend = self.pi.addLegend(offset=(-8, 8), labelTextColor=theme.TEXT, brush=pg.mkBrush(theme.BG1 + "cc"),
                                        pen=pg.mkPen(theme.BORDER))
        self._items: list = []
        self._last = None
        self.legend.setVisible(False)
        self.chk_pieces.toggled.connect(lambda _: self._redraw())
        self.chk_std.toggled.connect(lambda _: self._redraw())

    def set_axis(self, x: np.ndarray, label: str):
        self._x = np.asarray(x, np.float32)
        self.pi.setLabel("bottom", label, color=theme.MUTED)
        if len(self._x):
            self.pi.setXRange(float(self._x.min()), float(self._x.max()), padding=0.02)

    def clear(self):
        for it in self._items:
            self.pi.removeItem(it)
        self._items.clear()
        self.legend.clear()
        self.legend.setVisible(False)
        self._last = None
        self.info.setText("")

    def set_result(self, result, colors: list[str], ylabel: str, highlight_piece: int | None = None,
                   space: str = "raw"):
        """space: 'raw' = 원 세기, 'snv' = 픽셀별 SNV. 클러스터링에 쓴 정규화 공간이 아니라
        포인트 스펙트럼과 같은 공간에 그려서 서로 비교할 수 있게 한다."""
        self._last = (result, colors, ylabel, highlight_piece, space)
        self._redraw()

    def _redraw(self):
        if self._last is None:
            return
        result, colors, ylabel, highlight, space = self._last
        for it in self._items:
            self.pi.removeItem(it)
        self._items.clear()
        self.legend.clear()
        self.legend.setVisible(True)
        self.pi.setLabel("left", ylabel, color=theme.MUTED)
        disp = getattr(result, "display", None) or {}
        feats, means, stds = disp.get(space, (result.features, result.means, result.stds))
        x = getattr(self, "_x", None)
        b = means.shape[1]
        if x is None or len(x) != b:
            x = np.arange(b, dtype=np.float32)
        yarrays = [means]
        if self.chk_std.isChecked():
            yarrays += [means + stds, means - stds]
        if self.chk_pieces.isChecked() and feats.shape[0] == len(result.labels):
            yarrays.append(feats)
        _fit_y(self.pi, yarrays)

        if self.chk_pieces.isChecked() and feats.shape[0] == len(result.labels):
            for i, (feat, lab) in enumerate(zip(feats, result.labels)):
                col = QtGui.QColor(colors[int(lab) % len(colors)])
                is_hl = highlight is not None and (i + 1) == highlight
                col.setAlpha(255 if is_hl else 110)
                c = pg.PlotDataItem(x, feat, pen=pg.mkPen(col, width=2.5 if is_hl else 0.9))
                c.setZValue(9 if is_hl else 4)
                self.pi.addItem(c)
                self._items.append(c)

        names = (getattr(result, "extra", None) or {}).get("names") or []
        for c_idx in range(result.k):
            col = colors[c_idx % len(colors)]
            mu, sd = means[c_idx], stds[c_idx]
            cname = names[c_idx] if c_idx < len(names) else f"C{c_idx}"
            name = f"{cname}  (객체 {int(result.counts_pieces[c_idx])}개, {int(result.counts_px[c_idx])} px)"
            curve = pg.PlotDataItem(x, mu, pen=pg.mkPen(col, width=2.5), name=name)
            curve.setZValue(8)
            self.pi.addItem(curve)
            self._items.append(curve)
            if self.chk_std.isChecked():
                hi = pg.PlotDataItem(x, mu + sd)
                lo = pg.PlotDataItem(x, mu - sd)
                qc = QtGui.QColor(col)
                qc.setAlpha(40)
                fill = pg.FillBetweenItem(hi, lo, brush=pg.mkBrush(qc))
                fill.setZValue(5)
                self.pi.addItem(fill)
                self._items.append(fill)
        extra = []
        if result.silhouette is not None:
            extra.append(f"실루엣 {result.silhouette:.2f}")
        self.info.setText(f"{result.method} · K={result.k}" + (" · " + ", ".join(extra) if extra else ""))


class PieceSpectrumView(QtWidgets.QWidget):
    """탐지된 객체별 대표 스펙트럼. 전부 흐리게 깔고, 선택한 객체만 굵게 강조."""

    pieceClicked = QtCore.Signal(int)            # 클릭한 선의 객체 id (1부터)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(12, 8, 12, 0)
        self.title = QtWidgets.QLabel("객체 스펙트럼")
        self.title.setObjectName("CardTitle")
        top.addWidget(self.title)
        self.info = QtWidgets.QLabel("")
        self.info.setObjectName("Hint")
        top.addWidget(self.info, 1, QtCore.Qt.AlignRight)
        lay.addLayout(top)

        self.pw = pg.PlotWidget()
        _style_plot(self.pw, "밴드", "세기")
        lay.addWidget(self.pw, 1)
        self.pi = self.pw.getPlotItem()
        self._items: list = []
        self._x = None

    def set_axis(self, x: np.ndarray, label: str):
        self._x = np.asarray(x, np.float32)
        self.pi.setLabel("bottom", label, color=theme.MUTED)
        if len(self._x):
            self.pi.setXRange(float(self._x.min()), float(self._x.max()), padding=0.02)

    def clear(self):
        for it in self._items:
            self.pi.removeItem(it)
        self._items.clear()
        self.info.setText("")

    def set_pieces(self, feats: np.ndarray, colors: list[str], names: list[str], ylabel: str,
                   highlight: int | None = None):
        """feats: (N, B) 객체별 스펙트럼, colors/names: 객체마다 하나, highlight: 강조할 객체 id (1부터)."""
        self.clear()
        self.pi.setLabel("left", ylabel, color=theme.MUTED)
        if feats.shape[0] == 0:
            return
        b = feats.shape[1]
        x = self._x if self._x is not None and len(self._x) == b else np.arange(b, dtype=np.float32)
        _fit_y(self.pi, [feats])
        for i, (feat, col) in enumerate(zip(feats, colors)):
            is_hl = highlight is not None and (i + 1) == highlight
            qc = QtGui.QColor(col)
            if highlight is None:
                qc.setAlpha(255 if feats.shape[0] <= 12 else 120)   # 객체가 많으면 겹치니 반투명
            else:
                qc.setAlpha(255 if is_hl else 50)                    # 선택한 객체만 진하게
            c = pg.PlotDataItem(x, feat, pen=pg.mkPen(qc, width=2.8 if is_hl else 1.0), antialias=True)
            c.setZValue(9 if is_hl else 4)
            c.setCurveClickable(True, width=8)   # 가는 선도 클릭되도록 판정 폭 8px
            # pid=i + 1 기본값으로 지금의 i를 고정 (안 하면 모든 선이 마지막 i를 씀).
            # 선택하면 그래프를 다시 그려 이 선이 지워지므로, 클릭 처리가 끝난 뒤(singleShot 0)에 알린다
            c.sigClicked.connect(lambda _item, _ev, pid=i + 1:
                                 QtCore.QTimer.singleShot(0, lambda: self.pieceClicked.emit(pid)))
            self.pi.addItem(c)
            self._items.append(c)
        if highlight is not None and 1 <= highlight <= len(names):
            self.info.setText(f"객체 {len(names)}개 · 선택: 객체 {highlight} ({names[highlight - 1]})")
        else:
            self.info.setText(f"객체 {len(names)}개 · 이미지의 객체나 그래프의 선을 클릭하면 강조")
