"""pyqtgraph-based image canvas: base image, overlay, draggable point markers, ROIs, hover."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import theme


class _RemovableRectROI(pg.RectROI):
    """오른쪽 클릭하면 바로 삭제 요청(sigRemoveRequested)을 보내는 사각 ROI."""

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            if self.isMoving:
                self.cancelMove()
                return
            # 이벤트 처리 중에 자기 자신을 지우면 불안정하므로 처리가 끝난 뒤 삭제
            QtCore.QTimer.singleShot(0, lambda: self.sigRemoveRequested.emit(self))
            return
        super().mouseClickEvent(ev)


class _PickViewBox(pg.ViewBox):
    """ViewBox where a plain left click picks, left-drag near a marker moves it, else pans."""

    clicked = QtCore.Signal(float, float, int, object)   # x, y, button (1=left, 2=right), screen QPointF
    dragStarted = QtCore.Signal(float, float)
    dragMoved = QtCore.Signal(float, float)
    dragFinished = QtCore.Signal()
    boxDragged = QtCore.Signal(float, float, float, float, bool)   # x0, y0, x1, y1, finished

    def __init__(self, hit_test, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hit_test = hit_test           # callable(x, y) -> bool  (is a marker near?)
        self._dragging_marker = False
        self.box_tool = False
        self.setMenuEnabled(False)
        self.setMouseMode(pg.ViewBox.PanMode)

    def mouseClickEvent(self, ev):
        if ev.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            p = self.mapToView(ev.pos())
            ev.accept()
            self.clicked.emit(p.x(), p.y(), 1 if ev.button() == QtCore.Qt.LeftButton else 2, ev.screenPos())
            return
        super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.LeftButton and self.box_tool:
            ev.accept()
            p0 = self.mapToView(ev.buttonDownPos())
            p1 = self.mapToView(ev.pos())
            self.boxDragged.emit(p0.x(), p0.y(), p1.x(), p1.y(), ev.isFinish())
            return
        if ev.button() == QtCore.Qt.LeftButton:
            if ev.isStart():
                p = self.mapToView(ev.buttonDownPos())
                self._dragging_marker = bool(self._hit_test(p.x(), p.y()))
                if self._dragging_marker:
                    ev.accept()
                    self.dragStarted.emit(p.x(), p.y())
                    return
            if self._dragging_marker:
                ev.accept()
                p = self.mapToView(ev.pos())
                self.dragMoved.emit(p.x(), p.y())
                if ev.isFinish():
                    self._dragging_marker = False
                    self.dragFinished.emit()
                return
        if ev.button() == QtCore.Qt.RightButton:
            # right-drag: zoom like pyqtgraph default but keep it accepted
            super().mouseDragEvent(ev, axis)
            return
        super().mouseDragEvent(ev, axis)


class ImageCanvas(QtWidgets.QWidget):
    """Image viewer with picking, markers, overlays and ROIs."""

    pointAdded = QtCore.Signal(int, int)
    pointRemoveRequested = QtCore.Signal(int, int)
    pieceContextRequested = QtCore.Signal(int, object)   # piece id, global QPoint
    boxDrawn = QtCore.Signal(int, int, int, int)         # x0, y0, x1, y1 (pixel coords, inclusive-exclusive)
    pointMoved = QtCore.Signal(int, int, int)          # index, x, y
    pointMoveFinished = QtCore.Signal()
    hovered = QtCore.Signal(int, int)                  # x, y (or -1,-1 when leaving)
    aspectChanged = QtCore.Signal(bool)                # 비율 고정 상태가 바뀜 (툴바 체크 동기화)
    roiChanged = QtCore.Signal(int)                    # index of ROI changed
    roiRemoveRequested = QtCore.Signal(int)
    pieceClicked = QtCore.Signal(int)                  # piece id under click (0 = none)
    promptClicked = QtCore.Signal(int, int)            # 점·박스 모드에서 왼쪽 클릭한 픽셀 (x, y)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(theme.BG1)
        self.glw.ci.layout.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.glw)

        self.vb = _PickViewBox(self._hit_test, lockAspect=True, invertY=True, enableMenu=False)
        self.glw.addItem(self.vb)
        self.vb.setBackgroundColor(theme.BG0)
        self.vb.setDefaultPadding(0.02)

        self.img_item = pg.ImageItem(axisOrder="row-major")
        self.vb.addItem(self.img_item)
        self.overlay_item = pg.ImageItem(axisOrder="row-major")
        self.overlay_item.setZValue(5)
        self.overlay_item.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
        self.vb.addItem(self.overlay_item)
        self.overlay_item.hide()

        self.markers = pg.ScatterPlotItem(size=16, symbol="+", pxMode=True)
        self.markers.setZValue(20)
        self.vb.addItem(self.markers)

        self._bin_rects: list[QtWidgets.QGraphicsRectItem] = []
        self._labels: list[pg.TextItem] = []
        self._piece_labels: list[pg.TextItem] = []
        self._rois: list[pg.RectROI] = []
        self._highlight: QtWidgets.QGraphicsRectItem | None = None
        self._boxes: list[QtWidgets.QGraphicsRectItem] = []
        self._box_preview: QtWidgets.QGraphicsRectItem | None = None
        self._points: list[tuple[int, int, str]] = []
        self._drag_idx: int | None = None
        self._shape = None
        self._piece_label_map: np.ndarray | None = None
        self._aspect_locked = True
        self._auto_aspect = True

        # placeholder text
        self.placeholder = pg.TextItem(
            html=f'<div style="color:{theme.MUTED}; font-size:13pt; text-align:center;">'
                 f'<b>이미지가 없습니다</b><br/><span style="font-size:10pt;">파일 열기 (Ctrl+O) 또는 여기로 드래그</span></div>',
            anchor=(0.5, 0.5),
        )
        self.placeholder.setZValue(50)
        self.vb.addItem(self.placeholder)
        self.placeholder.setPos(0.5, 0.5)
        self.vb.setRange(xRange=(0, 1), yRange=(0, 1))

        self.vb.clicked.connect(self._on_click)
        self.vb.boxDragged.connect(self._on_box_drag)
        self.vb.dragStarted.connect(self._on_drag_start)
        self.vb.dragMoved.connect(self._on_drag_move)
        self.vb.dragFinished.connect(self._on_drag_finish)
        self.glw.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.setAcceptDrops(False)

    # ------------------------------------------------------------------ images
    def set_image(self, img: np.ndarray, levels=None, lut=None, reset_view: bool = False):
        """img: (H,W) float or (H,W,3) uint8."""
        first = self._shape is None or self._shape != img.shape[:2]
        self._shape = img.shape[:2]
        if first and self._auto_aspect:
            # 가로세로 비율이 극단적인 큐브(라인스캔 등)는 화면에 맞게 늘려서 보여준다
            h, w = self._shape
            self.set_aspect_locked(max(w / max(h, 1), h / max(w, 1)) <= 6.0)
        if img.ndim == 2:
            self.img_item.setImage(img, levels=levels, lut=lut, autoLevels=False)
        else:
            self.img_item.setImage(img, autoLevels=False, levels=None, lut=None)
        self.placeholder.hide()
        if first or reset_view:
            self.fit_view()

    def fit_view(self):
        if self._shape is None:
            return
        h, w = self._shape
        self.vb.setRange(xRange=(0, w), yRange=(0, h), padding=0.02)

    def set_aspect_locked(self, locked: bool):
        """True = 픽셀 정사각형 유지, False = 캔버스에 맞춰 늘림 (좌표는 그대로 데이터 픽셀)."""
        self._aspect_locked = bool(locked)
        self.vb.setAspectLocked(self._aspect_locked)
        self.aspectChanged.emit(self._aspect_locked)
        self.fit_view()

    def aspect_locked(self) -> bool:
        return self._aspect_locked

    def set_overlay(self, rgba: np.ndarray | None):
        if rgba is None:
            self.overlay_item.hide()
            return
        self.overlay_item.setImage(rgba, autoLevels=False, levels=None)
        self.overlay_item.show()

    def set_piece_labels(self, labels: list[tuple[float, float, str, str]] | None, label_map: np.ndarray | None = None):
        """labels: list of (x, y, text, color)."""
        for t in self._piece_labels:
            self.vb.removeItem(t)
        self._piece_labels.clear()
        self._piece_label_map = label_map
        if not labels:
            return
        for x, y, text, color in labels:
            t = pg.TextItem(html=f'<div style="color:{color}; font-weight:600; font-size:9pt; '
                                 f'background:rgba(0,0,0,0.45); padding:1px 3px; border-radius:3px;">{text}</div>',
                            anchor=(0.5, 0.5))
            t.setZValue(15)
            t.setPos(x, y)
            self.vb.addItem(t)
            self._piece_labels.append(t)

    def set_highlight(self, bbox: tuple[int, int, int, int] | None, color: str = theme.ACCENT):
        if self._highlight is not None:
            self.vb.removeItem(self._highlight)
            self._highlight = None
        if bbox is None:
            return
        x0, y0, x1, y1 = bbox
        r = QtWidgets.QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        pen = pg.mkPen(color, width=2, style=QtCore.Qt.DashLine)
        pen.setCosmetic(True)
        r.setPen(pen)
        r.setBrush(QtCore.Qt.NoBrush)
        r.setZValue(18)
        self.vb.addItem(r)
        self._highlight = r

    # ------------------------------------------------------------------ points
    def set_points(self, points: list[tuple[int, int, str]], colors: list[str], bin_size: int = 1):
        self._points = list(points)
        spots = []
        for (x, y, _label), c in zip(points, colors):
            spots.append({"pos": (x + 0.5, y + 0.5), "pen": pg.mkPen(c, width=2), "brush": pg.mkBrush(c)})
        self.markers.setData(spots)

        for r in self._bin_rects:
            self.vb.removeItem(r)
        self._bin_rects.clear()
        for t in self._labels:
            self.vb.removeItem(t)
        self._labels.clear()
        if self._shape is None:
            return
        h, w = self._shape
        half = bin_size // 2
        for (x, y, label), c in zip(points, colors):
            t = pg.TextItem(text=label, color=c, anchor=(0, 1))
            t.setZValue(21)
            t.setPos(x + 0.5 + 4 / max(1, self._px_per_unit()), y + 0.5)
            t.setFont(QtGui.QFont(theme.FONT_FAMILY, 9, QtGui.QFont.DemiBold))
            self.vb.addItem(t)
            self._labels.append(t)
            if bin_size > 1:
                x0, x1 = max(0, x - half), min(w, x + half + 1)
                y0, y1 = max(0, y - half), min(h, y + half + 1)
                r = QtWidgets.QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
                pen = pg.mkPen(c, width=1.2, style=QtCore.Qt.DashLine)
                pen.setCosmetic(True)
                r.setPen(pen)
                r.setBrush(QtCore.Qt.NoBrush)
                r.setZValue(19)
                self.vb.addItem(r)
                self._bin_rects.append(r)

    def _px_per_unit(self) -> float:
        try:
            return float(self.vb.viewPixelSize()[0]) ** -1
        except Exception:
            return 1.0

    def _nearest_point(self, x: float, y: float, max_px: float = 12.0) -> int | None:
        if not self._points:
            return None
        try:
            unit = self.vb.viewPixelSize()[0]   # view units per screen pixel
        except Exception:
            unit = 1.0
        tol = max_px * unit
        d = [np.hypot(x - (px + 0.5), y - (py + 0.5)) for px, py, _ in self._points]
        i = int(np.argmin(d))
        return i if d[i] <= tol else None

    def _hit_test(self, x: float, y: float) -> bool:
        return self._nearest_point(x, y) is not None

    def _inside(self, x: float, y: float) -> tuple[int, int] | None:
        if self._shape is None:
            return None
        h, w = self._shape
        xi, yi = int(np.floor(x)), int(np.floor(y))
        if 0 <= xi < w and 0 <= yi < h:
            return xi, yi
        return None

    # ------------------------------------------------------------------ events
    def _piece_at(self, xi: int, yi: int) -> int:
        if self._piece_label_map is None or not self.overlay_item.isVisible():
            return 0
        return int(self._piece_label_map[yi, xi])

    def _on_click(self, x: float, y: float, button: int, screen_pos):
        pos = self._inside(x, y)
        if pos is None:
            return
        xi, yi = pos
        if button == 1:
            if self.vb.box_tool:
                self.promptClicked.emit(xi, yi)
                return
            pid = self._piece_at(xi, yi)
            if pid > 0 and self._nearest_point(x, y) is None:
                self.pieceClicked.emit(pid)
                return
            self.pointAdded.emit(xi, yi)
        else:
            pid = self._piece_at(xi, yi)
            if pid > 0 and self._nearest_point(x, y) is None:
                try:
                    gp = QtCore.QPoint(int(screen_pos.x()), int(screen_pos.y()))
                except Exception:
                    gp = QtGui.QCursor.pos()
                self.pieceContextRequested.emit(pid, gp)
                return
            self.pointRemoveRequested.emit(xi, yi)

    # ------------------------------------------------------------------ box tool
    def set_box_tool(self, on: bool):
        self.vb.box_tool = bool(on)
        self.glw.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)

    def _on_box_drag(self, x0: float, y0: float, x1: float, y1: float, finished: bool):
        if self._shape is None:
            return
        h, w = self._shape
        x0, x1 = sorted((float(np.clip(x0, 0, w)), float(np.clip(x1, 0, w))))
        y0, y1 = sorted((float(np.clip(y0, 0, h)), float(np.clip(y1, 0, h))))
        if self._box_preview is None:
            r = QtWidgets.QGraphicsRectItem()
            pen = pg.mkPen(theme.WARN, width=2, style=QtCore.Qt.DashLine)
            pen.setCosmetic(True)
            r.setPen(pen)
            r.setBrush(pg.mkBrush(255, 200, 80, 40))
            r.setZValue(26)
            self.vb.addItem(r)
            self._box_preview = r
        self._box_preview.setRect(x0, y0, x1 - x0, y1 - y0)
        if finished:
            self.vb.removeItem(self._box_preview)
            self._box_preview = None
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                self.boxDrawn.emit(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))

    def set_boxes(self, boxes: list[tuple[int, int, int, int]]):
        for r in self._boxes:
            self.vb.removeItem(r)
        self._boxes.clear()
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            r = QtWidgets.QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
            pen = pg.mkPen(theme.WARN, width=2)
            pen.setCosmetic(True)
            r.setPen(pen)
            r.setBrush(QtCore.Qt.NoBrush)
            r.setZValue(24)
            self.vb.addItem(r)
            self._boxes.append(r)
            t = pg.TextItem(text=f"박스 {i + 1}", color=theme.WARN, anchor=(0, 1))
            t.setPos(x0, y0)
            t.setZValue(24)
            self.vb.addItem(t)
            self._boxes.append(t)

    def _on_drag_start(self, x: float, y: float):
        self._drag_idx = self._nearest_point(x, y)

    def _on_drag_move(self, x: float, y: float):
        if self._drag_idx is None:
            return
        pos = self._inside(x, y)
        if pos is None:
            return
        self.pointMoved.emit(self._drag_idx, pos[0], pos[1])

    def _on_drag_finish(self):
        self._drag_idx = None
        self.pointMoveFinished.emit()

    def _on_mouse_moved(self, scene_pos):
        if self._shape is None:
            return
        if not self.vb.sceneBoundingRect().contains(scene_pos):
            self.hovered.emit(-1, -1)
            return
        p = self.vb.mapSceneToView(scene_pos)
        pos = self._inside(p.x(), p.y())
        if pos is None:
            self.hovered.emit(-1, -1)
        else:
            self.hovered.emit(pos[0], pos[1])

    def leaveEvent(self, ev):
        self.hovered.emit(-1, -1)
        super().leaveEvent(ev)

    # ------------------------------------------------------------------ ROIs
    def add_roi(self, color: str, rect: tuple[int, int, int, int] | None = None) -> int:
        if self._shape is None:
            return -1
        h, w = self._shape
        if rect is None:
            # centre of the current view, 15% of the visible range
            (x0, x1), (y0, y1) = self.vb.viewRange()
            cw, ch = (x1 - x0) * 0.15, (y1 - y0) * 0.15
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rect = (int(cx - cw / 2), int(cy - ch / 2), int(cw), int(ch))
        x, y, rw, rh = rect
        # 이미지 밖으로 나가지 않게 (가로세로 비율이 극단적인 이미지에서 ROI가 이미지보다 커지는 것 방지)
        rw, rh = max(1, min(rw, w)), max(1, min(rh, h))
        x, y = int(np.clip(x, 0, w - rw)), int(np.clip(y, 0, h - rh))
        roi = _RemovableRectROI([x, y], [max(1, rw), max(1, rh)], pen=pg.mkPen(color, width=2),
                         hoverPen=pg.mkPen(color, width=3), handlePen=pg.mkPen(color),
                         maxBounds=QtCore.QRectF(0, 0, w, h), snapSize=1, translateSnap=True, scaleSnap=True)
        roi.setZValue(25)
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([0, 0], [1, 1])
        self.vb.addItem(roi)
        self._rois.append(roi)
        idx = len(self._rois) - 1
        roi.sigRegionChanged.connect(lambda r, i=idx: self._roi_changed(r))
        roi.sigRemoveRequested.connect(lambda r: self._roi_remove(r))
        return idx

    def _roi_changed(self, roi):
        if roi in self._rois:
            self.roiChanged.emit(self._rois.index(roi))

    def _roi_remove(self, roi):
        if roi in self._rois:
            self.roiRemoveRequested.emit(self._rois.index(roi))

    def roi_rect(self, idx: int) -> tuple[int, int, int, int]:
        roi = self._rois[idx]
        p, s = roi.pos(), roi.size()
        return int(round(p.x())), int(round(p.y())), int(round(p.x() + s.x())), int(round(p.y() + s.y()))

    def remove_roi(self, idx: int):
        roi = self._rois.pop(idx)
        self.vb.removeItem(roi)

    def clear_rois(self):
        for r in self._rois:
            self.vb.removeItem(r)
        self._rois.clear()

    def set_roi_color(self, idx: int, color: str):
        self._rois[idx].setPen(pg.mkPen(color, width=2))

    def roi_count(self) -> int:
        return len(self._rois)

    # ------------------------------------------------------------------ misc
    def clear_all(self):
        self.set_points([], [])
        self.clear_rois()
        self.set_boxes([])
        self.set_overlay(None)
        self.set_piece_labels(None)
        self.set_highlight(None)

    def grab_image(self) -> QtGui.QImage:
        return self.glw.grab().toImage()
