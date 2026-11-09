"""Labeler 그림 칸: 이미지를 보여주고 박스(드래그) 또는 붓(Segmentation)으로 라벨을 그린다.
붓 모드: 왼쪽 드래그 칠하기, 오른쪽 드래그 지우기, 마우스 휠 붓 크기.
SAM 3 모드: 왼쪽 클릭 = 포함 점, 오른쪽 클릭 = 제외 점, 드래그 = 박스 (신호만 보내고 SAM 호출은 app이 한다).
SAM 3 결과가 있을 때: Ctrl+클릭 = 그 결과 고르기(Shift를 같이 누르면 더하기), 숫자키 = 고른 결과의 클래스.
확대/이동: Ctrl+휠(붓 모드가 아니면 그냥 휠도) = 마우스 위치 기준 확대, 가운데 버튼 드래그 = 이동, fit_view() = 맞춤.
좌표는 모두 원본 이미지 픽셀 기준. 마스크는 클래스 번호+1 값(0은 배경)의 8비트 그레이 이미지."""
import numpy as np   # 마스크를 색 이미지로 바꿀 때 쓴다 (MATLAB의 배열 인덱싱처럼 lut[arr]로 한 번에 변환)
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from common import theme

UNDO_MAX = 30   # 되돌리기 최대 단계


class Canvas(QWidget):
    changed = Signal()          # 라벨이 추가/삭제/칠해졌다
    brush_changed = Signal(int)  # 붓 크기(이미지 픽셀)가 바뀌었다
    sam_point = Signal(float, float, bool)           # SAM 3 점 (x, y, 포함 여부)
    sam_box = Signal(float, float, float, float)     # SAM 3 박스 (x1, y1, x2, y2)
    sam_undo = Signal()                              # SAM 3 마지막 점 지우기 (오른쪽 클릭)
    sam_apply = Signal()                             # SAM 3 결과 적용 (Enter)
    sam_exit = Signal()                              # SAM 3 보조 모드 끄기 (Esc)
    sam_pick = Signal(float, float, bool)            # SAM 3 결과 고르기 (x, y, 더하기 여부)
    digit = Signal(int)                              # 숫자키 0~9
    hovered = Signal(int, int)                       # 마우스가 있는 이미지 픽셀 (x, y) (다파장 스펙트럼용)
    zoom_changed = Signal(float)                     # 확대 배율 (1 = 칸에 맞춤)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.pm = None
        self.tool = "box"            # "box"(객체 인식) 또는 "seg"(Segmentation 붓)
        self.cls = 0
        self.colors = []             # 클래스 번호 -> '#rrggbb'
        self.boxes = []              # (클래스, x1, y1, x2, y2)
        self.mask = None             # QImage Grayscale8, 클래스 번호+1 (0은 배경)
        self.overlay = None          # 마스크를 색으로 칠한 QImage ARGB32 (화면 표시용)
        self.brush = 20              # 붓 지름 (이미지 픽셀)
        self.opacity = 0.43          # 칠한 색/SAM 3 미리보기의 불투명도 0~1 (0.43이 예전 고정값과 같은 진하기)
        self.sam = False             # True면 클릭/드래그가 SAM 3 지정으로 쓰인다
        self.sam_pts = []            # SAM 3로 지정한 점 (x, y, 포함 여부)
        self.preview = None          # SAM 3 결과 미리보기 QImage ARGB32
        self._drag = None            # 박스 드래그 시작점/현재점 (이미지 좌표)
        self._stroke = None          # 붓으로 칠하는 중이면 (마지막 점, 지우기 여부)
        self._hover = None
        self.zoom = 1.0              # 1 = 칸에 맞춤. 2면 맞춤의 두 배
        self.pan = QPointF(0, 0)     # 맞춤 위치에서 옮긴 거리 (화면 픽셀)
        self._pan_start = None       # 가운데 버튼 드래그 시작 (마우스 위치, 그때의 pan)
        self.history = []            # 되돌리기용: 바꾸기 전 (박스 목록, 마스크) 복사본
        self.redo_list = []          # 다시 실행용: 되돌리기 직전 상태 복사본

    def push_undo(self):
        """지금 상태를 되돌리기 목록에 넣는다. 라벨을 바꾸기 직전에 부른다."""
        if self.mask is None:
            return
        self.history.append((list(self.boxes), self.mask.copy()))
        del self.history[:-UNDO_MAX]   # 오래된 것부터 버린다 (마스크 복사본이 메모리를 차지하므로)
        self.redo_list = []            # 새 작업을 하면 다시 실행할 것은 없어진다 (워드와 같음)

    def undo(self):
        """마지막 작업 전으로 돌린다. 돌릴 것이 없으면 False"""
        if not self.history:
            return False
        self.redo_list.append((list(self.boxes), self.mask.copy()))
        self.boxes, self.mask = self.history.pop()
        self._drag, self._stroke = None, None
        self.rebuild_overlay()
        self.changed.emit()
        return True

    def redo(self):
        """되돌린 작업을 다시 적용한다. 다시 할 것이 없으면 False"""
        if not self.redo_list:
            return False
        self.history.append((list(self.boxes), self.mask.copy()))
        self.boxes, self.mask = self.redo_list.pop()
        self._drag, self._stroke = None, None
        self.rebuild_overlay()
        self.changed.emit()
        return True

    def set_labels(self, boxes, mask):
        """현재 이미지의 라벨을 통째로 바꾼다 (되돌리기 가능)"""
        self.push_undo()
        self.boxes = list(boxes)
        if mask is not None and mask.size() == self.pm.size():
            self.mask = mask
        self.rebuild_overlay()
        self.changed.emit()

    def set_image(self, pm, boxes, mask=None):
        """mask는 QImage(Grayscale8)이거나 None(빈 마스크)"""
        self.pm, self.boxes = pm, list(boxes)
        self.history, self.redo_list = [], []
        self.zoom, self.pan = 1.0, QPointF(0, 0)   # 새 이미지는 맞춤으로
        self.zoom_changed.emit(self.zoom)
        self._drag, self._stroke = None, None
        self.sam_pts, self.preview = [], None
        if pm is None:
            self.mask = self.overlay = None
        else:
            if mask is None or mask.size() != pm.size():
                mask = QImage(pm.width(), pm.height(), QImage.Format_Grayscale8)
                mask.fill(0)
            self.mask = mask
            self.rebuild_overlay()
        self.update()

    def set_pixmap(self, pm):
        """보이는 그림만 바꾼다 (다파장 밴드/범위를 바꿀 때). 라벨과 되돌리기 기록은 그대로"""
        if self.pm is not None and pm.size() == self.pm.size():
            self.pm = pm
            self.update()

    def set_tool(self, tool):
        self.tool = tool
        self._drag, self._stroke = None, None
        self.update()

    def set_sam(self, on):
        self.sam = on
        self._drag, self._stroke = None, None
        self.update()

    def set_preview(self, items):
        """SAM 3 결과를 반투명으로 겹쳐 보여준다. items: (bool 마스크, 클래스 번호, 고름 여부) 목록.
        색은 그 클래스 색, 고른 결과는 하늘색 테두리. 빈 목록이면 지운다"""
        if not items or self.pm is None:
            self.preview = None
        else:
            rgba = np.zeros(items[0][0].shape + (4,), np.uint8)
            edge = QColor(theme.ACCENT)
            for m, c, sel in items:
                col = self._color(c)   # 진하기는 그릴 때 opacity로
                rgba[m] = (col.blue(), col.green(), col.red(), col.alpha())   # ARGB32는 메모리에 B,G,R,A 순서
                if sel:   # 테두리 = 마스크에서 안쪽(상하좌우 이웃이 모두 마스크인 픽셀)을 뺀 부분
                    inner = m.copy()
                    inner[1:, :] &= m[:-1, :]
                    inner[:-1, :] &= m[1:, :]
                    inner[:, 1:] &= m[:, :-1]
                    inner[:, :-1] &= m[:, 1:]
                    b = m & ~inner
                    b2 = b.copy()   # 한 픽셀 넓혀 두께 2로
                    b2[1:, :] |= b[:-1, :]
                    b2[:, 1:] |= b[:, :-1]
                    rgba[b2] = (edge.blue(), edge.green(), edge.red(), 255)
            h, w = rgba.shape[:2]
            self.preview = QImage(rgba.data, w, h, w * 4, QImage.Format_ARGB32).copy()
        self.update()

    def apply_mask(self, m, cls):
        """bool 배열 m 영역을 클래스 cls로 마스크에 칠한다 (되돌리기 기록은 부르는 쪽에서 push_undo)"""
        arr = self.mask_array()
        arr[m] = min(cls + 1, 255)
        h, w = arr.shape
        self.mask = QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
        self.rebuild_overlay()
        self.changed.emit()

    # ── 마스크 ──────────────────────────────────────────────
    def mask_array(self):
        """마스크를 numpy (높이, 너비) uint8 배열로 (복사본)"""
        if self.mask is None:
            return np.zeros((0, 0), np.uint8)
        m = self.mask
        buf = np.frombuffer(m.constBits(), np.uint8, m.height() * m.bytesPerLine())
        return buf.reshape(m.height(), m.bytesPerLine())[:, :m.width()].copy()

    def has_mask(self):
        return self.mask is not None and bool(self.mask_array().any())

    def rebuild_overlay(self):
        """마스크 값(클래스+1)을 클래스 색으로 바꿔 반투명 오버레이를 만든다"""
        if self.mask is None:
            return
        lut = np.zeros((256, 4), np.uint8)   # 값 -> (B, G, R, A). ARGB32는 메모리에 B,G,R,A 순서
        for v in range(1, 256):
            c = self._color(v - 1)   # 진하기는 그릴 때 opacity로
            lut[v] = (c.blue(), c.green(), c.red(), c.alpha())
        rgba = np.ascontiguousarray(lut[self.mask_array()])
        h, w = rgba.shape[:2]
        self.overlay = QImage(rgba.data, w, h, w * 4, QImage.Format_ARGB32).copy()
        self.update()

    def _paint_line(self, a, b, erase):
        """마스크와 오버레이에 a→b 둥근 붓 선을 긋는다 (지우기면 0/투명)"""
        v = 0 if erase else min(self.cls + 1, 255)
        for img, col in ((self.mask, QColor(v, v, v)),
                         (self.overlay, QColor(0, 0, 0, 0) if erase else self._color(self.cls))):
            p = QPainter(img)
            p.setCompositionMode(QPainter.CompositionMode_Source)   # 섞지 않고 값을 그대로 덮어쓴다
            if a == b:   # 길이 0인 선은 Qt가 그리지 않으므로 클릭 한 번은 원으로 찍는다
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                p.drawEllipse(QPointF(*a), self.brush / 2, self.brush / 2)
            else:
                p.setPen(QPen(col, self.brush, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                p.drawLine(QPointF(*a), QPointF(*b))
            p.end()

    # ── 화면 좌표 <-> 이미지 좌표 (이미지는 비율을 유지하며 칸에 맞춘다) ──
    def _geom(self):
        """(배율, 그림 왼쪽 위 x, y) 화면 좌표. 맞춤 배율에 zoom을 곱하고 pan만큼 옮긴다"""
        s = min(self.width() / self.pm.width(), self.height() / self.pm.height()) * self.zoom
        w, h = self.pm.width() * s, self.pm.height() * s
        return s, (self.width() - w) / 2 + self.pan.x(), (self.height() - h) / 2 + self.pan.y()

    def fit_view(self):
        """확대/이동을 처음(칸에 맞춤)으로"""
        self.zoom, self.pan = 1.0, QPointF(0, 0)
        self.zoom_changed.emit(self.zoom)
        self.update()

    def _zoom_at(self, pos, factor):
        """화면 위치 pos 아래의 이미지 점이 그대로 있도록 확대/축소한다"""
        s, ox, oy = self._geom()
        ix, iy = (pos.x() - ox) / s, (pos.y() - oy) / s       # pos 아래 이미지 좌표 (칸 밖이어도 그대로)
        max_zoom = max(1.0, 40 / s * self.zoom)                # 이미지 1픽셀이 화면 40픽셀이 될 때까지
        self.zoom = min(max(self.zoom * factor, 1.0), max_zoom)
        if self.zoom == 1.0:
            self.pan = QPointF(0, 0)
        else:
            s2, ox2, oy2 = self._geom()
            self.pan += QPointF(pos.x() - (ox2 + ix * s2), pos.y() - (oy2 + iy * s2))
        self.zoom_changed.emit(self.zoom)
        self.update()

    def _to_img(self, p):
        s, ox, oy = self._geom()
        x = min(max((p.x() - ox) / s, 0), self.pm.width())
        y = min(max((p.y() - oy) / s, 0), self.pm.height())
        return x, y

    def _to_view(self, x, y):
        s, ox, oy = self._geom()
        return QPointF(ox + x * s, oy + y * s)

    def set_opacity(self, v):
        """칠한 색, 박스 안쪽 색, SAM 3 미리보기의 불투명도 (0~1)"""
        self.opacity = min(max(v, 0.0), 1.0)
        self.update()

    def _color(self, c, alpha=255):
        col = QColor(self.colors[c] if c < len(self.colors) else theme.ACCENT)
        col.setAlpha(alpha)
        return col

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.BORDER))
        if self.pm is None:
            p.setPen(QColor(theme.MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "이미지 폴더를 열면 여기에 표시됩니다")
            return
        s, ox, oy = self._geom()
        target = QRectF(ox, oy, self.pm.width() * s, self.pm.height() * s)
        p.drawPixmap(target, self.pm, QRectF(self.pm.rect()))
        p.setOpacity(self.opacity)   # 아래 두 그림(칠한 색, SAM 3 미리보기)만 이 불투명도로
        p.drawImage(target, self.overlay)
        if self.preview is not None:
            p.setOpacity(min(1.0, self.opacity * 1.2))   # 미리보기는 칠한 색보다 조금 진하게 (예전 130 대 110)
            p.drawImage(target, self.preview)
        p.setOpacity(1.0)
        for c, x1, y1, x2, y2 in self.boxes:
            p.setPen(QPen(self._color(c), 2))       # 테두리는 불투명도와 상관없이 그대로
            p.setBrush(self._color(c, int(self.opacity * 93)))   # 안쪽 색: 예전 40 = 0.43 * 93
            p.drawRect(QRectF(self._to_view(x1, y1), self._to_view(x2, y2)))
        p.setPen(QPen(QColor(theme.ACCENT), 2, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        if self._drag:
            (x1, y1), (x2, y2) = self._drag
            p.drawRect(QRectF(self._to_view(x1, y1), self._to_view(x2, y2)))
        for x, y, pos in self.sam_pts:           # SAM 3 점: 포함은 하늘색, 제외는 흰색
            p.setPen(QPen(QColor(theme.TEXT), 1))
            p.setBrush(QColor(theme.ACCENT if pos else "#ffffff"))
            p.drawEllipse(self._to_view(x, y), 5, 5)
        if self.tool == "seg" and not self.sam and self._hover:   # 붓 크기 원
            p.setPen(QPen(QColor(theme.TEXT), 1))
            r = self.brush / 2 * s
            p.drawEllipse(self._to_view(*self._hover), r, r)

    def mousePressEvent(self, e):
        if self.pm is None:
            return
        if e.button() == Qt.MiddleButton:   # 가운데 버튼: 그림 이동
            self._pan_start = (e.position(), QPointF(self.pan))
            self.setCursor(Qt.ClosedHandCursor)
            return
        pt = self._to_img(e.position())
        if self.sam:
            if e.button() == Qt.RightButton:        # 오른쪽 클릭: 마지막으로 찍은 점 지우기
                self.sam_undo.emit()
            elif e.button() == Qt.LeftButton and e.modifiers() & Qt.ControlModifier:   # Ctrl+왼쪽 클릭: 제외 점
                self.sam_point.emit(pt[0], pt[1], False)
            elif e.button() == Qt.LeftButton:
                self._drag = (pt, pt)   # 짧게 누르면 점, 끌면 박스 (놓을 때 정한다)
        elif self.tool == "seg":
            if e.button() in (Qt.LeftButton, Qt.RightButton):
                erase = e.button() == Qt.RightButton
                self.push_undo()   # 붓 한 획(누르기~놓기)이 되돌리기 한 단계
                self._stroke = (pt, erase)
                self._paint_line(pt, pt, erase)
        elif e.button() == Qt.RightButton:      # 오른쪽 클릭: 그리는 중이면 취소, 아니면 클릭한 박스 삭제
            if self._drag:
                self._drag = None
            else:
                self._delete_box_at(pt)
        elif e.button() == Qt.LeftButton:
            self._drag = (pt, pt)
        self.update()

    def mouseMoveEvent(self, e):
        if self.pm is None:
            return
        if self._pan_start is not None:
            start, pan0 = self._pan_start
            self.pan = pan0 + (e.position() - start)
            self.update()
            return
        pt = self._to_img(e.position())
        if self._drag:
            self._drag = (self._drag[0], pt)
        if self._stroke:
            self._paint_line(self._stroke[0], pt, self._stroke[1])
            self._stroke = (pt, self._stroke[1])
        self._hover = pt
        self.hovered.emit(min(int(pt[0]), self.pm.width() - 1), min(int(pt[1]), self.pm.height() - 1))
        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton and self._pan_start is not None:
            self._pan_start = None
            self.unsetCursor()
            return
        if self._stroke:
            self._stroke = None
            self.changed.emit()
        elif self._drag and e.button() == Qt.LeftButton and self.sam:
            (x1, y1), (x2, y2) = self._drag
            self._drag = None
            s = self._geom()[0]
            if abs(x2 - x1) * s < 5 and abs(y2 - y1) * s < 5:   # 화면에서 5px 미만이면 클릭으로 본다
                self.sam_point.emit(x1, y1, True)
            else:
                self.sam_box.emit(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        elif self._drag and e.button() == Qt.LeftButton:
            (x1, y1), (x2, y2) = self._drag
            self._drag = None
            if abs(x2 - x1) > 2 and abs(y2 - y1) > 2:   # 너무 작은 드래그는 실수로 보고 버린다
                self.push_undo()
                self.boxes.append((self.cls, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                self.changed.emit()
        self.update()

    def wheelEvent(self, e):
        if self.pm is None:
            return
        up = e.angleDelta().y() > 0
        brush_mode = self.tool == "seg" and not self.sam
        if e.modifiers() & Qt.ControlModifier or not brush_mode:   # 확대/축소
            self._zoom_at(e.position(), 1.25 if up else 1 / 1.25)
            return
        b = int(self.brush * 1.15) + 1 if up else int(self.brush / 1.15)
        self.brush = min(max(b, 1), max(self.pm.width(), self.pm.height()))
        self.brush_changed.emit(self.brush)
        self.update()

    def leaveEvent(self, e):
        self._hover = None
        self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self._drag = None
            self.update()
            if self.sam:
                self.sam_exit.emit()
        elif e.key() in (Qt.Key_Return, Qt.Key_Enter) and self.sam:
            self.sam_apply.emit()
        elif Qt.Key_0 <= e.key() <= Qt.Key_9 and not e.modifiers() & Qt.ControlModifier:
            self.digit.emit(int(e.key() - Qt.Key_0))
        else:
            super().keyPressEvent(e)

    def _delete_box_at(self, pt):
        x, y = pt
        for i in range(len(self.boxes) - 1, -1, -1):
            _, x1, y1, x2, y2 = self.boxes[i]
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.push_undo()
                del self.boxes[i]
                self.changed.emit()
                return
