"""보조 대화상자: 배치 처리, K 추천, ROI 클래스 지정, AsLS / 미분 설정."""
from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .. import theme
from .common import ColorChip, form_row, hint, make_button, section_label


class BatchDialog(QtWidgets.QDialog):
    """폴더 안 모든 큐브에 현재 프롬프트/클러스터 설정을 적용."""

    def __init__(self, parent, settings_summary: str, has_refs: bool, last_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("배치 처리")
        self.setMinimumWidth(620)
        self.last_dir = last_dir
        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(10)

        lay.addWidget(section_label("입력"))
        row = QtWidgets.QHBoxLayout()
        self.in_dir = QtWidgets.QLineEdit()
        self.in_dir.setPlaceholderText("큐브 파일들이 있는 폴더")
        b = make_button("찾기…")
        b.clicked.connect(self._pick_in)
        row.addWidget(self.in_dir, 1)
        row.addWidget(b)
        lay.addLayout(row)
        self.files_lbl = QtWidgets.QLabel("")
        self.files_lbl.setObjectName("Hint")
        lay.addWidget(self.files_lbl)

        lay.addWidget(section_label("출력"))
        row = QtWidgets.QHBoxLayout()
        self.out_dir = QtWidgets.QLineEdit()
        self.out_dir.setPlaceholderText("결과 폴더 (비우면 입력폴더/hsi_batch)")
        b = make_button("찾기…")
        b.clicked.connect(self._pick_out)
        row.addWidget(self.out_dir, 1)
        row.addWidget(b)
        lay.addLayout(row)

        opts = QtWidgets.QHBoxLayout()
        self.chk_cluster = QtWidgets.QCheckBox("클러스터링")
        self.chk_cluster.setChecked(True)
        self.chk_classify = QtWidgets.QCheckBox("레퍼런스 분류")
        self.chk_classify.setChecked(has_refs)
        self.chk_classify.setEnabled(has_refs)
        self.chk_overlay = QtWidgets.QCheckBox("오버레이 PNG")
        self.chk_overlay.setChecked(True)
        self.chk_maps = QtWidgets.QCheckBox("라벨/클러스터 맵 (.npy)")
        self.chk_maps.setChecked(True)
        self.chk_yolo = QtWidgets.QCheckBox("YOLO 세그 라벨")
        for w in (self.chk_cluster, self.chk_classify, self.chk_overlay, self.chk_maps, self.chk_yolo):
            opts.addWidget(w)
        opts.addStretch(1)
        lay.addLayout(opts)
        self.yolo_by = QtWidgets.QComboBox()
        self.yolo_by.addItems(["클러스터 기준", "재질 기준"])
        lay.addLayout(form_row("YOLO 클래스", self.yolo_by, 90))

        lay.addWidget(hint("적용 설정: " + settings_summary))

        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("%v / %m")
        self.progress.setFixedHeight(14)
        self.progress.setStyleSheet("QProgressBar { color: %s; }" % theme.TEXT)
        lay.addWidget(self.progress)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        lay.addWidget(self.log, 1)

        brow = QtWidgets.QHBoxLayout()
        brow.addStretch(1)
        self.btn_close = make_button("닫기")
        self.btn_cancel = make_button("취소", "Danger")
        self.btn_cancel.setEnabled(False)
        self.btn_start = make_button("시작", "Primary")
        brow.addWidget(self.btn_close)
        brow.addWidget(self.btn_cancel)
        brow.addWidget(self.btn_start)
        lay.addLayout(brow)
        self.btn_close.clicked.connect(self.reject)
        self.in_dir.textChanged.connect(self._update_files)

    def _pick_in(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "입력 폴더", self.in_dir.text() or self.last_dir)
        if d:
            self.in_dir.setText(d)

    def _pick_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "출력 폴더", self.out_dir.text() or self.in_dir.text() or self.last_dir)
        if d:
            self.out_dir.setText(d)

    def _update_files(self):
        from ..batch import list_cubes

        d = self.in_dir.text().strip()
        if os.path.isdir(d):
            n = len(list_cubes(d))
            self.files_lbl.setText(f"큐브 파일 {n}개")
        else:
            self.files_lbl.setText("")

    def options(self) -> dict:
        in_dir = self.in_dir.text().strip()
        out_dir = self.out_dir.text().strip() or os.path.join(in_dir, "hsi_batch")
        return {
            "in_dir": in_dir,
            "out_dir": out_dir,
            "cluster": self.chk_cluster.isChecked(),
            "classify": self.chk_classify.isChecked(),
            "overlay": self.chk_overlay.isChecked(),
            "maps": self.chk_maps.isChecked(),
            "yolo": self.chk_yolo.isChecked(),
            "yolo_by": "material" if self.yolo_by.currentIndex() == 1 else "cluster",
        }

    def set_running(self, running: bool):
        self.btn_start.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.btn_close.setEnabled(not running)
        for w in (self.in_dir, self.out_dir, self.chk_cluster, self.chk_classify, self.chk_overlay, self.chk_maps, self.chk_yolo):
            w.setEnabled(not running)

    def append_log(self, text: str):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())


class KSuggestDialog(QtWidgets.QDialog):
    """K별 실루엣 점수를 보여주고 최적 K를 제안."""

    def __init__(self, parent, rows: list[tuple[int, float, float]]):
        super().__init__(parent)
        self.setWindowTitle("K 추천 (실루엣 점수)")
        self.setMinimumSize(520, 380)
        self.chosen: int | None = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(8)
        ks = [r[0] for r in rows]
        sil = [r[1] for r in rows]
        best = ks[int(np.nanargmax(sil))] if rows else None
        self.best = best

        pw = pg.PlotWidget()
        pw.setBackground(theme.BG1)
        pi = pw.getPlotItem()
        pi.showGrid(x=True, y=True, alpha=0.18)
        pi.setLabel("bottom", "K", color=theme.MUTED)
        pi.setLabel("left", "실루엣 점수 (높을수록 분리가 뚜렷)", color=theme.MUTED)
        pi.getViewBox().setBackgroundColor(theme.BG0)
        pi.getAxis("left").enableAutoSIPrefix(False)
        pi.setMenuEnabled(False)
        pi.hideButtons()
        if rows:
            bars = pg.BarGraphItem(x=ks, height=sil, width=0.6,
                                   brushes=[pg.mkBrush(theme.ACCENT if k == best else theme.BG3) for k in ks],
                                   pen=pg.mkPen(theme.BORDER))
            pi.addItem(bars)
            pi.plot(ks, sil, pen=pg.mkPen(theme.TEXT, width=1.5), symbol="o", symbolBrush=theme.TEXT, symbolSize=6)
            pi.getAxis("bottom").setTicks([[(k, str(k)) for k in ks]])
        lay.addWidget(pw, 1)

        if rows:
            txt = "  ·  ".join(f"K={k}: {s:.3f}" for k, s in zip(ks, sil))
            lab = QtWidgets.QLabel(f"추천 K = <b style='color:{theme.ACCENT}'>{best}</b>   ({txt})")
        else:
            lab = QtWidgets.QLabel("객체가 3개 이상 있어야 K를 비교할 수 있습니다.")
        lab.setWordWrap(True)
        lay.addWidget(lab)
        lay.addWidget(hint("실루엣 점수는 -1~1 범위이며 0.5 이상이면 비교적 잘 나뉜 것입니다. "
                           "inertia(팔꿈치법)도 함께 보려면 CSV로 저장하세요."))

        brow = QtWidgets.QHBoxLayout()
        self.k_spin = QtWidgets.QSpinBox()
        self.k_spin.setRange(1, 30)
        self.k_spin.setValue(best or 2)
        brow.addLayout(form_row("K 선택", self.k_spin, 50))
        brow.addStretch(1)
        b_close = make_button("닫기")
        b_use = make_button("이 K 사용", "Primary")
        brow.addWidget(b_close)
        brow.addWidget(b_use)
        lay.addLayout(brow)
        b_close.clicked.connect(self.reject)
        b_use.clicked.connect(self._use)
        self.rows = rows

    def _use(self):
        self.chosen = int(self.k_spin.value())
        self.accept()


class RoiClassDialog(QtWidgets.QDialog):
    """ROI 분류 전에 ROI마다 클래스 이름을 정한다. 같은 이름의 ROI는 한 클래스로 합쳐진다."""

    def __init__(self, parent, rois: list[dict]):
        """rois: [{"label": "R1", "color": "#..", "rect": (x0, y0, x1, y1), "cls": "R1"}, ...]"""
        super().__init__(parent)
        self.setWindowTitle("ROI 클래스 지정")
        self.setMinimumWidth(420)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(10)
        lay.addWidget(hint("ROI마다 클래스 이름을 적으세요. 같은 이름의 ROI는 한 클래스로 합쳐집니다.\n"
                           "한 클래스 안에서 모양이 다른 부분은 ROI를 따로 놓고 같은 이름을 주면 됩니다."))
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        self.edits: list[QtWidgets.QLineEdit] = []
        for i, r in enumerate(rois):
            chip = ColorChip(r["color"])
            x0, y0, x1, y1 = r["rect"]
            info = QtWidgets.QLabel(f"{r['label']}  ({x1 - x0}×{y1 - y0} px)")
            info.setObjectName("Muted")
            ed = QtWidgets.QLineEdit(r.get("cls") or r["label"])
            grid.addWidget(chip, i, 0)
            grid.addWidget(info, i, 1)
            grid.addWidget(ed, i, 2)
            self.edits.append(ed)
        grid.setColumnStretch(2, 1)
        lay.addLayout(grid)
        self.summary = QtWidgets.QLabel("")
        self.summary.setObjectName("Hint")
        lay.addWidget(self.summary)
        for ed in self.edits:
            ed.textChanged.connect(self._update)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.ok = btns.button(QtWidgets.QDialogButtonBox.Ok)
        self.ok.setText("분류")
        btns.button(QtWidgets.QDialogButtonBox.Cancel).setText("취소")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        self._update()

    def names(self) -> list[str]:
        return [ed.text().strip() for ed in self.edits]

    def _update(self):
        names = self.names()
        uniq = list(dict.fromkeys(n for n in names if n))
        ok = all(names) and len(uniq) >= 2
        self.ok.setEnabled(ok)
        self.summary.setText(f"클래스 {len(uniq)}개: {', '.join(uniq)}" if ok
                             else "이름을 모두 채우고, 서로 다른 이름이 2개 이상 있어야 합니다")


class _BandRows:
    """처리 대화상자 공통: 시작 밴드 + '잘린 큐브 저장' 체크 (B-106, 전에는 공간 크롭 B-098).
    처리는 시작 밴드부터 마지막 밴드까지만 하고 앞 밴드는 잘라낸다."""

    def __init__(self, lay: QtWidgets.QVBoxLayout, values: dict):
        self.start = QtWidgets.QSpinBox()
        self.start.setRange(0, 100000)   # 밴드 수 이상이면 대화상자/처리에서 오류로 알려 준다
        self.start.setValue(int(values["band_start"]))
        lay.addLayout(form_row("시작 밴드", self.start))
        self.save_cut = QtWidgets.QCheckBox("잘린 큐브 저장 (끄면 앞 밴드를 원본 값으로 붙여 저장)")
        self.save_cut.setChecked(bool(values["save_cut"]))
        lay.addWidget(self.save_cut)

    def values(self) -> dict:
        return {"band_start": self.start.value(), "save_cut": self.save_cut.isChecked()}


def _ok_cancel(dlg: QtWidgets.QDialog, lay: QtWidgets.QVBoxLayout) -> QtWidgets.QPushButton:
    btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    ok = btns.button(QtWidgets.QDialogButtonBox.Ok)
    ok.setText("처리")
    btns.button(QtWidgets.QDialogButtonBox.Cancel).setText("취소")
    btns.accepted.connect(dlg.accept)
    btns.rejected.connect(dlg.reject)
    lay.addWidget(btns)
    return ok


class AslsDialog(QtWidgets.QDialog):
    """AsLS 베이스라인 제거 설정 (B-097, 평활 벌점은 2차 차분 고정 B-099, 반복 5 고정 B-106).
    values: {"lam_exp", "p_exp", "band_start", "save_cut"}"""

    def __init__(self, parent, values: dict, device: str, shape: tuple):
        super().__init__(parent)
        self.setWindowTitle("AsLS 베이스라인 제거")
        self.setMinimumWidth(380)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(10)
        self.lam = QtWidgets.QDoubleSpinBox()
        self.lam.setRange(0.0, 12.0)
        self.lam.setDecimals(1)
        self.lam.setSingleStep(0.5)
        self.lam.setValue(float(values["lam_exp"]))
        self.p = QtWidgets.QDoubleSpinBox()
        self.p.setRange(0.5, 6.0)
        self.p.setDecimals(1)
        self.p.setSingleStep(0.5)
        self.p.setValue(float(values["p_exp"]))
        lay.addLayout(form_row("λ = 10^", self.lam))
        lay.addLayout(form_row("p = 10^-", self.p))
        self.bands = _BandRows(lay, values)
        H, W = shape[:2]
        self._n = shape[2]
        self.msg = hint("")
        lay.addWidget(self.msg)
        lay.addWidget(hint(f"큐브 {W}×{H} 픽셀, 밴드 {shape[2]}개, 계산 장치: {device}, 반복 5회\n"
                           "결과 = 원 신호 − 베이스라인. 늘 원본 큐브에서 다시 계산합니다 (쌓이지 않음).\n"
                           "시작 밴드는 배경 빼기, 미분과 같이 씁니다. 'AsLS 활성화'를 끄면 원본으로 돌아갑니다"))
        self.ok = _ok_cancel(self, lay)
        self.bands.start.valueChanged.connect(self._validate)
        self._validate()

    def _validate(self, *_):
        left = self._n - self.bands.start.value()   # 잘라낸 뒤 남는 밴드 수
        err = (f"시작 밴드가 밴드 수({self._n}) 이상입니다" if left <= 0
               else f"남는 밴드({left}개)가 너무 적습니다 (4개 이상)" if left <= 3 else "")
        self.msg.setText(err)
        self.ok.setEnabled(not err)

    def values(self) -> dict:
        return {"lam_exp": self.lam.value(), "p_exp": self.p.value(),
                **self.bands.values()}   # **: 두 dict를 하나로 합친다


class DerivDialog(QtWidgets.QDialog):
    """Savitzky-Golay 1차/2차 미분 설정 (B-099). values: {"order", "window", "poly", 시작 밴드 값들}"""

    def __init__(self, parent, values: dict, device: str, shape: tuple):
        super().__init__(parent)
        from ..derivative import check_params

        self._check, self._bands = check_params, shape[2]
        self.setWindowTitle("미분 (Savitzky-Golay)")
        self.setMinimumWidth(380)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(10)
        self.order = QtWidgets.QComboBox()
        self.order.addItems(["1차 미분", "2차 미분"])
        self.order.setCurrentIndex(int(values["order"]) - 1)
        self.window = QtWidgets.QSpinBox()
        self.window.setRange(3, 301)
        self.window.setSingleStep(2)
        self.window.setValue(int(values["window"]))
        self.poly = QtWidgets.QSpinBox()
        self.poly.setRange(1, 10)
        self.poly.setValue(int(values["poly"]))
        lay.addLayout(form_row("미분 차수", self.order))
        lay.addLayout(form_row("윈도우(홀수)", self.window))
        lay.addLayout(form_row("다항 차수", self.poly))
        self.bands = _BandRows(lay, values)
        H, W = shape[:2]
        self.msg = hint("")
        lay.addWidget(self.msg)
        lay.addWidget(hint(f"큐브 {W}×{H} 픽셀, 밴드 {shape[2]}개, 계산 장치: {device}\n"
                           "x 간격 = 밴드 1칸. 원본 큐브에서 다시 계산합니다 (AsLS가 켜져 있으면 AsLS 다음에 적용).\n"
                           "시작 밴드는 배경 빼기, AsLS와 같이 씁니다. 미분을 '끔'으로 하면 돌아갑니다"))
        self.ok = _ok_cancel(self, lay)
        for w in (self.window, self.poly, self.bands.start):
            w.valueChanged.connect(self._validate)
        self.order.currentIndexChanged.connect(self._validate)
        self._validate()

    def _validate(self, *_):   # *_: 시그널이 넘기는 값은 쓰지 않고 버린다
        v = self.values()
        b0 = v["band_start"]
        err = (f"시작 밴드({b0})가 밴드 수({self._bands}) 이상입니다" if b0 >= self._bands
               else self._check(self._bands - b0, v["order"], v["window"], v["poly"]))   # 잘린 밴드 수로 검사
        self.msg.setText(err)
        self.ok.setEnabled(not err)

    def values(self) -> dict:
        return {"order": self.order.currentIndex() + 1, "window": self.window.value(), "poly": self.poly.value(),
                **self.bands.values()}
