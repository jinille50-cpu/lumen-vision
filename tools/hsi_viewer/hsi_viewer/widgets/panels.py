"""우측 하단 옵션 패널 (한 화면에 들어가는 두 개의 카드) + 고급 설정 대화상자."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .. import theme
from ..clustering import METHODS
from .common import Card, LabeledSlider, SegmentedControl, form_row, make_button


# =============================================================================
class ViewPanel(Card):
    """이미지 표시 + 스펙트럼 옵션."""

    changed = QtCore.Signal()                 # 이미지 렌더 다시
    spectraChanged = QtCore.Signal()          # 스펙트럼 다시 계산
    bandChanged = QtCore.Signal(int, int)     # 채널 idx, 밴드 (스핀박스)
    addRoiRequested = QtCore.Signal()
    clearRequested = QtCore.Signal()
    exportRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__("이미지 · 스펙트럼", collapsible=False, parent=parent)
        self._block = False

        # 보기 모드 + 단일 밴드
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["False color (R/G/B)", "단일 밴드", "PCA 합성"])
        self.mode.setToolTip("False color: 세 밴드를 R/G/B로 합성 · 단일 밴드: 한 밴드를 컬러맵으로 · PCA: 주성분 1~3")
        row.addLayout(form_row("보기", self.mode, 36))
        self.band = QtWidgets.QSpinBox()
        self.band.setRange(0, 0)
        self.band.setToolTip("단일 밴드 모드에서 표시할 밴드")
        self.band_lbl = QtWidgets.QLabel("밴드")
        self.band_lbl.setObjectName("Muted")
        row.addWidget(self.band_lbl)
        row.addWidget(self.band)
        self.add(row)
        self.mode.currentIndexChanged.connect(self._on_mode)
        self.band.valueChanged.connect(lambda _: self._emit())

        # R/G/B
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.spins: list[QtWidgets.QSpinBox] = []
        for lab, col in (("R", theme.CURSOR_R), ("G", theme.CURSOR_G), ("B", theme.CURSOR_B)):
            l = QtWidgets.QLabel(lab)
            l.setStyleSheet(f"color:{col}; font-weight:700;")
            l.setFixedWidth(14)
            s = QtWidgets.QSpinBox()
            s.setRange(0, 0)
            s.setAlignment(QtCore.Qt.AlignCenter)
            s.setToolTip("이 채널의 밴드 번호. 그래프의 색 커서를 드래그해도 바뀝니다.")
            row.addWidget(l)
            row.addWidget(s, 1)
            self.spins.append(s)
        self.add(row)
        for i, s in enumerate(self.spins):
            s.valueChanged.connect(lambda v, i=i: self._on_spin(i, v))

        self.low = LabeledSlider("클립 하위", 0.0, 20.0, 0.1, 0.1, 1, " %", 64)
        self.high = LabeledSlider("클립 상위", 80.0, 100.0, 99.9, 0.1, 1, " %", 64)
        self.low.setToolTip("이 퍼센타일 아래를 0으로 (대비)")
        self.high.setToolTip("이 퍼센타일 위를 최대로 (대비)")
        self.add(self.low)
        self.add(self.high)
        self.low.valueChanged.connect(lambda _: self._emit())
        self.high.valueChanged.connect(lambda _: self._emit())

        # 반전 / 비닝 / SNV
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        self.chk_invert = QtWidgets.QCheckBox("반전")
        self.chk_invert.setToolTip("색 반전 (1 - x)")
        self.chk_invert.toggled.connect(lambda _: self._emit())
        row.addWidget(self.chk_invert)
        row.addStretch(1)
        self.bin = QtWidgets.QSpinBox()
        self.bin.setRange(1, 99)
        self.bin.setValue(1)
        self.bin.setFixedWidth(56)
        self.bin.setToolTip("포인트 주변 N×N 창 평균")
        self.bin.valueChanged.connect(lambda _: self.spectraChanged.emit())
        row.addLayout(form_row("비닝", self.bin, 30))
        self.chk_snv = QtWidgets.QCheckBox("SNV")
        self.chk_snv.setToolTip("스펙트럼별 표준정규변량 정규화")
        self.chk_snv.toggled.connect(lambda _: self.spectraChanged.emit())
        row.addWidget(self.chk_snv)
        self.add(row)

        # 버튼
        row = QtWidgets.QHBoxLayout()
        self.btn_roi = make_button("+ ROI", "", "사각 영역 추가 (영역 평균 스펙트럼). ROI 위에서 오른쪽 클릭 = 삭제")
        self.btn_clear = make_button("포인트 지우기", "Danger", "포인트/ROI 모두 삭제 (Ctrl+Shift+C)")
        self.btn_export = make_button("스펙트럼 저장", "Primary", "포인트/ROI 스펙트럼을 CSV로 (Ctrl+E)")
        row.addWidget(self.btn_roi)
        row.addWidget(self.btn_clear)
        row.addStretch(1)
        row.addWidget(self.btn_export)
        self.add(row)
        self.btn_roi.clicked.connect(self.addRoiRequested)
        self.btn_clear.clicked.connect(self.clearRequested)
        self.btn_export.clicked.connect(self.exportRequested)
        self.body_layout.addStretch(1)
        self._on_mode(0)

    def _on_mode(self, i: int):
        single = i == 1
        self.band.setVisible(single)
        self.band_lbl.setVisible(single)
        for s in self.spins:
            s.setEnabled(i == 0)
        self._emit()

    def _on_spin(self, i: int, v: int):
        if not self._block:
            self.bandChanged.emit(i, int(v))

    def _emit(self):
        if not self._block:
            self.changed.emit()

    def set_n_bands(self, n: int):
        self._block = True
        for s in self.spins:
            s.setRange(0, max(0, n - 1))
        self.band.setRange(0, max(0, n - 1))
        self._block = False

    def set_bands(self, bands):
        self._block = True
        for s, b in zip(self.spins, bands):
            s.setValue(int(b))
        self._block = False

    def state(self) -> dict:
        return {
            "mode": self.mode.currentIndex(),
            "low": self.low.value(),
            "high": self.high.value(),
            "invert": self.chk_invert.isChecked(),
            "band": int(self.band.value()),
            "bin": int(self.bin.value()),
            "snv": self.chk_snv.isChecked(),
        }


# =============================================================================
class SamPanel(Card):
    """SAM3 탐지 + 클러스터링 (핵심 파라미터만, 나머지는 고급 대화상자)."""

    runRequested = QtCore.Signal()
    clusterRequested = QtCore.Signal()
    clearRequested = QtCore.Signal()
    exportRequested = QtCore.Signal(str)
    overlayChanged = QtCore.Signal()
    loadModelRequested = QtCore.Signal()
    directToggled = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__("SAM3 탐지 · 클러스터링", collapsible=False, parent=parent)
        self.adv = {"nms_iou": 0.7, "overlap_iou": 0.7, "min_area": 30, "small_first": False, "snv": True,
                    "bg_norm": False, "stat": "median"}

        self.model_dot = QtWidgets.QLabel("●")
        self.model_lbl = QtWidgets.QLabel("")
        self.model_lbl.setObjectName("Hint")
        self.btn_load = make_button("모델 로딩", "Ghost", "sam3.pt를 지금 올립니다 (안 눌러도 첫 실행 때 자동)")
        self.btn_load.clicked.connect(self.loadModelRequested)
        self.header_extra.addWidget(self.model_dot)
        self.header_extra.addWidget(self.model_lbl)
        self.header_extra.addWidget(self.btn_load)

        self.prompt = QtWidgets.QLineEdit()
        self.prompt.setPlaceholderText("찾을 대상 (영문). 여러 대상은 쉼표로 구분  예: plastic piece, bottle cap")
        self.prompt.setToolTip("쉼표(,)로 나눈 각 문구를 별도 대상으로 한 번에 찾습니다.\nEnter = 탐지")
        self.prompt.returnPressed.connect(self.runRequested)
        row = form_row("프롬프트", self.prompt, 56)
        self.btn_direct = make_button("점·박스", "", "켜면 이미지에서 클릭(점) 또는 드래그(박스)한 물체 하나를\n"
                                      "SAM3로 잘라 객체로 추가합니다. 단축키 B")
        self.btn_direct.setCheckable(True)
        self.btn_direct.toggled.connect(self.directToggled)
        row.addWidget(self.btn_direct)
        self.add(row)

        self.conf = LabeledSlider("신뢰도", 0.01, 0.95, 0.20, 0.01, 2, "", 56)
        self.conf.setToolTip("이 점수 미만의 탐지는 버립니다. 놓치면 낮추고, 오탐이 많으면 올리세요.")
        self.add(self.conf)

        self.mode = SegmentedControl(["객체", "픽셀", "ROI"], 0)
        self.mode.setToolTip("탐지한 객체를 나누는 방법\n"
                             "객체: 객체마다 대표 스펙트럼 하나로 K개 그룹 (객체 단위)\n"
                             "픽셀: 객체 안 모든 픽셀을 K개로 자동 클러스터링\n"
                             "ROI:[+ ROI] 사각형이 클래스 예시. 객체 안 픽셀을 가장 비슷한 ROI 클래스로 분류")
        row = form_row("분류", self.mode, 56)
        self.add(row)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.k = QtWidgets.QSpinBox()
        self.k.setRange(1, 30)
        self.k.setValue(3)
        self.k.setFixedWidth(56)
        self.k.setToolTip("나눌 그룹 수")
        k_row = form_row("K", self.k, 56)
        self.k_label = k_row.itemAt(0).widget()
        row.addLayout(k_row)
        self.method = QtWidgets.QComboBox()
        self.method.addItems([m for m in METHODS if not m.startswith("By prompt")])
        self.method.setToolTip("클러스터링 방법")
        row.addWidget(self.method, 1)
        self.roi_hint = QtWidgets.QLabel("왼쪽 [+ ROI]로 클래스 예시를 그린 뒤 [분류]")
        self.roi_hint.setObjectName("Hint")
        row.addWidget(self.roi_hint, 1)
        self.btn_adv = make_button("고급…", "Ghost", "NMS/겹침 IoU, 최소 면적, 특징(SNV·배경정규화)")
        row.addWidget(self.btn_adv)
        self.add(row)
        self.btn_adv.clicked.connect(self._open_adv)
        self.mode.changed.connect(self._on_mode)
        self._on_mode(0)

        row = QtWidgets.QHBoxLayout()
        self.btn_run = make_button("탐지", "Primary", "Enter · 객체를 찾고 객체별 스펙트럼 표시")
        self.btn_cluster = make_button("분류", "", "위에서 고른 방법(객체 / 픽셀 / ROI)으로 객체를 나누기")
        self.btn_clear = make_button("지우기", "Danger", "탐지 결과 삭제")
        row.addWidget(self.btn_run, 2)
        row.addWidget(self.btn_cluster, 1)
        row.addWidget(self.btn_clear)
        self.add(row)
        self.btn_run.clicked.connect(self.runRequested)
        self.btn_cluster.clicked.connect(self.clusterRequested)
        self.btn_clear.clicked.connect(self.clearRequested)

        # 오버레이
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.chk_overlay = QtWidgets.QCheckBox("오버레이")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.setToolTip("단축키 O")
        row.addWidget(self.chk_overlay)
        self.overlay_kind = SegmentedControl(["객체별", "프롬프트별", "분류별"], 0)
        self.overlay_kind.setToolTip("색칠 기준. 프롬프트별 = 쉼표로 나눈 대상마다 한 색")
        row.addWidget(self.overlay_kind, 1)
        self.opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(55)
        self.opacity.setToolTip("오버레이 불투명도")
        row.addWidget(self.opacity, 1)
        self.add(row)
        self.chk_overlay.toggled.connect(lambda _: self.overlayChanged.emit())
        self.overlay_kind.changed.connect(lambda _: self.overlayChanged.emit())
        self.opacity.valueChanged.connect(lambda _: self.overlayChanged.emit())

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(5)
        self.progress.hide()
        self.add(self.progress)

        row = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("Hint")
        self.status.setWordWrap(True)
        row.addWidget(self.status, 1)
        self.btn_export = make_button("내보내기 ▾", "", "객체 표 / 클러스터 스펙트럼 / 라벨 맵 / 오버레이 저장")
        menu = QtWidgets.QMenu(self.btn_export)
        for key, text in (
            ("pieces_csv", "객체 표 (CSV)"),
            ("cluster_means_csv", "분류(클러스터) 평균 ± 표준편차 스펙트럼 (CSV)"),
            ("piece_spectra_csv", "객체별 스펙트럼 (CSV)"),
            ("label_map_npy", "객체 라벨 맵 (.npy)"),
            ("cluster_map_npy", "분류 맵 (.npy, 픽셀별 클래스 번호, -1 = 객체 밖)"),
            ("overlay_png", "오버레이 화면 (PNG)"),
            ("all", "전부 폴더에 저장…"),
        ):
            menu.addAction(text, lambda k=key: self.exportRequested.emit(k))
        self.btn_export.setMenu(menu)
        row.addWidget(self.btn_export, 0, QtCore.Qt.AlignTop)
        self.add(row)
        self.body_layout.addStretch(1)
        self.set_has_result(False)
        self.set_model_state("idle")

    # ------------------------------------------------------------------
    def prompts(self) -> list[str]:
        return [p.strip() for p in self.prompt.text().replace("\n", ",").split(",") if p.strip()]

    def params(self) -> dict:
        d = dict(self.adv)
        mode = self.mode.current()
        d.update({"prompts": self.prompts(), "conf": self.conf.value(), "k": int(self.k.value()),
                  "method": self.method.currentText(), "mode": mode, "pixel": mode == 1})
        return d

    def _on_mode(self, mode: int):
        self.k.setVisible(mode != 2)
        self.k_label.setVisible(mode != 2)
        self.method.setVisible(mode == 0)
        self.roi_hint.setVisible(mode == 2)

    def _open_adv(self):
        dlg = AdvancedDialog(self, self.adv)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.adv = dlg.values()

    def set_busy(self, busy: bool):
        for b in (self.btn_run, self.btn_load, self.btn_direct):
            b.setEnabled(not busy)
        self.btn_cluster.setEnabled(not busy and self._has_result)
        self.progress.setVisible(busy)

    def set_model_state(self, state: str):
        color = {"ready": theme.ACCENT, "loading": theme.WARN, "error": theme.DANGER}.get(state, theme.MUTED)
        text = {"ready": "모델 준비됨", "loading": "로딩 중…", "error": "로딩 실패", "missing": "sam3.pt 없음"}.get(state, "모델 미로딩")
        self.model_dot.setStyleSheet(f"color:{color}; font-size:11pt;")
        self.model_lbl.setText(text)
        self.btn_load.setVisible(state in ("idle", "error"))

    def set_status(self, text: str):
        self.status.setText(text)

    def set_has_result(self, has: bool):
        self._has_result = has
        self.btn_cluster.setEnabled(has)
        self.btn_export.setEnabled(has)
        self.btn_clear.setEnabled(has)

    def overlay_state(self) -> dict:
        return {"overlay": self.chk_overlay.isChecked(), "kind": self.overlay_kind.current(),
                "opacity": self.opacity.value() / 100.0}


# =============================================================================
class AdvancedDialog(QtWidgets.QDialog):
    def __init__(self, parent, values: dict):
        super().__init__(parent)
        self.setWindowTitle("고급 설정")
        self.setMinimumWidth(420)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(10)

        self.nms = LabeledSlider("NMS IoU", 0.10, 0.95, values["nms_iou"], 0.01, 2, "", 90)
        self.nms.setToolTip("박스 기준 중복 제거 임계값")
        self.overlap = LabeledSlider("겹침 IoU", 0.10, 0.95, values["overlap_iou"], 0.01, 2, "", 90)
        self.overlap.setToolTip("이미 채택한 마스크와 이 이상 겹치면 버림")
        self.min_area = LabeledSlider("최소 면적", 1, 5000, values["min_area"], 1, 0, " px", 90)
        self.min_area.setToolTip("이보다 작은 마스크는 노이즈로 간주")
        for w in (self.nms, self.overlap, self.min_area):
            lay.addWidget(w)
        self.chk_small = QtWidgets.QCheckBox("겹치면 작은 객체 우선")
        self.chk_small.setToolTip("끔(기본): 겹치는 픽셀은 점수가 높은 객체가 가짐.\n"
                                  "켬: 작은 객체가 가짐. 큰 객체 안에 있는 부분(예: 병 안의 뚜껑)을 따로 따고 싶을 때.\n"
                                  "끈 상태에서는 점수가 더 높은 큰 객체가 부분을 통째로 덮어 부분이 사라질 수 있습니다.")
        self.chk_small.setChecked(values.get("small_first", False))
        lay.addWidget(self.chk_small)

        row = QtWidgets.QHBoxLayout()
        self.chk_snv = QtWidgets.QCheckBox("SNV")
        self.chk_snv.setToolTip("스펙트럼 모양만 비교 (밝기 차이 제거)")
        self.chk_bg = QtWidgets.QCheckBox("배경 정규화")
        self.chk_bg.setToolTip("객체가 아닌 영역의 중앙값 스펙트럼으로 나눔 (조명 보정).\n"
                               "배경이 어두우면 값이 튀므로 흰 배경(레퍼런스)일 때만 권장")
        self.chk_snv.setChecked(values["snv"])
        self.chk_bg.setChecked(values["bg_norm"])
        for w in (self.chk_snv, self.chk_bg):
            row.addWidget(w)
        row.addStretch(1)
        lay.addLayout(row)

        self.stat = QtWidgets.QComboBox()
        self.stat.addItems(["median", "mean"])
        self.stat.setCurrentText(values["stat"])
        self.stat.setToolTip("객체 대표 스펙트럼 (median이 외곽 픽셀에 강함)")
        lay.addLayout(form_row("객체 대표값", self.stat, 90))

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.button(QtWidgets.QDialogButtonBox.Ok).setText("확인")
        btns.button(QtWidgets.QDialogButtonBox.Cancel).setText("취소")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def values(self) -> dict:
        return {"nms_iou": self.nms.value(), "overlap_iou": self.overlap.value(), "min_area": int(self.min_area.value()),
                "small_first": self.chk_small.isChecked(),
                "snv": self.chk_snv.isChecked(), "bg_norm": self.chk_bg.isChecked(),
                "stat": self.stat.currentText()}
