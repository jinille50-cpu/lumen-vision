"""Labeler 창: 위쪽 단계 표시줄 + 왼쪽 설정(폴더, 도구, 클래스, 이동) + 오른쪽 그림 칸.
객체 인식(박스)과 Segmentation(붓) 라벨링, 기존 라벨 불러오기, YOLO 형식 저장까지만 있다. 단계 표시줄의 회색 단계는 앞으로 붙일 자리(동작 없음)."""
import json
import os
import sys
import time

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QImage, QKeySequence, QPainter, QPixmap, QPolygonF, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QScrollArea,
    QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from common import settings, theme
from common.palette import class_colors
from tools.labeler.canvas import Canvas
from tools.labeler.compare import is_shadow, shadow_map
from tools.labeler import batch, cube, model_label, vlm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOOL = "labeler"
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
# 라벨을 찾는 위치 (이미지 폴더 기준): labels/ 아래, 이미지 옆, 상위 폴더의 labels/ (images/ + labels/ 구조)
LABEL_DIRS = ("labels", "", os.path.join("..", "labels"))
# 새 마스크는 labels/에 저장. 예전에 masks/에 저장한 것도 찾는다
MASK_DIRS = ("labels", "masks", os.path.join("..", "labels"), os.path.join("..", "masks"))
# 앞으로 붙일 단계: (이름, 항목 번호). 지금은 눌리지 않는다
FUTURE_STEPS = [("SAM 3 일괄", "B-010"), ("프롬프트 도우미", "B-013"), ("검수", "B-011"), ("모델 1차 라벨링", "B-014")]
# 이미 붙인 단계: 항목 번호 -> (누르면 부를 함수 이름, 툴팁)
READY_STEPS = {"B-010": ("_batch_open", "폴더 전체를 SAM 3 글자 프롬프트로 자동 라벨링"),
               "B-013": ("_focus_prompt", "SAM 3 보조의 프롬프트 칸으로: 한국어나 영어로 대상을 쓰고 '찾기'"),
               "B-011": ("_toggle_review", "자동 라벨을 점수 낮은 것부터 한 장씩 승인/수정/버리기 (다시 누르면 닫힘)"),
               "B-014": ("_model_open", "학습한 Segmentation/객체 인식 모델(.pt, .onnx)로 폴더를 1차 라벨링 → 검수에서 수정")}
STATUS_NAME = {"auto": "자동", "edited": "수정함", "approved": "확인", "rejected": "버림"}   # auto_labels.json의 status


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.folder = ""
        self.label_dir = ""      # 라벨 폴더를 따로 지정했으면 그 경로 (txt, 마스크 png, classes.txt를 모두 여기서). ""면 자동
        self.files = []
        self.idx = -1
        self.classes = ["class0"]
        self.dirty = False
        self.cands = []          # SAM 3 결과 후보 (Piece 목록)
        self.cand_cls = []       # 후보마다 넣을 클래스 번호 (cands와 같은 순서)
        self.cand_shadow = []    # 후보마다 그림자 의심 여부
        self._shadow = None      # 현재 이미지의 그림자 후보 픽셀 (bool 배열)
        self._vlm_thread = None  # 프롬프트 추천 요청 중인 스레드
        self._vlm_key = None     # 추천을 요청한 이미지 경로
        self._vlm_find = False   # True면 추천을 받은 뒤 바로 SAM 3로 찾는다
        self.auto = {}           # 자동 라벨 기록 {이미지 이름: 기록} (auto_labels.json)
        self._batch = None       # 일괄 라벨링 진행 상태
        self.batch_dlg = None
        self._mrun = None        # 모델 1차 라벨링 진행 상태
        self.cube = None         # 지금 이미지가 다파장 큐브면 Cube (data (H, W, 밴드), wavelengths)
        self.plot = None         # 스펙트럼 그래프 (pyqtgraph, 큐브를 처음 열 때 만든다)
        self._spec_timer = QTimer(self)   # 칠할 때마다 평균 스펙트럼을 다시 계산하지 않게 잠깐 모았다가 한 번
        self._spec_timer.setSingleShot(True)
        self._spec_timer.setInterval(300)
        self._spec_timer.timeout.connect(self._update_class_spectra)
        self.model_dlg = None
        self._sam_key = None     # 요청을 보낸 이미지 경로 (그 사이 이미지가 바뀌면 결과를 버린다)
        self._init_sam()
        self._build_ui()
        last = str(settings.get_value(TOOL, "folder", ""))
        if last and os.path.isdir(last):
            self._open_folder(last)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        bar = QFrame()
        bar.setObjectName("StepBar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(16, 8, 16, 8)
        cur = QPushButton("라벨링")
        cur.setObjectName("Step")
        cur.setProperty("current", True)
        bl.addWidget(cur)
        for name, num in FUTURE_STEPS:
            arrow = QLabel("→")
            arrow.setObjectName("StepArrow")
            bl.addWidget(arrow)
            b = QPushButton(name)
            b.setObjectName("Step")
            if num in READY_STEPS:
                fn, tip = READY_STEPS[num]
                b.clicked.connect(getattr(self, fn))   # getattr: 이름(문자열)으로 메서드를 찾는다
                b.setToolTip(tip)
            else:
                b.setEnabled(False)
                b.setToolTip(f"{num}에서 추가 예정")
            bl.addWidget(b)
        bl.addStretch(1)
        layout.addWidget(bar)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 8, 0)

        # 검수 칸: 단계 표시줄 '검수'를 누르면 보인다
        self.review_group = QGroupBox("검수 (최저 점수가 낮은 것부터)")
        vr = QVBoxLayout(self.review_group)
        self.review_info = QLabel("")
        self.review_info.setObjectName("Hint")
        self.review_info.setWordWrap(True)
        vr.addWidget(self.review_info)
        self.review_list = QListWidget()   # 최저 점수가 낮은 것부터
        self.review_list.setFixedHeight(150)
        self.review_list.itemClicked.connect(self._review_clicked)
        vr.addWidget(self.review_list)
        row = QHBoxLayout()
        self.approve_btn = QPushButton("승인")
        self.approve_btn.setObjectName("Primary")
        self.approve_btn.setToolTip("지금 라벨(고친 것 포함)을 저장하고 확인 처리, 다음 남은 이미지로")
        self.approve_btn.clicked.connect(self._review_approve)
        self.reject_btn = QPushButton("버리기")
        self.reject_btn.setToolTip("이 이미지의 라벨을 모두 비워 저장하고 버림 처리, 다음 남은 이미지로 (파일은 지우지 않음)")
        self.reject_btn.clicked.connect(self._review_reject)
        row.addWidget(self.approve_btn)
        row.addWidget(self.reject_btn)
        vr.addLayout(row)
        self.review_all = QCheckBox("확인/버림 처리한 것도 보기")
        self.review_all.toggled.connect(lambda _: self._fill_review())
        vr.addWidget(self.review_all)
        self.review_group.setVisible(False)
        lv.addWidget(self.review_group)

        g1 = QGroupBox("이미지 폴더")
        v1 = QVBoxLayout(g1)
        row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("이미지 폴더")
        self.folder_edit.editingFinished.connect(self._folder_typed)
        b = QPushButton("열기")
        b.clicked.connect(self._browse)
        row.addWidget(self.folder_edit, 1)
        row.addWidget(b)
        v1.addLayout(row)
        row = QHBoxLayout()
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("라벨 폴더 (비우면 labels/ 자동)")
        self.label_edit.setToolTip("박스 txt, 마스크 png, classes.txt를 이 폴더에서 이미지 이름으로 찾고 여기에 저장한다.\n"
                                   "이미지 폴더마다 기억한다. 칸을 비우고 Enter면 자동(labels/)으로")
        self.label_edit.editingFinished.connect(lambda: self._set_label_dir(self.label_edit.text().strip()))
        b = QPushButton("라벨 폴더")
        b.clicked.connect(self._browse_label_dir)
        row.addWidget(self.label_edit, 1)
        row.addWidget(b)
        v1.addLayout(row)
        b = QPushButton("이미지 파일 열기")   # 폴더 대신 개별 이미지 한 장만 작업할 때
        b.clicked.connect(self._browse_file)
        v1.addWidget(b)
        self.folder_info = QLabel("")
        self.folder_info.setObjectName("Hint")
        v1.addWidget(self.folder_info)
        lv.addWidget(g1)

        # 다파장 칸: 큐브 파일을 열었을 때만 보인다
        self.ms_group = QGroupBox("다파장")
        vm = QVBoxLayout(self.ms_group)
        self.ms_info = QLabel("")
        self.ms_info.setObjectName("Hint")
        self.ms_info.setWordWrap(True)
        vm.addWidget(self.ms_info)
        grid = QGridLayout()
        self.band_spins, self.band_labels = [], []
        for r, ch in enumerate("RGB"):
            grid.addWidget(QLabel(ch), r, 0)
            sp = QSpinBox()
            sp.valueChanged.connect(self._view_changed)
            lab = QLabel("")
            lab.setObjectName("Hint")
            grid.addWidget(sp, r, 1)
            grid.addWidget(lab, r, 2)
            self.band_spins.append(sp)
            self.band_labels.append(lab)
        vm.addLayout(grid)
        row = QHBoxLayout()
        row.addWidget(QLabel("범위 %"))
        self.lo_spin, self.hi_spin = QDoubleSpinBox(), QDoubleSpinBox()
        for sp, rng in ((self.lo_spin, (0.0, 49.0)), (self.hi_spin, (51.0, 100.0))):
            sp.setRange(*rng)
            sp.setDecimals(1)
            sp.setSingleStep(0.5)
            sp.valueChanged.connect(self._view_changed)
        self.lo_spin.setToolTip("밴드마다 이 퍼센타일 아래는 검정으로")
        self.hi_spin.setToolTip("밴드마다 이 퍼센타일 위는 흰색으로 (밝은 점 몇 개 때문에 전체가 어두워지지 않게)")
        row.addWidget(self.lo_spin)
        row.addWidget(QLabel("~"))
        row.addWidget(self.hi_spin)
        row.addStretch(1)
        vm.addLayout(row)
        row = QHBoxLayout()
        self.single_chk = QCheckBox("한 밴드")
        self.single_chk.toggled.connect(self._view_changed)
        self.band_slider = QSlider(Qt.Horizontal)
        self.band_slider.valueChanged.connect(self._view_changed)
        self.single_label = QLabel("")
        self.single_label.setObjectName("Hint")
        row.addWidget(self.single_chk)
        row.addWidget(self.band_slider, 1)
        row.addWidget(self.single_label)
        vm.addLayout(row)
        b = QPushButton("클래스별 평균 스펙트럼 내보내기")
        b.setToolTip("지금 이미지의 클래스마다 평균/표준편차 스펙트럼을 CSV로 저장")
        b.clicked.connect(self._export_spectra)
        vm.addWidget(b)
        self.ms_group.setVisible(False)
        lv.addWidget(self.ms_group)

        g2 = QGroupBox("모드")
        v2 = QVBoxLayout(g2)
        row = QHBoxLayout()
        self.btn_box = QPushButton("객체 인식")
        self.btn_seg = QPushButton("Segmentation")
        for b, t in ((self.btn_box, "box"), (self.btn_seg, "seg")):
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=t: self._set_tool(k))
            row.addWidget(b)
        v2.addLayout(row)
        self.hint = QLabel("")
        self.hint.setObjectName("Hint")
        self.hint.setWordWrap(True)
        self.hint.setMinimumHeight(40)   # 두 줄까지 안 잘리게
        v2.addWidget(self.hint)
        lv.addWidget(g2)

        g5 = self.sam_group = QGroupBox("SAM 3 보조")
        v5 = QVBoxLayout(g5)
        self.btn_sam = QPushButton("점/박스 지정")
        self.btn_sam.setCheckable(True)
        self.btn_sam.toggled.connect(self._toggle_sam)
        v5.addWidget(self.btn_sam)
        row = QHBoxLayout()
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("영어 단어(쉼표로 여러 개) 또는 한국어. 비우면 VLM이 그림을 보고 추천")
        self.prompt_edit.returnPressed.connect(self._prompt_enter)
        self.prompt_edit.setText(str(settings.get_value(TOOL, "prompt", "")))
        self.find_btn = QPushButton("찾기")
        self.find_btn.setToolTip("영어 단어면 그대로 SAM 3로 찾는다.\n"
                                 "한국어(예: 바이알)면 VLM이 영어 단어로 바꿔 칸에 채운 뒤 찾고, 비어 있으면 VLM이 그림을 보고 추천해 찾는다.\n"
                                 "결과는 모두 지금 클래스로 들어가고, 클래스는 결과마다 바꿀 수 있다")
        self.find_btn.clicked.connect(self._prompt_enter)
        row.addWidget(self.prompt_edit, 1)
        row.addWidget(self.find_btn)
        v5.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("점수 하한"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(float(settings.get_value(TOOL, "conf", 0.25)))
        self.conf_spin.setToolTip("글자 프롬프트 결과 중 이 점수보다 낮은 것은 버린다. 그림자 조각이 잡히면 올려 본다")
        row.addWidget(self.conf_spin)
        row.addStretch(1)
        v5.addLayout(row)
        self.cand_list = QListWidget()   # 체크한 결과만 적용된다. 줄을 고르면(여러 개 가능) 그림에 테두리
        self.cand_list.setFixedHeight(90)
        self.cand_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.cand_list.itemChanged.connect(lambda _: self._show_preview())
        self.cand_list.itemSelectionChanged.connect(self._show_preview)
        v5.addWidget(self.cand_list)
        for n in range(10):   # 목록에 커서가 있을 때 숫자키로 클래스 지정 (그림 칸은 canvas.digit 신호)
            sc = QShortcut(QKeySequence(str(n)), self.cand_list)
            sc.setContext(Qt.WidgetShortcut)
            sc.activated.connect(lambda k=n: self._set_cand_cls(k - self._cls_first()))
        row = QHBoxLayout()
        row.addWidget(QLabel("고른 결과의 클래스"))
        self.cand_cls_combo = QComboBox()
        self.cand_cls_combo.activated.connect(self._set_cand_cls)
        row.addWidget(self.cand_cls_combo, 1)
        v5.addLayout(row)
        row = QHBoxLayout()
        self.apply_btn = QPushButton("적용")
        self.apply_btn.clicked.connect(self._sam_apply)
        self.clear_btn = QPushButton("취소")
        self.clear_btn.clicked.connect(self._sam_clear)
        row.addWidget(self.apply_btn)
        row.addWidget(self.clear_btn)
        v5.addLayout(row)
        self.sam_hint = QLabel("")
        self.sam_hint.setObjectName("Hint")
        self.sam_hint.setWordWrap(True)
        v5.addWidget(self.sam_hint)
        lv.addWidget(g5)
        lv.addStretch(1)

        # 클래스, 이미지 목록은 그림 칸 오른쪽 세로 칸에 (왼쪽 아래에 있으면 잘 안 보여서)
        side = QWidget()
        sv = QVBoxLayout(side)
        sv.setContentsMargins(8, 0, 0, 0)
        g3 = QGroupBox("클래스")
        v3 = QVBoxLayout(g3)
        self.cls_list = QListWidget()
        self.cls_list.currentRowChanged.connect(self._class_changed)
        v3.addWidget(self.cls_list, 1)
        row = QHBoxLayout()
        for text, fn in (("추가", self._add_class), ("이름 변경", self._rename_class), ("삭제", self._del_class)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        v3.addLayout(row)
        sv.addWidget(g3, 1)

        g4 = QGroupBox("이미지")
        v4 = QVBoxLayout(g4)
        self.file_list = QListWidget()   # 라벨이 있는 이미지는 하늘색 + "라벨" 표시
        self.file_list.setMinimumHeight(90)
        self.file_list.currentRowChanged.connect(self._list_clicked)
        v4.addWidget(self.file_list, 1)
        row = QHBoxLayout()
        self.prev_btn = QPushButton("이전")
        self.next_btn = QPushButton("다음")
        self.prev_btn.clicked.connect(lambda: self._go(self.idx - 1))
        self.next_btn.clicked.connect(lambda: self._go(self.idx + 1))
        self.pos_label = QLabel("0 / 0")
        row.addWidget(self.prev_btn)
        row.addWidget(self.pos_label, 1, Qt.AlignCenter)
        row.addWidget(self.next_btn)
        v4.addLayout(row)
        self.load_label_btn = QPushButton("라벨 불러오기")   # 현재 이미지에 txt/마스크 png 적용
        self.load_label_btn.clicked.connect(self._load_label_file)
        v4.addWidget(self.load_label_btn)
        sv.addWidget(g4, 2)

        self.canvas = Canvas()
        self.canvas.changed.connect(self._mark_dirty)
        self.canvas.brush_changed.connect(lambda _: self._set_hint())
        self.canvas.sam_point.connect(self._sam_point)
        self.canvas.sam_undo.connect(self._sam_undo_point)
        self.canvas.sam_apply.connect(self._sam_apply)
        self.canvas.sam_exit.connect(lambda: self.btn_sam.setChecked(False))
        self.canvas.sam_box.connect(self._sam_box)
        self.canvas.sam_pick.connect(self._sam_pick)
        self.canvas.digit.connect(lambda k: self._set_cand_cls(k - self._cls_first()))
        self.canvas.hovered.connect(self._hover_spectrum)
        self.canvas.zoom_changed.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f}%"))
        self.canvas.changed.connect(lambda: self.cube is not None and self._spec_timer.start())
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 0, 0, 0)
        rv.addWidget(self.canvas, 1)
        self.plot_holder = QVBoxLayout()   # 스펙트럼 그래프가 들어갈 자리 (큐브일 때만)
        rv.addLayout(self.plot_holder)
        row = QHBoxLayout()
        self.name_label = QLabel("")
        self.name_label.setObjectName("Hint")
        row.addWidget(self.name_label, 1)
        row.addWidget(QLabel("불투명도"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setFixedWidth(120)
        self.opacity_slider.setToolTip("칠한 색, 박스 안쪽 색, SAM 3 결과 미리보기의 진하기 (박스 테두리는 그대로)")
        self.opacity_label = QLabel("")
        self.opacity_label.setObjectName("Hint")
        self.opacity_label.setFixedWidth(36)
        self.opacity_slider.valueChanged.connect(self._opacity_changed)
        self.opacity_slider.setValue(int(settings.get_value(TOOL, "overlay_opacity", 43)))
        self._opacity_changed(self.opacity_slider.value())   # 43이면 값이 안 바뀌어 신호가 없으므로 한 번 직접
        row.addWidget(self.opacity_slider)
        row.addWidget(self.opacity_label)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("Hint")
        self.zoom_label.setToolTip("Ctrl+휠 확대/축소 (붓 모드가 아니면 그냥 휠도), 가운데 버튼 드래그로 이동")
        row.addWidget(self.zoom_label)
        fit_btn = QPushButton("맞춤")
        fit_btn.setToolTip("그림 전체가 보이게 (확대/이동 처음으로)")
        fit_btn.clicked.connect(lambda: self.canvas.fit_view())
        row.addWidget(fit_btn)
        rv.addLayout(row)

        body = QWidget()
        bv = QVBoxLayout(body)
        bv.setContentsMargins(16, 16, 16, 16)
        sp = QSplitter(Qt.Horizontal)
        scroll = self.left_scroll = QScrollArea()   # 창이 낮으면 왼쪽 칸이 눌려 겹치지 않고 스크롤되게
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sp.addWidget(scroll)
        sp.addWidget(right)
        sp.addWidget(side)
        sp.setStretchFactor(0, 0)
        sp.setStretchFactor(1, 1)
        sp.setStretchFactor(2, 0)
        sp.setSizes([360, 760, 240])
        bv.addWidget(sp, 1)
        row = QHBoxLayout()
        self.status = QLabel("")
        self.status.setObjectName("Hint")
        row.addWidget(self.status, 1)
        self.undo_btn = QPushButton("되돌리기")
        self.undo_btn.setToolTip("Ctrl+Z")
        self.undo_btn.clicked.connect(self._undo)
        row.addWidget(self.undo_btn)
        # Ctrl+Z: 이 창(Labeler) 안에서만 동작. 글자 칸에 커서가 있으면 그 칸의 되돌리기가 먼저 쓰인다
        sc = QShortcut(QKeySequence.Undo, self)
        sc.setContext(Qt.WidgetWithChildrenShortcut)
        sc.activated.connect(self._undo)
        self.redo_btn = QPushButton("다시 실행")
        self.redo_btn.setToolTip("Ctrl+Y")
        self.redo_btn.clicked.connect(self._redo)
        row.addWidget(self.redo_btn)
        # Ctrl+Y: 되돌리기와 같은 범위. QKeySequence.Redo는 윈도우에서 Ctrl+Y
        sc = QShortcut(QKeySequence.Redo, self)
        sc.setContext(Qt.WidgetWithChildrenShortcut)
        sc.activated.connect(self._redo)
        self.save_btn = QPushButton("저장")
        self.save_btn.setObjectName("Primary")
        self.save_btn.clicked.connect(self._save)
        row.addWidget(self.save_btn)
        bv.addLayout(row)
        layout.addWidget(body, 1)

        self._set_tool("box")
        self._fill_classes()
        self._update_nav()
        self._update_sam_ui()

    # ── 도구/클래스 ──────────────────────────────────────────
    def _set_tool(self, tool):
        self.btn_box.setChecked(tool == "box")
        self.btn_seg.setChecked(tool == "seg")
        self.canvas.set_tool(tool)
        self._set_hint()
        if hasattr(self, "cls_list"):   # 모드마다 클래스 번호 시작이 달라 목록 글자를 다시 쓴다
            self._renumber_classes()

    def _cls_first(self):
        """화면에 보이는 클래스 번호의 시작. Segmentation은 배경이 0이라 1부터(마스크 png 값과 같음), 박스는 YOLO처럼 0부터"""
        return 1 if self.canvas.tool == "seg" else 0

    def _cls_text(self, c):
        name = self.classes[c] if 0 <= c < len(self.classes) else "?"
        return f"{c + self._cls_first()}  {name}"

    def _renumber_classes(self):
        """클래스 목록, 결과 클래스 칸의 번호 글자만 다시 쓴다 (고른 줄은 그대로)"""
        for i in range(min(self.cls_list.count(), len(self.classes))):
            self.cls_list.item(i).setText(self._cls_text(i))
        if hasattr(self, "cand_cls_combo"):
            for i in range(min(self.cand_cls_combo.count(), len(self.classes))):
                self.cand_cls_combo.setItemText(i, self._cls_text(i))
            self._refresh_cands()

    def _set_hint(self):
        if self.canvas.tool == "box":
            self.hint.setText("드래그로 박스를 그린다. 오른쪽 클릭으로 삭제. 휠 확대, 가운데 버튼 이동.")
        else:
            self.hint.setText(f"왼쪽 드래그 칠하기, 오른쪽 드래그 지우기, 휠로 붓 크기 (지금 {self.canvas.brush}px). "
                              "Ctrl+휠 확대, 가운데 버튼 이동.")

    def _fill_classes(self):
        # 클래스마다 색 칸을 붙인다. 0번(흰색 배경)은 빼고 1번부터 쓴다
        colors = class_colors(len(self.classes) + 1)[1:]
        self.canvas.colors = colors
        self.canvas.rebuild_overlay()
        self.cls_list.clear()
        for i, name in enumerate(self.classes):
            pm = QPixmap(14, 14)
            pm.fill(QColor(colors[i]))
            self.cls_list.addItem(QListWidgetItem(QIcon(pm), self._cls_text(i)))
        self.cls_list.setCurrentRow(0)
        self.canvas.update()
        if hasattr(self, "cand_cls_combo"):
            self.cand_cls_combo.clear()
            self.cand_cls_combo.addItems([self._cls_text(i) for i in range(len(self.classes))])
            self._refresh_cands()

    def _class_changed(self, row):
        self.canvas.cls = max(row, 0)

    def _add_class(self):
        name, ok = QInputDialog.getText(self, "클래스", "이름")
        if ok and name.strip():
            self.classes.append(name.strip())
            self._fill_classes()
            self.cls_list.setCurrentRow(len(self.classes) - 1)
            self._save_classes()

    def _rename_class(self):
        row = self.cls_list.currentRow()
        if row < 0:
            return
        name, ok = QInputDialog.getText(self, "클래스", "이름", text=self.classes[row])
        if ok and name.strip():
            self.classes[row] = name.strip()
            self._fill_classes()
            self.cls_list.setCurrentRow(row)
            self._save_classes()

    def _del_class(self):
        row = self.cls_list.currentRow()
        if row < 0 or len(self.classes) <= 1:
            return
        # 지우면 뒤 클래스 번호가 밀려 기존 라벨과 어긋나므로 마지막 클래스만 지울 수 있게 한다
        if row != len(self.classes) - 1:
            QMessageBox.warning(self, "클래스", "번호가 밀리지 않도록 마지막 클래스만 지울 수 있습니다.")
            return
        used = [c for c, *_ in self.canvas.boxes] + [int(v) - 1 for v in np.unique(self.canvas.mask_array())]
        if row in used:
            QMessageBox.warning(self, "클래스", "현재 이미지에서 쓰고 있는 클래스입니다.")
            return
        del self.classes[row]
        self._fill_classes()
        self._save_classes()

    # ── 폴더/이미지 ──────────────────────────────────────────
    def _browse(self):
        start = self.folder or str(settings.get_value(TOOL, "folder", ""))
        path = QFileDialog.getExistingDirectory(self, "이미지 폴더", start)
        if path:
            self._open_folder(path)

    def _browse_file(self):
        start = self.folder or str(settings.get_value(TOOL, "folder", ""))
        path, _ = QFileDialog.getOpenFileName(self, "이미지 파일", start,
                                              "이미지 (" + " ".join("*" + e for e in IMG_EXT + cube.CUBE_EXT) + ")")
        if path:
            self._open_folder(os.path.dirname(path), only=os.path.basename(path))

    def _folder_typed(self):
        path = self.folder_edit.text().strip()
        if path != self.folder:
            self._open_folder(path)

    def _open_folder(self, path, only=None):
        """폴더의 이미지를 모두 연다. only가 있으면 그 파일 한 장만 연다."""
        if not path or not os.path.isdir(path) or not self._confirm_leave():
            return
        self.folder = path
        self.files = [only] if only else sorted(f for f in os.listdir(path) if f.lower().endswith(IMG_EXT + cube.CUBE_EXT))
        self.folder_edit.setText(path)
        self.folder_info.setText(("이미지 1장 (파일)" if only else f"이미지 {len(self.files)}장") if self.files
                                 else "이미지가 없습니다")
        settings.set_value(TOOL, "folder", path)
        settings.push_recent(TOOL, path)
        ld = self._label_dirs().get(os.path.normcase(os.path.abspath(path)), "")
        self.label_dir = os.path.normpath(ld) if ld and os.path.isdir(ld) else ""
        self.label_edit.setText(self.label_dir)
        self._load_classes()
        self.auto = batch.load_auto(path)
        self.review_group.setVisible(False)
        self._fill_classes()
        self._fill_file_list()
        self.idx = -1
        self.dirty = False
        if self.files:
            self._go(0)
        else:
            self.canvas.set_image(None, [])
            self.name_label.setText("")
            self._update_nav()

    def _fill_file_list(self):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for name in self.files:
            self.file_list.addItem(QListWidgetItem(name))
        self.file_list.blockSignals(False)
        for i in range(len(self.files)):
            self._mark_item(i)

    def _mark_item(self, i):
        """라벨 파일이 있으면 목록 글자를 하늘색으로 하고 '라벨'을 붙인다"""
        stem = os.path.splitext(self.files[i])[0]
        item = self.file_list.item(i)
        has = self._has_label(stem)
        st = STATUS_NAME.get(self.auto.get(self.files[i], {}).get("status"), "")   # SAM 3 일괄로 만든 라벨의 검수 상태
        item.setText(self.files[i] + ("   라벨" if has else "") + (f"  {st}" if has and st else ""))
        item.setForeground(QColor(theme.ACCENT if has else theme.TEXT))

    def _list_clicked(self, row):
        if row != self.idx and row >= 0:
            self._go(row)
            self.file_list.blockSignals(True)   # 이동이 취소되면 목록 선택도 원래대로
            self.file_list.setCurrentRow(self.idx)
            self.file_list.blockSignals(False)

    def _confirm_leave(self):
        """저장하지 않은 변경이 있으면 저장할지 묻는다. 계속 진행해도 되면 True."""
        if not self.dirty:
            return True
        r = QMessageBox.question(self, "라벨", "저장하지 않은 변경이 있습니다. 저장할까요?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Save:
            self._save()
        self.dirty = False
        return True

    def _go(self, i):
        if not self.files or not (0 <= i < len(self.files)) or not self._confirm_leave():
            return
        path = os.path.join(self.folder, self.files[i])
        pm = QPixmap() if cube.is_cube_file(path) else QPixmap(path)
        cb = None
        if pm.isNull():   # 큐브 파일(다채널 tif 포함)
            try:
                QApplication.setOverrideCursor(Qt.BusyCursor)
                cb = cube.load(path)
            except Exception as e:  # noqa: BLE001  -- 형식 오류 등 어떤 이유든 알린다
                QMessageBox.warning(self, "이미지", f"이미지를 읽을 수 없습니다: {path}\n{e}")
                return
            finally:
                QApplication.restoreOverrideCursor()
            self.cube = cb
            self._setup_view()
            pm = self._cube_pixmap()
        self.cube = cb
        self.ms_group.setVisible(cb is not None)
        if self.plot is not None:
            self.plot.setVisible(cb is not None)
        self.idx = i
        boxes, polys = self._read_txt(self._txt_path(), pm.width(), pm.height())
        mask = self._read_mask(self._label_find(self._stem() + ".png"), pm.size())
        if polys:   # 예전 방식(다각형)으로 저장된 라벨은 마스크로 바꿔서 붓으로 고칠 수 있게 한다
            mask = self._paint_polys(mask, polys, pm.size())
        self._grow_classes(boxes, mask)
        self.canvas.set_image(pm, boxes, mask)
        self._clear_cands()
        self.name_label.setText(f"{path}  ({pm.width()} x {pm.height()}"
                                + (f" x {cb.n_bands}밴드)" if cb is not None else ")"))
        if cb is not None:
            self._update_class_spectra()
        self.dirty = False
        self.status.setText("")
        self.file_list.blockSignals(True)
        self.file_list.setCurrentRow(i)
        self.file_list.blockSignals(False)
        self._update_nav()
        if self.review_group.isVisible():
            self._fill_review()

    def _update_nav(self):
        n = len(self.files)
        self.pos_label.setText(f"{self.idx + 1} / {n}")
        self.prev_btn.setEnabled(self.idx > 0)
        self.next_btn.setEnabled(0 <= self.idx < n - 1)
        self.save_btn.setEnabled(self.idx >= 0)
        self.undo_btn.setEnabled(bool(self.canvas.history))
        self.redo_btn.setEnabled(bool(self.canvas.redo_list))
        self.load_label_btn.setEnabled(self.idx >= 0)
        if hasattr(self, "find_btn"):
            self._update_sam_ui()

    def _mark_dirty(self):
        self.dirty = True
        self.status.setText("저장하지 않은 변경 있음")
        self.undo_btn.setEnabled(bool(self.canvas.history))
        self.redo_btn.setEnabled(bool(self.canvas.redo_list))

    def _opacity_changed(self, v):
        self.canvas.set_opacity(v / 100)
        self.opacity_label.setText(f"{v}%")
        settings.set_value(TOOL, "overlay_opacity", v)

    def _undo(self):
        if self.canvas.undo():
            self.status.setText(f"되돌림 (남은 단계 {len(self.canvas.history)})")
        self.undo_btn.setEnabled(bool(self.canvas.history))
        self.redo_btn.setEnabled(bool(self.canvas.redo_list))

    def _redo(self):
        if self.canvas.redo():
            self.status.setText(f"다시 실행 (남은 단계 {len(self.canvas.redo_list)})")
        self.undo_btn.setEnabled(bool(self.canvas.history))
        self.redo_btn.setEnabled(bool(self.canvas.redo_list))

    # ── 저장/읽기 (YOLO 형식) ────────────────────────────────
    def _stem(self):
        return os.path.splitext(self.files[self.idx])[0]

    def _find(self, name, subdirs):
        """이미지 폴더 기준으로 subdirs 안에서 name 파일을 찾는다. 이미지 옆, labels/ 아래, 상위 폴더의 labels/
        (images/ + labels/ 구조) 모두 본다. 없으면 ''"""
        for d in subdirs:
            path = os.path.normpath(os.path.join(self.folder, d, name))
            if os.path.isfile(path):
                return path
        return ""

    def _label_find(self, name):
        """라벨 파일(.txt 또는 마스크 .png)을 찾는다. 라벨 폴더를 지정했으면 그 폴더에서만, 아니면 자동 위치에서. 없으면 ''"""
        if self.label_dir:
            path = os.path.join(self.label_dir, name)
            return path if os.path.isfile(path) else ""
        return self._find(name, LABEL_DIRS if name.endswith(".txt") else MASK_DIRS)

    def _txt_path(self, stem=None):
        """라벨 txt (기본은 현재 이미지). 라벨 폴더를 지정했으면 <라벨 폴더>/<이름>.txt.
        아니면 이미 있는 파일(수정하면 같은 곳에 저장), 없으면 <폴더>/labels/<이름>.txt"""
        stem = stem or self._stem()
        if self.label_dir:
            return os.path.join(self.label_dir, stem + ".txt")
        return self._label_find(stem + ".txt") or os.path.join(self.folder, "labels", stem + ".txt")

    def _mask_path(self, stem=None):
        stem = stem or self._stem()
        if self.label_dir:
            return os.path.join(self.label_dir, stem + ".png")
        return self._label_find(stem + ".png") or os.path.join(self.folder, "labels", stem + ".png")

    # ── 라벨 폴더 따로 지정 (B-088) ──────────────────────────
    def _label_dirs(self):
        """{이미지 폴더(소문자 정규화): 라벨 폴더} 설정. 설정에는 JSON 글자로 저장한다"""
        try:
            d = json.loads(str(settings.get_value(TOOL, "label_dirs", "{}")))
        except ValueError:   # 깨진 값이면 없는 것으로
            d = {}
        return d if isinstance(d, dict) else {}

    def _browse_label_dir(self):
        if not self.folder:
            self.status.setText("이미지 폴더를 먼저 여세요")
            return
        start = self.label_dir or os.path.dirname(self.folder)
        path = QFileDialog.getExistingDirectory(self, "라벨 폴더", start)
        if path:
            self._set_label_dir(path)

    def _set_label_dir(self, path):
        """라벨 폴더를 바꾸고(""면 자동) 이 이미지 폴더에 기억한 뒤 클래스와 지금 이미지 라벨을 다시 읽는다"""
        path = os.path.normpath(path) if path else ""
        if not self.folder or os.path.normcase(path) == os.path.normcase(self.label_dir):   # 그대로면 할 일 없음
            self.label_edit.setText(self.label_dir)
            return
        if path and not os.path.isdir(path):
            self.label_edit.setText(self.label_dir)
            self.status.setText(f"라벨 폴더가 없습니다: {path}")
            return
        if path and os.path.normcase(path) == os.path.normcase(os.path.normpath(self.folder)):
            self.label_edit.setText(self.label_dir)
            self.status.setText("라벨 폴더는 이미지 폴더와 달라야 합니다 (마스크 png가 이미지와 섞임)")
            return
        if not self._confirm_leave():   # 저장 안 한 라벨은 바꾸기 전 폴더에 저장하거나 버린다
            self.label_edit.setText(self.label_dir)
            return
        self.label_dir = path
        self.label_edit.setText(path)
        d = self._label_dirs()
        key = os.path.normcase(os.path.abspath(self.folder))
        if path:
            d[key] = path
        else:
            d.pop(key, None)   # pop(키, None): 있으면 지우고 없으면 그냥 넘어간다
        settings.set_value(TOOL, "label_dirs", json.dumps(d, ensure_ascii=False))
        self._load_classes()
        self._fill_classes()
        self._fill_file_list()
        n = sum(1 for f in self.files if self._has_label(os.path.splitext(f)[0]))
        self.dirty = False
        if 0 <= self.idx < len(self.files):
            self._go(self.idx)   # 지금 이미지의 라벨을 새 위치에서 다시 읽는다
        self.status.setText((f"라벨 폴더: {path}" if path else "라벨 폴더: 자동 (labels/)")
                            + f", 라벨 있는 이미지 {n}/{len(self.files)}장")

    def _classes_file(self):
        """읽을 classes.txt. 라벨 폴더를 지정했으면 거기를 먼저 본다"""
        if self.label_dir and os.path.isfile(os.path.join(self.label_dir, "classes.txt")):
            return os.path.join(self.label_dir, "classes.txt")
        return self._find("classes.txt", ("", "labels", ".."))

    def _load_classes(self):
        self.classes = ["class0"]
        cls_file = self._classes_file()
        if cls_file:
            with open(cls_file, encoding="utf-8") as f:
                self.classes = [l.strip() for l in f if l.strip()] or ["class0"]

    def _read_txt(self, path, w, h):
        """YOLO txt에서 읽는다. 값이 5개면 박스, 7개 이상이면 다각형(정규화 좌표 → 픽셀)."""
        boxes, polys = [], []
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    p = line.split()
                    try:
                        if len(p) == 5:
                            cx, cy, bw, bh = map(float, p[1:])
                            boxes.append((int(p[0]), (cx - bw / 2) * w, (cy - bh / 2) * h,
                                          (cx + bw / 2) * w, (cy + bh / 2) * h))
                        elif len(p) >= 7:
                            v = list(map(float, p[1:]))
                            polys.append((int(p[0]), [(v[k] * w, v[k + 1] * h) for k in range(0, len(v) - 1, 2)]))
                    except ValueError:
                        continue   # 숫자가 아닌 줄은 건너뛴다
        return boxes, polys

    def _read_mask(self, path, size):
        """마스크 png를 8비트 그레이로 읽는다. 없거나 크기가 다르면 None(빈 마스크)"""
        if not path:
            return None
        img = QImage(path)
        if img.isNull() or img.size() != size:
            return None
        return img.convertToFormat(QImage.Format_Grayscale8)

    def _paint_polys(self, mask, polys, size):
        """다각형을 마스크에 클래스 번호+1 값으로 채운다"""
        if mask is None:
            mask = QImage(size, QImage.Format_Grayscale8)
            mask.fill(0)
        p = QPainter(mask)
        p.setPen(Qt.NoPen)
        for c, pts in polys:
            v = min(c + 1, 255)
            p.setBrush(QColor(v, v, v))
            p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))
        p.end()
        return mask

    def _grow_classes(self, boxes, mask):
        """라벨에 목록보다 큰 클래스 번호가 있으면 class<번호>로 채워 넣는다"""
        top = max([c for c, *_ in boxes] + [0])
        if mask is not None:
            buf = np.frombuffer(mask.constBits(), np.uint8, mask.height() * mask.bytesPerLine())
            top = max(top, int(buf.max()) - 1)
        if top >= len(self.classes):
            self.classes += [f"class{i}" for i in range(len(self.classes), top + 1)]
            self._fill_classes()

    def _load_label_file(self):
        """고른 라벨 파일을 현재 이미지에 적용한다. .txt는 박스를 바꾸고 다각형은 마스크에 더하며, .png는 마스크를 바꾼다."""
        if self.idx < 0:
            return
        start = self._txt_path() if os.path.isfile(self._txt_path()) else self.folder
        path, _ = QFileDialog.getOpenFileName(self, "라벨 파일", os.path.dirname(start),
                                              "라벨 (*.txt *.png);;YOLO txt (*.txt);;마스크 png (*.png)")
        if not path:
            return
        pm = self.canvas.pm
        if path.lower().endswith(".png"):
            mask = self._read_mask(path, pm.size())
            if mask is None:
                QMessageBox.critical(self, "라벨", f"이미지와 크기가 같은 마스크가 아닙니다 ({pm.width()} x {pm.height()}).")
                return
            boxes = self.canvas.boxes
        else:
            boxes, polys = self._read_txt(path, pm.width(), pm.height())
            if not boxes and not polys:
                QMessageBox.critical(self, "라벨", "읽을 수 있는 라벨이 없습니다.")
                return
            mask = self._paint_polys(self.canvas.mask.copy(), polys, pm.size()) if polys else self.canvas.mask
            boxes = boxes if boxes else self.canvas.boxes
        self._grow_classes(boxes, mask)
        self.canvas.set_labels(boxes, mask)   # 되돌리기 가능
        self.status.setText(f"불러옴: {path} (저장하면 적용됨)")

    def _save_classes(self):
        if self.folder:
            with open(os.path.join(self.label_dir or self.folder, "classes.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(self.classes) + "\n")

    def _save(self):
        """박스는 labels/<이름>.txt, 붓으로 칠한 것은 labels/<이름>.png(클래스 번호+1 값, 0은 배경)에 쓴다."""
        if self.idx < 0:
            return
        w, h = self.canvas.pm.width(), self.canvas.pm.height()
        lines = [f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                 for c, x1, y1, x2, y2 in self.canvas.boxes]
        txt, mask_path = self._txt_path(), self._mask_path()
        saved = []
        try:
            if lines or os.path.isfile(txt):   # 박스를 모두 지운 경우에도 기존 파일을 비운다
                os.makedirs(os.path.dirname(txt), exist_ok=True)
                with open(txt, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))
                saved.append(txt)
            if self.canvas.has_mask() or os.path.isfile(mask_path):
                os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                if not self.canvas.mask.save(mask_path):
                    raise OSError(f"마스크를 쓸 수 없음: {mask_path}")
                saved.append(mask_path)
            self._save_classes()
        except OSError as e:
            QMessageBox.critical(self, "저장", f"저장하지 못했습니다: {e}")
            return
        self.dirty = False
        self.status.setText("저장함: " + (", ".join(saved) if saved else "라벨 없음"))
        rec = self.auto.get(self.files[self.idx])
        if rec and rec.get("status") == "auto":   # 자동 라벨을 손으로 고쳐 저장했다
            rec["status"] = "edited"
            try:
                batch.save_auto(self.folder, self.auto)
            except OSError:
                pass
        self._mark_item(self.idx)

    # ── SAM 3 보조 ───────────────────────────────────────────
    def _init_sam(self):
        """SAM 3 엔진(별도 프로세스)은 HSI Viewer 것을 그대로 쓴다. 모델은 처음 쓸 때 불러온다."""
        tool_dir = os.path.join(ROOT, "tools", "hsi_viewer")
        if tool_dir not in sys.path:
            sys.path.insert(0, tool_dir)   # hsi_viewer 패키지를 import 하려고 (main.build_hsi_viewer와 같은 방식)
        from hsi_viewer.sam_engine import SamEngine
        self.sam = SamEngine("", self)
        self.sam.status.connect(lambda t: self.status.setText(t))
        self.sam.finished.connect(self._sam_finished)
        self.sam.failed.connect(self._sam_failed)

    def set_sam_path(self, path):
        """메인 앱의 'SAM 3 로드'에서 고른 경로를 받는다 (모든 도구가 같이 씀)"""
        if not self.sam.set_model_path(path):
            QMessageBox.critical(self, "SAM 3", "SAM 3 작업 중에는 경로를 바꿀 수 없습니다.")
            return
        self._update_sam_ui()

    def _update_sam_ui(self):
        ok = self.sam.model_exists
        have = self.idx >= 0
        idle = not self.sam.busy and self._vlm_thread is None and self._batch is None and self._mrun is None
        self.btn_sam.setEnabled(ok and have and idle)
        for w in (self.prompt_edit, self.find_btn):   # 한국어/빈칸 → VLM 단어는 SAM 3 없이도 되므로 이미지만 있으면 켠다
            w.setEnabled(have and idle)
        self.apply_btn.setEnabled(bool(self.cands) and not self.sam.busy)
        self.cand_cls_combo.setEnabled(bool(self.cands))
        self.clear_btn.setEnabled(bool(self.cands) or bool(self.canvas.sam_pts))
        if not ok:
            self.sam_hint.setText("제목줄의 SAM 3 로드로 sam3.pt를 먼저 고르세요.")
        elif self.cands:
            self.sam_hint.setText("결과를 목록에서 고르고 숫자키나 위 칸으로 클래스를 정한 뒤 적용(Enter). Esc는 SAM 3 끄기.")
        elif self.btn_sam.isChecked():
            self.sam_hint.setText("왼쪽 클릭 포함 점, Ctrl+왼쪽 클릭 제외 점, 오른쪽 클릭 마지막 점 지우기, 드래그 박스. Enter로 적용, Esc로 SAM 3 끄기.")
        else:
            self.sam_hint.setText("글자 프롬프트로 찾거나 점/박스 지정을 켜세요. 적용 대상은 현재 모드와 클래스입니다.")

    def _toggle_sam(self, on):
        self.canvas.set_sam(on)
        self._sam_clear()

    def _rgb(self):
        """현재 이미지를 SAM 3에 넣을 RGB uint8 배열 (높이, 너비, 3)로"""
        img = self.canvas.pm.toImage().convertToFormat(QImage.Format_RGB888)
        buf = np.frombuffer(img.constBits(), np.uint8, img.height() * img.bytesPerLine())
        return buf.reshape(img.height(), img.bytesPerLine())[:, :img.width() * 3].reshape(img.height(), img.width(), 3).copy()

    def _sam_run(self, **job):
        if self.idx < 0 or self.sam.busy:
            return
        self._sam_key = os.path.join(self.folder, self.files[self.idx])
        rgb = self._rgb()
        self._shadow = shadow_map(rgb)
        ok = self.sam.segment(rgb=rgb, key=self._sam_key, **job)
        if ok:
            self.status.setText("SAM 3 실행 중…")
        self._update_sam_ui()

    def _sam_point(self, x, y, pos):
        if self.sam.busy:
            return
        self.canvas.sam_pts.append((x, y, pos))
        pts = self.canvas.sam_pts
        self._sam_run(prompts=["직접 지정"], points=[[a, b] for a, b, _ in pts],
                      point_labels=[1 if f else 0 for _, _, f in pts], min_area=1)

    def _sam_undo_point(self):
        """오른쪽 클릭: 마지막 점을 지우고 남은 점으로 다시 찾는다 (점이 없으면 결과를 지움)"""
        if self.sam.busy:
            return
        pts = self.canvas.sam_pts
        if not pts:
            self._sam_clear()
            return
        pts.pop()
        if not pts:
            self._sam_clear()
            return
        self._sam_run(prompts=["직접 지정"], points=[[a, b] for a, b, _ in pts],
                      point_labels=[1 if f else 0 for _, _, f in pts], min_area=1)

    def _sam_box(self, x1, y1, x2, y2):
        if self.sam.busy:
            return
        self.canvas.sam_pts = []
        self._sam_run(prompts=["직접 지정"], box=[[x1, y1, x2, y2]], min_area=1)

    def _sam_text(self):
        prompts = [t.strip() for t in self.prompt_edit.text().split(",") if t.strip()]
        if not prompts:
            self.status.setText("글자 프롬프트를 입력하세요 (예: crack, bolt)")
            return
        self.canvas.sam_pts = []
        self._save_prompt_settings()
        self._sam_run(prompts=prompts, conf=self.conf_spin.value(), nms_iou=0.7, min_area=30, overlap_iou=0.7)

    def _vlm_suggest(self):
        """VLM에 현재 이미지(와 한국어 설명)를 보내 영어 프롬프트 후보를 받는다"""
        if self.idx < 0 or self._vlm_thread is not None:
            return
        text = self.prompt_edit.text().strip()
        desc = text if vlm.has_korean(text) else ""
        model = str(settings.get_value(TOOL, "vlm_model", vlm.DEFAULT_MODEL)) or vlm.DEFAULT_MODEL   # 화면 칸 없이 설정값
        url = str(settings.get_value(TOOL, "vlm_url", vlm.DEFAULT_URL))
        self._vlm_key = os.path.join(self.folder, self.files[self.idx])
        self._vlm_find = self.sam.model_exists   # SAM 3 경로가 있으면 받은 단어로 바로 찾는다
        self._vlm_thread = vlm.SuggestThread(url, model, vlm.image_png_b64(self.canvas.pm), desc, self)
        self._vlm_thread.done.connect(self._vlm_done)
        self._vlm_thread.failed.connect(self._vlm_failed)
        self._vlm_thread.finished.connect(self._vlm_end)
        self._vlm_thread.start()
        self.status.setText(f"{model}에 프롬프트 추천 요청 중" + (f" (대상: {desc})" if desc else "") + "… 처음엔 모델 로딩으로 10초쯤 걸림")
        self._update_sam_ui()

    def _vlm_done(self, words, note, sec):
        cur = os.path.join(self.folder, self.files[self.idx]) if self.idx >= 0 else None
        if self._vlm_key != cur:
            return   # 그 사이 다른 이미지로 넘어갔으면 버린다
        if not words:
            self.status.setText("VLM이 후보를 내지 못했습니다. 한국어로 대상을 적고 다시 추천해 보세요")
            return
        self.prompt_edit.setText(", ".join(words))
        self.status.setText(f"추천 {len(words)}개 ({sec:.1f}s). {note}")
        if self._vlm_find:
            self._vlm_thread = None   # _sam_text 전에 '쉬는 중'으로 (끝 신호 _vlm_end보다 이 신호가 먼저 온다)
            self._sam_text()

    def _vlm_failed(self, msg):
        self.status.setText("프롬프트 추천 실패: " + msg)   # 서버가 없을 때 등은 상태 표시줄로만 알린다

    def _vlm_end(self):
        self._vlm_thread = None
        self._update_sam_ui()

    def _prompt_enter(self):
        """'찾기'와 글자 칸 Enter: 한국어거나 비어 있으면 VLM 단어를 받아 바로 찾고, 영어 단어면 그대로 찾는다"""
        if not self.find_btn.isEnabled():
            return
        text = self.prompt_edit.text().strip()
        if not text or vlm.has_korean(text):
            self._vlm_suggest()
        elif not self.sam.model_exists:
            self.status.setText("제목줄의 SAM 3 로드로 sam3.pt를 먼저 고르세요")
        else:
            self._sam_text()

    def _save_prompt_settings(self):
        settings.set_value(TOOL, "prompt", self.prompt_edit.text().strip())
        settings.set_value(TOOL, "conf", self.conf_spin.value())

    def _sam_finished(self, res):
        if self._batch is not None:   # 일괄 라벨링 중이면 그쪽에서 처리
            self._batch_result(res)
            return
        cur = os.path.join(self.folder, self.files[self.idx]) if self.idx >= 0 else None
        if self._sam_key != cur:
            self._update_sam_ui()   # 요청한 뒤 다른 이미지로 이동했으면 결과를 버린다
            return
        direct = res.params.get("prompts") == ["직접 지정"]
        if direct and res.params.get("points") and not res.pieces and len(self.canvas.sam_pts) > 1:
            # 점을 더했더니 아무것도 못 찾으면(예: 물체 안쪽에 제외 점) 그 점을 빼고 이전 결과를 그대로 둔다
            self.canvas.sam_pts.pop()
            self.canvas.update()
            self.status.setText("SAM 3: 이 점을 넣으면 찾지 못해 마지막 점을 뺐습니다")
            self._update_sam_ui()
            return
        pieces = res.pieces[:1] if direct else res.pieces   # 점/박스는 하나
        self._set_cands(pieces, check_shadow=not direct)
        self.status.setText(f"SAM 3: 결과 {len(self.cands)}개 ({res.elapsed:.1f}s)" if self.cands
                            else "SAM 3: 찾지 못했습니다. 점을 더 찍거나 박스로 지정해 보세요")
        self._update_sam_ui()

    def _set_cands(self, pieces, check_shadow=True):
        """결과 목록을 채운다. 처음엔 모두 지금 클래스, 모두 체크 (그림자 의심은 '그림자?' 표시만)"""
        self.cands = list(pieces)
        self.cand_cls = [self.canvas.cls] * len(self.cands)
        self.cand_shadow = [check_shadow and self._shadow is not None and is_shadow(q.mask, self._shadow)
                            for q in self.cands]
        self.cand_list.blockSignals(True)
        self.cand_list.clear()
        for i in range(len(self.cands)):
            it = QListWidgetItem("")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            self.cand_list.addItem(it)
        self.cand_list.blockSignals(False)
        self._refresh_cands()
        self._update_sam_ui()

    def _sam_failed(self, msg):
        if self._batch is not None:
            self._batch_end(f"SAM 3 오류로 멈춤: {msg.splitlines()[0]}")
        self._update_sam_ui()
        self.status.setText("SAM 3 오류")
        QMessageBox.critical(self, "SAM 3", msg)

    def _checked(self):
        """체크된 결과의 번호 목록"""
        return [i for i in range(len(self.cands)) if self.cand_list.item(i).checkState() == Qt.Checked]

    def _show_preview(self):
        sel = {ix.row() for ix in self.cand_list.selectedIndexes()}   # {}: 집합(set), 중복 없는 목록
        self.canvas.set_preview([(self.cands[i].mask, self.cand_cls[i], i in sel) for i in self._checked()])

    def _refresh_cands(self):
        """결과 목록 글자(클래스, 프롬프트, 점수, 면적)를 다시 쓰고 미리보기를 갱신한다"""
        self.cand_list.blockSignals(True)
        for i, q in enumerate(self.cands):
            c = self.cand_cls[i]
            shadow = "  그림자?" if i < len(self.cand_shadow) and self.cand_shadow[i] else ""
            self.cand_list.item(i).setText(f"[{self._cls_text(c)}]  {q.prompt}  점수 {q.score:.2f}  {q.area}px{shadow}")
        self.cand_list.blockSignals(False)
        self._show_preview()

    def _sam_pick(self, x, y, add):
        """그림에서 Ctrl+클릭한 위치의 결과를 목록에서 고른다 (add면 기존 선택에 더한다)"""
        hit = [i for i in self._checked() if self.cands[i].mask[int(min(y, self.cands[i].mask.shape[0] - 1)),
                                                               int(min(x, self.cands[i].mask.shape[1] - 1))]]
        if not add:
            self.cand_list.clearSelection()
        for i in hit[:1]:
            self.cand_list.item(i).setSelected(True)
            self.cand_list.scrollToItem(self.cand_list.item(i))

    def _set_cand_cls(self, c):
        """고른 결과(없으면 체크된 전체)의 클래스를 c로 바꾼다"""
        if not self.cands or not (0 <= c < len(self.classes)):
            return
        rows = sorted({ix.row() for ix in self.cand_list.selectedIndexes()}) or self._checked()
        for i in rows:
            self.cand_cls[i] = c
        self.cand_cls_combo.setCurrentIndex(c)
        self._refresh_cands()
        self.status.setText(f"결과 {len(rows)}개를 클래스 {self._cls_text(c)}로 지정")

    def _clear_cands(self):
        self.cands = []
        self.cand_cls = []
        self.cand_shadow = []
        self.cand_list.clear()
        self.canvas.sam_pts = []
        self.canvas.set_preview([])

    def _sam_clear(self):
        self._clear_cands()
        self._update_sam_ui()

    def _sam_apply(self):
        """체크한 결과를 현재 클래스로 넣는다. Segmentation 모드면 마스크에 칠하고, 객체 인식 모드면 박스로 넣는다."""
        if not self.cands or self.sam.busy:
            return
        sel = self._checked()
        if sel:
            self.canvas.push_undo()   # 한 번 적용한 것 전체가 되돌리기 한 단계
        for i in sel:
            q, c = self.cands[i], self.cand_cls[i]
            if self.canvas.tool == "seg":
                self.canvas.apply_mask(q.mask, c)
            else:
                x1, y1, x2, y2 = q.bbox
                self.canvas.boxes.append((c, x1, y1, x2, y2))
                self.canvas.changed.emit()
        used = sorted({self.cand_cls[i] for i in sel})
        self.status.setText(f"{len(sel)}개 적용함 (클래스 {', '.join(map(str, used))}, 저장해야 파일에 반영됨)")
        self._clear_cands()
        self._update_sam_ui()

    # ── SAM 3 일괄 라벨링 ────────────────────────────────────
    def _batch_open(self):
        if not self.files:
            self.status.setText("먼저 이미지 폴더를 여세요")
            return
        if not self.sam.model_exists:
            QMessageBox.critical(self, "SAM 3 일괄", "제목줄의 SAM 3 로드로 sam3.pt를 먼저 고르세요.")
            return
        if not self._confirm_leave():
            return
        words = list(dict.fromkeys(t.strip() for t in self.prompt_edit.text().split(",") if t.strip()))
        if self.batch_dlg is None:
            self.batch_dlg = batch.BatchDialog(self)
            self.batch_dlg.start_requested.connect(self._batch_start)
            self.batch_dlg.stop_requested.connect(self._batch_stop)
        n_lab = sum(1 for f in self.files if self._has_label(os.path.splitext(f)[0]))
        target = "마스크(labels/ png)" if self.canvas.tool == "seg" else "박스(labels/ YOLO txt)"
        info = (f"{self.folder}\n이미지 {len(self.files)}장 (라벨 있음 {n_lab}장) · 저장 형식: {target} — 모드 버튼으로 바꿈"
                + ("" if words else "\n프롬프트 칸에 단어를 쉼표로 넣고 다시 여세요 (한국어로 쓰고 '찾기'면 VLM이 영어 단어로 바꿔 줌)"))
        self.batch_dlg.setup(info, words, self.classes, self.canvas.cls, self.conf_spin.value(), self._cls_first())
        self.batch_dlg.show()

    def _has_label(self, stem):
        return bool(self._label_find(stem + ".txt") or self._label_find(stem + ".png"))

    def _batch_start(self, cfg):
        if self.sam.busy:
            return
        self._batch = {"cfg": cfg, "mode": self.canvas.tool, "i": 0, "stop": False, "t0": time.time(),
                       "done": 0, "skipped": 0, "empty": 0, "failed": 0, "pieces": 0}
        self.batch_dlg.set_running(True)
        self._update_sam_ui()
        self._batch_next()

    def _batch_stop(self):
        if self._batch is not None:
            self._batch["stop"] = True
            self.batch_dlg.progress(self._batch["i"], len(self.files), "멈춤 요청: 지금 이미지까지만 하고 멈춥니다")

    def _batch_next(self):
        """다음 이미지를 SAM 3에 보낸다. 건너뛸 이미지는 여기서 넘긴다"""
        b = self._batch
        n = len(self.files)
        while b["i"] < n and not b["stop"]:
            name = self.files[b["i"]]
            stem = os.path.splitext(name)[0]
            if b["cfg"]["skip"] and self._has_label(stem):
                b["skipped"] += 1
                b["i"] += 1
                self.batch_dlg.progress(b["i"], n, f"{name}: 라벨 있음, 건너뜀")
                continue
            path = os.path.join(self.folder, name)
            rgb = self._file_rgb(path)
            if rgb is None:
                b["failed"] += 1
                b["i"] += 1
                self.batch_dlg.progress(b["i"], n, f"{name}: 이미지를 읽지 못함")
                continue
            b["rgb_shadow"] = shadow_map(rgb)
            b["shape"] = rgb.shape[:2]
            self.batch_dlg.progress(b["i"], n, "")
            self.status.setText(f"SAM 3 일괄 {b['i'] + 1}/{n}: {name}")
            ok = self.sam.segment(rgb=rgb, key=path, prompts=list(b["cfg"]["words"]), conf=b["cfg"]["conf"],
                                  nms_iou=0.7, min_area=30, overlap_iou=0.5)   # 단어끼리 같은 물체(IoU 0.5 이상)는 점수 높은 것 하나
            if not ok:
                self._batch_end("SAM 3에 보내지 못해 멈춤")
            return
        self._batch_end("멈춤" if b["stop"] else "끝")

    def _batch_result(self, res):
        b = self._batch
        name = self.files[b["i"]]
        stem = os.path.splitext(name)[0]
        pieces = list(res.pieces)
        shadows = 0
        if b["cfg"]["no_shadow"]:
            keep = [q for q in pieces if not is_shadow(q.mask, b["rgb_shadow"])]
            shadows = len(pieces) - len(keep)
            pieces = keep
        words = b["cfg"]["words"]
        classes = [words.get(q.prompt, self.canvas.cls) for q in pieces]
        try:
            if pieces:
                h, w = b["shape"]
                if b["mode"] == "seg":
                    batch.write_mask(self._mask_path(stem), pieces, classes)
                else:
                    batch.write_boxes(self._txt_path(stem), pieces, classes, w, h)
                b["done"] += 1
                b["pieces"] += len(pieces)
                self.auto[name] = batch.record(pieces, classes, shadows, b["mode"])
                line = (f"{name}: {len(pieces)}개 (점수 {min(q.score for q in pieces):.2f}~{max(q.score for q in pieces):.2f})"
                        + (f", 그림자 {shadows}개 뺌" if shadows else ""))
            else:
                b["empty"] += 1
                line = f"{name}: 찾지 못함" + (f" (그림자 {shadows}개 뺌)" if shadows else "")
        except OSError as e:
            b["failed"] += 1
            line = f"{name}: 저장 실패 {e}"
        self._mark_item(b["i"])
        b["i"] += 1
        self.batch_dlg.progress(b["i"], len(self.files), line)
        self._batch_next()

    def _batch_end(self, why):
        b = self._batch
        self._batch = None
        try:
            batch.save_auto(self.folder, self.auto)
            self._save_classes()
        except OSError as e:
            self.batch_dlg.progress(b["i"], len(self.files), f"auto_labels.json 저장 실패: {e}")
        summary = (f"{why}: 라벨 만듦 {b['done']}장(조각 {b['pieces']}개), 못 찾음 {b['empty']}장, "
                   f"건너뜀 {b['skipped']}장, 실패 {b['failed']}장, {time.time() - b['t0']:.0f}s")
        self.batch_dlg.progress(b["i"], len(self.files), summary)
        self.batch_dlg.set_running(False)
        self.status.setText("SAM 3 일괄 " + summary)
        if self.idx >= 0:   # 지금 보던 이미지도 새 라벨로 다시 읽는다
            i, self.idx = self.idx, -1
            self._go(i)
        self._update_sam_ui()

    # ── 프로젝트 (main.py 제목줄의 프로젝트 저장/불러오기가 부른다, 파일 읽기/쓰기는 common/project.py) ──
    # 클래스(classes.txt)와 자동 라벨 검수 상태(auto_labels.json)는 이미지 폴더 안에 있어 폴더를 열면 같이 돌아온다
    def project_state(self):
        all_imgs = ([f for f in os.listdir(self.folder) if f.lower().endswith(IMG_EXT + cube.CUBE_EXT)]
                    if os.path.isdir(self.folder) else [])
        single = self.files[0] if len(self.files) == 1 and len(all_imgs) > 1 else ""   # 이미지 한 장만 연 경우
        values = {
            "single_file": single,
            "image": self.files[self.idx] if 0 <= self.idx < len(self.files) else "",   # 마지막으로 보던 이미지
            "tool": self.canvas.tool,
            "class_row": self.cls_list.currentRow(),
            "brush": self.canvas.brush,
            "classes": list(self.classes),
            "prompt": self.prompt_edit.text().strip(),
            "conf": self.conf_spin.value(),
        }
        return {"paths": {"folder": self.folder, "labels": self.label_dir}, "runs": [], "values": values}

    def apply_project(self, state):
        """프로젝트 값을 채운다. 돌려주는 값: 상태 표시줄에 붙일 짧은 안내 (없으면 "")."""
        folder = state.get("paths", {}).get("folder", "")
        v = state.get("values", {})
        if not os.path.isdir(folder):
            return "이미지 폴더를 찾지 못했습니다"
        single = v.get("single_file") or None
        if single and not os.path.isfile(os.path.join(folder, single)):
            single = None   # 그 파일이 없어졌으면 폴더 전체를 연다
        labels = state.get("paths", {}).get("labels", "")
        if labels and os.path.isdir(labels):
            labels = os.path.normpath(labels)   # 프로젝트의 라벨 폴더를 이 이미지 폴더에 기억해 두면 _open_folder가 꺼내 쓴다
            d = self._label_dirs()
            d[os.path.normcase(os.path.abspath(folder))] = labels
            settings.set_value(TOOL, "label_dirs", json.dumps(d, ensure_ascii=False))
        self._open_folder(folder, only=single)
        if os.path.normcase(os.path.abspath(self.folder)) != os.path.normcase(os.path.abspath(folder)):
            return "열기를 취소했습니다"   # 저장 안 한 변경이 있어 묻는 창에서 취소를 누른 경우
        if not self._classes_file() and v.get("classes"):
            self.classes = [str(c) for c in v["classes"]]   # 폴더에 classes.txt가 없으면 프로젝트에 적힌 목록을 쓴다
            self._fill_classes()
        self._set_tool("seg" if v.get("tool") == "seg" else "box")
        if v.get("brush"):
            self.canvas.brush = int(v["brush"])
            self._set_hint()
        row = int(v.get("class_row", 0))
        if 0 <= row < len(self.classes):
            self.cls_list.setCurrentRow(row)
        self.prompt_edit.setText(str(v.get("prompt", self.prompt_edit.text())))
        self.conf_spin.setValue(float(v.get("conf", self.conf_spin.value())))
        if v.get("image") in self.files:
            self._go(self.files.index(v["image"]))   # index: 목록에서 그 이름의 위치
        return ""

    def _focus_prompt(self):
        """프롬프트 도우미: 왼쪽 칸을 SAM 3 보조까지 내리고 프롬프트 칸에 커서를 둔다"""
        self.left_scroll.ensureWidgetVisible(self.sam_group)
        self.prompt_edit.setFocus()
        self.prompt_edit.selectAll()
        self.status.setText("프롬프트 칸에 한국어로 대상을 쓰고(예: 투명한 원판) '찾기'를 누르세요 (VLM이 영어 단어로 바꿔 찾음)")

    # ── 검수 (자동 라벨 승인/수정/버리기) ─────────────────────
    def _toggle_review(self):
        if self.review_group.isVisible():
            self.review_group.setVisible(False)
            return
        if not self.auto:
            self.status.setText("자동 라벨 기록이 없습니다. 먼저 SAM 3 일괄로 라벨을 만드세요")
            return
        self.review_group.setVisible(True)
        self.left_scroll.verticalScrollBar().setValue(0)
        left = [n for n in self._review_queue() if self.auto[n].get("status") == "auto"]
        if left:
            self._go(self.files.index(left[0]))   # 점수가 가장 낮은 남은 이미지부터
        self._fill_review()

    def _review_queue(self):
        """검수할 이미지 이름 목록. 최저 점수가 낮은(자신 없는) 것부터"""
        show_all = self.review_all.isChecked()
        names = [n for n in self.files if n in self.auto and (show_all or self.auto[n].get("status") == "auto")]
        return sorted(names, key=lambda n: self.auto[n].get("min_score") if self.auto[n].get("min_score") is not None else -1)

    def _fill_review(self):
        q = self._review_queue()
        cur = self.files[self.idx] if self.idx >= 0 else None
        self.review_list.clear()
        for n in q:
            r = self.auto[n]
            ms = r.get("min_score")
            it = QListWidgetItem(f"{n}   최저 {ms:.2f}   {r.get('n', 0)}개   {STATUS_NAME.get(r.get('status'), '')}"
                                 if ms is not None else f"{n}   {STATUS_NAME.get(r.get('status'), '')}")
            it.setData(Qt.UserRole, n)   # UserRole: 화면 글자와 별도로 파일 이름을 담아 둔다
            self.review_list.addItem(it)
            if n == cur:
                self.review_list.setCurrentItem(it)
        st = [r.get("status") for n, r in self.auto.items() if n in self.files]
        left = st.count("auto")
        self.review_info.setText(f"남은 것 {left}장 · 확인 {st.count('approved')} · 버림 {st.count('rejected')} · "
                                 f"수정함 {st.count('edited')}")
        on = cur in self.auto
        self.approve_btn.setEnabled(on)
        self.reject_btn.setEnabled(on)

    def _review_clicked(self, item):
        n = item.data(Qt.UserRole)
        if n in self.files:
            self._go(self.files.index(n))

    def _review_next(self):
        """현재 이미지를 뺀 첫 번째 남은(자동) 이미지로. 없으면 알린다"""
        cur = self.files[self.idx] if self.idx >= 0 else None
        left = [n for n in self._review_queue() if self.auto[n].get("status") == "auto" and n != cur]
        if left:
            self._go(self.files.index(left[0]))
        else:
            self._fill_review()
            self.status.setText("검수할 자동 라벨이 더 없습니다")

    def _review_mark(self, status):
        name = self.files[self.idx]
        self.auto.setdefault(name, {})["status"] = status
        try:
            batch.save_auto(self.folder, self.auto)
        except OSError as e:
            QMessageBox.critical(self, "검수", f"auto_labels.json을 쓰지 못했습니다: {e}")
        self._mark_item(self.idx)

    def _review_approve(self):
        if self.idx < 0:
            return
        self._save()
        if self.dirty:   # 저장 실패 (오류 창은 _save가 띄움)
            return
        self._review_mark("approved")
        self._review_next()

    def _review_reject(self):
        if self.idx < 0:
            return
        blank = QImage(self.canvas.pm.size(), QImage.Format_Grayscale8)
        blank.fill(0)
        self.canvas.set_labels([], blank)   # 저장하기 전에는 되돌리기로 살릴 수 있다
        self._save()
        if self.dirty:
            return
        self._review_mark("rejected")
        self.status.setText(f"버림: {self.files[self.idx]} (라벨을 비워 저장함)")
        self._review_next()

    # ── 모델로 1차 라벨링 ────────────────────────────────────
    def _model_open(self):
        if not self.files:
            self.status.setText("먼저 이미지 폴더를 여세요")
            return
        if not self._confirm_leave():
            return
        if self.model_dlg is None:
            self.model_dlg = model_label.ModelLabelDialog(self)
            self.model_dlg.start_requested.connect(self._model_start)
            self.model_dlg.stop_requested.connect(self._model_stop)
        seg = self.canvas.tool == "seg"
        target = ("저장: Segmentation 모델 → 마스크(labels/ png), 객체 인식 모델 → "
                  + ("YOLO-seg면 마스크, 아니면 박스(labels/)" if seg else "박스(labels/)") + " — 모드 버튼으로 바꿈")
        self.model_dlg.setup(self.classes, seg, target, str(settings.get_value(TOOL, "model_path", "")))
        self.model_dlg.show()

    def _model_start(self, cfg):
        mapping = list(cfg["map"])   # '새 클래스로 추가'(ADD)는 그 클래스가 실제로 나왔을 때 번호로 바꾼다 (_model_cls)
        settings.set_value(TOOL, "model_path", cfg["path"])
        items, skipped = [], 0
        for i, f in enumerate(self.files):
            if cfg["skip"] and self._has_label(os.path.splitext(f)[0]):
                skipped += 1
            else:
                items.append((i, os.path.join(self.folder, f)))
        as_mask = self.canvas.tool == "seg"
        self._mrun = {"cfg": cfg, "map": mapping, "n": len(items), "k": 0, "done": 0, "empty": 0, "failed": 0,
                      "skipped": skipped, "pieces": 0, "as_mask": as_mask, "t0": cfg["t0"]}
        dlg = self.model_dlg
        dlg.set_running(True)
        dlg.progress(0, len(items), f"대상 {len(items)}장, 라벨 있어서 건너뜀 {skipped}장")
        th = model_label.InferThread(cfg["path"], cfg["info"], items, cfg["conf"], as_mask, self._saved_view(), self)
        th.result.connect(self._model_result)
        th.log.connect(lambda t: dlg.progress(self._mrun["k"] if self._mrun else 0, len(items), t))
        th.done.connect(self._model_done)
        self._mrun["thread"] = th
        self._update_sam_ui()
        th.start()

    def _model_cls(self, mc):
        """모델 클래스 번호 → Labeler 클래스 번호 (넣지 않으면 -1). '새 클래스로 추가'는 여기서 처음 추가한다"""
        mp = self._mrun["map"]
        if mc >= len(mp):
            return -1
        if mp[mc] == model_label.ADD:
            n = str(self._mrun["cfg"]["info"]["names"][mc])
            if n not in self.classes:
                self.classes.append(n)
                self._mrun["new_cls"] = True
            mp[mc] = self.classes.index(n)
        return mp[mc]

    def _model_stop(self):
        if self._mrun is not None:
            self._mrun["thread"].stop()

    def _model_result(self, i, res):
        m = self._mrun
        m["k"] += 1
        name = self.files[i]
        stem = os.path.splitext(name)[0]
        try:
            if "error" in res:
                m["failed"] += 1
                line = f"{name}: {res['error']}"
            elif "label" in res:   # Segmentation 모델: 모델 클래스 → Labeler 클래스+1
                lab = res["label"]
                out = np.zeros(lab.shape, np.uint8)
                scores, classes = [], []
                for mc in sorted(res["cls_conf"]):
                    lc = self._model_cls(mc)
                    if lc >= 0:
                        out[lab == mc] = min(lc + 1, 255)
                        scores.append(round(res["cls_conf"][mc], 3))
                        classes.append(lc)
                line = self._model_write(name, stem, out if out.any() else None, None, scores, classes, "seg")
            else:                  # 객체 인식 모델
                h, w = res["shape"]
                keep = [(k, self._model_cls(c), s, x1, y1, x2, y2) for k, (c, s, x1, y1, x2, y2) in enumerate(res["boxes"])]
                keep = [t for t in keep if t[1] >= 0]
                scores = [round(s, 3) for _, _, s, *_ in keep]
                classes = [lc for _, lc, *_ in keep]
                if m["as_mask"] and res.get("masks") is not None and keep:   # YOLO-seg → 마스크 (점수 낮은 것부터 칠함)
                    out = np.zeros((h, w), np.uint8)
                    for k, lc, s, *_ in sorted(keep, key=lambda t: t[2]):
                        out[res["masks"][k]] = min(lc + 1, 255)
                    line = self._model_write(name, stem, out, None, scores, classes, "seg")
                else:
                    boxes = [(lc, x1, y1, x2, y2) for _, lc, _, x1, y1, x2, y2 in keep]
                    line = self._model_write(name, stem, None, (boxes, w, h) if boxes else None, scores, classes, "box")
        except OSError as e:
            m["failed"] += 1
            line = f"{name}: 저장 실패 {e}"
        self._mark_item(i)
        self.model_dlg.progress(m["k"], m["n"], line)

    def _model_write(self, name, stem, mask, boxes, scores, classes, mode):
        """마스크 또는 박스를 저장하고 auto_labels.json 기록을 남긴다. 로그 한 줄을 돌려준다"""
        m = self._mrun
        if mask is None and boxes is None:
            m["empty"] += 1
            return f"{name}: 찾지 못함"
        if mask is not None:
            model_label.write_mask_array(self._mask_path(stem), mask)
        else:
            model_label.write_box_txt(self._txt_path(stem), *boxes)
        m["done"] += 1
        m["pieces"] += len(scores)
        self.auto[name] = {"status": "auto", "mode": mode, "source": "model", "model": m["cfg"]["path"],
                           "time": time.strftime("%Y-%m-%d %H:%M"), "n": len(scores),
                           "min_score": min(scores) if scores else None, "scores": scores, "classes": classes}
        unit = "클래스" if m["cfg"]["info"]["kind"] == "seg" else "개"
        return f"{name}: {len(scores)}{unit} (점수 {min(scores):.2f}~{max(scores):.2f})" if scores else f"{name}: 저장"

    def _model_done(self, why):
        m = self._mrun
        self._mrun = None
        if m.get("new_cls"):   # 새로 생긴 클래스를 목록과 classes.txt에 반영
            self._fill_classes()
        try:
            self._save_classes()
            batch.save_auto(self.folder, self.auto)
        except OSError as e:
            self.model_dlg.progress(m["k"], m["n"], f"auto_labels.json 저장 실패: {e}")
        summary = (f"{why}: 라벨 만듦 {m['done']}장, 못 찾음 {m['empty']}장, 건너뜀 {m['skipped']}장, "
                   f"실패 {m['failed']}장, {time.time() - m['t0']:.0f}s")
        self.model_dlg.progress(m["k"], m["n"], summary)
        self.model_dlg.set_running(False)
        self.status.setText("모델 1차 라벨링 " + summary + " → 단계 표시줄 '검수'에서 확인")
        if self.idx >= 0:   # 지금 보던 이미지도 새 라벨로 다시 읽는다
            i, self.idx = self.idx, -1
            self._go(i)
        self._update_sam_ui()

    # ── 다파장 (큐브) ────────────────────────────────────────
    def _saved_view(self):
        """마지막 보기 설정 (밴드 번호는 큐브마다 fit_view로 맞춘다)"""
        v = settings.get_value(TOOL, "fc_view", None)
        return v if isinstance(v, dict) else {}

    def _setup_view(self):
        """새 큐브에 맞게 다파장 칸을 채운다 (신호는 잠시 막는다)"""
        cb = self.cube
        n = cb.n_bands
        v = cube.fit_view(self._saved_view(), n)
        widgets = self.band_spins + [self.lo_spin, self.hi_spin, self.single_chk, self.band_slider]
        for w in widgets:
            w.blockSignals(True)
        for sp, b in zip(self.band_spins, v["bands"]):
            sp.setRange(0, n - 1)
            sp.setValue(int(b))
        self.lo_spin.setValue(float(v["lo"]))
        self.hi_spin.setValue(float(v["hi"]))
        self.band_slider.setRange(0, n - 1)
        self.single_chk.setChecked(v.get("single") is not None)
        self.band_slider.setValue(int(v["single"]) if v.get("single") is not None else n // 2)
        for w in widgets:
            w.blockSignals(False)
        wl = (f" · 파장 {cb.wavelengths[0]:.0f}~{cb.wavelengths[-1]:.0f} nm" if cb.has_wavelengths else " · 파장 정보 없음")
        self.ms_info.setText(f"밴드 {n}개{wl}")
        self._update_band_labels()
        self._ensure_plot()
        self.hover_curve.setData([], [])   # 앞 큐브의 커서 스펙트럼이 남지 않게

    def _current_view(self):
        return {"bands": [sp.value() for sp in self.band_spins], "lo": self.lo_spin.value(), "hi": self.hi_spin.value(),
                "single": self.band_slider.value() if self.single_chk.isChecked() else None}

    def _update_band_labels(self):
        for sp, lab in zip(self.band_spins, self.band_labels):
            lab.setText(cube.band_label(self.cube, sp.value()))
        self.single_label.setText(cube.band_label(self.cube, self.band_slider.value()))
        for sp in self.band_spins:
            sp.setEnabled(not self.single_chk.isChecked())
        self.band_slider.setEnabled(self.single_chk.isChecked())

    def _cube_pixmap(self):
        rgb = np.ascontiguousarray(cube.view_rgb(self.cube.data, self._current_view()))
        h, w = rgb.shape[:2]
        return QPixmap.fromImage(QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy())

    def _view_changed(self, *_):
        if self.cube is None:
            return
        self._update_band_labels()
        settings.set_value(TOOL, "fc_view", self._current_view())
        self.canvas.set_pixmap(self._cube_pixmap())
        self._clear_cands()   # SAM 결과는 예전 그림 기준이므로 버린다

    def _file_rgb(self, path):
        """SAM 3 일괄용 RGB: 큐브면 지금 보기 설정으로 false color, 아니면 그림 그대로. 못 읽으면 None"""
        if cube.is_cube_file(path):
            try:
                cb = cube.load(path)
            except Exception:  # noqa: BLE001
                return None
            return cube.view_rgb(cb.data, cube.fit_view(self._saved_view(), cb.n_bands))
        return batch.read_rgb(path)

    def _ensure_plot(self):
        """스펙트럼 그래프를 처음 한 번 만든다 (pyqtgraph는 큐브를 열 때 처음 불러온다)"""
        if self.plot is not None:
            return
        import pyqtgraph as pg
        theme.apply_pyqtgraph_defaults()
        self.plot = pg.PlotWidget()
        self.plot.setFixedHeight(190)
        self.plot.setLabel("left", "값")
        self.plot.addLegend(offset=(-10, 10))
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.hover_curve = self.plot.plot([], [], pen=pg.mkPen(theme.MUTED, width=1, style=Qt.DashLine), name="커서")
        self.class_curves = []
        self.plot_holder.addWidget(self.plot)

    def _hover_spectrum(self, x, y):
        if self.cube is None or self.plot is None:
            return
        cb = self.cube
        if 0 <= y < cb.data.shape[0] and 0 <= x < cb.data.shape[1]:
            self.hover_curve.setData(cb.band_axis(), cb.data[y, x])

    def _update_class_spectra(self):
        """지금 마스크의 클래스별 평균 스펙트럼을 클래스 색으로 그린다"""
        if self.cube is None or self.plot is None:
            return
        import pyqtgraph as pg
        for c in self.class_curves:
            self.plot.removeItem(c)
        self.class_curves = []
        cb = self.cube
        self.plot.setLabel("bottom", "파장 (nm)" if cb.has_wavelengths else "밴드")
        arr = self.canvas.mask_array()
        if arr.shape != cb.data.shape[:2]:
            return
        for v, (mean, _std, n) in cube.class_means(cb.data, arr).items():
            c = v - 1
            name = self.classes[c] if c < len(self.classes) else f"class{c}"
            pen = pg.mkPen(self.canvas.colors[c] if c < len(self.canvas.colors) else theme.ACCENT, width=2)
            self.class_curves.append(self.plot.plot(cb.band_axis(), mean, pen=pen, name=f"{v} {name} ({n}px)"))

    def _export_spectra(self):
        """클래스별 평균/표준편차 스펙트럼을 CSV로 (첫 열 파장 또는 밴드 번호)"""
        if self.cube is None:
            return
        arr = self.canvas.mask_array()
        stats = cube.class_means(self.cube.data, arr)
        if not stats:
            self.status.setText("칠한 클래스가 없습니다 (평균 스펙트럼은 Segmentation 마스크 기준)")
            return
        default = os.path.join(self.folder, "spectra", self._stem() + "_class_mean.csv")
        path, _ = QFileDialog.getSaveFileName(self, "평균 스펙트럼 저장", default, "CSV (*.csv)")
        if not path:
            return
        cols, head = [self.cube.band_axis()], ["wavelength_nm" if self.cube.has_wavelengths else "band"]
        for v, (mean, std, n) in stats.items():
            name = self.classes[v - 1] if v - 1 < len(self.classes) else f"class{v - 1}"
            cols += [mean, std]
            head += [f"{name}_mean(n={n})", f"{name}_std"]
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # column_stack: 1차원 배열들을 열로 붙인다 (MATLAB의 [a(:), b(:)])
            np.savetxt(path, np.column_stack(cols), delimiter=",", header=",".join(head), comments="", fmt="%.6g")
        except OSError as e:
            QMessageBox.critical(self, "내보내기", f"저장하지 못했습니다: {e}")
            return
        self.status.setText(f"평균 스펙트럼 저장함: {path}")
