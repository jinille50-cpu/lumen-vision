import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QComboBox, QListWidget,
    QListWidgetItem, QGridLayout, QSpinBox, QRadioButton,
    QButtonGroup, QDoubleSpinBox, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView, QTextEdit, QSplitter, QCheckBox,
    QAbstractItemView, QApplication
)
from PySide6.QtCore import Qt, Signal

# YOLO Detection: 이미지는 png/jpg/bmp/tiff, 라벨은 txt 고정
# ("tiff" 선택 시 .tif, "jpg" 선택 시 .jpeg 파일도 함께 매칭됨)
IMG_EXTS = ["png", "jpg", "bmp", "tiff"]

# 학습 방식 라디오 버튼 ID → split_mode 문자열
SPLIT_MODES = {0: "manual", 1: "holdout", 2: "kfold", 3: "full"}


def _ext_suffixes(ext):
    """확장자 선택값 → 실제 매칭할 접미사 튜플 (tiff는 .tif, jpg는 .jpeg 포함)"""
    e = ext.lower()
    if e == "tiff":
        return (".tiff", ".tif")
    if e == "jpg":
        return (".jpg", ".jpeg")
    return (f".{e}",)


from common import settings
from tools.yolo.work_state import TOOL, load_last, pick_dir, save_last


class DataTab(QWidget):
    step_done = Signal()   # 단계 표시줄에 '끝남'을 알리는 신호

    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state
        self._build()
        self.load_last()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)

        # ── 왼쪽: 설정 ───────────────────────────────────────────
        left_w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget(); scroll.setWidget(inner)
        main = QVBoxLayout(inner)
        main.setSpacing(14); main.setContentsMargins(20, 20, 20, 20)

        # 확장자
        ext_group = QGroupBox("파일 형식")
        ext_v = QVBoxLayout(ext_group)
        ext_layout = QHBoxLayout()
        ext_layout.addWidget(QLabel("이미지 확장자:"))
        self.img_ext = QComboBox(); self.img_ext.addItems(IMG_EXTS)
        self.img_ext.currentTextChanged.connect(self._on_ext_changed)
        ext_layout.addWidget(self.img_ext)
        ext_layout.addSpacing(20)
        ext_layout.addWidget(QLabel("라벨 형식:"))
        lbl_fixed = QLabel("txt  (YOLO Detection)")
        lbl_fixed.setObjectName("section_title")
        ext_layout.addWidget(lbl_fixed)
        ext_layout.addStretch()
        ext_v.addLayout(ext_layout)

        # 다채널 TIFF 채널 선택 (원하는 채널을 골라서 사용, 최대 3개)
        ch_layout = QHBoxLayout()
        self.ch_check = QCheckBox("TIFF 특정 채널만 사용")
        self.ch_check.toggled.connect(self._on_ch_toggled)
        ch_layout.addWidget(self.ch_check)
        ch_layout.addSpacing(16)   # 체크박스 글자와 붙어 한 문장처럼 읽히지 않게
        ch_layout.addWidget(QLabel("채널:"))
        self.ch_box_widget = QWidget()
        self.ch_box_layout = QGridLayout(self.ch_box_widget)
        self.ch_box_layout.setContentsMargins(0, 0, 0, 0)
        self.ch_box_layout.setHorizontalSpacing(10)
        self.ch_box_layout.setVerticalSpacing(2)
        self.ch_checks = []
        self._rebuild_channel_checks(4)
        ch_layout.addWidget(self.ch_box_widget)
        ch_layout.addStretch()
        ext_v.addLayout(ch_layout)

        # false color: 채널을 정확히 3개 고르면 밴드마다 퍼센타일 하한~상한으로 늘린 RGB 그림으로 학습 (common/falsecolor.py)
        fc_layout = QHBoxLayout()
        self.fc_check = QCheckBox("false color (밴드 3개를 RGB로)")
        self.fc_check.setToolTip("체크한 순서대로 R, G, B 밴드가 됩니다. 밴드마다 하한~상한 퍼센타일을 0~255로 늘립니다.\n"
                                 "끄면 고른 채널을 원래 값(최소~최대) 그대로 N채널 입력으로 학습합니다.")
        fc_layout.addWidget(self.fc_check)
        fc_layout.addSpacing(16)
        fc_layout.addWidget(QLabel("하한 %:"))
        self.fc_lo = QDoubleSpinBox(); self.fc_lo.setRange(0.0, 49.0); self.fc_lo.setSingleStep(0.5); self.fc_lo.setValue(1.0); self.fc_lo.setFixedWidth(80)
        fc_layout.addWidget(self.fc_lo)
        fc_layout.addWidget(QLabel("상한 %:"))
        self.fc_hi = QDoubleSpinBox(); self.fc_hi.setRange(51.0, 100.0); self.fc_hi.setSingleStep(0.5); self.fc_hi.setValue(99.0); self.fc_hi.setFixedWidth(80)
        fc_layout.addWidget(self.fc_hi)
        fc_layout.addStretch()
        ext_v.addLayout(fc_layout)
        self.fc_check.toggled.connect(self._update_fc_enabled)
        self.ch_hint = QLabel("파일 스캔 시 채널 수 자동 감지 · 체크한 채널들이 순서대로 스택되어 N채널 입력으로 학습됩니다")
        self.ch_hint.setObjectName("hint")
        ext_v.addWidget(self.ch_hint)
        main.addWidget(ext_group)

        # 학습 방식
        split_group = QGroupBox("학습 방식")
        split_v = QVBoxLayout(split_group)
        self.split_btns = QButtonGroup()
        self.rb_manual  = QRadioButton("폴더로 분리  (Train / Val 폴더 따로 지정)")
        self.rb_holdout = QRadioButton("비율로 분리  (전체 데이터에서 랜덤 분할)")
        self.rb_kfold   = QRadioButton("K-fold  (교차검증, 데이터 적을 때 권장)")
        self.rb_full    = QRadioButton("Full Data  (교차검증 없이 전체 데이터로만 학습)")
        self.rb_full.setToolTip(
            "검증 분할 없이 전체 데이터를 train=val로 사용해 최종 모델만 학습합니다.\n"
            "K-Fold가 오류로 중단됐을 때 Full 학습만 따로 돌리는 용도.\n"
            "이전 K-Fold 결과(kfold_summary.json)가 있으면\n"
            "② 학습 탭에서 'K-Fold 평균 best epoch 사용'을 켜서 epoch를 맞출 수 있습니다.")
        self.rb_manual.setChecked(True)
        for i, rb in enumerate([self.rb_manual, self.rb_holdout,
                                self.rb_kfold, self.rb_full]):
            self.split_btns.addButton(rb, i)
            split_v.addWidget(rb)
        self.split_btns.buttonClicked.connect(self._on_split_changed)
        main.addWidget(split_group)

        # 경로
        path_group = QGroupBox("데이터 경로")
        pg = QGridLayout(path_group); pg.setSpacing(8)

        pg.addWidget(QLabel("Train 이미지:"), 0, 0)
        self.train_img_edit = QLineEdit(); self.train_img_edit.setPlaceholderText("이미지 폴더")
        pg.addWidget(self.train_img_edit, 0, 1)
        b = QPushButton("찾기"); b.setObjectName("secondary")
        b.clicked.connect(lambda: self._browse(self.train_img_edit)); pg.addWidget(b, 0, 2)

        pg.addWidget(QLabel("Train 라벨(txt):"), 1, 0)
        self.train_lbl_edit = QLineEdit(); self.train_lbl_edit.setPlaceholderText("txt 라벨 폴더")
        pg.addWidget(self.train_lbl_edit, 1, 1)
        b = QPushButton("찾기"); b.setObjectName("secondary")
        b.clicked.connect(lambda: self._browse(self.train_lbl_edit)); pg.addWidget(b, 1, 2)

        # 직접 입력해도 Validation 폴더를 찾는다 (찾기 버튼은 _browse에서 처리)
        self.train_img_edit.editingFinished.connect(self._autofill_val)
        self.train_lbl_edit.editingFinished.connect(self._autofill_val)

        # Val (manual 전용)
        self.val_img_label = QLabel("Val 이미지:")
        pg.addWidget(self.val_img_label, 2, 0)
        self.val_img_edit = QLineEdit(); self.val_img_edit.setPlaceholderText("이미지 폴더")
        pg.addWidget(self.val_img_edit, 2, 1)
        self.btn_vi = QPushButton("찾기"); self.btn_vi.setObjectName("secondary")
        self.btn_vi.clicked.connect(lambda: self._browse(self.val_img_edit)); pg.addWidget(self.btn_vi, 2, 2)

        self.val_lbl_label = QLabel("Val 라벨(txt):")
        pg.addWidget(self.val_lbl_label, 3, 0)
        self.val_lbl_edit = QLineEdit(); self.val_lbl_edit.setPlaceholderText("txt 라벨 폴더")
        pg.addWidget(self.val_lbl_edit, 3, 1)
        self.btn_vl = QPushButton("찾기"); self.btn_vl.setObjectName("secondary")
        self.btn_vl.clicked.connect(lambda: self._browse(self.val_lbl_edit)); pg.addWidget(self.btn_vl, 3, 2)

        # Hold-out 비율 (holdout 전용)
        self.holdout_label = QLabel("Val 비율:")
        pg.addWidget(self.holdout_label, 4, 0)
        self.holdout_spin = QDoubleSpinBox()
        self.holdout_spin.setRange(0.05, 0.5); self.holdout_spin.setSingleStep(0.05)
        self.holdout_spin.setValue(0.2); self.holdout_spin.setSuffix("   (20% = 0.2)")
        pg.addWidget(self.holdout_spin, 4, 1)

        # K-Fold K값 (kfold 전용)
        self.kfold_label = QLabel("Fold 수 (K):")
        pg.addWidget(self.kfold_label, 5, 0)
        self.kfold_spin = QSpinBox()
        self.kfold_spin.setRange(3, 20); self.kfold_spin.setValue(5)
        pg.addWidget(self.kfold_spin, 5, 1)

        # K-fold 옵션: 층화, fold 안에서 검증 따로 두기
        self.kfold_strat = QCheckBox("Stratified  (이미지에 있는 클래스 기준으로 fold에 고르게 배분)")
        self.kfold_strat.setChecked(True)
        pg.addWidget(self.kfold_strat, 7, 0, 1, 3)
        self.kfold_inner = QCheckBox("fold 안에서 검증 따로 두기")
        pg.addWidget(self.kfold_inner, 8, 0, 1, 3)
        self.kfold_note = QLabel()
        self.kfold_note.setObjectName("hint")
        self.kfold_spin.valueChanged.connect(self._update_kfold_note)
        self._update_kfold_note()
        pg.addWidget(self.kfold_note, 9, 0, 1, 3)

        # 분할 시드 (holdout/kfold 전용) — 채널 선택과 무관하게 동일 분할 보장
        self.split_seed_label = QLabel("분할 시드 (Seed):")
        pg.addWidget(self.split_seed_label, 6, 0)
        self.split_seed_spin = QSpinBox()
        self.split_seed_spin.setRange(0, 999999); self.split_seed_spin.setValue(42)
        self.split_seed_spin.setToolTip(
            "Hold-out / K-Fold 랜덤 분할 시드.\n"
            "분할은 채널 추출 전 원본 파일 목록에서 수행되므로,\n"
            "같은 시드면 TIFF 채널을 몇 개 선택하든 항상 동일하게 분할됩니다.\n"
            "→ 채널 조합 비교 실험 시 이 값을 고정하세요.")
        pg.addWidget(self.split_seed_spin, 6, 1)

        pg.setColumnStretch(1, 1)
        main.addWidget(path_group)

        # 클래스 (파일 스캔 때 라벨에서 자동 감지, 이름만 수정)
        cls_group = QGroupBox("클래스  (data.yaml names)")
        cv = QVBoxLayout(cls_group)
        self.detect_label = QLabel("클래스 -   (파일 스캔을 누르면 감지)")
        self.detect_label.setObjectName("hint")
        cv.addWidget(self.detect_label)
        self.class_table = QTableWidget(0, 2)
        self.class_table.setHorizontalHeaderLabels(["ID", "이름"])
        self.class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.class_table.setMaximumHeight(220)
        cv.addWidget(self.class_table)
        main.addWidget(cls_group)

        # 버튼
        btn_h = QHBoxLayout()
        scan_btn = QPushButton("파일 스캔"); scan_btn.setObjectName("secondary")
        scan_btn.clicked.connect(self._scan_files)
        verify_btn = QPushButton("데이터 검증")
        verify_btn.clicked.connect(self._verify)
        btn_h.addWidget(scan_btn); btn_h.addWidget(verify_btn); btn_h.addStretch()
        main.addLayout(btn_h)
        main.addStretch()

        lw = QVBoxLayout(left_w); lw.setContentsMargins(0, 0, 0, 0); lw.addWidget(scroll)
        splitter.addWidget(left_w)

        # ── 오른쪽: 목록 + 로그 ──────────────────────────────────
        right_w = QWidget()
        rv = QVBoxLayout(right_w); rv.setContentsMargins(16, 16, 16, 16); rv.setSpacing(8)

        two_col = QHBoxLayout()
        left_col = QVBoxLayout()
        ih = QLabel("이미지"); ih.setObjectName("CardTitle")
        left_col.addWidget(ih)
        self.file_list = QListWidget(); self.file_list.setMinimumHeight(180)
        self.file_list.setSelectionMode(QAbstractItemView.NoSelection)   # 보기만
        left_col.addWidget(self.file_list)
        self.img_count_label = QLabel("0개"); self.img_count_label.setObjectName("hint")
        left_col.addWidget(self.img_count_label)
        two_col.addLayout(left_col)

        right_col = QVBoxLayout()
        lh = QLabel("라벨(txt)"); lh.setObjectName("CardTitle")
        right_col.addWidget(lh)
        self.label_list = QListWidget(); self.label_list.setMinimumHeight(180)
        self.label_list.setSelectionMode(QAbstractItemView.NoSelection)
        right_col.addWidget(self.label_list)
        self.lbl_count_label = QLabel("0개"); self.lbl_count_label.setObjectName("hint")
        right_col.addWidget(self.lbl_count_label)
        two_col.addLayout(right_col)
        rv.addLayout(two_col)

        log_lbl = QLabel("검증 로그"); log_lbl.setObjectName("section_title")
        rv.addWidget(log_lbl)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        rv.addWidget(self.log_box, 1)

        splitter.addWidget(right_w)
        splitter.setSizes([600, 520])
        outer.addWidget(splitter, 1)

        # 저장 (오른쪽 아래, Segmentation 탭과 같은 위치/모양)
        save_btn = QPushButton("저장하고 다음")
        save_btn.setObjectName("Primary")
        save_btn.clicked.connect(self._save_state)
        bottom = QHBoxLayout()
        bottom.setContentsMargins(16, 8, 16, 12)
        bottom.addStretch()
        bottom.addWidget(save_btn)
        outer.addLayout(bottom)

        self._on_split_changed()
        self._on_ext_changed(self.img_ext.currentText())

    # ── helpers ──────────────────────────────────────────────────
    def _rebuild_channel_checks(self, n):
        """채널 수 n에 맞춰 채널 선택 체크박스 재생성 (기존 선택은 유지)"""
        prev = set(getattr(self, "_ch_order", []))
        for cb in self.ch_checks:
            self.ch_box_layout.removeWidget(cb)
            cb.deleteLater()
        self.ch_checks = []
        self._ch_order = []          # 체크한 순서 (R/G/B 배치 순서)
        per_row = 8
        enabled = self.ch_check.isChecked() and self.ch_check.isEnabled()
        for i in range(max(1, n)):
            cb = QCheckBox(str(i))
            cb.setEnabled(enabled)
            cb.toggled.connect(lambda on, idx=i: self._on_channel_pick(idx, on))
            self.ch_box_layout.addWidget(cb, i // per_row, i % per_row)
            self.ch_checks.append(cb)
        # 이전 선택 복원 (새 채널 범위 내에서만)
        for i in sorted(prev):
            if i < len(self.ch_checks):
                self.ch_checks[i].setChecked(True)

    def _on_channel_pick(self, idx, on):
        """채널 체크/해제 시 선택 순서 기록 (스택 순서 = 체크한 순서)"""
        if on:
            if idx not in self._ch_order:
                self._ch_order.append(idx)
        else:
            if idx in self._ch_order:
                self._ch_order.remove(idx)

    def _set_channel_checks_enabled(self, on):
        for cb in self.ch_checks:
            cb.setEnabled(on)

    def _on_ch_toggled(self, on):
        self._set_channel_checks_enabled(on)
        self._update_fc_enabled()

    def _update_fc_enabled(self, *_):
        """false color는 'TIFF 특정 채널만 사용'을 켰을 때만, 하한/상한은 false color를 켰을 때만"""
        on = self.ch_check.isChecked() and self.ch_check.isEnabled()
        self.fc_check.setEnabled(on)
        self.fc_lo.setEnabled(on and self.fc_check.isChecked())
        self.fc_hi.setEnabled(on and self.fc_check.isChecked())

    def false_color_error(self):
        """false color를 켰는데 채널이 3개가 아니면 오류 글자, 문제없으면 빈 글자"""
        if self.fc_check.isChecked() and self.fc_check.isEnabled() and len(self._ch_order) != 3:
            return f"false color는 채널을 정확히 3개 골라야 합니다 (지금 {len(self._ch_order)}개)"
        return ""

    def get_false_color(self):
        """false color 설정 {"lo", "hi"} (퍼센타일 %). 안 쓰면 None"""
        if self.get_channel() and len(self._ch_order) == 3 and self.fc_check.isChecked() and self.fc_check.isEnabled():
            return {"lo": self.fc_lo.value(), "hi": self.fc_hi.value()}
        return None

    def _on_ext_changed(self, ext):
        """채널 선택은 tiff에서만 의미 있음"""
        is_tiff = ext.lower() == "tiff"
        self.ch_check.setEnabled(is_tiff)
        self._set_channel_checks_enabled(is_tiff and self.ch_check.isChecked())
        self.ch_hint.setEnabled(is_tiff)
        if not is_tiff:
            self.ch_check.setChecked(False)
        self._update_fc_enabled()

    def get_channel(self):
        """선택된 TIFF 채널 리스트 — 체크한 순서대로 R/G/B 배치 (미사용 시 None)"""
        if self.img_ext.currentText().lower() == "tiff" and self.ch_check.isChecked():
            return list(self._ch_order) or None
        return None
    def _on_split_changed(self, btn=None):
        mode = self.split_btns.checkedId()  # 0=manual, 1=holdout, 2=kfold, 3=full
        manual  = (mode == 0)
        holdout = (mode == 1)
        kfold   = (mode == 2)
        for w in [self.val_img_label, self.val_img_edit, self.btn_vi,
                  self.val_lbl_label, self.val_lbl_edit, self.btn_vl]:
            w.setVisible(manual)
        self.holdout_label.setVisible(holdout)
        self.holdout_spin.setVisible(holdout)
        self.kfold_label.setVisible(kfold)
        self.kfold_spin.setVisible(kfold)
        self.split_seed_label.setVisible(holdout or kfold)
        self.split_seed_spin.setVisible(holdout or kfold)
        for w in (self.kfold_strat, self.kfold_inner, self.kfold_note):
            w.setVisible(kfold)

    def _browse(self, edit):
        # 시작 위치: 칸의 기존 경로 -> Train 이미지 폴더 -> 기본 위치
        path = pick_dir(self, edit.text(), self.state)
        if path:
            edit.setText(path)
            if edit in (self.train_img_edit, self.train_lbl_edit):
                self._autofill_val()

    def _find_val_dir(self, path):
        """path와 같은 상위 폴더 안에서 Validation / Val / Test 폴더를 대소문자 무시하고 찾는다.
        상위 폴더를 3단계까지 올라가며 보고, 찾으면 path와 같은 하위 구조(예: images)를 따라간다."""
        if not path or not os.path.isdir(path):
            return ""
        path = os.path.normpath(path)
        base = path
        for _ in range(3):
            parent = os.path.dirname(base)
            if parent == base:
                break
            rel = os.path.relpath(path, base).split(os.sep) if base != path else []
            subs = {d.lower(): d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))}
            for name in ("validation", "val", "test"):   # 이 순서로 우선
                if name in subs and os.path.join(parent, subs[name]) != base:
                    cand = os.path.join(parent, subs[name])
                    for part in rel:   # 하위 폴더 이름도 대소문자 무시
                        low = {d.lower(): d for d in os.listdir(cand) if os.path.isdir(os.path.join(cand, d))}
                        if part.lower() not in low:
                            cand = ""
                            break
                        cand = os.path.join(cand, low[part.lower()])
                    if cand:
                        return cand
            base = parent
        return ""

    def _autofill_val(self):
        """Train 이미지/라벨 폴더로 Val 이미지/라벨 칸을 채운다. 못 찾으면 비워 둔다."""
        self.val_img_edit.setText(self._find_val_dir(self.train_img_edit.text().strip()))
        self.val_lbl_edit.setText(self._find_val_dir(self.train_lbl_edit.text().strip()))

    def _update_kfold_note(self):
        k = self.kfold_spin.value()
        self.kfold_note.setText(
            f"끔: 학습 {k - 1}/{k}, 평가 1/{k} ({(k - 1) / k * 100:.0f}:{100 / k:.0f}). "
            "평가 fold로 best도 고르므로 점수가 약간 낙관적\n"
            f"켬: 학습 {k - 2}/{k}, 검증 1/{k}, 평가 1/{k} ({(k - 2) / k * 100:.0f}:{100 / k:.0f}:{100 / k:.0f}). "
            "검증으로 best를 고르고 평가는 점수에만 써서 점수가 정직함")

    def _detect_classes(self, lbl_path, lbl_files):
        """txt 라벨의 첫 값(클래스 ID)을 모아 클래스 표를 채운다. 이름은 기존 입력을 유지"""
        old = self.get_class_names() if self.class_table.rowCount() else []
        ids = set()
        QApplication.setOverrideCursor(Qt.WaitCursor)   # 라벨이 많으면 잠깐 걸린다
        try:
            for f in lbl_files:
                with open(os.path.join(lbl_path, f), encoding="utf-8", errors="ignore") as fp:
                    for line in fp:
                        t = line.split()
                        if t:
                            try:
                                ids.add(int(float(t[0])))
                            except ValueError:
                                pass
        finally:
            QApplication.restoreOverrideCursor()
        if not ids:
            self.detect_label.setText("클래스 -   (라벨에 객체 없음)")
            return
        n = max(ids) + 1   # 클래스 ID는 0부터 끊김 없이 쓴다
        self.class_table.setRowCount(n)
        for i in range(n):
            id_item = QTableWidgetItem(str(i)); id_item.setFlags(Qt.ItemIsEnabled)
            self.class_table.setItem(i, 0, id_item)
            self.class_table.setItem(i, 1, QTableWidgetItem(old[i] if i < len(old) else f"class{i}"))
        self.detect_label.setText(f"클래스 {n}개  (라벨 {len(lbl_files)}개 기준) · 이름만 수정할 수 있습니다")

    def get_class_names(self):
        names = []
        for i in range(self.class_table.rowCount()):
            item = self.class_table.item(i, 1)
            names.append(item.text() if item and item.text() else f"class{i}")
        return names

    def _scan_files(self):
        self.file_list.clear(); self.label_list.clear()
        img_path = self.train_img_edit.text().strip()
        lbl_path = self.train_lbl_edit.text().strip()
        ext = self.img_ext.currentText().lower()
        suffixes = _ext_suffixes(ext)

        if img_path and os.path.isdir(img_path):
            imgs = sorted([f for f in os.listdir(img_path)
                           if f.lower().endswith(suffixes)])
            for f in imgs:
                self.file_list.addItem(QListWidgetItem(f))
            self.img_count_label.setText(f"{len(imgs)}개")
            # tiff면 첫 이미지의 채널 수 감지
            if ext == "tiff" and imgs:
                from tools.yolo.yolo_module_wrap import get_channel_count
                n_ch = get_channel_count(os.path.join(img_path, imgs[0]))
                self.ch_hint.setText(f"감지된 채널 수: {n_ch}  (첫 이미지 기준) · 체크 순서대로 스택되어 입력됩니다")
                self._rebuild_channel_checks(n_ch)
                self.log_box.append(f"[스캔] TIFF 채널 수: {n_ch} ({imgs[0]} 기준)")
        else:
            self.img_count_label.setText("경로 없음")

        if lbl_path and os.path.isdir(lbl_path):
            lbls = sorted([f for f in os.listdir(lbl_path)
                           if f.lower().endswith(".txt")])
            for f in lbls:
                self.label_list.addItem(QListWidgetItem(f))
            self.lbl_count_label.setText(f"{len(lbls)}개")
            self._detect_classes(lbl_path, lbls)
        else:
            self.lbl_count_label.setText("경로 없음")

    def _verify(self):
        from tools.yolo.yolo_module_wrap import scan_dataset

        self.log_box.clear()
        img = self.train_img_edit.text().strip()
        lbl = self.train_lbl_edit.text().strip()
        ext = self.img_ext.currentText().lower()
        self.log_box.append("── Train 데이터 검증 ──")
        res = scan_dataset(img, lbl, ext)
        for m in res["messages"]:
            self.log_box.append(m)

        # manual이면 Val도 검증
        if self.split_btns.checkedId() == 0:
            vi = self.val_img_edit.text().strip()
            vl = self.val_lbl_edit.text().strip()
            if vi and vl:
                self.log_box.append("\n── Val 데이터 검증 ──")
                vres = scan_dataset(vi, vl, ext)
                for m in vres["messages"]:
                    self.log_box.append(m)

    def _save_state(self):
        if not self.class_table.rowCount():
            self.log_box.append("클래스가 없습니다. 라벨 폴더를 지정하고 파일 스캔을 먼저 누르세요")
            return
        if self.false_color_error():
            self.log_box.append(f"[오류] {self.false_color_error()}")
            return
        mode = SPLIT_MODES[self.split_btns.checkedId()]
        self.state["split_mode"]    = mode
        self.state["train_img_dir"] = self.train_img_edit.text().strip()
        self.state["train_lbl_dir"] = self.train_lbl_edit.text().strip()
        self.state["val_img_dir"]   = self.val_img_edit.text().strip()
        self.state["val_lbl_dir"]   = self.val_lbl_edit.text().strip()
        self.state["holdout_ratio"] = self.holdout_spin.value()
        self.state["kfold_k"]       = self.kfold_spin.value()
        self.state["split_seed"]    = self.split_seed_spin.value()
        self.state["img_ext"]       = self.img_ext.currentText()
        self.state["channel"]       = self.get_channel()
        self.state["false_color"]   = self.get_false_color()
        self.state["kfold_stratified"] = self.kfold_strat.isChecked()
        self.state["kfold_inner_val"]  = self.kfold_inner.isChecked()
        self.state["num_classes"]   = self.class_table.rowCount()
        ws = self.state["work"]
        ws.set("img_dir", self.val_img_edit.text() or self.train_img_edit.text())
        ws.set("lbl_dir", self.val_lbl_edit.text() or self.train_lbl_edit.text())
        self.state["class_names"]   = self.get_class_names()
        self._save_last()
        self.log_box.append(f"\n설정 저장 완료 (mode={mode}, "
                            f"classes={self.state['class_names']})")
        self.step_done.emit()

    # ── 마지막 설정값(다음에 열 때 그대로)과 프로젝트 파일이 같이 쓰는 값 모으기/채우기 ──
    def _save_last(self):
        save_last("data_tab", self.get_values())
        if self.train_img_edit.text().strip():
            settings.push_recent(TOOL, self.train_img_edit.text().strip())   # 홈 '최근 작업'에 표시

    def get_values(self):
        """데이터 탭 칸의 값 (경로 4개 포함)"""
        return {
            "split_id": self.split_btns.checkedId(),
            "img_ext": self.img_ext.currentText(),
            "train_img_dir": self.train_img_edit.text().strip(),
            "train_lbl_dir": self.train_lbl_edit.text().strip(),
            "val_img_dir": self.val_img_edit.text().strip(),
            "val_lbl_dir": self.val_lbl_edit.text().strip(),
            "holdout_ratio": self.holdout_spin.value(),
            "kfold_k": self.kfold_spin.value(),
            "split_seed": self.split_seed_spin.value(),
            "kfold_stratified": self.kfold_strat.isChecked(),
            "kfold_inner_val": self.kfold_inner.isChecked(),
            "channel_on": self.ch_check.isChecked(),
            "channels": list(self._ch_order),
            "fc_on": self.fc_check.isChecked(),
            "fc_lo": self.fc_lo.value(),
            "fc_hi": self.fc_hi.value(),
            "class_names": self.get_class_names(),
        }

    def load_last(self):
        """마지막으로 '저장하고 다음'을 눌렀을 때의 설정을 칸에 채운다"""
        d = load_last("data_tab")
        if d:
            self.set_values(d)

    def set_values(self, d):
        """get_values() 형식의 값을 칸에 채운다. 폴더가 있으면 파일 스캔까지 하고 클래스 이름, 채널 순서를 덮어쓴다"""
        btn = self.split_btns.button(int(d.get("split_id", 0)))
        if btn:
            btn.setChecked(True)
            self._on_split_changed()
        if d.get("img_ext") in IMG_EXTS:
            self.img_ext.setCurrentText(d["img_ext"])
        for key, edit in (("train_img_dir", self.train_img_edit), ("train_lbl_dir", self.train_lbl_edit),
                          ("val_img_dir", self.val_img_edit), ("val_lbl_dir", self.val_lbl_edit)):
            if os.path.isdir(d.get(key) or ""):   # 없어진 폴더는 채우지 않는다
                edit.setText(d[key])
        self.holdout_spin.setValue(float(d.get("holdout_ratio", self.holdout_spin.value())))
        self.kfold_spin.setValue(int(d.get("kfold_k", self.kfold_spin.value())))
        self.split_seed_spin.setValue(int(d.get("split_seed", self.split_seed_spin.value())))
        self.kfold_strat.setChecked(bool(d.get("kfold_stratified", True)))
        self.kfold_inner.setChecked(bool(d.get("kfold_inner_val", False)))
        if self.train_img_edit.text() or self.train_lbl_edit.text():
            self._scan_files()   # 파일 목록, 클래스 표, TIFF 채널 수
        names = d.get("class_names") or []
        for i in range(min(len(names), self.class_table.rowCount())):   # 고쳐 둔 클래스 이름 복원
            self.class_table.setItem(i, 1, QTableWidgetItem(str(names[i])))
        self.ch_check.setChecked(bool(d.get("channel_on")) and self.ch_check.isEnabled())
        for cb in self.ch_checks:   # 전부 끄고 저장된 순서대로 다시 켠다 (체크 순서 = 스택 순서)
            cb.setChecked(False)
        if self.ch_check.isChecked():
            for i in d.get("channels") or []:
                if 0 <= i < len(self.ch_checks):
                    self.ch_checks[i].setChecked(True)
        self.fc_lo.setValue(float(d.get("fc_lo", self.fc_lo.value())))
        self.fc_hi.setValue(float(d.get("fc_hi", self.fc_hi.value())))
        self.fc_check.setChecked(bool(d.get("fc_on", False)))
        self._update_fc_enabled()
