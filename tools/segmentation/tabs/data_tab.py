import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QComboBox, QListWidget,
    QListWidgetItem, QGridLayout, QSpinBox, QRadioButton,
    QButtonGroup, QAbstractItemView, QSplitter, QCheckBox,
    QDoubleSpinBox, QScrollArea, QFrame, QTableWidget,
    QTableWidgetItem, QHeaderView, QApplication
)
from PySide6.QtCore import Qt, Signal
from tools.segmentation.utils.file_dialog import pick_dir


SUPPORTED_EXTS = ["png", "jpg", "jpeg", "tiff", "tif", "mat", "npy", "hdr"]


class DataTab(QWidget):
    step_done = Signal()   # 단계 표시줄에 '끝남'을 알리는 신호
    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state
        self._build()

    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)

        main = QVBoxLayout(inner)
        main.setSpacing(14)
        main.setContentsMargins(20, 20, 20, 20)

        # ── 확장자 선택 ──────────────────────────────────────────
        ext_group = QGroupBox("파일 확장자 설정")
        ext_layout = QHBoxLayout(ext_group)
        ext_layout.addWidget(QLabel("이미지 확장자:"))
        self.img_ext = QComboBox()
        self.img_ext.addItems(SUPPORTED_EXTS)
        ext_layout.addWidget(self.img_ext)
        ext_layout.addSpacing(20)
        ext_layout.addWidget(QLabel("라벨 확장자:"))
        self.lbl_ext = QComboBox()
        self.lbl_ext.addItems(SUPPORTED_EXTS)
        self.lbl_ext.setCurrentText("png")
        ext_layout.addWidget(self.lbl_ext)
        ext_layout.addStretch()
        main.addWidget(ext_group)

        # ── 학습 방식 선택 ────────────────────────────────────────
        split_group = QGroupBox("학습 방식")
        split_v = QVBoxLayout(split_group)

        self.split_btns = QButtonGroup()
        self.rb_manual   = QRadioButton("폴더로 분리  (Train / Val 폴더 따로 지정)")
        self.rb_holdout  = QRadioButton("비율로 분리  (전체 데이터에서 랜덤 분할)")
        self.rb_kfold    = QRadioButton("K-fold  (교차검증, 데이터 적을 때 권장)")
        self.rb_manual.setChecked(True)
        for i, rb in enumerate([self.rb_manual, self.rb_holdout, self.rb_kfold]):
            self.split_btns.addButton(rb, i)
            split_v.addWidget(rb)

        self.split_btns.buttonClicked.connect(self._on_split_changed)
        main.addWidget(split_group)

        # ── 경로 설정 ─────────────────────────────────────────────
        path_group = QGroupBox("데이터 경로")
        path_grid = QGridLayout(path_group)
        path_grid.setSpacing(8)

        # Train image
        path_grid.addWidget(QLabel("Train 이미지:"), 0, 0)
        self.train_img_edit = QLineEdit(); self.train_img_edit.setPlaceholderText("폴더 경로")
        path_grid.addWidget(self.train_img_edit, 0, 1)
        btn_ti = QPushButton("찾기"); btn_ti.setObjectName("secondary")
        btn_ti.clicked.connect(lambda: self._browse(self.train_img_edit))
        path_grid.addWidget(btn_ti, 0, 2)

        # Train label
        path_grid.addWidget(QLabel("Train 라벨:"), 1, 0)
        self.train_lbl_edit = QLineEdit(); self.train_lbl_edit.setPlaceholderText("폴더 경로")
        path_grid.addWidget(self.train_lbl_edit, 1, 1)
        btn_tl = QPushButton("찾기"); btn_tl.setObjectName("secondary")
        btn_tl.clicked.connect(lambda: self._browse(self.train_lbl_edit))
        path_grid.addWidget(btn_tl, 1, 2)
        # 직접 입력해도 Validation 폴더를 찾는다 (찾기 버튼은 _browse에서 처리)
        # 칸에서 나가기만 해도 editingFinished가 오므로, 손으로 고쳤을 때(isModified)만 찾는다
        self.train_img_edit.editingFinished.connect(
            lambda: self._on_train_edited(self.train_img_edit))
        self.train_lbl_edit.editingFinished.connect(
            lambda: self._on_train_edited(self.train_lbl_edit))

        # Val image (manual 전용)
        self.val_img_label = QLabel("Val 이미지:")
        path_grid.addWidget(self.val_img_label, 2, 0)
        self.val_img_edit = QLineEdit(); self.val_img_edit.setPlaceholderText("폴더 경로")
        path_grid.addWidget(self.val_img_edit, 2, 1)
        self.btn_vi = QPushButton("찾기"); self.btn_vi.setObjectName("secondary")
        self.btn_vi.clicked.connect(lambda: self._browse(self.val_img_edit))
        path_grid.addWidget(self.btn_vi, 2, 2)

        # Val label (manual 전용)
        self.val_lbl_label = QLabel("Val 라벨:")
        path_grid.addWidget(self.val_lbl_label, 3, 0)
        self.val_lbl_edit = QLineEdit(); self.val_lbl_edit.setPlaceholderText("폴더 경로")
        path_grid.addWidget(self.val_lbl_edit, 3, 1)
        self.btn_vl = QPushButton("찾기"); self.btn_vl.setObjectName("secondary")
        self.btn_vl.clicked.connect(lambda: self._browse(self.val_lbl_edit))
        path_grid.addWidget(self.btn_vl, 3, 2)

        # 비율로 분리 비율 (holdout 전용)
        self.holdout_label = QLabel("Val 비율:")
        path_grid.addWidget(self.holdout_label, 4, 0)
        self.holdout_spin = QDoubleSpinBox()
        self.holdout_spin.setRange(0.05, 0.5); self.holdout_spin.setSingleStep(0.05)
        self.holdout_spin.setValue(0.2); self.holdout_spin.setPrefix("  "); self.holdout_spin.setSuffix("  (20% = 0.2)")
        path_grid.addWidget(self.holdout_spin, 4, 1)

        # K-Fold K값 (kfold 전용)
        self.kfold_label = QLabel("K (Fold 수):")
        path_grid.addWidget(self.kfold_label, 5, 0)
        self.kfold_spin = QSpinBox()
        self.kfold_spin.setRange(3, 20); self.kfold_spin.setValue(5)
        path_grid.addWidget(self.kfold_spin, 5, 1)

        # K-fold 옵션: 층화, fold 안에서 검증 따로 두기
        self.kfold_strat = QCheckBox("Stratified  (이미지에 있는 클래스 기준으로 fold에 고르게 배분)")
        self.kfold_strat.setChecked(True)
        path_grid.addWidget(self.kfold_strat, 6, 0, 1, 3)
        self.kfold_inner = QCheckBox("fold 안에서 검증 따로 두기")
        path_grid.addWidget(self.kfold_inner, 7, 0, 1, 3)
        self.kfold_note = QLabel(
            "끔: 학습 4/5, 평가 1/5 (K=5 기준, 8:2). 평가 fold로 best도 고르므로 점수가 약간 낙관적\n"
            "켬: 학습 3/5, 검증 1/5, 평가 1/5 (6:2:2). 검증으로 best를 고르고 평가는 점수에만 써서 점수가 정직함")
        self.kfold_note.setObjectName("hint")
        self.kfold_spin.valueChanged.connect(self._update_kfold_note)
        self._update_kfold_note()
        path_grid.addWidget(self.kfold_note, 8, 0, 1, 3)

        # Test 이미지/라벨은 ④ 평가탭, ⑤ 추론탭에서 직접 지정 → 제거됨

        path_grid.setColumnStretch(1, 1)
        main.addWidget(path_group)

        # ── 파일 목록 확인 ────────────────────────────────────────
        list_group = QGroupBox("파일 목록")
        list_v = QVBoxLayout(list_group)

        scan_btn = QPushButton("파일 스캔")
        scan_btn.clicked.connect(self._scan_files)
        list_v.addWidget(scan_btn)

        # 좌(이미지) / 우(라벨) 2열 레이아웃
        two_col = QHBoxLayout()

        # 왼쪽 — 이미지
        left_col = QVBoxLayout()
        img_header = QLabel("이미지")
        img_header.setObjectName("CardTitle")
        left_col.addWidget(img_header)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.NoSelection)   # 보기만
        self.file_list.setMinimumHeight(160)
        left_col.addWidget(self.file_list)
        self.img_count_label = QLabel("0개")
        self.img_count_label.setObjectName("hint")
        left_col.addWidget(self.img_count_label)
        two_col.addLayout(left_col)

        # 오른쪽 — 라벨
        right_col = QVBoxLayout()
        lbl_header = QLabel("라벨")
        lbl_header.setObjectName("CardTitle")
        right_col.addWidget(lbl_header)
        self.label_list = QListWidget()
        self.label_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.label_list.setMinimumHeight(160)
        right_col.addWidget(self.label_list)
        self.lbl_count_label = QLabel("0개")
        self.lbl_count_label.setObjectName("hint")
        right_col.addWidget(self.lbl_count_label)
        two_col.addLayout(right_col)

        list_v.addLayout(two_col)

        self.file_count_label = QLabel("이미지 0개 / 라벨 0개")
        self.file_count_label.setObjectName("hint")
        list_v.addWidget(self.file_count_label)

        # ── 라벨 클래스 (파일 스캔 때 자동 감지) ──────────────────
        cls_group = QGroupBox("라벨 클래스")
        cls_v = QVBoxLayout(cls_group)
        self.detect_label = QLabel("입력 채널 -  |  클래스 -   (파일 스캔을 누르면 감지)")
        self.detect_label.setObjectName("hint")
        cls_v.addWidget(self.detect_label)
        self.auto_w_btn = QPushButton("자동 가중치")
        self.auto_w_btn.setCheckable(True)   # 켜짐/꺼짐 버튼
        self.auto_w_btn.setToolTip("켜면 픽셀 비율로 계산(빈도 역수의 제곱근, 배경 1 기준, 상한 10), 끄면 모두 1.0")
        self.auto_w_btn.toggled.connect(self._apply_weights)
        cls_v.addWidget(self.auto_w_btn, 0, Qt.AlignLeft)
        self.ratios = []   # 감지한 클래스별 픽셀 비율
        self.class_table = QTableWidget(0, 4)
        self.class_table.setHorizontalHeaderLabels(["ID", "이름", "픽셀 비율", "가중치"])
        self.class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.class_table.setMinimumHeight(160)
        cls_v.addWidget(self.class_table)
        self.in_channels_found = None   # 감지한 입력 채널 수

        # ── 저장 (오른쪽 아래, 모델 탭과 같은 위치/모양) ──────────
        save_btn = QPushButton("저장하고 다음")
        save_btn.setObjectName("Primary")
        save_btn.clicked.connect(self._save_state)
        main.addStretch()

        # 두 칸: 왼쪽 위 설정(스크롤) + 왼쪽 아래 파일 목록 / 오른쪽 위 라벨 클래스 표
        left = QSplitter(Qt.Vertical)
        left.addWidget(scroll)
        left.addWidget(list_group)
        left.setSizes([420, 260])
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(16, 16, 16, 16)
        rv.addWidget(cls_group, 1)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([560, 580])
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter, 1)
        bottom = QHBoxLayout()
        bottom.setContentsMargins(16, 8, 16, 12)
        bottom.addStretch()
        bottom.addWidget(save_btn)
        outer.addLayout(bottom)
        self._on_split_changed()

    # ── helpers ──────────────────────────────────────────────────
    def _on_split_changed(self, btn=None):
        mode = self.split_btns.checkedId()  # 0=manual, 1=holdout, 2=kfold
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
        for w in (self.kfold_strat, self.kfold_inner, self.kfold_note):
            w.setVisible(kfold)

    def _browse(self, edit: QLineEdit):
        path = pick_dir(self, edit.text(), self.state)
        if path:
            edit.setText(path)
            if edit in (self.train_img_edit, self.train_lbl_edit):
                self._autofill_val(edit)

    def _find_val_dir(self, path):
        """path와 같은 상위 폴더 안에서 Validation / Val / Test 폴더를 대소문자 무시하고 찾는다.
        path의 상위 폴더를 3단계까지 올라가며 보고, 찾으면 path와 같은 하위 구조(예: images)를 따라간다."""
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

    def _on_train_edited(self, train_edit: QLineEdit):
        if train_edit.isModified():          # setText로 채운 값은 False, 키보드로 고치면 True
            train_edit.setModified(False)
            self._autofill_val(train_edit)

    def _autofill_val(self, train_edit: QLineEdit):
        """바꾼 Train 칸(이미지 또는 라벨)에 맞는 Val 칸 하나만 채운다. 못 찾으면 그대로 둔다."""
        val_edit = self.val_img_edit if train_edit is self.train_img_edit else self.val_lbl_edit
        found = self._find_val_dir(train_edit.text().strip())
        if found:
            val_edit.setText(found)

    def _scan_files(self):
        self.file_list.clear()
        self.label_list.clear()

        img_path = self.train_img_edit.text().strip()
        lbl_path = self.train_lbl_edit.text().strip()
        img_ext  = self.img_ext.currentText().lower()
        lbl_ext  = self.lbl_ext.currentText().lower()

        # 이미지 스캔
        if img_path and os.path.isdir(img_path):
            img_files = sorted([f for f in os.listdir(img_path)
                                 if f.lower().endswith(f".{img_ext}")])
            for f in img_files:
                self.file_list.addItem(QListWidgetItem(f))
            self.img_count_label.setText(f"{len(img_files)}개")
        else:
            img_files = []
            self.img_count_label.setText("경로 없음")

        # 라벨 스캔
        if lbl_path and os.path.isdir(lbl_path):
            lbl_files = sorted([f for f in os.listdir(lbl_path)
                                 if f.lower().endswith(f".{lbl_ext}")])
            for f in lbl_files:
                self.label_list.addItem(QListWidgetItem(f))
            self.lbl_count_label.setText(f"{len(lbl_files)}개")
        else:
            lbl_files = []
            self.lbl_count_label.setText("경로 없음")

        text = f"이미지 {self.file_list.count()}개 / 라벨 {self.label_list.count()}개"
        if img_files and lbl_files:
            from tools.segmentation.utils.pair_utils import pair_lists
            _, pinfo = pair_lists(img_files, lbl_files)
            if pinfo["mode"] == "index":
                text += "  |  파일명 매칭 실패 → 정렬 순서로 매칭"
            else:
                text += f"  |  짝 {pinfo['n_paired']}쌍"
                if pinfo["suffix"]:
                    text += f" (접미사 '{pinfo['suffix']}')"
                if pinfo["missing"]:
                    text += f", 라벨 없는 이미지 {len(pinfo['missing'])}개"
                if pinfo["orphan"]:
                    text += f", 이미지 없는 라벨 {len(pinfo['orphan'])}개"
        self.file_count_label.setText(text)
        self._detect_classes(img_path, img_ext, lbl_path, lbl_files)

    def _detect_classes(self, img_path, img_ext, lbl_path, lbl_files):
        """입력 채널 수와 라벨 클래스(픽셀 비율, 가중치 제안)를 감지해 표에 채운다"""
        from tools.segmentation.utils.label_utils import read_array, scan_classes, suggest_weights
        self.class_table.setRowCount(0)
        self.in_channels_found = None
        ch_text = "-"
        img_files = [f for f in os.listdir(img_path) if f.lower().endswith(f".{img_ext}")] \
            if img_path and os.path.isdir(img_path) else []
        try:
            if img_files:
                arr = read_array(os.path.join(img_path, sorted(img_files)[0]))
                self.in_channels_found = arr.shape[2] if arr.ndim == 3 else 1
                ch_text = f"{self.in_channels_found}  ({arr.shape[0]}x{arr.shape[1]})"
        except Exception as e:
            print(f"[DataTab] 채널 감지 실패: {e}")
        if not lbl_files:
            self.detect_label.setText(f"입력 채널 {ch_text}  |  클래스 -  (라벨 파일 없음)")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)   # 라벨이 많으면 잠깐 걸린다
        try:
            counts, n_read = scan_classes([os.path.join(lbl_path, f) for f in lbl_files])
        except Exception as e:
            self.detect_label.setText(f"입력 채널 {ch_text}  |  클래스 감지 실패: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        n = max(counts) + 1   # 클래스 ID는 0부터 끊김 없이 쓴다 (빈 ID는 비율 0)
        total = sum(counts.values())
        ratios = [counts.get(i, 0) / total for i in range(n)]
        self.ratios = ratios
        self.class_table.setRowCount(n)
        for i in range(n):
            id_item = QTableWidgetItem(str(i)); id_item.setFlags(Qt.ItemIsEnabled)
            r_item = QTableWidgetItem(f"{ratios[i] * 100:.2f}%"); r_item.setFlags(Qt.ItemIsEnabled)
            self.class_table.setItem(i, 0, id_item)
            self.class_table.setItem(i, 1, QTableWidgetItem("Background" if i == 0 else f"Class {i}"))
            self.class_table.setItem(i, 2, r_item)
            self.class_table.setItem(i, 3, QTableWidgetItem("1.00"))
        self._apply_weights()
        self.detect_label.setText(f"입력 채널 {ch_text}  |  클래스 {n}개  (라벨 {n_read}개 기준)")

    def _apply_weights(self):
        """자동 가중치 켬: 픽셀 비율로 계산해 채움, 끔: 모두 1.0"""
        from tools.segmentation.utils.label_utils import suggest_weights
        n = self.class_table.rowCount()
        if not n:
            return
        weights = suggest_weights(self.ratios) if self.auto_w_btn.isChecked() else [1.0] * n
        for i in range(n):
            self.class_table.setItem(i, 3, QTableWidgetItem(f"{weights[i]:.2f}"))

    def _update_kfold_note(self):
        k = self.kfold_spin.value()
        self.kfold_note.setText(
            f"끔: 학습 {k - 1}/{k}, 평가 1/{k} ({(k - 1) / k * 100:.0f}:{100 / k:.0f}). "
            "평가 fold로 best도 고르므로 점수가 약간 낙관적\n"
            f"켬: 학습 {k - 2}/{k}, 검증 1/{k}, 평가 1/{k} ({(k - 2) / k * 100:.0f}:{100 / k:.0f}:{100 / k:.0f}). "
            "검증으로 best를 고르고 평가는 점수에만 써서 점수가 정직함")

    def _save_state(self):
        mode_map = {0: "manual", 1: "holdout", 2: "kfold"}
        self.state["split_mode"]     = mode_map[self.split_btns.checkedId()]
        self.state["train_img_dir"]  = self.train_img_edit.text().strip()
        self.state["train_lbl_dir"]  = self.train_lbl_edit.text().strip()
        self.state["val_img_dir"]    = self.val_img_edit.text().strip()
        self.state["val_lbl_dir"]    = self.val_lbl_edit.text().strip()
        self.state["holdout_ratio"]  = self.holdout_spin.value()
        self.state["kfold_k"]        = self.kfold_spin.value()
        self.state["kfold_stratified"] = self.kfold_strat.isChecked()
        self.state["kfold_inner_val"]  = self.kfold_inner.isChecked()
        self.state["img_ext"]        = self.img_ext.currentText()
        if self.state.get("work") is not None:
            self.state["work"].set("img_ext", self.img_ext.currentText())   # 평가/추론/경량화가 이어받음
        self.state["lbl_ext"]        = self.lbl_ext.currentText()
        self.state["file_order"]     = [
            self.file_list.item(i).text()
            for i in range(self.file_list.count())
        ]
        # 감지한 클래스/채널 (감지 전이면 모델 탭 값을 그대로 둔다)
        if self.class_table.rowCount():
            names, weights = [], []
            for i in range(self.class_table.rowCount()):
                names.append(self.class_table.item(i, 1).text().strip() or f"Class {i}")
                try:
                    weights.append(float(self.class_table.item(i, 3).text()))
                except ValueError:
                    weights.append(1.0)
            self.state["detected_class_names"] = names
            self.state["detected_class_weights"] = weights
        self.state["detected_in_channels"] = self.in_channels_found
        print("[DataTab] 설정 저장 완료:", self.state)
        self.step_done.emit()
