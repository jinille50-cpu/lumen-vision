import os
import json
from datetime import datetime
from common import theme
from tools.yolo.work_state import load_last, save_config, save_last
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QScrollArea, QTextEdit, QProgressBar, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal


# 모델 계열 × 크기 (RT-DETR도 Ultralytics가 같은 라벨/데이터 형식으로 학습한다)
YOLO_VERSIONS = {
    "YOLOv8":  "yolov8{}.pt",
    "YOLOv10": "yolov10{}.pt",
    "YOLO11":  "yolo11{}.pt",
    "YOLO12":  "yolo12{}.pt",
    "YOLO26":  "yolo26{}.pt",
    "RT-DETR": "rtdetr-{}.pt",
}
YOLO_SIZES = ["n", "s", "m", "l"]
MODEL_SIZES = {"RT-DETR": ["l", "x"]}   # 여기 없는 계열은 YOLO_SIZES


# ═══════════════════════════════════════════════════════════════
#  학습 Worker
# ═══════════════════════════════════════════════════════════════
class TrainWorker(QThread):
    # epoch, total, box_loss, cls_loss, mAP50, mAP50_95
    epoch_done = Signal(int, int, object, object, object, object)
    fold_start = Signal(int, int)   # fold_idx, total_folds (0,0 = Full 학습)
    log_msg    = Signal(str)
    finished   = Signal(dict)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        from tools.yolo.yolo_module_wrap import build_yaml, train, train_kfold, train_full

        cfg = self.config
        aug = cfg.get("aug", None)
        # train()에 공통 전달할 학습 파라미터
        base = {
            "model": cfg["model"], "epochs": cfg["epochs"], "batch": cfg["batch"],
            "imgsz": cfg["imgsz"], "lr0": cfg["lr0"], "optimizer": cfg["optimizer"],
            "patience": cfg["patience"], "project": cfg["project"],
            "seed": cfg.get("seed", 42), "seed_on": cfg.get("seed_on", True),
        }
        try:
            if cfg["split_mode"] == "kfold":
                # ── K-Fold: build + K회 학습 + Full 재학습 ──────────
                self.log_msg.emit("[K-Fold] 데이터 분할 + 학습 시작...")
                kcfg = {
                    **base,
                    "split_seed":    cfg.get("split_seed"),
                    "kfold_k":       cfg.get("kfold_k", 5),
                    "kfold_stratified": cfg.get("kfold_stratified", False),
                    "kfold_inner_val":  cfg.get("kfold_inner_val", False),
                    "train_img_dir": cfg["train_img_dir"],
                    "train_lbl_dir": cfg["train_lbl_dir"],
                    "img_ext":       cfg.get("img_ext", "png"),
                    "channel":       cfg.get("channel"),
                    "false_color":   cfg.get("false_color"),
                    "class_names":   cfg["class_names"],
                    "aug":           aug,
                    "full_retrain":  True,
                }
                result = train_kfold(
                    kcfg,
                    on_epoch=self._on_epoch,
                    on_log=self.log_msg.emit,
                    should_stop=lambda: self._stop,
                    on_fold=self.fold_start.emit,
                )
            elif cfg["split_mode"] == "full":
                # ── Full Data 단독 학습: 교차검증 없이 전체 데이터로만 ──
                self.log_msg.emit("[Full Data] 전체 데이터 학습 시작...")
                fcfg = {
                    **base,
                    "name":             cfg["name"],
                    "train_img_dir":    cfg["train_img_dir"],
                    "train_lbl_dir":    cfg["train_lbl_dir"],
                    "img_ext":          cfg.get("img_ext", "png"),
                    "channel":          cfg.get("channel"),
                    "false_color":      cfg.get("false_color"),
                    "class_names":      cfg["class_names"],
                    "aug":              aug,
                    "use_kfold_epochs": cfg.get("use_kfold_epochs", False),
                    "kfold_run_dir":    cfg.get("kfold_run_dir"),
                }
                result = train_full(
                    fcfg,
                    on_epoch=self._on_epoch,
                    on_log=self.log_msg.emit,
                    should_stop=lambda: self._stop,
                    on_fold=self.fold_start.emit,
                )
            else:
                # ── manual / holdout: 단일 data.yaml + 1회 학습 ─────
                self.log_msg.emit("[1/2] data.yaml 생성 중...")
                yaml_path = build_yaml(
                    out_dir=cfg["project"],
                    class_names=cfg["class_names"],
                    split_mode=cfg["split_mode"],
                    train_img_dir=cfg["train_img_dir"],
                    train_lbl_dir=cfg["train_lbl_dir"],
                    val_img_dir=cfg.get("val_img_dir"),
                    val_lbl_dir=cfg.get("val_lbl_dir"),
                    img_ext=cfg.get("img_ext", "png"),
                    holdout_ratio=cfg.get("holdout_ratio", 0.2),
                    aug=aug,
                    # 분할 시드는 학습 시드와 분리 — 채널 조합 실험 시 동일 분할 보장
                    seed=cfg.get("split_seed", cfg.get("seed", 42)),
                    channel=cfg.get("channel"),
                    false_color=cfg.get("false_color"),
                    log=self.log_msg.emit,
                )
                self.log_msg.emit("[2/2] 학습 시작...")
                result = train(
                    {**base, "data_yaml": yaml_path, "name": cfg["name"], "aug": aug},
                    on_epoch=self._on_epoch,
                    on_log=self.log_msg.emit,
                    should_stop=lambda: self._stop,
                )
                # 평가탭 연동용 — 실제 학습에 사용한 data.yaml
                if isinstance(result, dict):
                    result.setdefault("data_yaml", yaml_path)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.log_msg.emit(f"[오류] {e}\n{traceback.format_exc()}")
            self.finished.emit({"success": False, "error": str(e)})

    def _on_epoch(self, ep, total, m):
        self.epoch_done.emit(ep, total, m.get("box_loss"), m.get("cls_loss"),
                             m.get("mAP50"), m.get("mAP50_95"))


# ═══════════════════════════════════════════════════════════════
#  TrainTab UI
# ═══════════════════════════════════════════════════════════════
class TrainTab(QWidget):
    step_done = Signal()   # 단계 표시줄에 '끝남'을 알리는 신호

    def __init__(self, shared_state):
        super().__init__()
        self.state = shared_state
        self.worker = None
        self._data_tab = None
        self.box_loss_data = []
        self.cls_loss_data = []
        self.map50_data    = []
        self.map5095_data  = []
        self._last_ep      = 0   # 중복 epoch 콜백(최종 검증) 방지용
        self._build()
        self._load_last()

    def set_data_tab(self, data_tab):
        self._data_tab = data_tab

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)

        # ── 왼쪽: 설정 ───────────────────────────────────────────
        left_w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); scroll.setWidget(inner)
        lv = QVBoxLayout(inner); lv.setSpacing(12); lv.setContentsMargins(16, 16, 16, 16)

        # 모델 선택
        mg = QGroupBox("모델 선택")
        m = QGridLayout(mg)
        m.addWidget(QLabel("모델 계열:"), 0, 0)
        self.version_combo = QComboBox(); self.version_combo.addItems(list(YOLO_VERSIONS.keys()))
        self.version_combo.setCurrentText("YOLOv8")
        m.addWidget(self.version_combo, 0, 1)
        m.addWidget(QLabel("모델 크기:"), 1, 0)
        self.size_combo = QComboBox(); self.size_combo.addItems(YOLO_SIZES)
        m.addWidget(self.size_combo, 1, 1)
        self.model_name_lbl = QLabel("→ yolov8n.pt")
        self.model_name_lbl.setObjectName("hint")
        m.addWidget(self.model_name_lbl, 2, 0, 1, 2)
        self.version_combo.currentTextChanged.connect(self._update_sizes)
        self.version_combo.currentTextChanged.connect(self._update_model_name)
        self.size_combo.currentTextChanged.connect(self._update_model_name)
        m.setColumnStretch(1, 1)
        lv.addWidget(mg)

        # 하이퍼파라미터
        hp = QGroupBox("하이퍼파라미터")
        hg = QGridLayout(hp)
        hg.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs = QSpinBox(); self.epochs.setRange(1, 5000); self.epochs.setValue(100)
        hg.addWidget(self.epochs, 0, 1)
        hg.addWidget(QLabel("Batch Size:"), 1, 0)
        # (Full Data 모드 전용 옵션은 아래 별도 그룹에 배치)
        self.batch = QSpinBox(); self.batch.setRange(1, 256); self.batch.setValue(16)
        hg.addWidget(self.batch, 1, 1)
        hg.addWidget(QLabel("Image Size:"), 2, 0)
        self.imgsz = QSpinBox(); self.imgsz.setRange(64, 2048); self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)
        hg.addWidget(self.imgsz, 2, 1)
        hg.addWidget(QLabel("LR0:"), 3, 0)
        self.lr0 = QDoubleSpinBox(); self.lr0.setDecimals(5); self.lr0.setRange(1e-5, 1.0)
        self.lr0.setSingleStep(0.001); self.lr0.setValue(0.01)
        hg.addWidget(self.lr0, 3, 1)
        hg.addWidget(QLabel("Optimizer:"), 4, 0)
        self.optimizer = QComboBox()
        self.optimizer.addItems(["auto", "SGD", "Adam", "AdamW", "RMSProp"])
        hg.addWidget(self.optimizer, 4, 1)
        hg.addWidget(QLabel("Patience:"), 5, 0)
        self.patience = QSpinBox(); self.patience.setRange(0, 1000); self.patience.setValue(50)
        hg.addWidget(self.patience, 5, 1)
        self.seed_check = QCheckBox("시드 고정")
        self.seed_check.setChecked(True)
        hg.addWidget(self.seed_check, 6, 0)
        self.seed = QSpinBox(); self.seed.setRange(0, 999999); self.seed.setValue(42)
        self.seed.setToolTip("Ultralytics 학습 재현성 시드.\n"
                             "데이터 분할(비율/OOF) 시드는 ① 데이터 탭의 '분할 시드'에서 설정")
        self.seed_check.toggled.connect(self.seed.setEnabled)
        hg.addWidget(self.seed, 6, 1)
        lv.addWidget(hp)

        # Full Data 학습 옵션 (① 데이터 탭에서 Full Data 선택 시에만 적용)
        self.full_group = QGroupBox("Full Data 학습 옵션")
        fg = QVBoxLayout(self.full_group)
        self.use_kfold_ep = QCheckBox("K-Fold 평균 best epoch 사용")
        self.use_kfold_ep.setToolTip(
            "결과 저장 폴더의 kfold_summary.json에서\n"
            "fold별 best epoch의 평균을 읽어 그 값만큼만 학습합니다.\n"
            "(파일이 없으면 위 Epochs 설정값을 그대로 사용)")
        self.use_kfold_ep.toggled.connect(self._update_kfold_epoch_hint)
        fg.addWidget(self.use_kfold_ep)
        self.kfold_ep_hint = QLabel(""); self.kfold_ep_hint.setObjectName("hint")
        self.kfold_ep_hint.setWordWrap(True)
        fg.addWidget(self.kfold_ep_hint)
        lv.addWidget(self.full_group)

        # Augmentation
        aug_g = QGroupBox("Augmentation  (Ultralytics 내장)")
        ag = QGridLayout(aug_g)
        ag.addWidget(QLabel("항목"), 0, 0)
        ag.addWidget(QLabel("사용"), 0, 1)
        ag.addWidget(QLabel("수치"), 0, 2)
        # (key, label, default, range_max)
        aug_defs = [
            ("fliplr",   "좌우 반전 (fliplr)",     0.5,  1.0),
            ("flipud",   "상하 반전 (flipud)",     0.0,  1.0),
            ("mosaic",   "Mosaic",                1.0,  1.0),
            ("mixup",    "MixUp",                 0.0,  1.0),
            ("hsv_h",    "HSV 색조 (hsv_h)",       0.015, 1.0),
            ("hsv_s",    "HSV 채도 (hsv_s)",       0.7,  1.0),
            ("hsv_v",    "HSV 명도 (hsv_v)",       0.4,  1.0),
            ("degrees",  "회전 (degrees)",         0.0,  180.0),
            ("translate","이동 (translate)",       0.1,  1.0),
            ("scale",    "스케일 (scale)",          0.5,  1.0),
        ]
        self.aug_checks = {}; self.aug_spins = {}
        for i, (key, label, val, rmax) in enumerate(aug_defs):
            r = i + 1
            ag.addWidget(QLabel(label), r, 0)
            cb = QCheckBox(); cb.setChecked(val > 0)
            ag.addWidget(cb, r, 1); self.aug_checks[key] = cb
            sp = QDoubleSpinBox(); sp.setDecimals(3); sp.setRange(0.0, rmax)
            sp.setValue(val); sp.setFixedWidth(90)
            ag.addWidget(sp, r, 2); self.aug_spins[key] = sp
        lv.addWidget(aug_g)

        # 결과 저장 경로
        pr = QGroupBox("체크포인트 저장")
        pg = QVBoxLayout(pr)
        self.ckpt_label = QLabel("저장: 라벨 폴더 상위 / checkpoints / <모델>_<날짜-시각> /")
        self.ckpt_label.setObjectName("hint")
        self.ckpt_label.setWordWrap(True)
        pg.addWidget(self.ckpt_label)
        lv.addWidget(pr)
        lv.addStretch()

        lw = QVBoxLayout(left_w); lw.setContentsMargins(0, 0, 0, 0); lw.addWidget(scroll)
        splitter.addWidget(left_w)

        # ── 오른쪽: 모니터 ───────────────────────────────────────
        right_w = QWidget()
        rv = QVBoxLayout(right_w); rv.setContentsMargins(16, 16, 16, 16); rv.setSpacing(8)

        bh = QHBoxLayout()
        self.start_btn = QPushButton("학습 시작"); self.start_btn.clicked.connect(self._start_train)
        self.stop_btn = QPushButton("중단"); self.stop_btn.setObjectName("secondary")
        self.stop_btn.clicked.connect(self._stop_train); self.stop_btn.setEnabled(False)
        bh.addWidget(self.start_btn); bh.addWidget(self.stop_btn)
        rv.addLayout(bh)

        self.progress = QProgressBar(); self.progress.setRange(0, 100)
        rv.addWidget(self.progress)
        self.epoch_label = QLabel("대기 중..."); self.epoch_label.setObjectName("hint")
        rv.addWidget(self.epoch_label)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=(6, 4), facecolor=theme.BG0)
            self.ax1 = self.fig.add_subplot(121)
            self.ax2 = self.fig.add_subplot(122)
            self.canvas = Canvas(self.fig); rv.addWidget(self.canvas, 1)
            self.has_mpl = True
        except ImportError:
            rv.addWidget(QLabel("matplotlib 미설치")); self.has_mpl = False

        log_lbl = QLabel("학습 로그"); log_lbl.setObjectName("section_title")
        rv.addWidget(log_lbl)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(160)
        rv.addWidget(self.log_box)

        splitter.addWidget(right_w)
        splitter.setSizes([400, 600])
        outer.addWidget(splitter)

        self._update_model_name()
        self._update_kfold_epoch_hint()

    # ── helpers ──────────────────────────────────────────────────
    def _update_sizes(self, version):
        """계열마다 있는 크기가 달라(RT-DETR은 l, x) 크기 목록을 바꾼다. 같은 크기가 있으면 그대로 둔다"""
        cur = self.size_combo.currentText()
        sizes = MODEL_SIZES.get(version, YOLO_SIZES)
        self.size_combo.blockSignals(True)   # 목록을 바꾸는 동안 이름 갱신이 여러 번 불리지 않게
        self.size_combo.clear()
        self.size_combo.addItems(sizes)
        self.size_combo.setCurrentText(cur if cur in sizes else sizes[0])
        self.size_combo.blockSignals(False)

    def _update_model_name(self):
        v = self.version_combo.currentText()
        s = self.size_combo.currentText()
        name = YOLO_VERSIONS[v].format(s)
        from tools.yolo.yolo_module_wrap import local_weight_path
        where = ("models/weights에 있음" if local_weight_path(name)
                 else "models/weights에 없음 — 처음 학습할 때 자동으로 받음")
        self.model_name_lbl.setText(f"→ {name}  ({where})")
        return name

    def _ckpt_root(self):
        """체크포인트 폴더: 라벨 폴더의 바로 상위 / checkpoints (라벨 폴더가 비면 None)"""
        lbl = self._get_data_config().get("train_lbl_dir", "")
        if not lbl:
            return None
        return os.path.join(os.path.dirname(os.path.normpath(lbl)), "checkpoints")

    def _update_kfold_epoch_hint(self, *_):
        """kfold_summary.json에서 평균 best epoch를 읽어 안내 문구 갱신"""
        if not self.use_kfold_ep.isChecked():
            self.kfold_ep_hint.setText(
                "체크하면 이전 K-Fold 결과의 평균 best epoch만큼만 학습합니다")
            return
        from tools.yolo.yolo_module_wrap import read_kfold_best_epoch
        from tools.yolo.yolo_module_wrap import latest_kfold_run
        root = self._ckpt_root()
        run = latest_kfold_run(root) if root else None
        fe, me = read_kfold_best_epoch(run) if run else (None, None)
        if fe:
            self.kfold_ep_hint.setText(
                f"{os.path.basename(run)}의 kfold_summary.json — 평균 best epoch {me} → Epochs = {fe}")
        else:
            self.kfold_ep_hint.setText(
                f"[경고] checkpoints 안에 kfold_summary.json 없음 "
                f"→ 위 Epochs 설정값({self.epochs.value()}) 사용")

    def _get_data_config(self):
        """data_tab UI에서 직접 읽기 (저장 안 눌렀어도 동작)"""
        if self._data_tab is None:
            return {}
        dt = self._data_tab
        from tools.yolo.tabs.data_tab import SPLIT_MODES
        mode = SPLIT_MODES[dt.split_btns.checkedId()]
        return {
            "split_mode":    mode,
            "train_img_dir": dt.train_img_edit.text().strip(),
            "train_lbl_dir": dt.train_lbl_edit.text().strip(),
            "val_img_dir":   dt.val_img_edit.text().strip(),
            "val_lbl_dir":   dt.val_lbl_edit.text().strip(),
            "holdout_ratio": dt.holdout_spin.value(),
            "kfold_k":       dt.kfold_spin.value(),
            "kfold_stratified": dt.kfold_strat.isChecked(),
            "kfold_inner_val":  dt.kfold_inner.isChecked(),
            "split_seed":    dt.split_seed_spin.value(),
            "img_ext":       dt.img_ext.currentText(),
            "channel":       dt.get_channel(),
            "false_color":   dt.get_false_color(),   # 밴드 3개 false color (안 쓰면 None)
            "class_names":   dt.get_class_names(),
        }

    # ── 마지막 설정값(학습 시작할 때 저장, 다음에 열 때 복원)과 프로젝트 파일이 같이 쓰는 값 모으기/채우기 ──
    def _save_last(self):
        save_last("train_tab", self.get_values())

    def get_values(self):
        return {
            "version": self.version_combo.currentText(), "size": self.size_combo.currentText(),
            "epochs": self.epochs.value(), "batch": self.batch.value(), "imgsz": self.imgsz.value(),
            "lr0": self.lr0.value(), "optimizer": self.optimizer.currentText(),
            "patience": self.patience.value(), "seed_on": self.seed_check.isChecked(), "seed": self.seed.value(),
            "use_kfold_epochs": self.use_kfold_ep.isChecked(),
            "aug_on": {k: cb.isChecked() for k, cb in self.aug_checks.items()},
            "aug": {k: sp.value() for k, sp in self.aug_spins.items()},
        }

    def _load_last(self):
        d = load_last("train_tab")
        if d:
            self.set_values(d)

    def set_values(self, d):
        """get_values() 형식의 값을 칸에 채운다. 없는 키는 지금 값 그대로"""
        if d.get("version") in YOLO_VERSIONS:
            self.version_combo.setCurrentText(d["version"])   # 크기 목록도 같이 바뀐다
        if d.get("size") in [self.size_combo.itemText(i) for i in range(self.size_combo.count())]:
            self.size_combo.setCurrentText(d["size"])
        for key, box in (("epochs", self.epochs), ("batch", self.batch), ("imgsz", self.imgsz),
                         ("patience", self.patience), ("seed", self.seed)):
            if key in d:
                box.setValue(int(d[key]))
        if "lr0" in d:
            self.lr0.setValue(float(d["lr0"]))
        if self.optimizer.findText(str(d.get("optimizer"))) >= 0:   # findText: 목록에 없으면 -1
            self.optimizer.setCurrentText(d["optimizer"])
        self.seed_check.setChecked(bool(d.get("seed_on", True)))
        self.use_kfold_ep.setChecked(bool(d.get("use_kfold_epochs", False)))
        for k, on in (d.get("aug_on") or {}).items():
            if k in self.aug_checks:
                self.aug_checks[k].setChecked(bool(on))
        for k, v in (d.get("aug") or {}).items():
            if k in self.aug_spins:
                self.aug_spins[k].setValue(float(v))

    def _collect_aug(self):
        aug = {}
        for key, cb in self.aug_checks.items():
            aug[key] = self.aug_spins[key].value() if cb.isChecked() else 0.0
        return aug

    def _start_train(self):
        self.box_loss_data.clear(); self.cls_loss_data.clear()
        self.map50_data.clear(); self.map5095_data.clear()
        self._last_ep = 0
        self.log_box.clear(); self.progress.setValue(0)

        data_cfg = self._get_data_config()
        if not data_cfg.get("train_img_dir"):
            self.log_box.append("[오류] ① 데이터 탭에서 Train 이미지 폴더를 지정하세요"); return
        if not data_cfg.get("train_lbl_dir"):
            self.log_box.append("[오류] ① 데이터 탭에서 Train 라벨(txt) 폴더를 지정하세요"); return
        if data_cfg["split_mode"] == "manual" and not data_cfg.get("val_img_dir"):
            self.log_box.append("[오류] manual 방식은 Val 이미지 폴더가 필요합니다\n"
                                "→ Hold-out으로 바꾸거나 Val 폴더를 지정하세요"); return
        if not data_cfg.get("class_names"):
            self.log_box.append("[오류] 클래스 이름이 비어있습니다"); return
        if self._data_tab is not None and self._data_tab.false_color_error():
            self.log_box.append(f"[오류] {self._data_tab.false_color_error()}"); return

        # 저장 폴더: <라벨 폴더 상위>/checkpoints/<모델>_<YYYYMMDD-HHMM>/
        model_name = self._update_model_name()
        run_dir = os.path.join(self._ckpt_root(),
                               f"{os.path.splitext(model_name)[0]}_{datetime.now():%Y%m%d-%H%M}")
        self.run_dir = run_dir
        self.ckpt_label.setText(f"저장: {run_dir}")
        from tools.yolo.yolo_module_wrap import latest_kfold_run
        kfold_run = latest_kfold_run(self._ckpt_root()) if self.use_kfold_ep.isChecked() else None

        config = {
            **data_cfg,
            "model":     model_name,
            "epochs":    self.epochs.value(),
            "batch":     self.batch.value(),
            "imgsz":     self.imgsz.value(),
            "lr0":       self.lr0.value(),
            "optimizer": self.optimizer.currentText(),
            "patience":  self.patience.value(),
            "seed":      self.seed.value(),
            "seed_on":   self.seed_check.isChecked(),
            "project":   run_dir,
            "name":      "full" if data_cfg["split_mode"] == "full" else "train",
            "kfold_run_dir": kfold_run,
            "aug":       self._collect_aug(),
            "use_kfold_epochs": self.use_kfold_ep.isChecked(),
        }
        self.state["last_project"] = config["project"]
        self.state["last_name"]    = config["name"]
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        self.state["class_names"]  = config["class_names"]
        self._config = config   # 학습 끝나면 best/last 옆에 모델이름_config.json으로 저장
        self._save_last()
        self.total_epochs = config["epochs"]

        ch_info = f"  채널={config['channel']}" if config.get("channel") is not None else ""
        if config.get("false_color"):
            fc = config["false_color"]
            ch_info += f" (false color, 하한 {fc['lo']}% / 상한 {fc['hi']}%)"
        split_info = (f"  분할시드={config['split_seed']}"
                      if config["split_mode"] in ("holdout", "kfold") else "")
        self.log_box.append(f"[설정] 모델={config['model']}  "
                            f"분할={config['split_mode']}  "
                            f"seed={config['seed'] if config['seed_on'] else '고정 안 함'}"
                            f"{split_info}{ch_info}  클래스={config['class_names']}")
        if config["split_mode"] == "full":
            self.log_box.append("[안내] Full Data 모드 — 전체 데이터를 train=val로 사용합니다. "
                                "표시되는 mAP는 학습 데이터 기준이므로 성능 지표로 쓰지 마세요.")
        self.worker = TrainWorker(config)
        self.worker.epoch_done.connect(self._on_epoch)
        self.worker.fold_start.connect(self._on_fold_start)
        self.worker.log_msg.connect(self.log_box.append)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)

    def _stop_train(self):
        if self.worker:
            self.worker.stop()
            self.epoch_label.setText("중단 요청됨 — 현재 epoch 종료 후 best를 저장하고 평가로 이동")

    def _on_fold_start(self, fold_idx, total):
        """fold 시작 시 그래프 초기화"""
        self.box_loss_data.clear(); self.cls_loss_data.clear()
        self.map50_data.clear(); self.map5095_data.clear()
        self._last_ep = 0
        self.progress.setValue(0)
        if total == 0:
            self.epoch_label.setText("Full Data 학습 중...")
        else:
            self.epoch_label.setText(f"Fold {fold_idx} / {total} 학습 중...")
        if self.has_mpl:
            self.ax1.cla(); self.ax2.cla(); self.canvas.draw()

    def _on_epoch(self, ep, total, box_loss, cls_loss, map50, map5095):
        bl = box_loss if box_loss is not None else float('nan')
        cl = cls_loss if cls_loss is not None else float('nan')
        m50 = map50 if map50 is not None else float('nan')
        m5095 = map5095 if map5095 is not None else float('nan')
        if ep == self._last_ep and self.box_loss_data:
            # 같은 epoch 재호출(최종 검증 콜백) → 마지막 점 덮어쓰기
            self.box_loss_data[-1] = bl; self.cls_loss_data[-1] = cl
            self.map50_data[-1] = m50;   self.map5095_data[-1] = m5095
        else:
            self.box_loss_data.append(bl); self.cls_loss_data.append(cl)
            self.map50_data.append(m50);   self.map5095_data.append(m5095)
            self._last_ep = ep
        self.progress.setValue(int(ep / max(total, 1) * 100))
        parts = [f"Epoch {ep}/{total}"]
        if box_loss is not None: parts.append(f"box={box_loss:.4f}")
        if cls_loss is not None: parts.append(f"cls={cls_loss:.4f}")
        if map50 is not None: parts.append(f"mAP50={map50:.4f}")
        if map5095 is not None: parts.append(f"mAP50-95={map5095:.4f}")
        self.epoch_label.setText("  |  ".join(parts))
        if self.has_mpl:
            self._update_plot()

    def _update_plot(self):
        self.ax1.cla(); self.ax2.cla()
        xs = list(range(1, len(self.box_loss_data) + 1))
        # 색은 common 테마에서만: 첫 번째 선 하늘색, 두 번째 선 주황
        self.ax1.plot(xs, self.box_loss_data, label="box_loss", color=theme.ACCENT, lw=1.5)
        self.ax1.plot(xs, self.cls_loss_data, label="cls_loss", color=theme.WARN, lw=1.5)
        self.ax1.set_title("Loss", fontsize=10); self.ax1.legend(fontsize=8)
        self.ax1.grid(alpha=0.3); self.ax1.set_xlabel("Epoch")
        self.ax2.plot(xs, self.map50_data, label="mAP50", color=theme.ACCENT, lw=1.5)
        self.ax2.plot(xs, self.map5095_data, label="mAP50-95", color=theme.WARN, lw=1.5)
        self.ax2.set_title("mAP", fontsize=10); self.ax2.legend(fontsize=8)
        self.ax2.set_ylim(0, 1); self.ax2.grid(alpha=0.3); self.ax2.set_xlabel("Epoch")
        self.fig.tight_layout(); self.canvas.draw()

    def _on_finished(self, result):
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        if result.get("success"):
            if not result.get("stopped"):
                self.progress.setValue(100)
            best = result.get("best_pt", "")
            save_dir = result.get("save_dir", "")
            # K-Fold 요약 출력
            if "mean_map50" in result:
                self.log_box.append(
                    f"K-Fold 평균 mAP50 = {result['mean_map50']:.4f} "
                    f"± {result['std_map50']:.4f}  |  "
                    f"mAP50-95 = {result['mean_map5095']:.4f} "
                    f"± {result['std_map5095']:.4f}")
            self.log_box.append("학습 중단 — 현재까지의 모델을 저장했습니다"
                                if result.get("stopped") else "학습 완료")
            self.step_done.emit()
            if best:
                self.state["best_pt"] = best
                self.state["save_dir"] = save_dir
                self.log_box.append(f"Best 모델 → 평가/추론 탭에 자동 반영: {best}")
                self._save_model_configs(best, result.get("data_yaml"))
                # 평가/추론 탭에 모델 경로 공유
                self._propagate_model(best, save_dir, result.get("data_yaml"))
                win = self.window()
                if hasattr(win, "go_step"):
                    win.go_step(2)   # 평가 탭으로 이동
        else:
            self.log_box.append(f"학습 실패: {result.get('error', '알 수 없음')}")

    def _save_model_configs(self, best_pt, data_yaml=None):
        """best.pt, last.pt 옆에 모델이름_config.json 저장 (work_state.config_path_for 규칙)"""
        cfg = getattr(self, "_config", None)
        if not cfg:
            return
        cfg = {**cfg, "data_yaml": data_yaml or ""}
        for p in (best_pt, os.path.join(os.path.dirname(best_pt), "last.pt")):
            if os.path.isfile(p):
                save_config(p, cfg)

    def _propagate_model(self, best_pt, save_dir, data_yaml=None):
        """학습 완료 후 best.pt를 평가/추론 탭에 전달"""
        win = self.window()
        ws = self.state.get("work")
        if ws:
            ws.set_model(best_pt)   # 평가/추론 탭이 changed 신호로 따라 바뀐다
        # 평가탭에 data.yaml 경로 전달 — 실제 학습에 사용한 yaml 우선
        # (채널 추출 학습 시 project/data.yaml이 과거 실행의 다른 채널 수 yaml일 수
        #  있어, 추측 대신 학습 결과에서 받은 경로를 사용해야 채널이 일치한다)
        tab_eval = getattr(win, "tab_eval", None)
        if tab_eval and hasattr(tab_eval, "set_data_yaml"):
            if data_yaml and os.path.isfile(data_yaml):
                tab_eval.set_data_yaml(data_yaml)
                self.log_box.append(f"평가탭 data.yaml 자동 반영: {data_yaml}")
                return
            proj = self.state.get("last_project", "")
            for cand in ("data_full.yaml", "data.yaml"):
                yaml_path = os.path.join(proj, cand)
                if os.path.isfile(yaml_path):
                    tab_eval.set_data_yaml(yaml_path)
                    break
