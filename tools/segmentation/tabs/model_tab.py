import os
import importlib
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QScrollArea,
    QLineEdit, QSizePolicy, QSplitter, QListWidget, QListWidgetItem, QMessageBox
)
from PySide6.QtCore import Qt, Signal

from common import settings

TOOL = "segmentation"  # common 설정에 저장할 때 쓰는 이름


# 전이학습 가중치 매핑 (모델별)
PRETRAINED_WEIGHTS = {
    # ── SMP CNN 계열 ──────────────────────────────────────────
    "UNet":       ["없음 (scratch)", "resnet34-ImageNet", "resnet50-ImageNet", "efficientnet-b0"],
    "UNet++":     ["없음 (scratch)", "resnet34-ImageNet", "resnet50-ImageNet"],
    "MAnet":      ["없음 (scratch)", "resnet34-ImageNet", "resnet50-ImageNet", "resnet101-ImageNet"],
    "FPN":        ["없음 (scratch)", "resnet34-ImageNet", "resnet50-ImageNet"],
    "PSPNet":     ["없음 (scratch)", "resnet34-ImageNet", "resnet50-ImageNet"],
    "PAN":        ["없음 (scratch)", "resnet34-ImageNet", "resnet50-ImageNet"],
    "Linknet":    ["없음 (scratch)", "resnet34-ImageNet", "resnet50-ImageNet"],
    "DeepLabV3":  ["없음 (scratch)", "resnet50-ImageNet", "resnet101-ImageNet"],
    "DeepLabV3+": ["없음 (scratch)", "resnet50-ImageNet", "resnet101-ImageNet", "xception-ImageNet"],
    # ── Transformer 계열 ──────────────────────────────────────
    "SegFormer-B0": ["없음 (scratch)", "nvidia/mit-b0 (ImageNet)"],
    "SegFormer-B2": ["없음 (scratch)", "nvidia/mit-b2 (ImageNet)", "nvidia/mit-b2 (ImageNet-22K)"],
    "SegFormer-B4": ["없음 (scratch)", "nvidia/mit-b4 (ImageNet-22K)"],
    "MobileViT":    ["없음 (scratch)", "apple/deeplabv3-mobilevit-small"],
    "Mask2Former":  ["없음 (scratch)", "swin-T-ImageNet", "swin-B-ImageNet-22K"],
    # ── DINO 백본 + 분할 디코더 (소규모 데이터는 사전학습 권장) ──
    "DINOv2-S": ["DINO 사전학습 (LVD-142M)", "없음 (scratch)"],
    "DINOv2-B": ["DINO 사전학습 (LVD-142M)", "없음 (scratch)"],
    "DINOv3-S": ["DINO 사전학습 (LVD-1689M)", "없음 (scratch)"],
    "DINOv3-B": ["DINO 사전학습 (LVD-1689M)", "없음 (scratch)"],
}

# 표시명 → smp 실제 encoder_name 매핑
ENCODER_MAP = {
    "기본값 (resnet34)":    "resnet34",
    "resnet18":             "resnet18",
    "resnet34":             "resnet34",
    "resnet50":             "resnet50",
    "resnet101":            "resnet101",
    "resnet152":            "resnet152",
    "resnext50_32x4d":      "resnext50_32x4d",
    "resnext101_32x4d":     "resnext101_32x4d",
    "efficientnet-b0":      "efficientnet-b0",
    "efficientnet-b1":      "efficientnet-b1",
    "efficientnet-b2":      "efficientnet-b2",
    "efficientnet-b3":      "efficientnet-b3",
    "efficientnet-b4":      "efficientnet-b4",
    "efficientnet-b5":      "efficientnet-b5",
    "densenet121":          "densenet121",
    "densenet169":          "densenet169",
    "densenet201":          "densenet201",
    "mobilenet_v2":         "mobilenet_v2",
    "xception":             "xception",
    "mit_b0":               "mit_b0",
    "mit_b1":               "mit_b1",
    "mit_b2":               "mit_b2",
    "mit_b3":               "mit_b3",
    "mit_b4":               "mit_b4",
    "mit_b5":               "mit_b5",
    "timm-efficientnet-b0": "timm-efficientnet-b0",
    "timm-efficientnet-b4": "timm-efficientnet-b4",
    "vgg16":                "vgg16",
    "vgg19":                "vgg19",
}


class ModelTab(QWidget):
    step_done = Signal()   # 단계 표시줄에 '끝남'을 알리는 신호
    def __init__(self, shared_state):
        super().__init__()
        self.state     = shared_state
        self._data_tab = None
        self._build()

    def set_data_tab(self, data_tab):
        self._data_tab = data_tab
        # 데이터탭에서 이미지 경로 변경 시 자동으로 채널/크기 감지
        if hasattr(data_tab, 'train_img_edit'):
            data_tab.train_img_edit.textChanged.connect(self._on_data_path_changed)
        if hasattr(data_tab, 'img_ext'):
            data_tab.img_ext.currentTextChanged.connect(self._on_data_path_changed)

    def _on_data_path_changed(self, *args):
        """데이터탭 경로/확장자 변경 시 자동 감지 (디바운스 처리)"""
        # 경로가 유효한 폴더일 때만 시도
        img_dir, ext = self._get_img_dir_and_ext()
        if img_dir and os.path.isdir(img_dir):
            # 짧은 딜레이로 연속 입력 흡수 (선택사항)
            self._auto_detect_all()

    def _get_img_dir_and_ext(self):
        """shared_state 우선, 없으면 data_tab UI에서 직접 읽기"""
        img_dir = self.state.get("train_img_dir", "").strip()
        ext     = self.state.get("img_ext", "png")
        if not img_dir and self._data_tab is not None:
            img_dir = self._data_tab.train_img_edit.text().strip()
            ext     = self._data_tab.img_ext.currentText()
        return img_dir, ext

    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)

        main = QVBoxLayout(inner)
        main.setSpacing(14)
        main.setContentsMargins(20, 20, 20, 20)

        # ── 모델 선택 ─────────────────────────────────────────────
        model_group = QGroupBox("모델 선택")
        mg = QGridLayout(model_group)

        mg.addWidget(QLabel("Segmentation 모델:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(PRETRAINED_WEIGHTS.keys()))
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        mg.addWidget(self.model_combo, 0, 1)

        mg.addWidget(QLabel("전이학습 가중치:"), 1, 0)
        self.weight_combo = QComboBox()
        mg.addWidget(self.weight_combo, 1, 1)

        mg.addWidget(QLabel("인코더 백본:"), 2, 0)
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems(list(ENCODER_MAP.keys()))
        mg.addWidget(self.encoder_combo, 2, 1)

        hint = QLabel("※ Transformer 계열 모델은 인코더 선택 무관")
        hint.setObjectName("hint")
        mg.addWidget(hint, 3, 1)
        mg.setColumnStretch(1, 1)
        main.addWidget(model_group)

        # ── 입력 설정 ─────────────────────────────────────────────
        input_group = QGroupBox("입력 설정  (데이터 로드 시 자동 감지 → 필요 시 수동 변경)")
        ig = QGridLayout(input_group)

        ig.addWidget(QLabel("입력 채널 수:"), 0, 0)
        self.in_channels = QSpinBox()
        self.in_channels.setRange(1, 500)
        self.in_channels.setValue(3)
        ig.addWidget(self.in_channels, 0, 1)

        ig.addWidget(QLabel("Input Size (H × W):"), 1, 0)
        size_h = QHBoxLayout()
        self.input_h = QSpinBox(); self.input_h.setRange(32, 2048); self.input_h.setValue(512)
        self.input_w = QSpinBox(); self.input_w.setRange(32, 2048); self.input_w.setValue(512)
        size_h.addWidget(self.input_h)
        size_h.addWidget(QLabel("×"))
        size_h.addWidget(self.input_w)
        size_h.addStretch()
        ig.addLayout(size_h, 1, 1)

        # 통합 다시 감지 버튼
        redetect_btn = QPushButton("데이터에서 다시 감지")
        redetect_btn.setObjectName("secondary")
        redetect_btn.clicked.connect(self._auto_detect_all)
        ig.addWidget(redetect_btn, 2, 0, 1, 2)

        ig.setColumnStretch(1, 1)
        main.addWidget(input_group)

        # ── W&B 기록 (상단 표시줄에서 옮겨 옴) ─────────────────────
        wb_group = QGroupBox("W&&B 기록")   # && = 글자 & 표시
        wg = QGridLayout(wb_group)
        self.wandb_check = QCheckBox("학습 기록을 W&&B에 올리기")
        self.wandb_check.toggled.connect(self._set_wandb_enabled)
        wg.addWidget(self.wandb_check, 0, 0, 1, 2)
        wg.addWidget(QLabel("프로젝트:"), 1, 0)
        self.wandb_project = QLineEdit()
        self.wandb_project.editingFinished.connect(self._set_wandb_project)
        wg.addWidget(self.wandb_project, 1, 1)
        wg.setColumnStretch(1, 1)
        main.addWidget(wb_group)

        # ── ONNX 만들기: 학습이 끝나면 자동으로 만든다 ───────────────
        onnx_group = QGroupBox("ONNX 내보내기")
        og = QGridLayout(onnx_group)
        og.addWidget(QLabel("학습 후 자동 생성:"), 0, 0)
        self.onnx_combo = QComboBox()
        self.onnx_combo.addItem("ONNX만", "onnx")
        self.onnx_combo.addItem("ONNX + 경량화(INT8)", "int8")
        og.addWidget(self.onnx_combo, 0, 1)
        onnx_hint = QLabel("최종 모델을 학습 이미지 100장으로 경량화합니다")
        onnx_hint.setObjectName("hint")
        og.addWidget(onnx_hint, 1, 1)
        og.setColumnStretch(1, 1)
        main.addWidget(onnx_group)

        # ── 클래스 수 (이름, 가중치는 데이터 탭에서만 정한다) ─────
        cls_group = QGroupBox("클래스")
        cv = QHBoxLayout(cls_group)
        cv.addWidget(QLabel("클래스 수:"))
        self.num_classes = QSpinBox()
        self.num_classes.setRange(2, 256)
        self.num_classes.setValue(2)
        cv.addWidget(self.num_classes)
        hint2 = QLabel("이름과 가중치는 데이터 탭에서 정합니다")
        hint2.setObjectName("hint")
        cv.addWidget(hint2)
        cv.addStretch()

        # ── 입력 채널 선택 (3채널 RGB면 접어 둔다) ──────────────────
        self.total_channels = 0   # 데이터에서 감지한 전체 채널 수
        self.ch_group = QGroupBox("입력 채널 선택")
        chv = QVBoxLayout(self.ch_group)
        self.ch_info = QLabel("데이터에서 채널을 감지하면 여기에 표시됩니다")
        self.ch_info.setObjectName("hint")
        chv.addWidget(self.ch_info)
        self.ch_buttons = QWidget()
        chb = QHBoxLayout(self.ch_buttons)
        chb.setContentsMargins(0, 0, 0, 0)
        all_btn = QPushButton("전체 선택"); all_btn.setObjectName("secondary")
        none_btn = QPushButton("전체 해제"); none_btn.setObjectName("secondary")
        all_btn.clicked.connect(lambda: self._set_all_channels(True))
        none_btn.clicked.connect(lambda: self._set_all_channels(False))
        chb.addWidget(all_btn); chb.addWidget(none_btn); chb.addStretch()
        chv.addWidget(self.ch_buttons)
        self.ch_list = QListWidget()
        self.ch_list.itemChanged.connect(self._on_channel_toggled)
        chv.addWidget(self.ch_list, 1)
        self.ch_buttons.setVisible(False)
        self.ch_list.setVisible(False)

        # ── 저장 (오른쪽 아래, 데이터 탭과 같은 위치/모양) ────────
        save_btn = QPushButton("저장하고 다음")
        save_btn.setObjectName("Primary")
        save_btn.clicked.connect(self._save_state)
        main.addStretch()

        # 두 칸: 왼쪽 모델/입력 설정(스크롤), 오른쪽 클래스 표
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(16, 16, 16, 16)
        rv.addWidget(self.ch_group, 1)
        rv.addWidget(cls_group)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setSizes([520, 620])
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter, 1)
        bottom = QHBoxLayout()
        bottom.setContentsMargins(16, 8, 16, 12)
        bottom.addStretch()
        bottom.addWidget(save_btn)
        outer.addLayout(bottom)
        self._on_model_changed(self.model_combo.currentText())

    # ── W&B 설정 (common 설정에 저장) ────────────────────────────
    def load_wandb_settings(self):
        from tools.segmentation.utils.wandb_logger import DEFAULT_PROJECT
        # QSettings는 값을 문자열로 돌려줄 수 있어 'true' 와 비교한다
        enabled = str(settings.get_value(TOOL, "wandb_enabled", False)).lower() == "true"
        self.state["wandb_enabled"] = enabled
        self.state["wandb_project"] = str(settings.get_value(TOOL, "wandb_project", DEFAULT_PROJECT))
        self.wandb_check.setChecked(enabled)
        self.wandb_project.setText(self.state["wandb_project"])

    def _set_wandb_enabled(self, enabled):
        self.state["wandb_enabled"] = enabled
        settings.set_value(TOOL, "wandb_enabled", enabled)

    def _set_wandb_project(self):
        name = self.wandb_project.text().strip()
        if name:
            self.state["wandb_project"] = name
            settings.set_value(TOOL, "wandb_project", name)
        else:
            self.wandb_project.setText(self.state["wandb_project"])

    def _on_model_changed(self, model_name):
        self.weight_combo.clear()
        weights = PRETRAINED_WEIGHTS.get(model_name, ["없음 (scratch)"])
        self.weight_combo.addItems(weights)

    def apply_detected(self, in_channels, names):
        """데이터 탭에서 감지한 입력 채널과 클래스 수를 이 탭의 칸에 채운다"""
        if in_channels:
            self._set_total_channels(in_channels)
        if not names:
            return
        self.num_classes.setValue(max(2, len(names)))  # 클래스 수 칸의 최소값이 2

    def _auto_detect_all(self):
        """채널 + 크기 한 번에 자동 감지 (샘플 1개만 읽음)"""
        img_dir, ext = self._get_img_dir_and_ext()
        if not img_dir or not os.path.isdir(img_dir):
            print(f"[AutoDetect] 경로 없음 또는 유효하지 않음: '{img_dir}'")
            return
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(f".{ext}")]
        if not files:
            print(f"[AutoDetect] {img_dir} 에 .{ext} 파일 없음")
            return
        try:
            import numpy as np
            sample = os.path.join(img_dir, files[0])
            if ext == "mat":
                import scipy.io as sio
                d   = sio.loadmat(sample)
                arr = list(d.values())[-1]
            elif ext == "npy":
                arr = np.load(sample)
            else:
                from PIL import Image
                arr = np.array(Image.open(sample))
            # 채널
            ch = arr.shape[2] if arr.ndim == 3 else 1
            self._set_total_channels(ch)
            # 크기
            h, w = arr.shape[:2]
            self.input_h.setValue(h)
            self.input_w.setValue(w)
            print(f"[AutoDetect] {files[0]}  →  채널 {ch}, 크기 {h}x{w}")
        except Exception as e:
            print(f"[AutoDetect] 감지 실패: {e}")

    # ── 입력 채널 선택 ────────────────────────────────────────────
    def _find_wavelengths(self, n):
        """데이터 폴더(또는 상위 폴더)의 wavelength*.txt/csv/npy 에서 파장 n개를 읽는다. 없으면 None"""
        import glob, re
        import numpy as np
        img_dir, _ = self._get_img_dir_and_ext()
        if not img_dir or not os.path.isdir(img_dir):
            return None
        for folder in (img_dir, os.path.dirname(os.path.normpath(img_dir))):
            for pat in ("wavelength*", "Wavelength*", "wl*"):
                for path in sorted(glob.glob(os.path.join(folder, pat))):
                    try:
                        if path.lower().endswith(".npy"):
                            vals = np.load(path).ravel().tolist()
                        elif path.lower().endswith((".txt", ".csv")):
                            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                                vals = [float(v) for v in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", f.read())]
                        else:
                            continue
                    except Exception:
                        continue
                    if len(vals) == n:
                        return vals
        return None

    def _set_total_channels(self, n, selected=None):
        """전체 채널 수 n을 받아 체크박스 목록을 만든다. 3채널(RGB)이면 접어 두고 전부 사용.
        selected: 체크할 채널 번호 목록 (없으면 전부)"""
        if selected is None and n == self.total_channels and self.ch_list.count():
            return   # 같은 데이터를 다시 감지한 경우 사용자가 고른 선택을 유지한다
        self.total_channels = n
        self.ch_list.blockSignals(True)   # 목록을 만드는 동안 변경 신호는 막는다
        self.ch_list.clear()
        if n == 3 or n < 1:
            self.ch_list.blockSignals(False)
            self.ch_buttons.setVisible(False)
            self.ch_list.setVisible(False)
            self.in_channels.setReadOnly(False)
            self.in_channels.setValue(max(n, 1))
            self.ch_info.setText("3채널(RGB) 데이터: 채널 3개를 모두 사용합니다" if n == 3
                                 else "데이터에서 채널을 감지하면 여기에 표시됩니다")
            return
        waves = self._find_wavelengths(n)
        chosen = set(range(n)) if selected is None else set(selected)
        for i in range(n):
            text = f"{i}" if waves is None else f"{i}    {waves[i]:g} nm"
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if i in chosen else Qt.Unchecked)
            self.ch_list.addItem(item)
        self.ch_list.blockSignals(False)
        self.ch_buttons.setVisible(True)
        self.ch_list.setVisible(True)
        self.in_channels.setReadOnly(True)   # 선택한 개수가 자동 반영되므로 직접 입력은 막는다
        self._on_channel_toggled()

    def selected_channels(self):
        """체크한 채널 번호 목록. 목록을 접어 둔 경우(RGB 등)는 None(전부 사용)"""
        if self.ch_list.isHidden():   # 접어 둔 상태(RGB 등)
            return None
        return [i for i in range(self.ch_list.count())
                if self.ch_list.item(i).checkState() == Qt.Checked]

    def _set_all_channels(self, checked):
        self.ch_list.blockSignals(True)
        for i in range(self.ch_list.count()):
            self.ch_list.item(i).setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.ch_list.blockSignals(False)
        self._on_channel_toggled()

    def _on_channel_toggled(self, *args):
        n_sel = sum(self.ch_list.item(i).checkState() == Qt.Checked for i in range(self.ch_list.count()))
        self.in_channels.setValue(max(n_sel, 1))   # 입력 채널 수 칸의 최소값이 1
        self.ch_info.setText(f"선택 {n_sel} / 전체 {self.total_channels}")

    def _auto_detect_channels(self):
        """기존 호환용 — _auto_detect_all로 위임"""
        self._auto_detect_all()

    def _auto_detect_size(self):
        """기존 호환용 — _auto_detect_all로 위임"""
        self._auto_detect_all()

    def _save_state(self):
        channels = self.selected_channels()
        if channels is not None and not channels:
            QMessageBox.warning(self, "입력 채널", "입력 채널을 1개 이상 선택하세요")
            return
        # 이름, 가중치는 데이터 탭에서 정한 값 (감지 전이면 기본값)
        n = self.num_classes.value()
        names = list(self.state.get("detected_class_names") or [])
        weights = list(self.state.get("detected_class_weights") or [])
        names += [f"Class {i}" for i in range(len(names), n)]
        weights += [1.0] * (n - len(weights))
        names, weights = names[:n], weights[:n]

        # 표시명 → smp 실제 encoder 이름으로 변환
        display_enc = self.encoder_combo.currentText()
        real_enc    = ENCODER_MAP.get(display_enc, "resnet34")

        self.state["model_name"]    = self.model_combo.currentText()
        self.state["pretrained"]    = self.weight_combo.currentText()
        self.state["encoder"]       = real_enc          # ← smp 실제 키값
        self.state["in_channels"]   = self.in_channels.value()
        self.state["channels"]      = channels   # 선택한 채널 번호 (None이면 전부)
        self.state["input_h"]       = self.input_h.value()
        self.state["input_w"]       = self.input_w.value()
        self.state["num_classes"]   = self.num_classes.value()
        self.state["class_names"]   = names
        self.state["class_weights"] = weights
        self.state["onnx_mode"]     = self.onnx_combo.currentData()   # "onnx" / "int8"
        print("[ModelTab] 설정 저장:",
              f"model={self.state['model_name']} | encoder={real_enc} | "
              f"ch={self.state['in_channels']} | classes={self.state['num_classes']}")
        self.step_done.emit()
