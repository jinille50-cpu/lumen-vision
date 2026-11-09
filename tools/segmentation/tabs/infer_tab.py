import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QLineEdit,
    QScrollArea, QComboBox, QCheckBox,
    QSplitter, QSlider, QColorDialog, QListWidget, QListWidgetItem,
    QProgressBar, QTextEdit, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal
from tools.segmentation.utils.file_dialog import pick_dir, pick_open
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from common import theme

from tools.segmentation.utils.config_path import find_config, read_onnx_config
from common.palette import class_colors

ALL_MODELS = [
    # SMP CNN 계열
    "UNet","UNet++","MAnet","FPN","PSPNet","PAN","Linknet",
    "DeepLabV3","DeepLabV3+",
    # Transformer 계열
    "SegFormer-B0","SegFormer-B2","SegFormer-B4",
    "MobileViT","Mask2Former",
    # DINO 백본
    "DINOv2-S", "DINOv2-B", "DINOv3-S", "DINOv3-B"
]


# ═══════════════════════════════════════════════════════════════
#  실제 추론 Worker
# ═══════════════════════════════════════════════════════════════
class InferWorker(QThread):
    progress_update = Signal(int, int)
    result_ready    = Signal(int, object, object, object)  # idx, vis, mask, gt(None=Unknown)
    log_msg         = Signal(str)
    finished        = Signal()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._stop  = False

    def stop(self): self._stop = True

    def run(self):
        try:
            self._run_infer()
        except Exception as e:
            import traceback
            self.log_msg.emit(f"[오류] {e}\n{traceback.format_exc()}")
        finally:
            self.finished.emit()

    def _load_model(self, model_name, encoder, in_ch, n_cls, model_path, device):
        import torch, torch.nn as nn
        import segmentation_models_pytorch as smp
        from transformers import (
            SegformerForSemanticSegmentation, SegformerConfig,
            MobileViTForSemanticSegmentation, MobileViTConfig,
            Mask2FormerForUniversalSegmentation, Mask2FormerConfig,
        )

        def _replace_first_conv(model, target_in_channels):
            if target_in_channels == 3: return model
            # Conv2d 탐색
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                    parts  = name.split(".")
                    parent = model
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    orig = getattr(parent, parts[-1])
                    setattr(parent, parts[-1], nn.Conv2d(
                        target_in_channels, orig.out_channels,
                        kernel_size=orig.kernel_size, stride=orig.stride,
                        padding=orig.padding, bias=(orig.bias is not None),
                    ))
                    return model
            # Linear patch embedding 탐색 (DPT/BEiT)
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and "patch" in name.lower():
                    if module.in_features % 3 == 0:
                        patch_pix = module.in_features // 3
                        new_in    = patch_pix * target_in_channels
                        parts  = name.split(".")
                        parent = model
                        for p in parts[:-1]:
                            parent = getattr(parent, p)
                        orig = getattr(parent, parts[-1])
                        setattr(parent, parts[-1], nn.Linear(
                            new_in, orig.out_features,
                            bias=(orig.bias is not None),
                        ))
                        return model
            return model

        SMP_MAP = {"UNet":"Unet","UNet++":"UnetPlusPlus","MAnet":"MAnet",
                   "DeepLabV3":"DeepLabV3","DeepLabV3+":"DeepLabV3Plus",
                   "FPN":"FPN","PSPNet":"PSPNet","PAN":"PAN","Linknet":"Linknet"}

        self.log_msg.emit(f"  모델 구조 생성 중: {model_name}")
        is_tr = False
        try:
            if model_name.startswith("DINO"):
                from tools.segmentation.models.dino_seg import build_dino_seg
                model = build_dino_seg(model_name, in_ch, n_cls)
            elif model_name in SMP_MAP:
                model = getattr(smp, SMP_MAP[model_name])(
                    encoder_name=encoder, encoder_weights=None,
                    in_channels=in_ch, classes=n_cls)
            elif "SegFormer" in model_name:
                SFCFG = {
                    "B0":{"depths":[2,2,2,2],"hidden_sizes":[32,64,160,256]},
                    "B2":{"depths":[3,4,6,3],"hidden_sizes":[64,128,320,512]},
                    "B4":{"depths":[3,8,27,3],"hidden_sizes":[64,128,320,512]},
                }
                v = model_name.split("-")[-1]
                vc = SFCFG.get(v, SFCFG["B2"])
                sc = SegformerConfig(num_channels=in_ch, num_labels=n_cls,
                                     depths=vc["depths"], hidden_sizes=vc["hidden_sizes"])
                model = SegformerForSemanticSegmentation(sc); is_tr = True
            elif model_name == "MobileViT":
                model = MobileViTForSemanticSegmentation(MobileViTConfig(num_channels=in_ch, num_labels=n_cls))
                model = _replace_first_conv(model, in_ch); is_tr = True
            elif model_name == "Mask2Former":
                model = Mask2FormerForUniversalSegmentation(Mask2FormerConfig(num_labels=n_cls))
                model = _replace_first_conv(model, in_ch); is_tr = True
            else:
                model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                                 in_channels=in_ch, classes=n_cls)
        except Exception as e:
            raise RuntimeError(f"모델 구조 생성 실패: {e}")

        self.log_msg.emit(f"  가중치 로드 중: {os.path.basename(model_path)}")
        try:
            state = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            elif isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                self.log_msg.emit(f"  [경고] 없는 키 {len(missing)}개")
            if unexpected:
                self.log_msg.emit(f"  [경고] 예상 외 키 {len(unexpected)}개")
        except Exception as e:
            raise RuntimeError(f"가중치 로드 실패: {e}\n"
                               f"→ 모델 구조({model_name})와 .pth 파일이 일치하는지 확인하세요")

        model = model.to(device); model.eval()
        self.log_msg.emit(f"  모델 로드 완료")
        return model, is_tr

    def _run_infer(self):
        import torch, numpy as np, glob
        from PIL import Image
        import scipy.io as sio
        import torch.nn.functional as F
        from tools.segmentation.utils.pad_utils import get_pad_multiple, pad_tensor, unpad_tensor, read_input_scale
        from tools.segmentation.utils.pair_utils import pair_lists
        from tools.segmentation.utils.label_utils import load_label

        cfg        = self.config
        model_path = cfg["model_path"]
        use_onnx   = model_path.lower().endswith(".onnx")
        # ONNX 는 메타데이터 -> 옆의 _config.json (_int8 이름은 FP32 와 같은 config) 순으로 찾는다
        in_scale   = read_input_scale(model_path)   # 학습 때 입력 정규화 (구버전 모델은 1.0)
        model_name = cfg["model_name"]
        img_dir    = cfg["img_dir"]
        img_ext    = cfg.get("img_ext","png")
        in_ch      = cfg.get("in_channels",3)
        n_cls      = cfg.get("num_classes",2)
        encoder    = cfg.get("encoder","resnet34")
        out_dir    = cfg.get("out_dir","./infer_results")
        os.makedirs(out_dir, exist_ok=True)
        # _config.json에서 pad_multiple 읽기, 없으면 모델명으로 자동 결정
        multiple = cfg.get("pad_multiple") or get_pad_multiple(model_name)
        # 안전장치: SMP CNN 계열은 무조건 32 이상
        SMP_MODELS = ["UNet","UNet++","MAnet","DeepLabV3","DeepLabV3+",
                      "FPN","PSPNet","PAN","Linknet"]
        if model_name in SMP_MODELS:
            if not multiple or multiple < 32:
                multiple = 32
        elif not multiple or multiple < 4:
            multiple = 32
        # DINO 는 패치 크기의 배수만 받으므로 항상 모델 값을 쓴다 (32 로 패딩하면 DINOv2 는 오류)
        if model_name.startswith("DINO"):
            multiple = get_pad_multiple(model_name)

        if use_onnx:
            # onnxruntime 은 numpy 로 입력받으므로 텐서는 CPU 에 둔다
            device = torch.device("cpu")
            self.log_msg.emit(f"[추론 시작]  형식: ONNX ({os.path.basename(model_path)})")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_msg.emit(f"[추론 시작]  디바이스: {device}")
        self.log_msg.emit(f"  패딩 배수: {multiple}  |  입력 정규화: / {in_scale:g}")

        # 모델 로드 (안전 처리)
        if use_onnx:
            from tools.segmentation.utils.quantize_onnx import OnnxModel
            # 고른 실행 방식 먼저, 안 되면 CPU (실제 쓰인 장치는 세션에서 다시 읽어 로그에 남김)
            providers = [cfg.get("onnx_provider") or "CPUExecutionProvider"]
            if "CPUExecutionProvider" not in providers:
                providers.append("CPUExecutionProvider")
            model, is_tr = OnnxModel(model_path, providers=providers, threads=0), False
            self.log_msg.emit(f"  실행 장치: {model.provider}")
        else:
            model, is_tr = self._load_model(model_name, encoder, in_ch, n_cls, model_path, device)

        img_files = cfg.get("img_files",[])
        if not img_files:
            img_files = sorted(glob.glob(os.path.join(img_dir, f"*.{img_ext}")))
        total = len(img_files)
        self.log_msg.emit(f"  추론 대상: {total}장")

        colors  = cfg.get("class_colors",{})
        alpha   = cfg.get("alpha",0.5)
        n_cls_c = cfg.get("num_classes",2)
        defs    = class_colors(max(n_cls_c, 1))

        def hex2rgb(h):
            h = h.lstrip("#")
            return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

        lbl_dir  = cfg.get("lbl_dir", "")
        lbl_ext  = cfg.get("lbl_ext", "png")
        has_gt   = bool(lbl_dir and os.path.isdir(lbl_dir))
        # 파일명 기준으로 GT 라벨 짝짓기 (짝 없는 이미지는 GT 없이 추론)
        gt_files = []
        if has_gt:
            lbl_all = sorted([
                os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir)
                if f.lower().endswith(f".{lbl_ext}")
            ])
            self.log_msg.emit(f"  GT 라벨 폴더: {lbl_dir}  ({len(lbl_all)}개)")
            pairs, pinfo = pair_lists(img_files, lbl_all)
            for m in pinfo["messages"]:
                self.log_msg.emit(m)
            img_files = [p[0] for p in pairs]   # index fallback 시에도 순서 일치
            gt_files  = [p[1] for p in pairs]
        else:
            self.log_msg.emit(f"  GT 없음 (Unknown 모드)")

        with torch.no_grad():
            for idx, ip in enumerate(img_files):
                if self._stop:
                    self.log_msg.emit("[중단]"); break

                ext = ip.rsplit(".",1)[-1].lower()
                if ext=="mat":   raw = sio.loadmat(ip)['image'].astype(np.float32)
                elif ext=="npy": raw = np.load(ip).astype(np.float32)
                else:            raw = np.array(Image.open(ip)).astype(np.float32)
                from tools.segmentation.utils.pad_utils import select_channels
                raw = select_channels(raw, cfg.get("channels"))

                t = torch.tensor(raw).float() / in_scale
                if t.ndim==3 and t.shape[-1]<t.shape[0]: t = t.permute(2,0,1)
                elif t.ndim==2: t = t.unsqueeze(0)
                t = t.unsqueeze(0).to(device)

                # 패딩 적용
                t_pad, orig_h, orig_w = pad_tensor(t, multiple=multiple)

                out    = model(t_pad)
                logits = out.logits if is_tr else out

                # 패딩 제거 → 원본 크기
                logits = unpad_tensor(logits, orig_h, orig_w)

                if logits.shape[-2:] != (orig_h, orig_w):
                    logits = F.interpolate(logits, size=(orig_h, orig_w),
                                           mode='bilinear', align_corners=False)
                mask = torch.argmax(logits,1).squeeze().cpu().numpy().astype(np.uint8)

                # ── 시각화용 원본 (초분광 대응) ───────────────────
                if raw.ndim == 3:
                    ch = raw.shape[-1] if raw.shape[-1] < raw.shape[0] else raw.shape[0]
                    if raw.shape[-1] < raw.shape[0]:
                        # (H,W,C) 형태
                        arr = raw
                    else:
                        # (C,H,W) 형태 → (H,W,C)
                        arr = raw.transpose(1,2,0)
                        ch  = arr.shape[-1]

                    if ch >= 3:
                        # 초분광: 균등 간격 3밴드 선택 (R,G,B 대표)
                        b_r = ch - 1
                        b_g = ch // 2
                        b_b = 0
                        vis_raw = np.stack([arr[:,:,b_r], arr[:,:,b_g], arr[:,:,b_b]], axis=-1)
                    else:
                        vis_raw = np.stack([arr[:,:,0]]*3, axis=-1)
                else:
                    vis_raw = np.stack([raw]*3, axis=-1)

                # 0~255 정규화 (min-max)
                vmin, vmax = vis_raw.min(), vis_raw.max()
                if vmax > vmin:
                    vis = ((vis_raw - vmin) / (vmax - vmin) * 255).astype(np.uint8)
                else:
                    vis = np.zeros_like(vis_raw, dtype=np.uint8)

                # ── GT 라벨 로드 (파일명 매칭 결과) ─────────────────
                gt = None
                if has_gt and idx < len(gt_files):
                    lp = gt_files[idx]
                    if lp and os.path.isfile(lp):
                        gt = load_label(lp)

                fname = os.path.splitext(os.path.basename(ip))[0]

                # ① Segmentation Result — 클래스 색상 RGB (alpha 없음)
                if cfg.get("save_seg_color", True):
                    seg_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
                    for c in range(n_cls_c):
                        hx = colors.get(c, defs[c])
                        r,g,b = hex2rgb(hx)
                        seg_rgb[mask==c, 0] = r
                        seg_rgb[mask==c, 1] = g
                        seg_rgb[mask==c, 2] = b
                    Image.fromarray(seg_rgb).save(
                        os.path.join(out_dir, f"{fname}_seg_color.png"))

                # ② Overlay — 원본 + 클래스 색상 (alpha 적용)
                if cfg.get("save_overlay", True):
                    ovl = vis.copy().astype(float)
                    for c in range(n_cls_c):
                        hx = colors.get(c, defs[c])
                        r,g,b = hex2rgb(hx); m=(mask==c)
                        ovl[m,0]=ovl[m,0]*(1-alpha)+r*alpha
                        ovl[m,1]=ovl[m,1]*(1-alpha)+g*alpha
                        ovl[m,2]=ovl[m,2]*(1-alpha)+b*alpha
                    Image.fromarray(ovl.clip(0,255).astype(np.uint8)).save(
                        os.path.join(out_dir, f"{fname}_overlay.png"))

                # ③ GT — 라벨 있을 때만 (Unknown 모드 아닐 때)
                if gt is not None:
                    gt_rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
                    for c in range(n_cls_c):
                        hx = colors.get(c, defs[c])
                        r,g,b = hex2rgb(hx)
                        gt_rgb[gt==c, 0] = r
                        gt_rgb[gt==c, 1] = g
                        gt_rgb[gt==c, 2] = b
                    Image.fromarray(gt_rgb).save(
                        os.path.join(out_dir, f"{fname}_gt.png"))

                self.progress_update.emit(idx+1, total)
                self.result_ready.emit(idx, vis, mask, gt)  # gt=None이면 Unknown

        self.log_msg.emit(f"추론 완료  →  {out_dir}")


# ═══════════════════════════════════════════════════════════════
#  InferTab UI
# ═══════════════════════════════════════════════════════════════
class InferTab(QWidget):
    step_done = Signal()   # 단계 표시줄에 '끝남'을 알리는 신호
    def __init__(self, shared_state):
        super().__init__()
        self.state        = shared_state
        self.class_colors = {}
        self.worker       = None
        self.preview_imgs = []   # [(vis_np, mask_np, gt_np or None), ...]
        self.preview_idx  = 0
        self._zoom        = 1.0
        self._build()
        self._connect_work()
        # 위젯 렌더링 완료 후 초기 visible 상태 적용
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: self._on_unknown_toggled(self.unknown_mode.isChecked()))

    # ── 평가탭에서 호출하는 공유 setter ──────────────────────────
    def _connect_work(self):
        """현재 작업 상태(모델/config/폴더)를 평가·경량화 탭과 같이 쓴다"""
        self.work = self.state["work"]
        self.work.changed.connect(self._on_work)
        # 추론 탭이 ONNX를 직접 고른 경우는 작업 모델(.pth)로 올리지 않는다
        self.model_path.textChanged.connect(
            lambda t: self.work.set_model(t) if t.lower().endswith((".pth", ".pt")) else None)
        self.onnx_edit.textChanged.connect(lambda t: self.work.set("onnx_path", t))
        self.config_path_edit.textChanged.connect(lambda t: self.work.set("config_path", t))
        self.input_dir.textChanged.connect(lambda t: self.work.set("img_dir", t))
        self.lbl_dir_edit.textChanged.connect(lambda t: self.work.set("lbl_dir", t))
        self.ext_combo.currentTextChanged.connect(lambda t: self.work.set("img_ext", t))
        self._on_work("model_path")   # 탭이 만들어지기 전에 정해진 값도 반영
        for k in ("onnx_path", "config_path", "img_dir", "lbl_dir", "img_ext"):
            self._on_work(k)

    def _on_work(self, key):
        val = self.work.get(key)
        if key == "img_ext":   # 확장자는 칸이 아니라 콤보박스
            if val and self.ext_combo.findText(val) >= 0: self.ext_combo.setCurrentText(val)
            return
        edit = {"model_path": self.model_path, "onnx_path": self.onnx_edit, "config_path": self.config_path_edit,
                "img_dir": self.input_dir, "lbl_dir": self.lbl_dir_edit}[key]
        if not val or edit.text().strip() == val: return
        edit.setText(val)
        if key == "model_path":
            self._model_path_synced = val
        elif key == "config_path":
            self._load_config(val)
            self.log_box.append(f"config 자동 로드: {os.path.basename(val)}")

    def set_model_path(self, path):
        self.model_path.setText(path)
        self._model_path_synced = path

    def set_model_arch(self, arch):
        idx = self.model_arch.findText(arch)
        if idx >= 0: self.model_arch.setCurrentIndex(idx)

    def set_img_dir(self, path):
        self.input_dir.setText(path)

    def set_img_ext(self, ext):
        idx = self.ext_combo.findText(ext)
        if idx >= 0: self.ext_combo.setCurrentIndex(idx)

    def set_lbl_dir(self, path):
        self.lbl_dir_edit.setText(path)

    def set_model_params(self, in_channels, num_classes, encoder):
        """모델탭/평가탭에서 채널·클래스·인코더 자동 반영"""
        if in_channels and in_channels > 0:
            self.in_ch_spin.setValue(in_channels)
        if num_classes and num_classes > 0:
            self.n_cls_spin.setValue(num_classes)
        if encoder:
            self.encoder_edit.setText(encoder)

    def set_config_path(self, path):
        """평가탭에서 _config.json 로드 시 추론탭에도 같은 경로 표시"""
        if hasattr(self, 'config_path_edit') and path:
            self.config_path_edit.setText(path)

    def _on_unknown_toggled(self, checked):
        """Unknown 모드 토글 시 라벨 폴더 위젯 숨김/표시"""
        if not hasattr(self, 'lbl_dir_label'): return
        for w in [self.lbl_dir_label, self.lbl_dir_edit,
                  self.btn_lbl, self.lbl_ext_label, self.lbl_ext_combo]:
            w.setVisible(not checked)

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0,0,0,0)
        splitter = QSplitter(Qt.Horizontal)

        # ── 왼쪽 ─────────────────────────────────────────────────
        left_w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner  = QWidget(); scroll.setWidget(inner)
        lv = QVBoxLayout(inner)
        lv.setSpacing(12); lv.setContentsMargins(16,16,16,16)

        # 모델 (평가탭 연동 + 직접 입력 가능)
        mg = QGroupBox("모델  (평가 탭에서 자동 반영 또는 직접 입력)")
        mg_g = QGridLayout(mg)
        mg_g.addWidget(QLabel("모델 파일 (.pth):"), 0, 0)
        self.model_path = QLineEdit(); self.model_path.setPlaceholderText("평가탭에서 자동 반영")
        mg_g.addWidget(self.model_path, 0, 1)
        btn_m = QPushButton("찾기"); btn_m.setObjectName("secondary")
        btn_m.clicked.connect(self._browse_model)
        mg_g.addWidget(btn_m, 0, 2)

        mg_g.addWidget(QLabel("모델 옵션 (_config.json):"), 1, 0)
        self.config_path_edit = QLineEdit(); self.config_path_edit.setPlaceholderText("평가탭에서 자동 반영")
        mg_g.addWidget(self.config_path_edit, 1, 1)
        btn_cfg = QPushButton("찾기"); btn_cfg.setObjectName("secondary")
        btn_cfg.clicked.connect(self._browse_config)
        mg_g.addWidget(btn_cfg, 1, 2)

        mg_g.addWidget(QLabel("모델 구조:"), 2, 0)
        self.model_arch = QComboBox(); self.model_arch.addItems(ALL_MODELS)
        mg_g.addWidget(self.model_arch, 2, 1)

        mg_g.addWidget(QLabel("인코더:"), 3, 0)
        self.encoder_edit = QLineEdit(); self.encoder_edit.setPlaceholderText("예: resnet34, mit_b2")
        mg_g.addWidget(self.encoder_edit, 3, 1)

        mg_g.addWidget(QLabel("입력 채널 수:"), 4, 0)
        from PySide6.QtWidgets import QSpinBox
        self.in_ch_spin = QSpinBox(); self.in_ch_spin.setRange(1, 500); self.in_ch_spin.setValue(3)
        hint_ch = QLabel("※ 학습 채널 수와 일치 필요")
        hint_ch.setObjectName("hint")
        ch_h = QHBoxLayout()
        ch_h.addWidget(self.in_ch_spin); ch_h.addWidget(hint_ch); ch_h.addStretch()
        mg_g.addLayout(ch_h, 4, 1)

        mg_g.addWidget(QLabel("클래스 수:"), 5, 0)
        self.n_cls_spin = QSpinBox(); self.n_cls_spin.setRange(2, 256); self.n_cls_spin.setValue(2)
        mg_g.addWidget(self.n_cls_spin, 5, 1)

        mg_g.addWidget(QLabel("모델 형식:"), 6, 0)
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["PyTorch", "ONNX FP32", "ONNX INT8"])
        self.fmt_combo.setToolTip("PyTorch는 위의 모델(.pt)을, ONNX는 아래 ONNX 파일(INT8은 같은 이름_int8.onnx)을 사용합니다")
        mg_g.addWidget(self.fmt_combo, 6, 1)

        # 경량화에서 만든 .onnx (작업 상태에 따로 저장, 모델 .pt와 별개)
        mg_g.addWidget(QLabel("ONNX 파일:"), 7, 0)
        self.onnx_edit = QLineEdit(); self.onnx_edit.setPlaceholderText("경량화에서 만든 .onnx")
        mg_g.addWidget(self.onnx_edit, 7, 1)
        btn_onnx = QPushButton("찾기"); btn_onnx.setObjectName("secondary"); btn_onnx.clicked.connect(self._browse_onnx)
        mg_g.addWidget(btn_onnx, 7, 2)

        # ONNX 실행 방식: 이 PC의 onnxruntime 이 실제로 쓸 수 있는 것만 보인다
        from tools.segmentation.utils.quantize_onnx import available_providers
        avail = available_providers()
        mg_g.addWidget(QLabel("ONNX 실행 방식:"), 8, 0)
        self.provider_combo = QComboBox()
        for p in avail:
            self.provider_combo.addItem(p.replace("ExecutionProvider", ""), p)
        mg_g.addWidget(self.provider_combo, 8, 1)
        if "CUDAExecutionProvider" not in avail:
            gpu_hint = QLabel("GPU 실행은 onnxruntime-gpu 설치 시 사용 가능")
            gpu_hint.setObjectName("hint")
            mg_g.addWidget(gpu_hint, 9, 1)

        btn_exp = QPushButton("내보낸 모델 열기"); btn_exp.setObjectName("secondary")
        btn_exp.setToolTip("최종 모델 내보내기로 만든 export 폴더를 고르면 모델 형식, config, 색을 자동으로 불러옵니다")
        btn_exp.clicked.connect(self._browse_export)
        mg_g.addWidget(btn_exp, 10, 1)

        mg_g.setColumnStretch(1,1)
        lv.addWidget(mg)

        # 입력
        inp = QGroupBox("입력 폴더  (평가 탭에서 자동 반영)")
        ig = QGridLayout(inp)
        ig.addWidget(QLabel("이미지 폴더:"), 0, 0)
        self.input_dir = QLineEdit(); self.input_dir.setPlaceholderText("평가탭에서 자동 반영")
        ig.addWidget(self.input_dir, 0, 1)
        btn_i = QPushButton("찾기"); btn_i.setObjectName("secondary")
        btn_i.clicked.connect(lambda: self._browse_dir(self.input_dir))
        ig.addWidget(btn_i, 0, 2)

        ig.addWidget(QLabel("확장자:"), 1, 0)
        self.ext_combo = QComboBox()
        self.ext_combo.addItems(["png","jpg","tiff","tif","mat","npy"])
        ig.addWidget(self.ext_combo, 1, 1)

        # 라벨 폴더 (Unknown 모드 아닐 때 표시)
        self.lbl_dir_label = QLabel("라벨 폴더:")
        ig.addWidget(self.lbl_dir_label, 2, 0)
        self.lbl_dir_edit = QLineEdit()
        self.lbl_dir_edit.setPlaceholderText("평가탭 라벨 폴더 자동 반영 (직접 입력 가능)")
        ig.addWidget(self.lbl_dir_edit, 2, 1)
        self.btn_lbl = QPushButton("찾기"); self.btn_lbl.setObjectName("secondary")
        self.btn_lbl.clicked.connect(lambda: self._browse_dir(self.lbl_dir_edit))
        ig.addWidget(self.btn_lbl, 2, 2)

        self.lbl_ext_label = QLabel("라벨 확장자:")
        ig.addWidget(self.lbl_ext_label, 3, 0)
        self.lbl_ext_combo = QComboBox()
        self.lbl_ext_combo.addItems(["png","jpg","tiff","tif"])
        ig.addWidget(self.lbl_ext_combo, 3, 1)

        self.unknown_mode = QCheckBox("Unknown 모드  (라벨 없이 예측 — GT 표시 안 함)")
        self.unknown_mode.toggled.connect(self._on_unknown_toggled)
        ig.addWidget(self.unknown_mode, 4, 0, 1, 3)

        scan_btn = QPushButton("파일 스캔"); scan_btn.setObjectName("secondary")
        scan_btn.clicked.connect(self._scan_input)
        ig.addWidget(scan_btn, 5, 0, 1, 3)
        self.input_list = QListWidget(); self.input_list.setMaximumHeight(110)
        ig.addWidget(self.input_list, 6, 0, 1, 3)
        ig.setColumnStretch(1,1)
        lv.addWidget(inp)

        # 출력
        out = QGroupBox("출력 설정")
        og = QGridLayout(out)
        og.addWidget(QLabel("저장 폴더:"), 0, 0)
        self.output_dir = QLineEdit(); self.output_dir.setPlaceholderText("./infer_results")
        og.addWidget(self.output_dir, 0, 1)
        btn_o = QPushButton("찾기"); btn_o.setObjectName("secondary")
        btn_o.clicked.connect(lambda: self._browse_dir(self.output_dir))
        og.addWidget(btn_o, 0, 2)
        self.save_seg   = QCheckBox("Segmentation Result 저장  (클래스 색상, alpha 없음)"); self.save_seg.setChecked(True)
        self.save_overlay = QCheckBox("Overlay 저장  (원본 + 클래스 색상)"); self.save_overlay.setChecked(True)
        og.addWidget(self.save_seg, 1, 0, 1, 2)
        og.addWidget(self.save_overlay, 2, 0, 1, 2)
        og.addWidget(QLabel("오버레이 투명도:"), 3, 0)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(10,90); self.alpha_slider.setValue(50)
        self.alpha_lbl = QLabel("50%")
        self.alpha_slider.valueChanged.connect(lambda v: self.alpha_lbl.setText(f"{v}%"))
        ah = QHBoxLayout(); ah.addWidget(self.alpha_slider); ah.addWidget(self.alpha_lbl)
        og.addLayout(ah, 3, 1)
        og.setColumnStretch(1,1)
        lv.addWidget(out)

        # 시각화
        vis = QGroupBox("시각화 옵션")
        vv = QVBoxLayout(vis)
        self.show_axis  = QCheckBox("Axis 표시");  self.show_axis.setChecked(False)
        self.show_title = QCheckBox("Title 표시"); self.show_title.setChecked(False)
        vv.addWidget(self.show_axis); vv.addWidget(self.show_title)
        cl = QLabel("클래스별 색상  (더블클릭 변경)"); cl.setObjectName("hint")
        vv.addWidget(cl)
        self.color_list = QListWidget(); self.color_list.setMaximumHeight(90)
        self.color_list.itemDoubleClicked.connect(self._pick_color)
        vv.addWidget(self.color_list)
        rc_btn = QPushButton("색상 목록 갱신"); rc_btn.setObjectName("secondary")
        rc_btn.clicked.connect(self._refresh_class_colors)
        vv.addWidget(rc_btn)
        lv.addWidget(vis)

        self.run_btn = QPushButton("추론 시작")
        self.run_btn.clicked.connect(self._run_infer)
        lv.addWidget(self.run_btn)
        self.infer_progress = QProgressBar(); self.infer_progress.setValue(0)
        lv.addWidget(self.infer_progress)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(80)
        lv.addWidget(self.log_box)
        lv.addStretch()

        lw = QVBoxLayout(left_w); lw.setContentsMargins(0,0,0,0); lw.addWidget(scroll)
        splitter.addWidget(left_w)

        # ── 오른쪽 ───────────────────────────────────────────────
        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setContentsMargins(16,16,16,16); rv.setSpacing(6)

        # 제목 + Zoom 컨트롤
        title_h = QHBoxLayout()
        prev_l = QLabel("결과 미리보기"); prev_l.setObjectName("section_title")
        title_h.addWidget(prev_l)
        title_h.addStretch()
        zoom_lbl = QLabel("Zoom:")
        title_h.addWidget(zoom_lbl)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 300); self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(120)
        self.zoom_pct_lbl = QLabel("100%")
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        title_h.addWidget(self.zoom_slider); title_h.addWidget(self.zoom_pct_lbl)
        btn_reset_zoom = QPushButton("1:1"); btn_reset_zoom.setObjectName("secondary")
        btn_reset_zoom.setFixedWidth(40)
        btn_reset_zoom.clicked.connect(lambda: self.zoom_slider.setValue(100))
        title_h.addWidget(btn_reset_zoom)
        rv.addLayout(title_h)

        # Canvas 컨테이너
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
            from matplotlib.figure import Figure
            self._Canvas = Canvas; self._Figure = Figure
            self.canvas_container = QVBoxLayout()
            cw = QWidget(); cw.setLayout(self.canvas_container)
            rv.addWidget(cw, 1)
            self.canvas = None; self.has_mpl = True
        except ImportError:
            rv.addWidget(QLabel("matplotlib 미설치")); self.has_mpl = False

        # 이전/다음
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("이전"); self.prev_btn.setObjectName("secondary")
        self.next_btn = QPushButton("다음"); self.next_btn.setObjectName("secondary")
        self.img_idx_lbl = QLabel("0 / 0")
        self.prev_btn.clicked.connect(self._prev_preview)
        self.next_btn.clicked.connect(self._next_preview)
        nav.addWidget(self.prev_btn); nav.addWidget(self.img_idx_lbl); nav.addWidget(self.next_btn)
        rv.addLayout(nav)

        splitter.addWidget(right_w)
        splitter.setSizes([400,600])
        outer.addWidget(splitter)

        # 초기 상태 적용 (Unknown 체크 해제 = 라벨 폴더 표시)
        self._on_unknown_toggled(self.unknown_mode.isChecked())

    # ── helpers ──────────────────────────────────────────────────
    def _browse_model(self):
        path = pick_open(self, "모델 파일", "모델 (*.pt *.pth *.onnx);;All (*)",
                         self.model_path.text(), self.state)
        if not path: return
        if path.lower().endswith(".onnx"):   # .onnx 는 모델(.pt) 칸이 아니라 ONNX 칸으로
            self._set_onnx(path); return
        self.model_path.setText(path)
        auto_cfg = find_config(path)
        if auto_cfg and auto_cfg != self.config_path_edit.text().strip():
            self.config_path_edit.setText(auto_cfg)
            self._load_config(auto_cfg)
            self.log_box.append(f"config 자동 로드: {os.path.basename(auto_cfg)}")

    def _browse_onnx(self):
        path = pick_open(self, "ONNX 파일", "ONNX (*.onnx);;All (*)",
                         self.onnx_edit.text() or self.model_path.text(), self.state)
        if path:
            self._set_onnx(path)

    def _set_onnx(self, path):
        """ONNX 파일을 정하고 형식을 맞춘 뒤 config 를 찾는다:
        ONNX 메타데이터 -> 같은 이름의 _config.json(예전 config.json) -> 고르기 창"""
        if path.endswith("_int8.onnx"):
            # 작업 상태에는 FP32 이름으로 저장 (INT8은 같은 이름 + _int8)
            self.onnx_edit.setText(path[:-len("_int8.onnx")] + ".onnx")
            self.fmt_combo.setCurrentIndex(2)
        else:
            self.onnx_edit.setText(path)
            self.fmt_combo.setCurrentIndex(1)
        meta = read_onnx_config(path)
        if meta is not None:
            self.config_path_edit.setText("")
            self._apply_config(meta)
            self.log_box.append("config 자동 로드: ONNX 메타데이터")
            return
        auto_cfg = find_config(path)
        if not auto_cfg:
            self.log_box.append("[알림] config 가 없습니다. config 파일을 골라주세요")
            self._browse_config()
        elif auto_cfg != self.config_path_edit.text().strip():
            self.config_path_edit.setText(auto_cfg)
            self._load_config(auto_cfg)
            self.log_box.append(f"config 자동 로드: {os.path.basename(auto_cfg)}")

    def _browse_export(self):
        folder = pick_dir(self, self.model_path.text() or self.onnx_edit.text(), self.state)
        if folder:
            self.open_exported(folder)

    def open_exported(self, folder):
        """최종 모델 내보내기 폴더를 불러온다: 모델 형식 자동 판별, config와 색 자동 로드. 성공하면 True"""
        from tools.segmentation.utils.export_bundle import find_exported_model
        path, fmt = find_exported_model(folder)
        if not path:
            self.log_box.append(f"[오류] 내보낸 모델 파일(.pth/.pt/.onnx)이 없습니다: {folder}")
            return False
        self.log_box.append(f"내보낸 모델 열기: {os.path.basename(path)}")
        if fmt == 0:
            self.model_path.setText(path)      # work.set_model이 같은 이름 config를 찾아 불러온다
            cfg = find_config(path)
            self.config_path_edit.setText(cfg)
            if cfg: self._load_config(cfg)
            self.fmt_combo.setCurrentIndex(0)
        else:
            self._set_onnx(path)               # 형식 콤보, ONNX 메타데이터/config 처리
        return True

    def _resolve_model(self):
        """모델 형식 콤보에 맞춰 실행할 파일을 돌려준다.
        PyTorch: 모델(.pt) 칸, ONNX: ONNX 칸(INT8이면 같은 이름_int8.onnx)"""
        fmt = self.fmt_combo.currentIndex()
        if fmt == 0:
            return self.model_path.text().strip()
        onnx = self.onnx_edit.text().strip()
        if fmt == 2 and onnx:
            onnx = os.path.splitext(onnx)[0] + "_int8.onnx"
        return onnx

    def _browse_config(self):
        path = pick_open(self, "모델 옵션 파일", "JSON (*.json);;All (*)",
                         self.config_path_edit.text() or self.model_path.text(), self.state)
        if path:
            self.config_path_edit.setText(path)
            self._load_config(path)   # 칸 글자가 같으면 _on_work는 다시 읽지 않으므로 여기서 읽는다

    def _load_config(self, path):
        """_config.json 읽어서 SpinBox/콤보박스 자동 채우기"""
        import json
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self._apply_config(cfg)
        except Exception as e:
            self.log_box.append(f"  [config 로드 오류] {e}")

    def _apply_config(self, cfg):
        """config dict 로 SpinBox/콤보박스 채우기 (json 파일과 ONNX 메타데이터 공통)"""
        try:
            model_name  = cfg.get("model_name","")
            encoder     = cfg.get("encoder","")
            in_channels = cfg.get("in_channels",3)
            num_classes = cfg.get("num_classes",2)
            self.state["channels"] = cfg.get("channels")   # 학습 때 고른 입력 채널 (없으면 전부)

            idx = self.model_arch.findText(model_name)
            if idx >= 0: self.model_arch.setCurrentIndex(idx)
            if encoder:     self.encoder_edit.setText(encoder)
            if in_channels: self.in_ch_spin.setValue(in_channels)
            if num_classes: self.n_cls_spin.setValue(num_classes)
            if cfg.get("class_names"):          # 내보낸 폴더의 config는 클래스 이름도 담고 있다
                self.state["class_names"] = list(cfg["class_names"])
            saved = cfg.get("class_colors")     # 학습 때 저장한 색 목록 (없으면 공통 팔레트)
            if saved:
                self.class_colors = {i: hx for i, hx in enumerate(saved)}
                self._refresh_class_colors()

            self.log_box.append(
                f"  모델: {model_name}  인코더: {encoder}  "
                f"채널: {in_channels}  클래스: {num_classes}"
            )
        except Exception as e:
            self.log_box.append(f"  [config 로드 오류] {e}")

    def _browse_dir(self, edit):
        path = pick_dir(self, edit.text(), self.state)
        if path: edit.setText(path)

    def _scan_input(self):
        self.input_list.clear()
        path = self.input_dir.text().strip()
        ext  = self.ext_combo.currentText()
        if not path or not os.path.isdir(path): return
        files = sorted([f for f in os.listdir(path) if f.lower().endswith(f".{ext}")])
        for f in files:
            self.input_list.addItem(QListWidgetItem(f))

    def _refresh_class_colors(self):
        self.color_list.clear()
        names = self.state.get("class_names",["Class 0","Class 1"])
        defs  = class_colors(len(names))
        for i, name in enumerate(names):
            color = self.class_colors.get(i, defs[i])
            self.class_colors[i] = color
            item = QListWidgetItem(f"Class {i}: {name}")
            # 글자는 항상 테마 글자색, 색은 작은 칸(아이콘)으로 보여준다. 흰색 칸도 보이게 회색 테두리
            pm = QPixmap(14, 14); pm.fill(QColor(color))
            pt = QPainter(pm); pt.setPen(QColor(theme.MUTED)); pt.drawRect(0, 0, 13, 13); pt.end()
            item.setIcon(QIcon(pm)); item.setForeground(QColor(theme.TEXT))
            item.setData(Qt.UserRole, i)
            self.color_list.addItem(item)

    def _pick_color(self, item):
        idx   = item.data(Qt.UserRole)
        color = QColorDialog.getColor(QColor(self.class_colors.get(idx,"#ffffff")), self)
        if color.isValid():
            self.class_colors[idx] = color.name()
            self._refresh_class_colors()

    def _on_zoom_changed(self, val):
        self._zoom = val / 100.0
        self.zoom_pct_lbl.setText(f"{val}%")
        self._show_preview()

    def _run_infer(self):
        img = self.input_dir.text().strip()
        run_path = self._resolve_model()
        if not run_path or not os.path.isfile(run_path):
            self.log_box.append(f"[오류] 선택한 형식의 모델 파일이 없습니다: {run_path or '(비어 있음)'}"); return
        if not img or not os.path.isdir(img):
            self.log_box.append("[오류] 이미지 폴더를 선택해주세요"); return

        ext   = self.ext_combo.currentText()
        files = [os.path.join(img, self.input_list.item(i).text())
                 for i in range(self.input_list.count())]
        if not files:
            files = sorted([os.path.join(img,f) for f in os.listdir(img)
                            if f.lower().endswith(f".{ext}")])

        self.preview_imgs.clear(); self.preview_idx = 0
        self.run_btn.setEnabled(False); self.log_box.clear()
        self.infer_progress.setValue(0)

        config = {
            "model_path":   run_path,
            "onnx_provider": self.provider_combo.currentData(),
            "model_name":   self.model_arch.currentText(),
            "img_dir":      img, "img_files": files, "img_ext": ext,
            # UI 직접 입력 우선, 없으면 state에서
            "in_channels":  self.in_ch_spin.value(),
            "channels":     self.state.get("channels"),
            "num_classes":  self.n_cls_spin.value(),
            "encoder":      self.encoder_edit.text().strip() or self.state.get("encoder","resnet34"),
            "out_dir":      self.output_dir.text().strip() or "./infer_results",
            "save_seg_color": self.save_seg.isChecked(),
            "save_overlay": self.save_overlay.isChecked(),
            "alpha":        self.alpha_slider.value()/100.0,
            "class_colors": dict(self.class_colors),
            # GT 라벨 (Unknown 모드 체크 시 비워서 GT 없음)
            "lbl_dir":  "" if self.unknown_mode.isChecked() else self.lbl_dir_edit.text().strip(),
            "lbl_ext":  self.lbl_ext_combo.currentText(),
            "pad_multiple": self.state.get("pad_multiple", None),  # None → Worker에서 모델명으로 자동 결정
        }
        self.worker = InferWorker(config)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.result_ready.connect(self._on_result)
        self.worker.log_msg.connect(self.log_box.append)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, cur, total):
        self.infer_progress.setRange(0,total); self.infer_progress.setValue(cur)

    def _on_result(self, idx, vis, mask, gt):
        self.preview_imgs.append((vis, mask, gt))  # gt=None이면 Unknown
        if idx == 0:
            self.preview_idx = 0; self._show_preview()
        self.img_idx_lbl.setText(f"{self.preview_idx+1} / {len(self.preview_imgs)}")

    def _on_finished(self):
        self.run_btn.setEnabled(True)
        self.step_done.emit()
        total = len(self.preview_imgs)
        self.img_idx_lbl.setText(f"{min(self.preview_idx+1,total)} / {total}")

    def _show_preview(self):
        if not self.has_mpl or not self.preview_imgs: return
        import numpy as np
        import matplotlib
        for font in ["Malgun Gothic","AppleGothic","NanumGothic","DejaVu Sans"]:
            try:
                matplotlib.rcParams['font.family'] = font
                matplotlib.rcParams['axes.unicode_minus'] = False
                break
            except Exception: continue

        vis, mask, gt = self.preview_imgs[self.preview_idx]
        # UI SpinBox 우선 (state보다 직접 입력값이 정확)
        n_cls  = self.n_cls_spin.value()
        alpha  = self.alpha_slider.value() / 100.0
        defs   = class_colors(n_cls)
        zoom   = self._zoom
        show_ax    = self.show_axis.isChecked()
        show_title = self.show_title.isChecked()

        def hex2rgb(hx):
            hx = hx.lstrip("#")
            return int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)

        def make_color_mask(label_arr):
            rgb = np.zeros((*label_arr.shape, 3), dtype=np.uint8)
            for c in range(n_cls):
                hx = self.class_colors.get(c, defs[c])
                r,g,b = hex2rgb(hx)
                rgb[label_arr==c, 0]=r; rgb[label_arr==c, 1]=g; rgb[label_arr==c, 2]=b
            return rgb

        seg_rgb = make_color_mask(mask)

        # Overlay
        overlay = vis.copy().astype(float)
        for c in range(n_cls):
            hx = self.class_colors.get(c, defs[c])
            r,g,b = hex2rgb(hx); m=(mask==c)
            overlay[m,0]=overlay[m,0]*(1-alpha)+r*alpha
            overlay[m,1]=overlay[m,1]*(1-alpha)+g*alpha
            overlay[m,2]=overlay[m,2]*(1-alpha)+b*alpha
        overlay = overlay.clip(0,255).astype(np.uint8)

        # Figure 재생성
        if self.canvas is not None:
            self.canvas_container.removeWidget(self.canvas)
            self.canvas.setParent(None); self.canvas.deleteLater(); self.canvas = None

        has_gt = gt is not None

        if has_gt:
            # ── 2x2 레이아웃 ─────────────────────────────────────
            # 위: 원본 | Overlay
            # 아래: GT  | Seg Result
            gt_rgb = make_color_mask(gt)
            fig = self._Figure(figsize=(5.0*zoom, 5.0*zoom), facecolor='white')
            ax_orig    = fig.add_subplot(2, 2, 1)
            ax_overlay = fig.add_subplot(2, 2, 2)
            ax_gt      = fig.add_subplot(2, 2, 3)
            ax_seg     = fig.add_subplot(2, 2, 4)

            panels = [
                (ax_orig,    vis,     "Original"),
                (ax_overlay, overlay, "Overlay"),
                (ax_gt,      gt_rgb,  "GT"),
                (ax_seg,     seg_rgb, "Seg Result"),
            ]
        else:
            # ── 1x3 레이아웃 (Unknown 모드) ───────────────────────
            # 원본 | Seg Result | Overlay
            fig = self._Figure(figsize=(7.5*zoom, 3.0*zoom), facecolor='white')
            ax_orig = fig.add_subplot(1, 3, 1)
            ax_seg  = fig.add_subplot(1, 3, 2)
            ax_ovl  = fig.add_subplot(1, 3, 3)

            panels = [
                (ax_orig, vis,     "Original"),
                (ax_seg,  seg_rgb, "Seg Result"),
                (ax_ovl,  overlay, "Overlay"),
            ]

        for ax, img, title in panels:
            ax.imshow(img)
            if show_title: ax.set_title(title, fontsize=9)
            if not show_ax: ax.axis("off")

        # 범례: 그림 오른쪽에 클래스 이름과 색 (흰색 칸은 테두리를 그려 보이게 한다)
        from matplotlib.patches import Patch
        names = self.state.get("class_names") or []
        handles = [Patch(facecolor=self.class_colors.get(c, defs[c]), edgecolor=theme.MUTED,
                         label=names[c] if c < len(names) else f"Class {c}")
                   for c in range(n_cls)]
        fig.legend(handles=handles, loc="center right", fontsize=8, frameon=False, labelcolor=theme.TEXT)
        fig.tight_layout(pad=0.5, rect=(0, 0, 0.82, 1))
        self.canvas = self._Canvas(fig)
        self.canvas_container.addWidget(self.canvas)
        self.canvas.draw()

    def _prev_preview(self):
        if self.preview_idx > 0:
            self.preview_idx -= 1; self._show_preview()
            self.img_idx_lbl.setText(f"{self.preview_idx+1} / {len(self.preview_imgs)}")

    def _next_preview(self):
        if self.preview_idx < len(self.preview_imgs)-1:
            self.preview_idx += 1; self._show_preview()
            self.img_idx_lbl.setText(f"{self.preview_idx+1} / {len(self.preview_imgs)}")
