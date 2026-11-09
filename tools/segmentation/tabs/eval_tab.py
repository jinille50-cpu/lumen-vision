import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QGridLayout, QLineEdit,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QCheckBox, QTextEdit, QFrame, QSpinBox,
    QButtonGroup, QRadioButton, QSlider
)
from PySide6.QtCore import Qt, QThread, Signal
from tools.segmentation.utils.file_dialog import pick_dir, pick_open

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
#  EvalWorker — CM + 확률 정보 함께 반환
# ═══════════════════════════════════════════════════════════════
class EvalWorker(QThread):
    result_ready = Signal(dict)
    cm_ready     = Signal(object, object)  # cm, prob_data dict
    log_msg      = Signal(str)
    finished     = Signal()

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            self._run_eval()
        except Exception as e:
            import traceback
            self.log_msg.emit(f"[오류] {e}\n{traceback.format_exc()}")
        finally:
            self.finished.emit()

    def _load_model(self, model_name, encoder, in_ch, n_cls, model_path, device):
        """모델 빌드 + 가중치 로드 — 에러 세분화"""
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
            # weights_only=False — PyTorch 2.x 경고 방지
            state = torch.load(model_path, map_location=device, weights_only=False)
            # state_dict 키가 'model' 등으로 감싸진 경우 자동 처리
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            elif isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            # strict=False → 키 불일치 시에도 가능한 것만 로드
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                self.log_msg.emit(f"  [경고] 없는 키 {len(missing)}개 (정상 범위면 무시)")
            if unexpected:
                self.log_msg.emit(f"  [경고] 예상 외 키 {len(unexpected)}개")
        except Exception as e:
            raise RuntimeError(f"가중치 로드 실패: {e}\n"
                               f"→ 모델 구조({model_name})와 .pth 파일이 일치하는지 확인하세요")

        model = model.to(device); model.eval()
        self.log_msg.emit(f"  모델 로드 완료")
        return model, is_tr

    def _run_eval(self):
        import torch, numpy as np, glob
        from PIL import Image
        import scipy.io as sio
        import torch.nn.functional as F
        from tools.segmentation.utils.pad_utils import get_pad_multiple, pad_tensor, unpad_tensor, read_input_scale
        from tools.segmentation.utils.pair_utils import pair_dirs
        from tools.segmentation.utils.label_utils import load_label

        cfg = self.config
        model_path = cfg["model_path"]
        in_scale   = read_input_scale(model_path)   # 학습 때 입력 정규화 (구버전 모델은 1.0)
        model_name = cfg["model_name"]
        img_dir = cfg["img_dir"]
        lbl_dir = cfg["lbl_dir"]
        img_ext = cfg.get("img_ext","png")
        lbl_ext = cfg.get("lbl_ext","png")
        in_ch   = cfg.get("in_channels",3)
        n_cls   = cfg.get("num_classes",2)
        encoder = cfg.get("encoder","resnet34")
        # _config.json에서 pad_multiple 읽기, 없으면 모델명으로 자동 결정
        multiple = cfg.get("pad_multiple", get_pad_multiple(model_name))
        # 안전장치: SMP CNN 계열은 무조건 32 이상 (16 이하면 에러 발생)
        SMP_MODELS = ["UNet","UNet++","MAnet","DeepLabV3","DeepLabV3+",
                      "FPN","PSPNet","PAN","Linknet"]
        if model_name in SMP_MODELS:
            if not multiple or multiple < 32:
                multiple = 32
        elif not multiple or multiple < 4:
            multiple = 32  # fallback
        # DINO 는 패치 크기의 배수만 받으므로 항상 모델 값을 쓴다 (32 로 패딩하면 DINOv2 는 오류)
        if model_name.startswith("DINO"):
            multiple = get_pad_multiple(model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_msg.emit(f"[평가 시작]  디바이스: {device}")
        self.log_msg.emit(f"  패딩 배수: {multiple}")

        # 모델 로드 (세분화된 에러 처리)
        model, is_tr = self._load_model(model_name, encoder, in_ch, n_cls, model_path, device)

        def load_img(p):
            ext = p.rsplit(".",1)[-1].lower()
            if ext=="mat":   return sio.loadmat(p)['image'].astype(np.float32)
            elif ext=="npy": return np.load(p).astype(np.float32)
            else:            return np.array(Image.open(p)).astype(np.float32)

        from tools.segmentation.utils.pad_utils import select_channels
        _load_raw = load_img
        def load_img(p):
            return select_channels(_load_raw(p), cfg.get("channels"))

        if not glob.glob(os.path.join(img_dir, f"*.{img_ext}")):
            self.log_msg.emit(f"[오류] 이미지 없음: {img_dir}"); return
        # 파일명 기준으로 이미지 ↔ 라벨 짝짓기 (라벨 없는 이미지는 제외)
        img_files, lbl_files, pinfo = pair_dirs(img_dir, lbl_dir, img_ext, lbl_ext)
        for m in pinfo["messages"]:
            self.log_msg.emit(m)
        if not img_files:
            self.log_msg.emit(f"[오류] 이미지 ↔ 라벨 짝이 하나도 없습니다: {lbl_dir}"); return
        self.log_msg.emit(f"  이미지 {len(img_files)}개 평가 중...")

        cm = np.zeros((n_cls, n_cls), dtype=np.int64)
        # PR/ROC용 확률 및 라벨 수집
        all_probs  = [[] for _ in range(n_cls)]
        all_labels = [[] for _ in range(n_cls)]

        with torch.no_grad():
            for ip, lp in zip(img_files, lbl_files):
                img = load_img(ip)
                lbl = load_label(lp)
                t = torch.tensor(img).float() / in_scale
                if t.ndim==3 and t.shape[-1]<t.shape[0]: t=t.permute(2,0,1)
                elif t.ndim==2: t=t.unsqueeze(0)
                t = t.unsqueeze(0).to(device)

                # 패딩 적용
                t_pad, orig_h, orig_w = pad_tensor(t, multiple=multiple)

                out    = model(t_pad)
                logits = out.logits if is_tr else out

                # 패딩 제거 후 원본 크기로
                logits = unpad_tensor(logits, orig_h, orig_w)

                if logits.shape[-2:] != torch.Size(list(lbl.shape)):
                    logits = F.interpolate(logits, size=lbl.shape,
                                           mode='bilinear', align_corners=False)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()  # (C,H,W)
                pred  = np.argmax(probs, axis=0)
                lf = lbl.flatten(); pf = pred.flatten()
                for tc in range(n_cls):
                    for pc in range(n_cls):
                        cm[tc,pc] += int(((lf==tc)&(pf==pc)).sum())
                # 클래스별 확률/라벨 수집 (PR/ROC용)
                for c in range(n_cls):
                    all_probs[c].append(probs[c].flatten())
                    all_labels[c].append((lbl.flatten()==c).astype(np.uint8))

        # 지표 계산
        total = cm.sum(); diag = np.diag(cm)
        results = {}
        results["Pixel Accuracy"] = float(diag.sum()/total) if total>0 else 0.0
        iou_list = []
        for c in range(n_cls):
            d = cm[c,:].sum()+cm[:,c].sum()-cm[c,c]
            iou_list.append(float(cm[c,c]/d) if d>0 else 0.0)
        results["mIoU"]          = float(np.mean(iou_list))
        results["Per-class IoU"] = iou_list
        f1_list, prec_list, rec_list = [], [], []
        for c in range(n_cls):
            tp=cm[c,c]; fp=cm[:,c].sum()-tp; fn=cm[c,:].sum()-tp
            d=2*tp+fp+fn; f1_list.append(float(2*tp/d) if d>0 else 0.0)
            prec_list.append(float(tp/(tp+fp)) if (tp+fp)>0 else 0.0)
            rec_list.append(float(tp/(tp+fn))  if (tp+fn)>0 else 0.0)
        results["F1 Score (Dice)"] = float(np.mean(f1_list))
        results["Precision"] = float(np.mean(prec_list))
        results["Recall"]    = float(np.mean(rec_list))
        p_o = diag.sum()/total
        p_e = sum((cm[c,:].sum()*cm[:,c].sum()) for c in range(n_cls))/(total**2)
        results["Kappa Coefficient"] = float((p_o-p_e)/(1-p_e)) if (1-p_e)>0 else 0.0

        prob_data = {
            "probs":  [np.concatenate(all_probs[c])  for c in range(n_cls)],
            "labels": [np.concatenate(all_labels[c]) for c in range(n_cls)],
            "n_cls":  n_cls,
        }

        self.log_msg.emit("  평가 완료")
        self.result_ready.emit(results)
        self.cm_ready.emit(cm, prob_data)


# ═══════════════════════════════════════════════════════════════
#  EvalTab UI
# ═══════════════════════════════════════════════════════════════
class EvalTab(QWidget):
    step_done = Signal()   # 단계 표시줄에 '끝남'을 알리는 신호
    def __init__(self, shared_state):
        super().__init__()
        self.state      = shared_state
        self.worker     = None
        self._infer_tab = None
        self._last_cm   = None
        self._last_prob = None
        self._build()
        # 현재 작업 상태(모델/config/폴더)를 다른 탭과 같이 쓴다
        self.state = shared_state
        self.work = shared_state["work"]
        self.work.changed.connect(self._on_work)
        self.model_path.textChanged.connect(lambda t: self.work.set_model(t))
        self.config_path.textChanged.connect(lambda t: self.work.set("config_path", t))
        self.eval_img.textChanged.connect(lambda t: self.work.set("img_dir", t))
        self.eval_lbl.textChanged.connect(lambda t: self.work.set("lbl_dir", t))
        self.img_ext.currentTextChanged.connect(lambda t: self.work.set("img_ext", t))
        if self.work.get("img_ext"):
            self.img_ext.setCurrentText(self.work.get("img_ext"))

    def set_infer_tab(self, tab):
        self._infer_tab = tab

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0,0,0,0)
        splitter = QSplitter(Qt.Horizontal)

        # ── 왼쪽 설정 패널 ───────────────────────────────────────
        left_w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner  = QWidget(); scroll.setWidget(inner)
        lv = QVBoxLayout(inner)
        lv.setSpacing(10); lv.setContentsMargins(14,14,14,14)

        # 모델
        mg_g = QGroupBox("모델 설정  (평가 · 추론 공통)")
        mg = QGridLayout(mg_g)

        mg.addWidget(QLabel("모델 파일 (.pth):"), 0, 0)
        self.model_path = QLineEdit(); self.model_path.setPlaceholderText("모델 가중치 경로")
        self.model_path.textChanged.connect(self._sync_to_infer)
        mg.addWidget(self.model_path, 0, 1)
        btn_m = QPushButton("찾기"); btn_m.setObjectName("secondary")
        btn_m.clicked.connect(self._browse_model); mg.addWidget(btn_m, 0, 2)

        mg.addWidget(QLabel("모델 옵션 (_config.json):"), 1, 0)
        self.config_path = QLineEdit(); self.config_path.setPlaceholderText("_config.json 경로 (선택)")
        mg.addWidget(self.config_path, 1, 1)
        btn_cfg = QPushButton("찾기"); btn_cfg.setObjectName("secondary")
        btn_cfg.clicked.connect(self._browse_config); mg.addWidget(btn_cfg, 1, 2)

        mg.addWidget(QLabel("모델 구조:"), 2, 0)
        self.model_arch = QComboBox(); self.model_arch.addItems(ALL_MODELS)
        self.model_arch.currentTextChanged.connect(self._sync_to_infer)
        mg.addWidget(self.model_arch, 2, 1)

        hint_cfg = QLabel("추론탭 자동 반영  |  _config.json 로드 시 구조 자동 설정")
        hint_cfg.setObjectName("hint")
        mg.addWidget(hint_cfg, 3, 1)
        mg.setColumnStretch(1,1); lv.addWidget(mg_g)

        # 데이터
        dg_g = QGroupBox("데이터 설정  (평가 · 추론 공통)")
        dg = QGridLayout(dg_g)
        dg.addWidget(QLabel("이미지 폴더:"), 0, 0)
        self.eval_img = QLineEdit(); self.eval_img.setPlaceholderText("train / val / test 어디든")
        self.eval_img.textChanged.connect(self._sync_to_infer)
        dg.addWidget(self.eval_img, 0, 1)
        btn_i = QPushButton("찾기"); btn_i.setObjectName("secondary")
        btn_i.clicked.connect(lambda: self._browse_dir(self.eval_img)); dg.addWidget(btn_i, 0, 2)
        dg.addWidget(QLabel("라벨 폴더:"), 1, 0)
        self.eval_lbl = QLineEdit(); self.eval_lbl.setPlaceholderText("정답 마스크 폴더")
        dg.addWidget(self.eval_lbl, 1, 1)
        btn_l = QPushButton("찾기"); btn_l.setObjectName("secondary")
        btn_l.clicked.connect(lambda: self._browse_dir(self.eval_lbl)); dg.addWidget(btn_l, 1, 2)
        dg.addWidget(QLabel("이미지 확장자:"), 2, 0)
        self.img_ext = QComboBox()
        self.img_ext.addItems(["png","jpg","tiff","tif","mat","npy"])
        self.img_ext.currentTextChanged.connect(self._sync_to_infer)
        dg.addWidget(self.img_ext, 2, 1)
        dg.setColumnStretch(1,1); lv.addWidget(dg_g)

        # 저장 폴더
        sv_g = QGroupBox("Figure 저장 폴더")
        sg = QGridLayout(sv_g)
        self.save_dir = QLineEdit(); self.save_dir.setPlaceholderText("./eval_results  (빈칸이면 저장 안 함)")
        sg.addWidget(self.save_dir, 0, 0)
        btn_s = QPushButton("찾기"); btn_s.setObjectName("secondary"); btn_s.setFixedWidth(44)
        btn_s.clicked.connect(lambda: self._browse_dir(self.save_dir))
        sg.addWidget(btn_s, 0, 1); lv.addWidget(sv_g)

        line = QFrame(); line.setFrameShape(QFrame.HLine)
        lv.addWidget(line)

        # 평가 지표
        mt_g = QGroupBox("평가 지표  (체크 → 저장)")
        mv = QVBoxLayout(mt_g)
        metrics = ["mIoU","Pixel Accuracy","F1 Score (Dice)",
                   "Precision","Recall","Confusion Matrix",
                   "Per-class IoU","Kappa Coefficient"]
        self.metric_checks = {}
        for m in metrics:
            cb = QCheckBox(m)
            cb.setChecked(m in ["mIoU","F1 Score (Dice)","Confusion Matrix"])
            self.metric_checks[m] = cb; mv.addWidget(cb)
        lv.addWidget(mt_g)

        # Confusion Matrix 옵션
        cm_g = QGroupBox("Confusion Matrix 표시 형식")
        cm_v = QVBoxLayout(cm_g)
        self.cm_mode_btns = QButtonGroup()
        # 번호(id)는 그리는 코드와 맞춰 0=Count, 1=Normalized, 2=둘 다. 화면 순서는 Normalized가 맨 위(기본)
        for i, label in ((1, "Normalized (비율)"), (0, "숫자 (Count)"), (2, "둘 다")):
            rb = QRadioButton(label)
            if i == 1: rb.setChecked(True)
            self.cm_mode_btns.addButton(rb, i)
            cm_v.addWidget(rb)
        lv.addWidget(cm_g)

        # 추가 Plot
        plot_g = QGroupBox("추가 Plot  (체크 → 저장)")
        pv = QVBoxLayout(plot_g)
        self.plot_checks = {}
        for p in ["PR Curve","ROC Curve","F1-Threshold Curve","Per-class Bar Chart"]:
            cb = QCheckBox(p); self.plot_checks[p] = cb; pv.addWidget(cb)
        lv.addWidget(plot_g)

        # 폰트 크기
        font_g = QGroupBox("Figure 폰트 크기")
        fg = QHBoxLayout(font_g)
        fg.addWidget(QLabel("크기:"))
        self.font_spin = QSpinBox(); self.font_spin.setRange(6,24); self.font_spin.setValue(10)
        fg.addWidget(self.font_spin); fg.addStretch()
        lv.addWidget(font_g)

        self.run_btn = QPushButton("평가 시작")
        self.run_btn.clicked.connect(self._run_eval)
        lv.addWidget(self.run_btn)

        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(80); lv.addWidget(self.log_box)
        lv.addStretch()

        lw = QVBoxLayout(left_w); lw.setContentsMargins(0,0,0,0); lw.addWidget(scroll)
        splitter.addWidget(left_w)

        # ── 오른쪽 결과 ──────────────────────────────────────────
        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setContentsMargins(14,14,14,14); rv.setSpacing(8)

        res_l = QLabel("평가 결과"); res_l.setObjectName("section_title")
        rv.addWidget(res_l)

        self.result_table = QTableWidget(0,3)
        self.result_table.setHorizontalHeaderLabels(["지표","값","비고"])
        hh = self.result_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setMaximumHeight(200)
        rv.addWidget(self.result_table)

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

        splitter.addWidget(right_w)
        splitter.setSizes([380,620])
        outer.addWidget(splitter)

    # ── helpers ──────────────────────────────────────────────────
    def _sync_to_infer(self):
        if self._infer_tab is None: return
        self._infer_tab.set_model_arch(self.model_arch.currentText())
        self._infer_tab.set_img_ext(self.img_ext.currentText())
        self._infer_tab.set_model_params(
            self.state.get("in_channels", 0),
            self.state.get("num_classes", 0),
            self.state.get("encoder", ""),
        )
        self.state["lbl_dir"] = self.eval_lbl.text().strip()
        self.state["lbl_ext"] = self.state.get("lbl_ext", "png")

    def _on_work(self, key):
        """다른 탭이나 학습 결과가 현재 작업 상태를 바꿨을 때 이 탭의 칸을 맞춘다"""
        if key == "onnx_path": return   # ONNX 경로는 추론/경량화 탭만 쓴다
        val = self.work.get(key)
        if key == "img_ext":   # 확장자는 칸이 아니라 콤보박스
            if val and self.img_ext.findText(val) >= 0: self.img_ext.setCurrentText(val)
            return
        edit = {"model_path": self.model_path, "config_path": self.config_path,
                "img_dir": self.eval_img, "lbl_dir": self.eval_lbl}[key]
        if edit.text().strip() == val: return
        edit.setText(val)
        if key == "config_path" and val:
            self._load_config(val)
            self.log_box.append(f"  config 자동 로드: {os.path.basename(val)}")

    def _browse_model(self):
        path = pick_open(self, "모델 파일", "PyTorch (*.pth *.pt);;All (*)",
                         self.model_path.text(), self.state)
        if path:
            self.work.set_model(path)   # 같은 이름의 config(없으면 예전 config.json)도 같이 따라온다

    def load_trained(self, path, val_dirs=None):
        """학습이 끝난(또는 중단된) 모델과 config를 현재 작업 상태에 올린다.
        val_dirs=(이미지 폴더, 라벨 폴더)는 '폴더로 분리'였을 때 Validation 폴더를 평가 칸에 채운다"""
        if not path: return
        self.work.set_model(path)
        self.log_box.append(f"  학습 결과 자동 로드: {path}")
        if val_dirs and val_dirs[0] and val_dirs[1]:
            self.work.set("img_dir", val_dirs[0])
            self.work.set("lbl_dir", val_dirs[1])
            self.log_box.append(f"  Validation 폴더 자동 입력: {val_dirs[0]}")

    def _browse_config(self):
        path = pick_open(self, "모델 옵션 파일", "JSON (*.json);;All (*)",
                         self.config_path.text() or self.model_path.text(), self.state)
        if path:
            self.config_path.setText(path)
            self._load_config(path)   # 칸 글자가 같으면 _on_work는 다시 읽지 않으므로 여기서 읽는다

    def _load_config(self, path):
        """_config.json 읽어서 모델 구조/채널/클래스 state + 추론탭에 반영"""
        import json
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            # state 업데이트
            if "model_name"  in cfg: self.state["model_name"]  = cfg["model_name"]
            if "encoder"     in cfg: self.state["encoder"]      = cfg["encoder"]
            if "in_channels" in cfg: self.state["in_channels"]  = cfg["in_channels"]
            self.state["channels"] = cfg.get("channels")   # 학습 때 고른 입력 채널 (없으면 전부)
            if "num_classes" in cfg: self.state["num_classes"]  = cfg["num_classes"]
            if "pad_multiple" in cfg: self.state["pad_multiple"] = cfg["pad_multiple"]

            # 모델 구조 콤보박스 업데이트
            model_name = cfg.get("model_name","")
            idx = self.model_arch.findText(model_name)
            if idx >= 0: self.model_arch.setCurrentIndex(idx)

            # 추론탭 SpinBox 자동 반영
            if self._infer_tab:
                self._infer_tab.set_model_params(
                    cfg.get("in_channels",3),
                    cfg.get("num_classes",2),
                    cfg.get("encoder","resnet34"),
                )
                self._infer_tab.set_model_arch(model_name)

            # 로그 출력
            self.log_box.append(
                f"  모델: {cfg.get('model_name','?')}  "
                f"인코더: {cfg.get('encoder','?')}  "
                f"채널: {cfg.get('in_channels','?')}  "
                f"클래스: {cfg.get('num_classes','?')}"
            )
        except Exception as e:
            self.log_box.append(f"  [config 로드 오류] {e}")

    def _browse_dir(self, edit):
        path = pick_dir(self, edit.text(), self.state)
        if path: edit.setText(path)

    def _run_eval(self):
        mp  = self.model_path.text().strip()
        img = self.eval_img.text().strip()
        lbl = self.eval_lbl.text().strip()
        if not mp or not os.path.isfile(mp):
            self.log_box.append("[오류] 모델 파일 선택해주세요"); return
        if not img or not os.path.isdir(img):
            self.log_box.append("[오류] 이미지 폴더 선택해주세요"); return
        if not lbl or not os.path.isdir(lbl):
            self.log_box.append("[오류] 라벨 폴더 선택해주세요"); return

        self.run_btn.setEnabled(False); self.log_box.clear()
        config = {
            "model_path": mp, "model_name": self.model_arch.currentText(),
            "img_dir": img, "lbl_dir": lbl,
            "img_ext": self.img_ext.currentText(),
            "lbl_ext": self.state.get("lbl_ext","png"),
            "in_channels": self.state.get("in_channels",3),
            "channels": self.state.get("channels"),
            "num_classes": self.state.get("num_classes",2),
            "encoder": self.state.get("encoder","resnet34"),
            "pad_multiple": self.state.get("pad_multiple", None),  # None → Worker에서 모델명으로 자동 결정
        }
        self.worker = EvalWorker(config)
        self.worker.result_ready.connect(self._on_result)
        self.worker.cm_ready.connect(self._on_cm_and_plots)
        self.worker.log_msg.connect(self.log_box.append)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_result(self, results):
        self._last_results = results
        selected = [k for k,cb in self.metric_checks.items() if cb.isChecked()]
        self.result_table.setRowCount(0)
        n_cls = self.state.get("num_classes",2)
        names = self.state.get("class_names",[f"C{i}" for i in range(n_cls)])
        for m in selected:
            if m == "Confusion Matrix": continue
            if m == "Per-class IoU":
                for i,iou in enumerate(results.get("Per-class IoU",[])):
                    row = self.result_table.rowCount()
                    self.result_table.insertRow(row)
                    nm = names[i] if i<len(names) else f"C{i}"
                    self.result_table.setItem(row,0,QTableWidgetItem(f"IoU [{nm}]"))
                    self.result_table.setItem(row,1,QTableWidgetItem(f"{iou:.4f}"))
                    self.result_table.setItem(row,2,QTableWidgetItem(""))
            elif m in results:
                row = self.result_table.rowCount()
                self.result_table.insertRow(row)
                self.result_table.setItem(row,0,QTableWidgetItem(m))
                self.result_table.setItem(row,1,QTableWidgetItem(f"{results[m]:.4f}"))
                self.result_table.setItem(row,2,QTableWidgetItem(""))

    def _on_cm_and_plots(self, cm, prob_data):
        self._last_cm   = cm
        self._last_prob = prob_data
        save_dir = self.save_dir.text().strip()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fs = self.font_spin.value()
        self._save_results(cm)   # 체크포인트 폴더/Results/에 항상 저장

        # CM 그리기 (UI 표시)
        if self.metric_checks.get("Confusion Matrix", QCheckBox()).isChecked():
            self._draw_cm(cm, fs, save_dir)

        # 추가 Plot 저장
        if save_dir:
            self._save_extra_plots(prob_data, fs, save_dir)

    def _save_results(self, cm):
        """체크포인트 폴더 안 Results/에 summary.json, 혼동 행렬 그림(confusion_matrix.png) 저장.
        모델이 full/ 또는 foldN/ 안에 있으면 그 위(학습 폴더)를 체크포인트 폴더로 본다"""
        import json, re
        import numpy as np
        mp = self.model_path.text().strip()
        if not mp or not self._last_results: return
        ckpt = os.path.dirname(os.path.abspath(mp))
        if os.path.basename(ckpt) == "full" or re.fullmatch(r"fold\d+", os.path.basename(ckpt)):
            ckpt = os.path.dirname(ckpt)
        out = os.path.join(ckpt, "Results")
        try:
            os.makedirs(out, exist_ok=True)
            n = cm.shape[0]
            names = list(self.state.get("class_names", []))
            names = [names[i] if i < len(names) else f"C{i}" for i in range(n)]
            dice = []
            for c in range(n):
                tp = cm[c, c]; d = 2 * tp + (cm[:, c].sum() - tp) + (cm[c, :].sum() - tp)
                dice.append(float(2 * tp / d) if d > 0 else 0.0)
            r = self._last_results
            summary = {
                "model_path": mp, "img_dir": self.eval_img.text().strip(),
                "lbl_dir": self.eval_lbl.text().strip(), "class_names": names,
                "mIoU": r["mIoU"], "Pixel Accuracy": r["Pixel Accuracy"],
                "F1 Score (Dice)": r["F1 Score (Dice)"], "Precision": r["Precision"],
                "Recall": r["Recall"], "Kappa Coefficient": r["Kappa Coefficient"],
                "per_class_iou": dict(zip(names, r["Per-class IoU"])),
                "per_class_dice": dict(zip(names, dice)),
                "confusion_matrix": cm.tolist(),
            }
            with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            if self.has_mpl:
                cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
                fig = self._Figure(figsize=(10, 4), facecolor="white")
                for k, (data, title, fmt) in enumerate(
                        ((cm.astype(float), "Confusion Matrix (Count)", "{:.0f}"),
                         (cm_norm, "Confusion Matrix (Normalized)", "{:.2f}"))):
                    ax = fig.add_subplot(1, 2, k + 1)
                    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max())
                    fig.colorbar(im, ax=ax)
                    ax.set_xticks(range(n)); ax.set_xticklabels(names)
                    ax.set_yticks(range(n)); ax.set_yticklabels(names)
                    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                    for i in range(n):
                        for j in range(n):
                            ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                                    color="white" if data[i, j] > data.max() * 0.6 else "black")
                fig.tight_layout()
                fig.savefig(os.path.join(out, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
            self.log_box.append(f"  Results 저장: {out}")
        except Exception as e:
            self.log_box.append(f"[Results 저장 오류] {e}")

    def _draw_cm(self, cm, fs, save_dir):
        import numpy as np
        if not self.has_mpl: return

        mode = self.cm_mode_btns.checkedId()  # 0=count, 1=norm, 2=both
        n    = cm.shape[0]
        names = self.state.get("class_names",[f"C{i}" for i in range(n)])
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        def _draw_single(ax, data, title, fmt):
            im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max())
            ax.figure.colorbar(im, ax=ax)
            ax.set_xticks(range(n)); ax.set_xticklabels(names, fontsize=fs)
            ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=fs)
            ax.set_title(title, fontsize=fs+1)
            ax.set_xlabel("Predicted", fontsize=fs)
            ax.set_ylabel("True", fontsize=fs)
            for i in range(n):
                for j in range(n):
                    val = data[i,j]
                    txt = fmt.format(val)
                    ax.text(j,i,txt,ha='center',va='center',fontsize=fs,
                            color='white' if val>data.max()*0.6 else 'black')

        # Canvas 재생성
        if self.canvas is not None:
            self.canvas_container.removeWidget(self.canvas)
            self.canvas.setParent(None); self.canvas.deleteLater(); self.canvas = None

        if mode == 2:  # 둘 다
            fig = self._Figure(figsize=(10,4), facecolor='white')
            ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)
            _draw_single(ax1, cm.astype(float), "Confusion Matrix (Count)", "{:.0f}")
            _draw_single(ax2, cm_norm, "Confusion Matrix (Normalized)", "{:.2f}")
        elif mode == 1:  # normalized
            fig = self._Figure(figsize=(5,4), facecolor='white')
            ax  = fig.add_subplot(111)
            _draw_single(ax, cm_norm, "Confusion Matrix (Normalized)", "{:.2f}")
        else:  # count
            fig = self._Figure(figsize=(5,4), facecolor='white')
            ax  = fig.add_subplot(111)
            _draw_single(ax, cm.astype(float), "Confusion Matrix", "{:.0f}")

        fig.tight_layout()
        self.canvas = self._Canvas(fig)
        self.canvas_container.addWidget(self.canvas)
        self.canvas.draw()

        # 저장
        if save_dir:
            fig.savefig(os.path.join(save_dir, "confusion_matrix.png"),
                        dpi=150, bbox_inches='tight')
            self.log_box.append(f"  confusion_matrix.png 저장")

    def _save_extra_plots(self, prob_data, fs, save_dir):
        import numpy as np
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            for font in ["Malgun Gothic","AppleGothic","NanumGothic","DejaVu Sans"]:
                try:
                    matplotlib.rcParams['font.family'] = font
                    matplotlib.rcParams['axes.unicode_minus'] = False
                    break
                except Exception: continue
        except ImportError:
            self.log_box.append("  [경고] matplotlib 없음 — Plot 저장 불가"); return

        n_cls  = prob_data["n_cls"]
        probs  = prob_data["probs"]
        labels = prob_data["labels"]
        names  = self.state.get("class_names",[f"C{i}" for i in range(n_cls)])
        colors = ["#2563eb","#ef4444","#10b981","#f59e0b","#8b5cf6","#06b6d4"]

        # ── PR Curve ──────────────────────────────────────────────
        if self.plot_checks.get("PR Curve",QCheckBox()).isChecked():
            try:
                from sklearn.metrics import precision_recall_curve, average_precision_score
                fig, ax = plt.subplots(figsize=(6,5))
                for c in range(n_cls):
                    p,r,_ = precision_recall_curve(labels[c], probs[c])
                    ap    = average_precision_score(labels[c], probs[c])
                    ax.plot(r, p, color=colors[c%len(colors)], lw=2,
                            label=f"{names[c]}  AP={ap:.3f}")
                ax.set_xlabel("Recall", fontsize=fs); ax.set_ylabel("Precision", fontsize=fs)
                ax.set_title("PR Curve", fontsize=fs+2)
                ax.legend(fontsize=fs-1); ax.grid(alpha=0.3)
                ax.tick_params(labelsize=fs-1)
                fig.tight_layout()
                fig.savefig(os.path.join(save_dir,"pr_curve.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)
                self.log_box.append("  pr_curve.png 저장")
            except Exception as e:
                self.log_box.append(f"  [PR Curve 오류] {e}")

        # ── ROC Curve ─────────────────────────────────────────────
        if self.plot_checks.get("ROC Curve",QCheckBox()).isChecked():
            try:
                from sklearn.metrics import roc_curve, auc
                fig, ax = plt.subplots(figsize=(6,5))
                for c in range(n_cls):
                    fpr,tpr,_ = roc_curve(labels[c], probs[c])
                    roc_auc   = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=colors[c%len(colors)], lw=2,
                            label=f"{names[c]}  AUC={roc_auc:.3f}")
                ax.plot([0,1],[0,1],'k--',lw=1)
                ax.set_xlabel("FPR", fontsize=fs); ax.set_ylabel("TPR", fontsize=fs)
                ax.set_title("ROC Curve", fontsize=fs+2)
                ax.legend(fontsize=fs-1); ax.grid(alpha=0.3)
                ax.tick_params(labelsize=fs-1)
                fig.tight_layout()
                fig.savefig(os.path.join(save_dir,"roc_curve.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)
                self.log_box.append("  roc_curve.png 저장")
            except Exception as e:
                self.log_box.append(f"  [ROC Curve 오류] {e}")

        # ── F1-Threshold Curve ────────────────────────────────────
        if self.plot_checks.get("F1-Threshold Curve",QCheckBox()).isChecked():
            try:
                from sklearn.metrics import precision_recall_curve
                fig, ax = plt.subplots(figsize=(6,5))
                for c in range(n_cls):
                    p,r,thr = precision_recall_curve(labels[c], probs[c])
                    f1 = 2*p[:-1]*r[:-1]/(p[:-1]+r[:-1]+1e-9)
                    ax.plot(thr, f1, color=colors[c%len(colors)], lw=2, label=names[c])
                ax.set_xlabel("Threshold", fontsize=fs); ax.set_ylabel("F1 Score", fontsize=fs)
                ax.set_title("F1-Threshold Curve", fontsize=fs+2)
                ax.legend(fontsize=fs-1); ax.grid(alpha=0.3)
                ax.tick_params(labelsize=fs-1)
                fig.tight_layout()
                fig.savefig(os.path.join(save_dir,"f1_threshold.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)
                self.log_box.append("  f1_threshold.png 저장")
            except Exception as e:
                self.log_box.append(f"  [F1-Threshold 오류] {e}")

        # ── Per-class Bar Chart ───────────────────────────────────
        if self.plot_checks.get("Per-class Bar Chart",QCheckBox()).isChecked():
            try:
                cm = self._last_cm
                if cm is not None:
                    fig, ax = plt.subplots(figsize=(max(6, n_cls*1.5), 5))
                    iou_list, f1_list, prec_list, rec_list = [], [], [], []
                    for c in range(n_cls):
                        tp=cm[c,c]; fp=cm[:,c].sum()-tp; fn=cm[c,:].sum()-tp
                        d_iou=cm[c,:].sum()+cm[:,c].sum()-cm[c,c]
                        iou_list.append(float(cm[c,c]/d_iou) if d_iou>0 else 0.0)
                        d_f1=2*tp+fp+fn
                        f1_list.append(float(2*tp/d_f1) if d_f1>0 else 0.0)
                        prec_list.append(float(tp/(tp+fp)) if (tp+fp)>0 else 0.0)
                        rec_list.append(float(tp/(tp+fn)) if (tp+fn)>0 else 0.0)
                    import numpy as np
                    x = np.arange(n_cls); w = 0.2
                    ax.bar(x-1.5*w, iou_list, w, label="IoU",       color="#2563eb")
                    ax.bar(x-0.5*w, f1_list,  w, label="F1",        color="#ef4444")
                    ax.bar(x+0.5*w, prec_list, w, label="Precision", color="#10b981")
                    ax.bar(x+1.5*w, rec_list,  w, label="Recall",    color="#f59e0b")
                    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=fs)
                    ax.set_ylim(0,1.05); ax.set_ylabel("Score", fontsize=fs)
                    ax.set_title("Per-class Metrics", fontsize=fs+2)
                    ax.legend(fontsize=fs-1); ax.grid(axis='y', alpha=0.3)
                    ax.tick_params(labelsize=fs-1)
                    fig.tight_layout()
                    fig.savefig(os.path.join(save_dir,"per_class_bar.png"),
                                dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    self.log_box.append("  per_class_bar.png 저장")
            except Exception as e:
                self.log_box.append(f"  [Bar Chart 오류] {e}")

    def _on_finished(self):
        self.run_btn.setEnabled(True)
        self.log_box.append("평가 완료")
        self.step_done.emit()
        save_dir = self.save_dir.text().strip()
        if save_dir:
            self.log_box.append(f"저장 위치: {save_dir}")
