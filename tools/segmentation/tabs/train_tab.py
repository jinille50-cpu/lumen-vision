from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QScrollArea, QTextEdit, QProgressBar, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal


# ═══════════════════════════════════════════════════════════════
#  실제 학습 Worker
# ═══════════════════════════════════════════════════════════════
class TrainWorker(QThread):
    epoch_done = Signal(int, float, float, float, float)
    log_msg    = Signal(str)
    fold_start = Signal(int, int)   # fold_idx, total_folds (0,0 = Full 학습)
    finished   = Signal()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._stop  = False
        self._wb    = None   # W&B 기록 (설정에서 켰을 때만 동작)
        self._aug_pipe = None

    def stop(self):
        self._stop = True

    def _build_model(self, model_name, in_channels, num_classes):
        import segmentation_models_pytorch as smp
        from transformers import (
            SegformerForSemanticSegmentation, SegformerConfig,
            MobileViTForSemanticSegmentation, MobileViTConfig,
            Mask2FormerForUniversalSegmentation, Mask2FormerConfig,
        )

        def _replace_first_conv(model, target_in_channels):
            """3채널 입력층(Conv2d 또는 Linear patch embedding)을 자동 탐색해서 교체"""
            import torch.nn as nn
            if target_in_channels == 3:
                return model

            # ① Conv2d 우선 탐색 (대부분의 CNN/Swin 백본)
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                    parts  = name.split(".")
                    parent = model
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    orig = getattr(parent, parts[-1])
                    setattr(parent, parts[-1], nn.Conv2d(
                        target_in_channels,
                        orig.out_channels,
                        kernel_size=orig.kernel_size,
                        stride=orig.stride,
                        padding=orig.padding,
                        bias=(orig.bias is not None),
                    ))
                    return model

            # ② Linear patch embedding 탐색 (DPT/BEiT/ViT 계열)
            # ViT 계열은 (patch_size × patch_size × 3) → hidden_dim 형태
            # 이름에 "patch_embed" 또는 "embeddings" 포함된 Linear 찾기
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and "patch" in name.lower():
                    in_features = module.in_features
                    # patch_size를 추정 (in_features = patch_size * patch_size * 3)
                    if in_features % 3 == 0:
                        patch_pix      = in_features // 3
                        new_in_features = patch_pix * target_in_channels
                        parts  = name.split(".")
                        parent = model
                        for p in parts[:-1]:
                            parent = getattr(parent, p)
                        orig = getattr(parent, parts[-1])
                        setattr(parent, parts[-1], nn.Linear(
                            new_in_features,
                            orig.out_features,
                            bias=(orig.bias is not None),
                        ))
                        return model

            # ③ 못 찾았으면 경고만 (에러 대신)
            print(f"[Warning] 3채널 입력층을 찾지 못했습니다. "
                  f"이 모델은 {target_in_channels}채널 입력에서 동작 안 할 수 있어요.")
            return model

        SMP_MAP = {
            "UNet": "Unet", "UNet++": "UnetPlusPlus", "MAnet": "MAnet",
            "DeepLabV3": "DeepLabV3", "DeepLabV3+": "DeepLabV3Plus",
            "FPN": "FPN", "PSPNet": "PSPNet", "PAN": "PAN", "Linknet": "Linknet",
        }
        SEGFORMER_CONFIGS = {
            "B0": {"depths": [2,2,2,2], "hidden_sizes": [32,64,160,256]},
            "B2": {"depths": [3,4,6,3], "hidden_sizes": [64,128,320,512]},
            "B4": {"depths": [3,8,27,3], "hidden_sizes": [64,128,320,512]},
        }
        encoder    = self.config.get("encoder", "resnet34")
        pretrained = self.config.get("pretrained", "없음 (scratch)")
        enc_w      = None if "scratch" in pretrained else "imagenet"
        FALLBACK = {
            "기본값 (resnet34)": "resnet34",
            "기본값 (모델 내장)": "resnet34",
            "": "resnet34",
        }
        encoder = FALLBACK.get(encoder, encoder)

        # ── DINO 백본 ────────────────────────────────────────────
        if model_name.startswith("DINO"):
            from tools.segmentation.models.dino_seg import build_dino_seg
            return build_dino_seg(model_name, in_channels, num_classes,
                                  pretrained="사전학습" in pretrained), False

        # ── SMP CNN 계열 ─────────────────────────────────────────
        if model_name in SMP_MAP:
            model = getattr(smp, SMP_MAP[model_name])(
                encoder_name=encoder, encoder_weights=enc_w,
                in_channels=in_channels, classes=num_classes,
            )
            return model, False

        # ── SegFormer ────────────────────────────────────────────
        elif "SegFormer" in model_name:
            v = model_name.split("-")[-1]
            vcfg = SEGFORMER_CONFIGS.get(v, SEGFORMER_CONFIGS["B2"])
            cfg = SegformerConfig(num_channels=in_channels, num_labels=num_classes,
                                  depths=vcfg["depths"], hidden_sizes=vcfg["hidden_sizes"])
            return SegformerForSemanticSegmentation(cfg), True

        # ── MobileViT (모바일 친화적 가벼운 모델) ────────────────
        elif model_name == "MobileViT":
            cfg = MobileViTConfig(num_channels=in_channels, num_labels=num_classes)
            model = MobileViTForSemanticSegmentation(cfg)
            model = _replace_first_conv(model, in_channels)
            return model, True

        # ── Mask2Former (Swin 백본) ──────────────────────────────
        elif model_name == "Mask2Former":
            cfg_m2f = Mask2FormerConfig(num_labels=num_classes)
            model   = Mask2FormerForUniversalSegmentation(cfg_m2f)
            model   = _replace_first_conv(model, in_channels)
            return model, True

        # ── fallback ─────────────────────────────────────────────
        else:
            model = smp.Unet(encoder_name="resnet34", encoder_weights=enc_w,
                             in_channels=in_channels, classes=num_classes)
            return model, False

    def _build_optimizer(self, model):
        import torch.optim as optim
        lr   = self.config.get("lr", 6e-4)
        name = self.config.get("optimizer", "AdamW")
        if name == "AdamW":
            return optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.999), weight_decay=0.01)
        elif name == "Adam":
            return optim.Adam(model.parameters(), lr=lr)
        elif name == "SGD":
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        return optim.RMSprop(model.parameters(), lr=lr)

    def _build_scheduler(self, optimizer):
        from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
        name   = self.config.get("scheduler", "ReduceLROnPlateau")
        p1     = self.config.get("sch_param1", 10)
        p2     = self.config.get("sch_param2", 0.5)
        epochs = self.config.get("epochs", 300)
        if name == "StepLR":
            return StepLR(optimizer, step_size=int(p1), gamma=p2), False
        elif name == "CosineAnnealingLR":
            return CosineAnnealingLR(optimizer, T_max=int(p1), eta_min=p2), False
        elif name == "ReduceLROnPlateau":
            return ReduceLROnPlateau(optimizer, mode='min', patience=int(p1), factor=p2), True
        elif name == "OneCycleLR":
            return OneCycleLR(optimizer, max_lr=p2, epochs=epochs, steps_per_epoch=1), False
        return None, False

    def _build_loss(self, device):
        import torch, torch.nn as nn
        import segmentation_models_pytorch as smp
        name  = self.config.get("loss_fn", "CrossEntropyLoss")
        cls_w = self.config.get("class_weights", None)
        w = torch.tensor(cls_w, dtype=torch.float).to(device) if cls_w else None
        if name == "CrossEntropyLoss":
            return nn.CrossEntropyLoss(weight=w)
        elif name == "DiceLoss":
            return smp.losses.DiceLoss(mode='multiclass')
        elif name == "Dice + CE":
            dice = smp.losses.DiceLoss(mode='multiclass')
            ce   = nn.CrossEntropyLoss(weight=w)
            class Combined(nn.Module):
                def forward(self, p, t): return dice(p, t) + ce(p, t)
            return Combined()
        elif name == "FocalLoss":
            return smp.losses.FocalLoss(mode='multiclass')
        elif name == "IoULoss":
            return smp.losses.JaccardLoss(mode='multiclass')
        return nn.CrossEntropyLoss(weight=w)

    def _build_dataset(self):
        import torch, numpy as np, glob, os
        from PIL import Image
        import scipy.io as sio
        from torch.utils.data import DataLoader, TensorDataset, random_split, Subset

        # 모델별 패딩 배수 자동 결정
        try:
            from tools.segmentation.utils.pad_utils import get_pad_multiple, pad_np, common_pad_size, input_scale_for
            multiple = get_pad_multiple(self.config.get("model_name","UNet"))
        except Exception:
            multiple = 32
            def pad_np(arr, multiple=32, is_label=False):
                h, w = arr.shape[:2]
                ph = (multiple - h % multiple) % multiple
                pw = (multiple - w % multiple) % multiple
                if ph == 0 and pw == 0:
                    return arr, h, w
                pad_width = ((0,ph),(0,pw),(0,0)) if arr.ndim == 3 else ((0,ph),(0,pw))
                mode = 'constant' if is_label else 'reflect'
                kwargs = {'constant_values': 0} if is_label else {}
                return np.pad(arr, pad_width, mode=mode, **kwargs), h, w

        img_dir = self.config.get("train_img_dir", "")
        lbl_dir = self.config.get("train_lbl_dir", "")
        img_ext = self.config.get("img_ext", "png")
        lbl_ext = self.config.get("lbl_ext", "png")
        batch   = self.config.get("batch", 2)
        split   = self.config.get("split_mode", "manual")

        def load_img(path):
            ext = path.rsplit(".", 1)[-1].lower()
            if ext == "mat":
                return sio.loadmat(path)['image'].astype(np.float32)
            elif ext == "npy":
                return np.load(path).astype(np.float32)
            else:
                # 원본 dtype(uint8) 유지 — float 변환은 패딩 배열에 채울 때 (메모리 절약)
                return np.array(Image.open(path))

        # 모델 탭에서 고른 입력 채널만 사용 (없으면 전부)
        from tools.segmentation.utils.pad_utils import select_channels
        channels = self.config.get("channels")
        _load_raw = load_img
        def load_img(path):
            return select_channels(_load_raw(path), channels)

        from tools.segmentation.utils.label_utils import load_label as load_lbl   # 0/255 → 0/1 변환 포함

        # 패딩 적용 + 원본 크기 저장
        # 이미지 크기가 제각각이어도 한 텐서로 묶을 수 있도록 (최대 H,W)의 multiple 배수로 통일
        orig_size = None
        def pad_imgs_lbls(imgs, lbls):
            nonlocal orig_size
            hw = common_pad_size(imgs, multiple=multiple)
            # 미리 할당한 배열에 바로 채움 (리스트 → np.array → tensor 복사로 메모리가 3배 되는 것 방지)
            out_i = np.empty((len(imgs), *hw, *imgs[0].shape[2:]), dtype=np.float32)
            out_l = np.empty((len(lbls), *hw), dtype=lbls[0].dtype)
            for k in range(len(imgs)):
                p, h, w = pad_np(imgs[k], multiple=multiple, is_label=False, target_hw=hw)
                out_i[k] = p
                imgs[k] = None
                if orig_size is None:
                    orig_size = (h, w)
            for k in range(len(lbls)):
                out_l[k] = pad_np(lbls[k], multiple=multiple, is_label=True, target_hw=hw)[0]
            return out_i, out_l

        def to_tensor(imgs, lbls):
            it = torch.from_numpy(np.asarray(imgs, dtype=np.float32))
            lbls = np.asarray(lbls)
            # uint8 라벨은 그대로 보관 (long은 8배 메모리) → 학습 루프에서 GPU로 옮길 때 .long()
            lt = torch.from_numpy(lbls) if lbls.dtype == np.uint8 else torch.from_numpy(lbls.astype(np.int64))
            # (N,H,W,C) → (N,C,H,W)
            if it.ndim == 4 and it.shape[-1] < it.shape[-2]:
                it = it.permute(0, 3, 1, 2)
            elif it.ndim == 3:
                it = it.unsqueeze(1)
            return it, lt

        from tools.segmentation.utils.pair_utils import pair_dirs
        if not glob.glob(os.path.join(img_dir, f"*.{img_ext}")):
            raise FileNotFoundError(f"이미지 없음: {img_dir}/*.{img_ext}")
        if not glob.glob(os.path.join(lbl_dir, f"*.{lbl_ext}")):
            raise FileNotFoundError(f"라벨 없음: {lbl_dir}/*.{lbl_ext}")
        # 파일명 기준으로 이미지 ↔ 라벨 짝짓기 (라벨 없는 이미지는 제외)
        img_files, lbl_files, pinfo = pair_dirs(img_dir, lbl_dir, img_ext, lbl_ext)
        for m in pinfo["messages"]:
            self.log_msg.emit(m)
        if not img_files:
            raise FileNotFoundError(f"이미지 ↔ 라벨 짝이 하나도 없습니다: {img_dir} / {lbl_dir}")

        self.log_msg.emit(f"  Train 이미지 {len(img_files)}개 / 라벨 {len(lbl_files)}개")
        imgs = [load_img(p) for p in img_files]
        lbls = [load_lbl(p) for p in lbl_files]
        # 입력 정규화 분모 (uint8 → 255) — _config.json 에 저장되어 평가/추론에서도 동일 적용
        in_scale = input_scale_for(imgs[0])

        # 패딩 적용
        imgs, lbls = pad_imgs_lbls(imgs, lbls)
        if orig_size is not None:
            new_h, new_w = imgs[0].shape[:2]
            if (new_h, new_w) != orig_size:
                self.log_msg.emit(
                    f"  자동 패딩: {orig_size[0]}x{orig_size[1]} → {new_h}x{new_w} "
                    f"(multiple={multiple})"
                )

        img_t, lbl_t = to_tensor(imgs, lbls)
        del imgs, lbls
        img_t /= in_scale
        self.log_msg.emit(f"  입력 정규화: / {in_scale:g}")
        dataset = TensorDataset(img_t, lbl_t)

        # 나중에 config.json에 저장하기 위해 보관
        self.config["_pad_multiple"] = multiple
        self.config["_orig_size"]    = list(orig_size) if orig_size else None
        self.config["_input_scale"]  = in_scale

        if split == "manual":
            vi_dir = self.config.get("val_img_dir", "")
            vl_dir = self.config.get("val_lbl_dir", "")
            vi, vl, vinfo = pair_dirs(vi_dir, vl_dir, img_ext, lbl_ext)
            for m in vinfo["messages"]:
                self.log_msg.emit(f"  (Val){m}")
            if not vi:
                raise FileNotFoundError(f"Val 이미지 없음: {vi_dir}")
            v_imgs = [load_img(p) for p in vi]
            v_lbls = [load_lbl(p) for p in vl]
            # Val 데이터에도 동일 multiple로 패딩
            v_imgs, v_lbls = pad_imgs_lbls(v_imgs, v_lbls)
            vi_t, vl_t = to_tensor(v_imgs, v_lbls)
            vi_t /= in_scale
            self.log_msg.emit(f"  Val 이미지 {len(vi)}개")
            val_ds = TensorDataset(vi_t, vl_t)
            return [(DataLoader(dataset, batch_size=batch, shuffle=True,
                                num_workers=0, pin_memory=True),
                     DataLoader(val_ds,  batch_size=batch, shuffle=False,
                                num_workers=0, pin_memory=True))]

        elif split == "holdout":
            ratio = self.config.get("holdout_ratio", 0.2)
            n_v   = max(1, int(len(dataset) * ratio))
            n_t   = len(dataset) - n_v
            tr, vl = random_split(dataset, [n_t, n_v], generator=torch.Generator().manual_seed(self._seed()))
            self.log_msg.emit(f"  비율로 분리: Train {n_t} / Val {n_v}")
            return [(DataLoader(tr, batch_size=batch, shuffle=True,
                                num_workers=0, pin_memory=True),
                     DataLoader(vl, batch_size=batch, shuffle=False,
                                num_workers=0, pin_memory=True))]

        elif split == "kfold":
            from sklearn.model_selection import KFold
            k      = self.config.get("kfold_k", 5)
            strat  = self.config.get("kfold_stratified", True)
            inner  = self.config.get("kfold_inner_val", False)
            seed   = self._seed()
            n      = len(dataset)
            # 이미지마다 "있는 클래스" 중 가장 드문 클래스를 층화 기준으로 쓴다
            # (희귀 클래스가 있는 이미지가 fold마다 고르게 들어가게)
            present = [set(torch.unique(lbl_t[i]).tolist()) for i in range(n)]
            n_img_of = {}
            for s_ in present:
                for c in s_:
                    n_img_of[c] = n_img_of.get(c, 0) + 1
            key = [min(s_, key=lambda c: n_img_of[c]) for s_ in present]

            def split_idx(idx, n_splits):
                """idx(번호 배열)를 n_splits로 나눠 (나머지, 빠진 것) 쌍을 차례로 돌려준다"""
                idx = np.asarray(idx)
                if strat:
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")   # 희귀 클래스가 fold 수보다 적을 때 경고
                            from sklearn.model_selection import StratifiedKFold
                            sp = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                            return [(idx[a], idx[b]) for a, b in sp.split(idx, [key[i] for i in idx])]
                    except ValueError:
                        self.log_msg.emit("  [경고] 층화 불가(이미지 수 부족) → 일반 분할로 대체")
                sp = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                return [(idx[a], idx[b]) for a, b in sp.split(idx)]

            def loader(idx, shuffle):
                return DataLoader(Subset(dataset, [int(i) for i in idx]), batch_size=batch,
                                  shuffle=shuffle, num_workers=0, pin_memory=True)

            folds = []
            for rest, ev in split_idx(range(n), k):
                if inner:
                    # 남은 것에서 다시 검증용을 뗀다 (학습 : 검증 : 평가 = K-2 : 1 : 1)
                    tr, va = split_idx(rest, k - 1)[0]
                    folds.append((loader(tr, True), loader(va, False), loader(ev, False)))
                else:
                    folds.append((loader(rest, True), loader(ev, False), loader(ev, False)))
            self.log_msg.emit(
                f"  K-fold: K={k}  층화 {'켬' if strat else '끔'}  "
                f"검증 따로 {'켬 (학습/검증/평가)' if inner else '끔 (학습/평가)'}")
            return folds
        return []

    def _seed(self):
        """분할에 쓰는 시드 (시드 고정을 껐으면 무작위)"""
        import random
        return self.config.get("seed") if self.config.get("seed_on") else random.randrange(2**31)

    def _augment(self, data, labels):
        """Albumentations 파이프라인(self._aug_pipe)을 배치에 적용. 증강이 없으면 그대로."""
        if self._aug_pipe is None:
            return data, labels
        from tools.segmentation.utils.augment import apply_batch
        return apply_batch(self._aug_pipe, data, labels)

    def _make_aug(self, loader):
        """데이터 크기와 채널 수에 맞춰 증강 파이프라인을 만든다 (첫 배치 모양으로 판단)"""
        from tools.segmentation.utils.augment import build_pipeline
        x, _ = next(iter(loader))
        n_ch, hw = x.shape[1], tuple(x.shape[-2:])
        seed = self.config.get("seed") if self.config.get("seed_on") else None
        aug = self.config.get("augmentation", {})
        self._aug_pipe = build_pipeline(aug, hw, n_ch, seed=seed)
        if n_ch != 3 and any(aug.get(k, {}).get("on") for k in ("Jitter", "Blur", "Gray")):
            self.log_msg.emit(f"  입력 채널 {n_ch}개: 색 변환은 끄고 기하 변환만 사용")
        self.log_msg.emit("  증강: " + (" / ".join(t.__class__.__name__ for t in self._aug_pipe.transforms)
                                      if self._aug_pipe else "없음"))

    @staticmethod
    def _get_logits(out, is_tr, model_name, labels):
        """모델 출력에서 (N,C,H,W) logits 추출 — Mask2Former 특수 처리 포함"""
        import torch, torch.nn.functional as F

        if "Mask2Former" in (model_name or ""):
            # Mask2Former: class_queries_logits (N,Q,C) × masks_queries_logits (N,Q,H,W)
            cls_logits  = out.class_queries_logits   # (N, Q, num_labels+1)
            mask_logits = out.masks_queries_logits   # (N, Q, H', W')
            # 각 픽셀에 대해 query별 class score × mask score 합산
            cls_prob  = cls_logits[:, :, :-1].softmax(-1)   # (N, Q, C)  마지막은 no-obj
            mask_prob = mask_logits.sigmoid()                # (N, Q, H', W')
            # einsum: N Q C, N Q H W -> N C H W
            seg = torch.einsum("bqc,bqhw->bchw", cls_prob, mask_prob)
            if seg.shape[-2:] != labels.shape[-2:]:
                seg = F.interpolate(seg, size=labels.shape[1:],
                                    mode='bilinear', align_corners=False)
            return seg

        logits = out.logits if is_tr else out
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[1:],
                                   mode='bilinear', align_corners=False)
        return logits

    @staticmethod
    def _calc_acc(logits, labels):
        import torch
        return (torch.argmax(logits, dim=1) == labels).float().mean().item()

    @staticmethod
    def _resize(logits, labels):
        import torch.nn.functional as F
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[1:],
                                   mode='bilinear', align_corners=False)
        return logits

    def run(self):
        try:
            self._run_train()
        except Exception as e:
            import traceback
            self.log_msg.emit(f"[오류] {e}\n{traceback.format_exc()}")
        finally:
            if self._wb:
                self._wb.finish()
            self.finished.emit()

    def _run_train(self):
        import os, json, torch, numpy as np
        cfg      = self.config
        epochs   = cfg.get("epochs", 300)
        ckpt_dir = cfg["ckpt_dir"]
        os.makedirs(ckpt_dir, exist_ok=True)

        if cfg.get("seed_on"):
            import random
            seed = cfg.get("seed", 42)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.log_msg.emit(f"  시드 고정: {seed}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_msg.emit(f"[학습 시작]  디바이스: {device}")
        self.log_msg.emit(f"  모델: {cfg.get('model_name','?')}  |  Epochs: {epochs}  |  LR: {cfg.get('lr')}")
        self.log_msg.emit(f"  저장 경로: {ckpt_dir}")

        folds = self._build_dataset()
        if not folds:
            self.log_msg.emit("[오류] 데이터셋 생성 실패 — 데이터 탭 경로를 확인하세요")
            return

        self._make_aug(folds[0][0])

        from tools.segmentation.utils.wandb_logger import WandbRun
        self._wb = WandbRun(cfg, log=self.log_msg.emit)

        in_ch      = cfg.get("in_channels", 3)
        n_cls      = cfg.get("num_classes", 2)
        is_kfold   = cfg.get("split_mode") == "kfold" and len(folds) > 1
        fold_results = []   # K-Fold 각 fold 결과 수집
        oof_conf = torch.zeros(n_cls, n_cls, dtype=torch.long)   # OOF: 모든 fold의 검증 예측을 모은 혼동 행렬 (행=정답, 열=예측)
        oof_n_img = 0

        # ── K-Fold / Hold-out / Manual 학습 루프 ─────────────────
        for fold_idx, fold_loaders in enumerate(folds):
            train_loader, val_loader = fold_loaders[0], fold_loaders[1]
            eval_loader = fold_loaders[2] if len(fold_loaders) > 2 else val_loader   # K-fold: 평가 fold
            if self._stop: break

            # OOF면 fold1/ ... foldK/ 하위 폴더, 아니면 체크포인트 폴더에 바로 저장
            out_dir = os.path.join(ckpt_dir, f"fold{fold_idx+1}") if len(folds) > 1 else ckpt_dir
            os.makedirs(out_dir, exist_ok=True)
            best_path = os.path.join(out_dir, "best.pt")

            fold_tag = f"[Fold {fold_idx+1}/{len(folds)}] " if len(folds) > 1 else ""
            if fold_tag:
                self.log_msg.emit(f"\n── Fold {fold_idx+1}/{len(folds)} 시작 ──")

            # Fold 시작 시 플롯 초기화 시그널
            self.fold_start.emit(fold_idx + 1, len(folds))

            model, is_tr = self._build_model(cfg.get("model_name","UNet"), in_ch, n_cls)
            model        = model.to(device)
            optimizer    = self._build_optimizer(model)
            scheduler, sch_on_loss = self._build_scheduler(optimizer)
            criterion    = self._build_loss(device)

            # AMP (Automatic Mixed Precision) — GPU에서 1.5~3배 속도 향상
            use_amp = (device.type == "cuda")
            scaler  = torch.amp.GradScaler('cuda') if use_amp else None
            if use_amp and fold_idx == 0:
                self.log_msg.emit(f"  AMP (Mixed Precision) 활성화 — GPU 학습 가속")

            use_es      = cfg.get("use_early_stopping", True)
            es_patience = cfg.get("es_patience", 50)
            es_counter  = 0
            best_vl     = float('inf')
            best_va     = 0.0

            for ep in range(1, epochs + 1):
                if self._stop:
                    self.log_msg.emit("[중단]"); break

                model.train()
                tl = ta = 0.0
                n_bad = 0
                model_name = cfg.get("model_name","UNet")
                for bd, bl in train_loader:
                    bd, bl = self._augment(bd, bl)
                    bd = bd.to(device, non_blocking=True)
                    bl = bl.to(device, non_blocking=True).long()

                    optimizer.zero_grad(set_to_none=True)
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            out    = model(bd)
                            logits = self._get_logits(out, is_tr, model_name, bl)
                            loss   = criterion(logits, bl)
                        if not torch.isfinite(loss):
                            n_bad += 1; continue      # 수치 폭주 배치는 업데이트하지 않음
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        out    = model(bd)
                        logits = self._get_logits(out, is_tr, model_name, bl)
                        loss   = criterion(logits, bl)
                        if not torch.isfinite(loss):
                            n_bad += 1; continue
                        loss.backward()
                        optimizer.step()

                    acc = self._calc_acc(logits, bl)
                    tl += loss.item(); ta += acc
                n_ok   = max(len(train_loader) - n_bad, 1)
                avg_tl = tl / n_ok
                avg_ta = ta / n_ok
                if n_bad:
                    self.log_msg.emit(f"  [경고] 손실 NaN/inf 배치 {n_bad}개 건너뜀")

                # Validation은 fp32로 (AMP fp16은 큰 activation에서 inf → val_loss NaN 발생)
                model.eval()
                vl = va = 0.0
                inter = torch.zeros(n_cls, device=device); union = torch.zeros(n_cls, device=device)
                with torch.no_grad():
                    for bd, bl in val_loader:
                        bd = bd.to(device, non_blocking=True)
                        bl = bl.to(device, non_blocking=True).long()
                        out    = model(bd)
                        logits = self._get_logits(out, is_tr, model_name, bl)
                        loss   = criterion(logits, bl)
                        vl += loss.item()
                        va += self._calc_acc(logits, bl)
                        pred = logits.argmax(1)
                        for c in range(n_cls):
                            p_c, t_c = pred == c, bl == c
                            inter[c] += (p_c & t_c).sum(); union[c] += (p_c | t_c).sum()
                avg_vl = vl / len(val_loader)
                avg_va = va / len(val_loader)
                # 배경(0) 제외 클래스 평균 IoU — 결함처럼 작은 영역은 pixel acc로는 안 보임
                iou    = (inter / union.clamp(min=1)).tolist()
                fg_iou = float(np.mean(iou[1:])) if n_cls > 1 else iou[0]

                if scheduler:
                    scheduler.step(avg_vl) if sch_on_loss else scheduler.step()
                cur_lr = optimizer.param_groups[0]['lr']

                if avg_vl < best_vl:
                    best_vl    = avg_vl
                    best_va    = avg_va
                    es_counter = 0
                    torch.save(model.state_dict(), best_path)

                    # 모델 구조 정보 json — 추후 로드 시 자동 복원용
                    self._write_config(out_dir, fold=fold_idx + 1 if len(folds) > 1 else None,
                                       best_epoch=ep, val_loss=round(avg_vl, 6), val_acc=round(avg_va, 6))

                    self.log_msg.emit(
                        f"  Best ep{ep} | val_loss={avg_vl:.4f} val_acc={avg_va*100:.2f}%"
                    )
                else:
                    es_counter += 1

                self.epoch_done.emit(ep, avg_tl, avg_vl, avg_ta, avg_va)
                wb_prefix = f"fold{fold_idx+1}/" if len(folds) > 1 else ""
                self._wb.log({
                    f"{wb_prefix}epoch": ep, f"{wb_prefix}train_loss": avg_tl,
                    f"{wb_prefix}val_loss": avg_vl, f"{wb_prefix}train_acc": avg_ta,
                    f"{wb_prefix}val_acc": avg_va, f"{wb_prefix}val_fg_iou": fg_iou,
                    f"{wb_prefix}lr": cur_lr,
                })
                self.log_msg.emit(
                    f"{fold_tag}Ep {ep:03d}/{epochs} | "
                    f"Loss {avg_tl:.4f}/{avg_vl:.4f} | "
                    f"Acc {avg_ta*100:.1f}%/{avg_va*100:.1f}% | "
                    f"IoU(fg) {fg_iou*100:.1f}% | "
                    f"LR {cur_lr:.2e} | ES {es_counter}/{es_patience}"
                )

                if use_es and es_counter >= es_patience:
                    self.log_msg.emit(f"  Early Stopping! (ep{ep})")
                    break

            torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
            self._write_config(out_dir, model_file="last.pt", fold=fold_idx + 1 if len(folds) > 1 else None,
                               best_epoch=None)   # last는 마지막 에폭 모델이라 best_epoch 없음
            if not os.path.exists(best_path):
                # 첫 에폭이 끝나기 전에 중단된 경우: best가 없으니 last를 best로 저장
                torch.save(model.state_dict(), best_path)
                self._write_config(out_dir, fold=fold_idx + 1 if len(folds) > 1 else None, best_epoch=0)

            # OOF: 이 fold의 평가 데이터를 best 모델로 예측해 혼동 행렬에 더함
            if is_kfold and os.path.exists(best_path):
                model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
                model.eval()
                with torch.no_grad():
                    for bd, bl in eval_loader:
                        bd = bd.to(device); bl = bl.to(device).long()
                        logits = self._get_logits(model(bd), is_tr, cfg.get("model_name", "UNet"), bl)
                        pred = logits.argmax(1)
                        # bincount로 (정답, 예측) 쌍을 한 번에 세기. MATLAB의 accumarray와 비슷
                        idx = (bl * n_cls + pred).flatten()
                        oof_conf += torch.bincount(idx, minlength=n_cls * n_cls).reshape(n_cls, n_cls).cpu()
                oof_n_img += len(eval_loader.dataset)

            # Fold 결과 기록
            fold_results.append({
                "fold":      fold_idx + 1,
                "best_val_loss": round(best_vl, 6),
                "best_val_acc":  round(best_va, 6),
                "best_model":    best_path,
            })
            self._export_onnx(best_path, final=not is_kfold)   # K-fold 의 경량화는 마지막 Full 모델에서만
            self._wb.log_model(best_path, alias=f"fold{fold_idx+1}" if len(folds) > 1 else None)

        # ── K-Fold 결과 요약 저장 ─────────────────────────────────
        if is_kfold and not self._stop:
            accs   = [r["best_val_acc"]  for r in fold_results]
            losses = [r["best_val_loss"] for r in fold_results]
            summary = {
                "model":        cfg.get("model_name","?"),
                "encoder":      cfg.get("encoder","?"),
                "k_folds":      len(folds),
                "epochs":       epochs,
                "fold_results": fold_results,
                "mean_val_acc":  round(float(np.mean(accs)),  6),
                "std_val_acc":   round(float(np.std(accs)),   6),
                "mean_val_loss": round(float(np.mean(losses)),6),
                "std_val_loss":  round(float(np.std(losses)), 6),
            }
            summary_path = os.path.join(ckpt_dir, "kfold_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            self.log_msg.emit(f"\n{'='*50}")
            self.log_msg.emit(f"[K-Fold 결과 요약]")
            for r in fold_results:
                self.log_msg.emit(
                    f"  Fold {r['fold']}: val_acc={r['best_val_acc']*100:.2f}%  "
                    f"val_loss={r['best_val_loss']:.4f}"
                )
            self.log_msg.emit(f"  평균 Acc : {summary['mean_val_acc']*100:.2f}% "
                              f"± {summary['std_val_acc']*100:.2f}%")
            self.log_msg.emit(f"  평균 Loss: {summary['mean_val_loss']:.4f} "
                              f"± {summary['std_val_loss']:.4f}")
            self.log_msg.emit(f"  저장: {summary_path}")
            self._save_oof_metrics(ckpt_dir, oof_conf, oof_n_img, len(folds))

            # ── Full Data 학습 ────────────────────────────────────
            if not self._stop:
                self.log_msg.emit(f"\n{'='*50}")
                self.log_msg.emit(f"[Full Data 학습 시작]  전체 데이터로 최종 모델 생성")
                self.fold_start.emit(0, 0)  # 플롯 초기화

                full_loader, _ = self._build_full_dataset()
                if full_loader:
                    model_f, is_tr_f = self._build_model(
                        cfg.get("model_name","UNet"), in_ch, n_cls)
                    model_f    = model_f.to(device)
                    optimizer_f = self._build_optimizer(model_f)
                    scheduler_f, sch_on_loss_f = self._build_scheduler(optimizer_f)
                    criterion_f = self._build_loss(device)
                    best_full_loss = float('inf')
                    full_dir  = os.path.join(ckpt_dir, "full")
                    full_path = os.path.join(full_dir, "best.pt")
                    use_amp_f = (device.type == "cuda")
                    scaler_f  = torch.amp.GradScaler('cuda') if use_amp_f else None

                    for ep in range(1, epochs + 1):
                        if self._stop: break
                        model_f.train()
                        tl = ta = 0.0
                        for bd, bl in full_loader:
                            bd, bl = self._augment(bd, bl)
                            bd = bd.to(device, non_blocking=True)
                            bl = bl.to(device, non_blocking=True).long()

                            optimizer_f.zero_grad(set_to_none=True)
                            if use_amp_f:
                                with torch.amp.autocast('cuda'):
                                    out    = model_f(bd)
                                    logits = self._get_logits(out, is_tr_f,
                                                              cfg.get("model_name","UNet"), bl)
                                    loss   = criterion_f(logits, bl)
                                scaler_f.scale(loss).backward()
                                scaler_f.step(optimizer_f)
                                scaler_f.update()
                            else:
                                out    = model_f(bd)
                                logits = self._get_logits(out, is_tr_f,
                                                          cfg.get("model_name","UNet"), bl)
                                loss   = criterion_f(logits, bl)
                                loss.backward()
                                optimizer_f.step()

                            acc = self._calc_acc(logits, bl)
                            tl += loss.item(); ta += acc
                        avg_tl = tl / len(full_loader)
                        avg_ta = ta / len(full_loader)

                        if scheduler_f:
                            scheduler_f.step(avg_tl) if sch_on_loss_f else scheduler_f.step()
                        cur_lr = optimizer_f.param_groups[0]['lr']

                        if avg_tl < best_full_loss:
                            best_full_loss = avg_tl
                            full_dir  = os.path.join(ckpt_dir, "full")
                            os.makedirs(full_dir, exist_ok=True)
                            full_path = os.path.join(full_dir, "best.pt")
                            torch.save(model_f.state_dict(), full_path)
                            self._write_config(full_dir, fold="full", best_epoch=ep,
                                               train_loss=round(avg_tl, 6), train_acc=round(avg_ta, 6))

                        self.epoch_done.emit(ep, avg_tl, avg_tl, avg_ta, avg_ta)
                        self._wb.log({"full/epoch": ep, "full/train_loss": avg_tl,
                                      "full/train_acc": avg_ta, "full/lr": cur_lr})
                        self.log_msg.emit(
                            f"[Full] Ep {ep:03d}/{epochs} | "
                            f"Loss {avg_tl:.4f} | Acc {avg_ta*100:.1f}% | LR {cur_lr:.2e}"
                        )

                    os.makedirs(full_dir, exist_ok=True)
                    torch.save(model_f.state_dict(), os.path.join(full_dir, "last.pt"))
                    self._write_config(full_dir, model_file="last.pt", fold="full", best_epoch=None)
                    if not os.path.exists(full_path):   # 중단으로 best가 없으면 last를 best로
                        torch.save(model_f.state_dict(), full_path)
                        self._write_config(full_dir, fold="full", best_epoch=0)
                    self.log_msg.emit(f"  Full 모델 저장: {full_path}")
                    self._export_onnx(full_path, final=True)
                    self._wb.log_model(full_path, alias="full")
                    # summary에 full 결과 추가
                    summary["full_model_path"] = full_path
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)

        # ── 최종 ─────────────────────────────────────────────────
        self.log_msg.emit(f"\n전체 학습 완료")

    def _write_config(self, out_dir, model_file="best.pt", **extra):
        """모델 폴더에 config.json 저장: 모델 구조, 클래스, 채널, 정규화, 증강, 학습 설정.
        (모델 복원에 쓰는 키는 예전 _config.json과 같은 이름 그대로)"""
        import os, json
        from common.palette import class_colors
        cfg = self.config
        data = {
            "model_name":   cfg.get("model_name", "UNet"),
            "encoder":      cfg.get("encoder", "resnet34"),
            "pretrained":   cfg.get("pretrained"),
            "in_channels":  cfg.get("in_channels", 3),
            "channels":     cfg.get("channels"),   # 선택한 입력 채널 번호 (None이면 전부)
            "num_classes":  cfg.get("num_classes", 2),
            "class_names":  cfg.get("class_names"),
            "class_colors": cfg.get("class_colors") or class_colors(cfg.get("num_classes", 2)),
            "class_weights": cfg.get("class_weights"),
            "pad_multiple": cfg.get("_pad_multiple", 32),
            "orig_size":    cfg.get("_orig_size", None),
            "input_scale":  cfg.get("_input_scale", 1.0),
            "augmentation": cfg.get("augmentation", {}),
            "seed":         cfg.get("seed") if cfg.get("seed_on") else None,
            "train": {k: cfg.get(k) for k in (
                "split_mode", "kfold_k", "kfold_stratified", "kfold_inner_val", "holdout_ratio", "epochs", "batch", "lr", "optimizer",
                "loss_fn", "scheduler", "sch_param1", "sch_param2", "use_early_stopping", "es_patience")},
            **extra,
        }
        from tools.segmentation.utils.config_path import config_path_for
        # 모델 파일과 같은 이름 규칙: best.pt -> best_config.json, last.pt -> last_config.json
        with open(config_path_for(os.path.join(out_dir, model_file)), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_oof_metrics(self, ckpt_dir, conf, n_img, k):
        """모든 fold의 검증 예측을 합친 혼동 행렬로 전체 지표(클래스별 IoU, Dice 등)를 계산해
        ckpt_dir/oof_metrics.json 에 저장"""
        import os, json
        c = conf.double()
        tp = c.diag(); fp = c.sum(0) - tp; fn = c.sum(1) - tp
        iou  = tp / (tp + fp + fn).clamp(min=1)
        dice = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
        prec = tp / (tp + fp).clamp(min=1)
        rec  = tp / (tp + fn).clamp(min=1)
        names = self.config.get("class_names") or [f"Class {i}" for i in range(len(tp))]
        present = (c.sum(1) > 0)               # 정답에 한 번도 안 나온 클래스는 평균에서 뺀다
        fg = present.clone(); fg[0] = False    # 배경(0) 제외 평균
        def mean(v, m): return round(float(v[m].mean()), 6) if m.any() else None
        out = {
            "k_folds": k, "stratified": self.config.get("kfold_stratified", True),
            "inner_val": self.config.get("kfold_inner_val", False), "n_images": n_img, "n_pixels": int(c.sum()),
            "pixel_acc": round(float(tp.sum() / c.sum().clamp(min=1)), 6),
            "mean_iou": mean(iou, present), "mean_dice": mean(dice, present),
            "fg_mean_iou": mean(iou, fg), "fg_mean_dice": mean(dice, fg),
            "per_class": [{"id": i, "name": names[i] if i < len(names) else f"Class {i}",
                           "iou": round(float(iou[i]), 6), "dice": round(float(dice[i]), 6),
                           "precision": round(float(prec[i]), 6), "recall": round(float(rec[i]), 6),
                           "pixels": int(c.sum(1)[i])} for i in range(len(tp))],
            "confusion_matrix": conf.tolist(),   # 행 = 정답, 열 = 예측
        }
        path = os.path.join(ckpt_dir, "oof_metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        self.log_msg.emit(f"[OOF 전체 지표]  mIoU {out['mean_iou']}  mDice {out['mean_dice']}  "
                          f"픽셀 정확도 {out['pixel_acc']}")
        for pc in out["per_class"]:
            self.log_msg.emit(f"  {pc['name']}: IoU {pc['iou']:.4f}  Dice {pc['dice']:.4f}")
        self.log_msg.emit(f"  저장: {path}")

    def _export_onnx(self, pth_path, final=False):
        """best .pth 를 ONNX 로 내보낸다 (모델 탭의 'ONNX 만들기' 선택에 따라 INT8 도 만든다).
        실패해도 .pth 는 이미 저장돼 있다. 대신 원인을 화면 로그에 그대로 남긴다.
        final=True: 최종 모델(INT8 경량화 대상)"""
        import os, traceback
        try:
            from tools.segmentation.utils.onnx_export import export_onnx
            res = export_onnx(pth_path, log=self.log_msg.emit)
            if not os.path.isfile(res["path"]):   # 예외 없이 끝났는데 파일이 없는 경우도 알린다
                raise FileNotFoundError(f"ONNX 파일이 만들어지지 않았습니다: {res['path']}")
        except Exception as e:
            # 종류와 메시지를 함께 보여 주고, 자세한 추적은 마지막 줄들만 남긴다
            self.log_msg.emit(f"  [오류] ONNX 내보내기 실패 (.pt는 정상 저장됨): {type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}")
            self.log_msg.emit("  " + "\n  ".join(traceback.format_exc().strip().splitlines()[-4:]))
            return
        if self.config.get("onnx_mode") == "int8" and final:
            try:
                from tools.segmentation.utils.quantize_onnx import (
                    default_excluded, pick_calibration_files, quantize_int8)
                self.log_msg.emit("  INT8 경량화 시작")
                files = pick_calibration_files(self.config["train_img_dir"], self.config.get("img_ext", "png"), 100)
                if not files:
                    raise FileNotFoundError(f"캘리브레이션 이미지 없음: {self.config['train_img_dir']}")
                quantize_int8(res["path"], files, default_excluded(res["path"]), log=self.log_msg.emit)
            except Exception as e:
                self.log_msg.emit(f"  [오류] INT8 경량화 실패 (FP32 .onnx는 저장됨): {type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}")
                self.log_msg.emit("  " + "\n  ".join(traceback.format_exc().strip().splitlines()[-4:]))

    def _build_full_dataset(self):
        """K-Fold용 전체 데이터 DataLoader 생성"""
        import torch, numpy as np, glob, os
        from PIL import Image
        import scipy.io as sio
        from torch.utils.data import DataLoader, TensorDataset

        try:
            from tools.segmentation.utils.pad_utils import get_pad_multiple, pad_np, common_pad_size, input_scale_for
            multiple = get_pad_multiple(self.config.get("model_name","UNet"))
        except Exception:
            multiple = 32
            def pad_np(arr, multiple=32, is_label=False):
                h, w = arr.shape[:2]
                ph = (multiple - h % multiple) % multiple
                pw = (multiple - w % multiple) % multiple
                if ph == 0 and pw == 0:
                    return arr, h, w
                pad_width = ((0,ph),(0,pw),(0,0)) if arr.ndim == 3 else ((0,ph),(0,pw))
                mode = 'constant' if is_label else 'reflect'
                kwargs = {'constant_values': 0} if is_label else {}
                return np.pad(arr, pad_width, mode=mode, **kwargs), h, w

        cfg     = self.config
        img_dir = cfg.get("train_img_dir","")
        lbl_dir = cfg.get("train_lbl_dir","")
        img_ext = cfg.get("img_ext","png")
        lbl_ext = cfg.get("lbl_ext","png")
        batch   = cfg.get("batch",2)

        def load_img(p):
            ext = p.rsplit(".",1)[-1].lower()
            if ext=="mat":   return sio.loadmat(p)['image'].astype(np.float32)
            elif ext=="npy": return np.load(p).astype(np.float32)
            else:            return np.array(Image.open(p))   # uint8 유지 (메모리 절약)

        from tools.segmentation.utils.pad_utils import select_channels
        _load_raw = load_img
        def load_img(p):
            return select_channels(_load_raw(p), cfg.get("channels"))

        from tools.segmentation.utils.pair_utils import pair_dirs
        img_files, lbl_files, _ = pair_dirs(img_dir, lbl_dir, img_ext, lbl_ext)
        if not img_files: return None, None

        from tools.segmentation.utils.label_utils import load_label
        imgs_raw = [load_img(p) for p in img_files]
        lbls_raw = [load_label(p) for p in lbl_files]
        # 패딩 적용 (크기가 제각각이어도 공통 크기로)
        in_scale = cfg.get("_input_scale") or input_scale_for(imgs_raw[0])
        hw = common_pad_size(imgs_raw, multiple=multiple)
        # 미리 할당한 배열에 바로 채움 (메모리 복사 최소화)
        imgs = np.empty((len(imgs_raw), *hw, *imgs_raw[0].shape[2:]), dtype=np.float32)
        lbls = np.empty((len(lbls_raw), *hw), dtype=lbls_raw[0].dtype)
        for k in range(len(imgs_raw)):
            imgs[k] = pad_np(imgs_raw[k], multiple=multiple, is_label=False, target_hw=hw)[0]
            imgs_raw[k] = None
        for k in range(len(lbls_raw)):
            lbls[k] = pad_np(lbls_raw[k], multiple=multiple, is_label=True, target_hw=hw)[0]

        it = torch.from_numpy(imgs)
        it /= in_scale
        lt = torch.from_numpy(lbls) if lbls.dtype == np.uint8 else torch.from_numpy(lbls.astype(np.int64))
        if it.ndim==4 and it.shape[-1]<it.shape[-2]: it = it.permute(0,3,1,2)
        elif it.ndim==3: it = it.unsqueeze(1)
        ds = TensorDataset(it, lt)
        self.log_msg.emit(f"  Full 데이터: {len(ds)}장 (multiple={multiple})")
        return DataLoader(ds, batch_size=batch, shuffle=True,
                          num_workers=0, pin_memory=True), None


# ═══════════════════════════════════════════════════════════════
#  TrainTab UI
# ═══════════════════════════════════════════════════════════════
class TrainTab(QWidget):
    step_done = Signal()   # 단계 표시줄에 '끝남'을 알리는 신호
    def __init__(self, shared_state):
        super().__init__()
        self.state  = shared_state
        self.worker = None
        self._data_tab = None
        self.train_loss_data = []
        self.val_loss_data   = []
        self.train_acc_data  = []
        self.val_acc_data    = []
        self._build()

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)

        left_w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner  = QWidget(); scroll.setWidget(inner)
        left_v = QVBoxLayout(inner)
        left_v.setSpacing(12); left_v.setContentsMargins(16, 16, 16, 16)

        # 하이퍼파라미터
        hp = QGroupBox("하이퍼파라미터")
        hg = QGridLayout(hp)
        hg.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs = QSpinBox(); self.epochs.setRange(1,5000); self.epochs.setValue(300)
        hg.addWidget(self.epochs, 0, 1)
        hg.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch = QSpinBox(); self.batch.setRange(1,256); self.batch.setValue(2)
        hg.addWidget(self.batch, 1, 1)
        hg.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(6); self.lr.setRange(1e-7,1.0)
        self.lr.setValue(0.0006); self.lr.setSingleStep(0.0001)
        hg.addWidget(self.lr, 2, 1)
        hg.addWidget(QLabel("Optimizer:"), 3, 0)
        self.optimizer = QComboBox()
        self.optimizer.addItems(["AdamW","Adam","SGD","RMSprop"])
        hg.addWidget(self.optimizer, 3, 1)
        hg.addWidget(QLabel("Loss Function:"), 4, 0)
        self.loss_fn = QComboBox()
        self.loss_fn.addItems(["CrossEntropyLoss","DiceLoss","Dice + CE","FocalLoss","IoULoss"])
        hg.addWidget(self.loss_fn, 4, 1)
        left_v.addWidget(hp)

        # LR Scheduler
        sch_g = QGroupBox("LR Scheduler")
        sg = QGridLayout(sch_g)
        sg.addWidget(QLabel("Scheduler:"), 0, 0)
        self.scheduler = QComboBox()
        self.scheduler.addItems(["없음","StepLR","CosineAnnealingLR","ReduceLROnPlateau","OneCycleLR"])
        self.scheduler.setCurrentText("ReduceLROnPlateau")
        self.scheduler.currentTextChanged.connect(self._on_scheduler_changed)
        sg.addWidget(self.scheduler, 0, 1)
        self.sch_p1_lbl = QLabel("Patience:"); sg.addWidget(self.sch_p1_lbl, 1, 0)
        self.sch_p1 = QSpinBox(); self.sch_p1.setRange(1,1000); self.sch_p1.setValue(10)
        sg.addWidget(self.sch_p1, 1, 1)
        self.sch_p2_lbl = QLabel("Factor:"); sg.addWidget(self.sch_p2_lbl, 2, 0)
        self.sch_p2 = QDoubleSpinBox(); self.sch_p2.setDecimals(3)
        self.sch_p2.setRange(0.001,1.0); self.sch_p2.setValue(0.5)
        sg.addWidget(self.sch_p2, 2, 1)
        left_v.addWidget(sch_g)

        # Early Stopping
        es_g = QGroupBox("Early Stopping")
        eg = QGridLayout(es_g)
        self.use_es = QCheckBox("Early Stopping 사용"); self.use_es.setChecked(True)
        eg.addWidget(self.use_es, 0, 0, 1, 2)
        eg.addWidget(QLabel("Patience:"), 1, 0)
        self.es_patience = QSpinBox(); self.es_patience.setRange(1,500); self.es_patience.setValue(50)
        eg.addWidget(self.es_patience, 1, 1)
        left_v.addWidget(es_g)

        # Augmentation (Albumentations)
        from tools.segmentation.utils.augment import ITEMS, PRESETS
        aug_g = QGroupBox("Augmentation")
        av = QVBoxLayout(aug_g)
        top = QHBoxLayout()
        top.addWidget(QLabel("프리셋:"))
        self.aug_preset = QComboBox(); self.aug_preset.addItems(list(PRESETS))
        self.aug_preset.setCurrentText("보통")
        self.aug_preset.currentTextChanged.connect(self._apply_preset)
        top.addWidget(self.aug_preset, 1)
        self.aug_adv_btn = QPushButton("고급"); self.aug_adv_btn.setObjectName("secondary")
        self.aug_adv_btn.setCheckable(True)
        top.addWidget(self.aug_adv_btn)
        av.addLayout(top)
        # 고급: 항목별 켜고 끄기
        self.aug_adv = QWidget(); ag = QGridLayout(self.aug_adv); ag.setContentsMargins(0, 0, 0, 0)
        self.aug_checks = {}; self.aug_spins = {}
        for r, (key, label, vlabel, is_color, lo, hi, step) in enumerate(ITEMS):
            cb = QCheckBox(label); ag.addWidget(cb, r, 0); self.aug_checks[key] = cb
            sp = QDoubleSpinBox(); sp.setDecimals(2); sp.setRange(lo, hi); sp.setSingleStep(step)
            sp.setPrefix(f"{vlabel}="); sp.setFixedWidth(120)
            ag.addWidget(sp, r, 1); self.aug_spins[key] = sp
        self.aug_color_hint = QLabel(""); self.aug_color_hint.setObjectName("hint")
        self.aug_color_hint.setWordWrap(True)
        ag.addWidget(self.aug_color_hint, len(ITEMS), 0, 1, 2)
        self.aug_adv.setVisible(False)
        self.aug_adv_btn.toggled.connect(self.aug_adv.setVisible)
        av.addWidget(self.aug_adv)
        left_v.addWidget(aug_g)
        self._apply_preset("보통")

        # 재현성 / 저장 위치
        ck_g = QGroupBox("저장 / 시드")
        ck = QGridLayout(ck_g)
        self.seed_check = QCheckBox("시드 고정"); self.seed_check.setChecked(True)
        ck.addWidget(self.seed_check, 0, 0)
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 2**31 - 1); self.seed_spin.setValue(42)
        ck.addWidget(self.seed_spin, 0, 1)
        self.seed_check.toggled.connect(self.seed_spin.setEnabled)
        self.ckpt_label = QLabel("저장: 라벨 폴더 상위 / checkpoints / <모델>_<날짜-시각> /")
        self.ckpt_label.setObjectName("hint"); self.ckpt_label.setWordWrap(True)
        ck.addWidget(self.ckpt_label, 1, 0, 1, 2)
        left_v.addWidget(ck_g)
        left_v.addStretch()

        lw = QVBoxLayout(left_w); lw.setContentsMargins(0,0,0,0); lw.addWidget(scroll)
        splitter.addWidget(left_w)

        # 오른쪽 모니터
        right_w = QWidget()
        rv = QVBoxLayout(right_w); rv.setContentsMargins(16,16,16,16); rv.setSpacing(8)

        btn_h = QHBoxLayout()
        self.start_btn = QPushButton("학습 시작"); self.start_btn.clicked.connect(self._start_train)
        self.stop_btn  = QPushButton("중단"); self.stop_btn.setObjectName("secondary")
        self.stop_btn.clicked.connect(self._stop_train); self.stop_btn.setEnabled(False)
        btn_h.addWidget(self.start_btn); btn_h.addWidget(self.stop_btn)
        rv.addLayout(btn_h)

        self.progress = QProgressBar(); self.progress.setRange(0,100); rv.addWidget(self.progress)
        self.epoch_label = QLabel("대기 중..."); self.epoch_label.setObjectName("hint")
        rv.addWidget(self.epoch_label)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=(6,4), facecolor='white')
            self.ax1 = self.fig.add_subplot(121); self.ax2 = self.fig.add_subplot(122)
            self.canvas = Canvas(self.fig); rv.addWidget(self.canvas); self.has_mpl = True
        except ImportError:
            rv.addWidget(QLabel("matplotlib 미설치")); self.has_mpl = False

        prev_h = QHBoxLayout()
        prev_lbl = QLabel("증강 미리보기"); prev_lbl.setObjectName("section_title")
        prev_h.addWidget(prev_lbl); prev_h.addStretch(1)
        prev_btn = QPushButton("미리보기"); prev_btn.setObjectName("secondary")
        prev_btn.clicked.connect(self._preview_aug)
        prev_h.addWidget(prev_btn)
        rv.addLayout(prev_h)
        if self.has_mpl:
            self.pfig = Figure(figsize=(6, 1.6), facecolor='white')
            self.pcanvas = Canvas(self.pfig); self.pcanvas.setFixedHeight(130)
            rv.addWidget(self.pcanvas)

        log_lbl = QLabel("학습 로그"); log_lbl.setObjectName("section_title"); rv.addWidget(log_lbl)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(180)
        rv.addWidget(self.log_box)

        splitter.addWidget(right_w); splitter.setSizes([380,620]); outer.addWidget(splitter)

    def _on_scheduler_changed(self, name):
        param_map = {
            "StepLR":            ("Step Size:", "Gamma:"),
            "CosineAnnealingLR": ("T_max:",     "eta_min:"),
            "ReduceLROnPlateau": ("Patience:",  "Factor:"),
            "OneCycleLR":        ("Max LR:",    "Pct Start:"),
            "없음":              ("",           ""),
        }
        p1, p2 = param_map.get(name, ("Param1:", "Param2:"))
        self.sch_p1_lbl.setText(p1); self.sch_p2_lbl.setText(p2)
        visible = name != "없음"
        for w in [self.sch_p1_lbl, self.sch_p1, self.sch_p2_lbl, self.sch_p2]:
            w.setVisible(visible)

    def _apply_preset(self, name):
        """프리셋을 고급 항목(켜기/수치)에 반영"""
        from tools.segmentation.utils.augment import PRESETS, DEFAULTS, ITEM_KEYS
        preset = PRESETS.get(name, {})
        for key in ITEM_KEYS:
            self.aug_checks[key].setChecked(key in preset)
            self.aug_spins[key].setValue(preset.get(key, DEFAULTS[key]))

    def _aug_config(self):
        return {k: {"on": cb.isChecked(), "value": self.aug_spins[k].value()}
                for k, cb in self.aug_checks.items()}

    def _update_color_enabled(self):
        """입력 채널이 3이 아니면(다파장) 색 변환 항목을 끄고 잠근다"""
        from tools.segmentation.utils.augment import COLOR_KEYS
        n_ch = self.state.get("in_channels", 3)
        multi = (n_ch != 3)
        for k in COLOR_KEYS:
            if multi:
                self.aug_checks[k].setChecked(False)
            self.aug_checks[k].setEnabled(not multi); self.aug_spins[k].setEnabled(not multi)
        self.aug_color_hint.setText(f"입력 채널 {n_ch}개: 색 변환은 쓰지 않고 기하 변환만 사용" if multi else "")

    def showEvent(self, e):
        super().showEvent(e)
        self._update_color_enabled()   # 모델 탭에서 채널 수를 바꿨을 수 있다

    def _preview_aug(self):
        """Train 첫 이미지에 지금 증강 설정을 4번 적용해 보여준다 (라벨은 윤곽선)"""
        import numpy as np
        self._update_color_enabled()
        dc = self._get_data_config()
        try:
            from tools.segmentation.utils.pair_utils import pair_dirs
            from tools.segmentation.utils.label_utils import load_label
            from tools.segmentation.utils.pad_utils import input_scale_for
            from tools.segmentation.utils.augment import build_pipeline, apply_one
            from PIL import Image
            import scipy.io as sio
            imgs, lbls, _ = pair_dirs(dc.get("train_img_dir", ""), dc.get("train_lbl_dir", ""),
                                       dc.get("img_ext", "png"), dc.get("lbl_ext", "png"))
            if not imgs:
                raise FileNotFoundError("데이터 탭에서 Train 이미지/라벨 폴더를 먼저 지정하세요")
            p = imgs[0]; ext = p.rsplit(".", 1)[-1].lower()
            img = (sio.loadmat(p)['image'] if ext == "mat" else np.load(p) if ext == "npy"
                   else np.array(Image.open(p))).astype(np.float32)
            if img.ndim == 2: img = img[..., None]
            img = img / input_scale_for(img)
            lbl = load_label(lbls[0]).astype(np.uint8)
            n_ch = img.shape[2]
            pipe = build_pipeline(self._aug_config(), img.shape[:2], n_ch)
        except Exception as ex:
            self.log_box.append(f"[미리보기 오류] {ex}"); return
        if not self.has_mpl: return
        self.pfig.clear()
        for k in range(4):
            ax = self.pfig.add_subplot(1, 4, k + 1)
            a, m = (img, lbl) if (pipe is None or k == 0) else apply_one(pipe, img, lbl)
            show = a[..., :3] if a.shape[2] >= 3 else np.repeat(a[..., :1], 3, 2)
            lo, hi = np.percentile(show, 1), np.percentile(show, 99)
            ax.imshow(np.clip((show - lo) / max(hi - lo, 1e-9), 0, 1))
            if m.max() > 0: ax.contour(m > 0, levels=[0.5], linewidths=0.8)
            ax.set_title("원본" if k == 0 else f"증강 {k}", fontsize=8); ax.axis("off")
        self.pfig.tight_layout(pad=0.3); self.pcanvas.draw()

    def set_data_tab(self, data_tab):
        """main.py에서 호출 — 데이터탭 직접 참조로 저장 버튼 없이도 경로 읽기"""
        self._data_tab = data_tab

    def _get_data_config(self):
        """shared_state 우선, 없으면 data_tab UI에서 직접 읽기"""
        # shared_state에 경로가 있으면 그걸 씀
        if self.state.get("train_img_dir", "").strip():
            return {
                "train_img_dir": self.state.get("train_img_dir", ""),
                "train_lbl_dir": self.state.get("train_lbl_dir", ""),
                "val_img_dir":   self.state.get("val_img_dir", ""),
                "val_lbl_dir":   self.state.get("val_lbl_dir", ""),
                "img_ext":       self.state.get("img_ext", "png"),
                "lbl_ext":       self.state.get("lbl_ext", "png"),
                "split_mode":    self.state.get("split_mode", "manual"),
                "holdout_ratio": self.state.get("holdout_ratio", 0.2),
                "kfold_k":       self.state.get("kfold_k", 5),
                "kfold_stratified": self.state.get("kfold_stratified", True),
                "kfold_inner_val":  self.state.get("kfold_inner_val", False),
            }
        # data_tab UI에서 직접 읽기 (저장 버튼 안 눌렀어도 동작)
        if hasattr(self, '_data_tab') and self._data_tab is not None:
            dt = self._data_tab
            mode_map = {0: "manual", 1: "holdout", 2: "kfold"}
            return {
                "train_img_dir": dt.train_img_edit.text().strip(),
                "train_lbl_dir": dt.train_lbl_edit.text().strip(),
                "val_img_dir":   dt.val_img_edit.text().strip(),
                "val_lbl_dir":   dt.val_lbl_edit.text().strip(),
                "img_ext":       dt.img_ext.currentText(),
                "lbl_ext":       dt.lbl_ext.currentText(),
                "split_mode":    mode_map.get(dt.split_btns.checkedId(), "manual"),
                "holdout_ratio": dt.holdout_spin.value(),
                "kfold_k":       dt.kfold_spin.value(),
                "kfold_stratified": dt.kfold_strat.isChecked(),
                "kfold_inner_val":  dt.kfold_inner.isChecked(),
            }
        return {}

    def _start_train(self):
        self.train_loss_data.clear(); self.val_loss_data.clear()
        self.train_acc_data.clear();  self.val_acc_data.clear()
        self.log_box.clear(); self.progress.setValue(0)

        # 데이터 경로 읽기 (저장 버튼 없이도 동작)
        data_cfg = self._get_data_config()

        # 경로 검증
        train_img = data_cfg.get("train_img_dir", "").strip()
        train_lbl = data_cfg.get("train_lbl_dir", "").strip()
        split_mode = data_cfg.get("split_mode", "manual")

        if not train_img:
            self.log_box.append("[오류] Train 이미지 폴더가 비어있어요\n"
                                "→ 데이터 탭에서 경로를 설정하거나\n"
                                "→ '데이터 설정 저장' 버튼을 눌러주세요")
            return
        if not train_lbl:
            self.log_box.append("[오류] Train 라벨 폴더가 비어있어요")
            return
        if split_mode == "manual":
            val_img = data_cfg.get("val_img_dir", "").strip()
            if not val_img:
                self.log_box.append("[오류] Val 이미지 폴더가 비어있어요\n"
                                    "→ 비율로 분리 또는 OOF 방식으로 변경하거나\n"
                                    "→ Val 폴더를 지정해주세요")
                return

        self._update_color_enabled()
        # 저장 폴더: 라벨 폴더의 바로 상위 / checkpoints / <모델>_<날짜-시각>
        import os, datetime
        model_name = self.state.get("model_name", "UNet")
        run_dir = os.path.join(os.path.dirname(os.path.normpath(train_lbl)), "checkpoints",
                               f"{model_name}_{datetime.datetime.now():%Y%m%d-%H%M}")
        self.run_dir = run_dir
        self.ckpt_label.setText(f"저장: {run_dir}")

        config = {
            **data_cfg,
            "model_name":         self.state.get("model_name", "UNet"),
            "encoder":            self.state.get("encoder", "resnet34"),
            "pretrained":         self.state.get("pretrained", "없음 (scratch)"),
            "in_channels":        self.state.get("in_channels", 3),
            "channels":           self.state.get("channels"),
            "num_classes":        self.state.get("num_classes", 2),
            "class_weights":      self.state.get("class_weights", None),
            "epochs":             self.epochs.value(),
            "batch":              self.batch.value(),
            "lr":                 self.lr.value(),
            "optimizer":          self.optimizer.currentText(),
            "loss_fn":            self.loss_fn.currentText(),
            "scheduler":          self.scheduler.currentText(),
            "sch_param1":         self.sch_p1.value(),
            "sch_param2":         self.sch_p2.value(),
            "use_early_stopping": self.use_es.isChecked(),
            "es_patience":        self.es_patience.value(),
            "class_names":        self.state.get("class_names"),
            "augmentation":       self._aug_config(),
            "onnx_mode":          self.state.get("onnx_mode", "onnx"),
            "seed_on":            self.seed_check.isChecked(),
            "seed":               self.seed_spin.value(),
            "ckpt_dir":           run_dir,
            "wandb_enabled":      self.state.get("wandb_enabled", False),
            "wandb_project":      self.state.get("wandb_project", ""),
        }

        self.log_box.append(f"[경로 확인]")
        self.log_box.append(f"  Train 이미지: {config['train_img_dir']}")
        self.log_box.append(f"  Train 라벨:   {config['train_lbl_dir']}")
        self.log_box.append(f"  분할 방식:    {config['split_mode']}")
        self.total_epochs = config["epochs"]
        self.worker = TrainWorker(config)
        self.worker.epoch_done.connect(self._on_epoch)
        self.worker.fold_start.connect(self._on_fold_start)
        self.worker.log_msg.connect(self.log_box.append)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)

    def _on_fold_start(self, fold_idx, total):
        """Fold 시작 시 플롯 초기화"""
        self.train_loss_data.clear(); self.val_loss_data.clear()
        self.train_acc_data.clear();  self.val_acc_data.clear()
        self.progress.setValue(0)
        if total == 0:
            self.epoch_label.setText("Full Data 학습 중...")
        else:
            self.epoch_label.setText(f"Fold {fold_idx} / {total} 학습 중...")
        if self.has_mpl:
            self.ax1.cla(); self.ax2.cla()
            self.canvas.draw()

    def _stop_train(self):
        if self.worker: self.worker.stop()

    def _on_epoch(self, ep, tl, vl, ta, va):
        self.train_loss_data.append(tl); self.val_loss_data.append(vl)
        self.train_acc_data.append(ta);  self.val_acc_data.append(va)
        self.progress.setValue(int(ep / self.total_epochs * 100))
        self.epoch_label.setText(
            f"Epoch {ep}/{self.total_epochs}  |  Val Loss: {vl:.4f}  Val Acc: {va*100:.2f}%"
        )
        if self.has_mpl: self._update_plot()

    def _update_plot(self):
        self.ax1.cla(); self.ax2.cla()
        xs = list(range(1, len(self.train_loss_data)+1))
        self.ax1.plot(xs, self.train_loss_data, label="Train", color="#2563eb", lw=1.5)
        self.ax1.plot(xs, self.val_loss_data,   label="Val",   color="#ef4444", lw=1.5)
        self.ax1.set_title("Loss", fontsize=10); self.ax1.legend(fontsize=8); self.ax1.grid(alpha=0.3)
        self.ax2.plot(xs, self.train_acc_data, label="Train", color="#2563eb", lw=1.5)
        self.ax2.plot(xs, self.val_acc_data,   label="Val",   color="#ef4444", lw=1.5)
        self.ax2.set_title("Accuracy", fontsize=10); self.ax2.legend(fontsize=8)
        self.ax2.set_ylim(0,1); self.ax2.grid(alpha=0.3)
        self.fig.tight_layout(); self.canvas.draw()

    def _on_finished(self):
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        stopped = self.worker is not None and self.worker._stop
        self.log_box.append("학습 중단 — 현재까지의 모델을 저장했습니다" if stopped else "학습 완료")
        if not stopped: self.progress.setValue(100)
        self._save_train_snapshot()
        self.step_done.emit()

    def result_model_path(self):
        """학습 결과 모델 경로: full/best.pt → best.pt → fold1/best.pt 순서로 찾는다 (없으면 "")"""
        import os
        run_dir = getattr(self, "run_dir", None)
        if not run_dir: return ""
        for sub in ("full", "", "fold1"):
            p = os.path.join(run_dir, sub, "best.pt")
            if os.path.isfile(p): return p
        return ""

    def _save_train_snapshot(self):
        """학습 Loss/Accuracy 플롯 + 하이퍼파라미터를 checkpoint 폴더에 저장"""
        import os, json, datetime
        ckpt_dir = getattr(self, "run_dir", None)
        if not ckpt_dir or not self.has_mpl or not self.train_loss_data:
            return
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            for font in ["Malgun Gothic","AppleGothic","NanumGothic","DejaVu Sans"]:
                try:
                    matplotlib.rcParams['font.family'] = font
                    matplotlib.rcParams['axes.unicode_minus'] = False
                    break
                except Exception: continue

            xs = list(range(1, len(self.train_loss_data)+1))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.plot(xs, self.train_loss_data, label="Train Loss", color="#2563eb", lw=1.5)
            ax1.plot(xs, self.val_loss_data,   label="Val Loss",   color="#ef4444", lw=1.5)
            ax1.set_title("Loss", fontsize=12); ax1.legend(); ax1.grid(alpha=0.3)
            ax1.set_xlabel("Epoch")

            ax2.plot(xs, self.train_acc_data, label="Train Acc", color="#2563eb", lw=1.5)
            ax2.plot(xs, self.val_acc_data,   label="Val Acc",   color="#ef4444", lw=1.5)
            ax2.set_title("Accuracy", fontsize=12); ax2.legend()
            ax2.set_ylim(0,1); ax2.grid(alpha=0.3); ax2.set_xlabel("Epoch")

            # 하이퍼파라미터 텍스트 박스
            params_text = "\n".join([
                f"Model: {self.state.get('model_name','?')}",
                f"Encoder: {self.state.get('encoder','?')}",
                f"In_channels: {self.state.get('in_channels','?')}",
                f"Classes: {self.state.get('num_classes','?')}",
                f"Epochs: {self.epochs.value()}",
                f"Batch: {self.batch.value()}",
                f"LR: {self.lr.value():.6f}",
                f"Optimizer: {self.optimizer.currentText()}",
                f"Loss: {self.loss_fn.currentText()}",
                f"Scheduler: {self.scheduler.currentText()}",
                f"ES Patience: {self.es_patience.value()}",
                f"Split: {self.state.get('split_mode','?')}",
            ])
            fig.text(0.5, -0.05, params_text, ha='center', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.8))

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(ckpt_dir, f"train_snapshot_{timestamp}.png")
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.log_box.append(f"학습 스냅샷 저장: {plot_path}")

            # 하이퍼파라미터 json 별도 저장
            meta = {
                "timestamp": timestamp,
                "model_name": self.state.get("model_name","?"),
                "encoder":    self.state.get("encoder","?"),
                "in_channels": self.state.get("in_channels","?"),
                "num_classes": self.state.get("num_classes","?"),
                "epochs":     self.epochs.value(),
                "batch_size": self.batch.value(),
                "lr":         self.lr.value(),
                "optimizer":  self.optimizer.currentText(),
                "loss_fn":    self.loss_fn.currentText(),
                "scheduler":  self.scheduler.currentText(),
                "sch_param1": self.sch_p1.value(),
                "sch_param2": self.sch_p2.value(),
                "early_stopping": self.use_es.isChecked(),
                "es_patience":    self.es_patience.value(),
                "split_mode": self.state.get("split_mode","?"),
                "final_train_loss": round(self.train_loss_data[-1],6) if self.train_loss_data else None,
                "final_val_loss":   round(self.val_loss_data[-1],6)   if self.val_loss_data   else None,
                "final_train_acc":  round(self.train_acc_data[-1],6)  if self.train_acc_data  else None,
                "final_val_acc":    round(self.val_acc_data[-1],6)    if self.val_acc_data    else None,
                "augmentation": self._aug_config(),
            }
            json_path = os.path.join(ckpt_dir, f"hyperparams_{timestamp}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            self.log_box.append(f"하이퍼파라미터 저장: {json_path}")

        except Exception as e:
            self.log_box.append(f"[스냅샷 오류] {e}")
