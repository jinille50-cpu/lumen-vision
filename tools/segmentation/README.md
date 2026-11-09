# SegToolkit

A PyQt5-based GUI application for semantic segmentation model training, evaluation, and inference.  
Supports hyperspectral (multi-channel), grayscale, and RGB images.

---

## Features

- **Multi-format support**: PNG, JPG, TIFF, MAT, NPY, HDR
- **Model selection**: UNet, UNet++, SegFormer-B0/B2/B4, DeepLabV3+, Mask2Former
- **Hyperspectral-specialized models**: `HSI-SpectralUNet (3D)`, `HSI-SpectralAttnUNet` — spectral-spatial learning over arbitrary band counts (30~100+)
- **Pretrained weights**: ImageNet / ImageNet-22K via HuggingFace
- **Split strategies**: Manual Train/Val split · Hold-out · Stratified K-Fold · Patient-level group split (no patient overlap between Train/Val; filename = patient ID + 2-digit slice number)
- **Class weights**: Per-class loss weighting for imbalanced datasets
- **Real-time training plot**: Loss & mIoU curves
- **Independent evaluation**: Load any `.pth` without re-training
- **Batch inference**: Free folder selection (train / val / test)
- **Overlay export**: Predicted mask overlaid on original image, per-class color customizable
- **Unified model builder**: All model construction goes through `utils/model_builder.py` (shared by train / eval / infer)

---

## Installation

```bash
git clone https://github.com/yourname/SegToolkit.git
cd SegToolkit
pip install -r requirements.txt
python main.py
```

---

## Workflow

```
① Data   →  ② Model   →  ③ Train   →  ④ Eval   →  ⑤ Infer
(path/split)  (arch/weight) (hyperparam)  (metrics)  (overlay save)
```

Evaluation and Inference tabs work independently — load any saved `.pth` directly.

---

## Adding a New Model

All model construction is centralized in `utils/model_builder.py`. To add one:

1. Implement an `nn.Module` (e.g. in `models/hyperspectral.py`) whose `forward`
   takes `(N, C, H, W)` and returns raw logits `(N, num_classes, H, W)`.
2. Register it in the model's `HSI_MODELS` registry (or add a branch in
   `build_model()` for non-HSI / HuggingFace models).
3. Add the display name to `PRETRAINED_WEIGHTS` in `tabs/model_tab.py`
   (drives the dropdown) and its pad multiple to `_PAD_MAP` in
   `utils/pad_utils.py`.

Because all three tabs (train / eval / infer) call the same `build_model()`,
one edit propagates everywhere.

---

## License

MIT
