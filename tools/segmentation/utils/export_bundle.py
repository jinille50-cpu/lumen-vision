"""최종 모델 내보내기: 체크포인트 폴더 아래 export/<모델이름>_<YYYYMMDD-HHMM>/ 에
모델 파일 + config + 클래스 색 범례 + README + Results 요약을 한 폴더로 모은다.
추론 탭은 이 폴더만 골라 바로 불러온다 (find_exported_model)."""
import glob
import json
import os
import re
import shutil
import tempfile
from datetime import datetime

FORMATS = ("PyTorch", "ONNX", "ONNX INT8")


def checkpoint_dir(model_path):
    """모델이 full/ 또는 foldN/ 안에 있으면 그 위(학습 폴더)를 체크포인트 폴더로 본다 (평가 탭과 같은 규칙)"""
    ckpt = os.path.dirname(os.path.abspath(model_path))
    if os.path.basename(ckpt) == "full" or re.fullmatch(r"fold\d+", os.path.basename(ckpt)):
        ckpt = os.path.dirname(ckpt)
    return ckpt


def _save_legend(path, names, colors):
    """클래스 색 범례 그림 (colors 는 '#rrggbb' 목록)"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from common import theme
    fig = Figure(figsize=(3.2, 0.4 + 0.32 * len(names)), facecolor="white")
    FigureCanvasAgg(fig)   # savefig 가 그릴 수 있게 캔버스를 붙인다 (창은 만들지 않음)
    ax = fig.add_subplot(111); ax.axis("off")
    handles = [Patch(facecolor=c, edgecolor=theme.MUTED, label=f"{i}: {n}")
               for i, (n, c) in enumerate(zip(names, colors))]
    ax.legend(handles=handles, loc="center left", frameon=False, fontsize=10, labelcolor=theme.TEXT)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _readme(cfg, name, fmt, model_file, names, colors, miou, hw, results_files):
    chans = cfg.get("channels")
    lines = [
        f"# {name} 내보낸 모델", "",
        f"- 내보낸 시각: {datetime.now():%Y-%m-%d %H:%M}",
        f"- 형식: {fmt} ({model_file})",
        f"- 모델 구조: {cfg.get('model_name', '')}  인코더: {cfg.get('encoder', '')}",
        f"- 입력 채널 수: {cfg.get('in_channels')}",
        f"- 선택한 채널: {', '.join(map(str, chans)) if chans else '전부 (선택 없음)'}",
        f"- 입력 크기: {hw[0]} x {hw[1]} (높이 x 너비, 학습 크기를 {cfg.get('pad_multiple') or 32} 배수로 올림. 다른 크기도 입력 가능)",
        f"- 입력 정규화 분모(input_scale): {cfg.get('input_scale', '없음 (정규화 안 함)')}",
        f"- mIoU: {f'{miou:.4f}' if miou is not None else '평가 기록 없음 (평가 탭을 먼저 실행)'}",
        "", "## 클래스", "", "| 번호 | 이름 | 색 |", "|---|---|---|",
    ]
    lines += [f"| {i} | {n} | {c} |" for i, (n, c) in enumerate(zip(names, colors))]
    lines += ["", "## 파일", "", f"- {model_file}: 모델",
              f"- {name}_config.json: 모델 옵션 (ONNX는 파일 안에도 들어 있음)",
              "- legend.png: 클래스 색 범례"]
    lines += [f"- Results/{f}" for f in results_files]
    lines += ["", "추론 탭의 '내보낸 모델 열기'로 이 폴더를 고르면 모델 형식, config, 색이 자동으로 불러와집니다.", ""]
    return "\n".join(lines)


def export_bundle(model_path, fmt, calib_dir="", img_ext="png", calib_count=100, log=None):
    """model_path(.pth/.pt)를 fmt(FORMATS 중 하나)로 내보내고 폴더 경로를 돌려준다.
    INT8 은 calib_dir(캘리브레이션 이미지 폴더)가 필요하다."""
    from common.palette import class_colors
    from tools.segmentation.utils.config_path import config_path_for, load_config

    def _log(msg):
        if log:
            log(msg)

    if fmt not in FORMATS:
        raise ValueError(f"알 수 없는 형식: {fmt}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
    cfg = load_config(model_path)
    if cfg is None:
        raise FileNotFoundError("모델 옆에 <모델이름>_config.json 이 필요합니다")
    cfg = dict(cfg)

    n_cls = int(cfg.get("num_classes", 2))
    names = list(cfg.get("class_names") or [])
    names = [names[i] if i < len(names) else f"Class {i}" for i in range(n_cls)]
    colors = list(cfg.get("class_colors") or [])
    colors = (colors + class_colors(n_cls)[len(colors):])[:n_cls]   # 모자란 색은 공통 팔레트로 채운다
    cfg["class_names"], cfg["class_colors"] = names, colors

    ckpt = checkpoint_dir(model_path)
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", str(cfg.get("model_name") or "model"))
    base_out = os.path.join(ckpt, "export", f"{name}_{datetime.now():%Y%m%d-%H%M}")
    out, n = base_out, 1
    while os.path.exists(out):   # 같은 분에 다시 내보내도 다른 형식 파일과 섞이지 않게 _2, _3 을 붙인다
        n += 1
        out = f"{base_out}_{n}"
    os.makedirs(out)
    _log(f"내보내기 폴더: {out}")

    # 모델 파일
    if fmt == "PyTorch":
        model_file = f"{name}.pth"
        shutil.copy2(model_path, os.path.join(out, model_file))
    else:
        # export_onnx 는 .pth 옆의 config 로 모델을 복원하므로 임시 폴더에 같은 이름으로 놓고 만든다
        with tempfile.TemporaryDirectory() as tmp:
            tmp_pth = os.path.join(tmp, f"{name}.pth")
            shutil.copy2(model_path, tmp_pth)
            with open(config_path_for(tmp_pth), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            from tools.segmentation.utils.onnx_export import export_onnx
            res = export_onnx(tmp_pth, log=log)
            if fmt == "ONNX":
                model_file = f"{name}.onnx"
                shutil.move(res["path"], os.path.join(out, model_file))
            else:
                from tools.segmentation.utils.quantize_onnx import (
                    default_excluded, pick_calibration_files, quantize_int8)
                files = pick_calibration_files(calib_dir, img_ext, calib_count) if calib_dir else []
                if not files:
                    raise FileNotFoundError(f"INT8 캘리브레이션 이미지 없음: {calib_dir or '(폴더 미지정)'}")
                model_file = f"{name}_int8.onnx"
                quantize_int8(res["path"], files, default_excluded(res["path"]), log=log,
                              out_path=os.path.join(out, model_file))

    # config (INT8 도 FP32 와 같은 이름: config_path_for 가 _int8 을 뗀다)
    with open(config_path_for(os.path.join(out, model_file)), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    _save_legend(os.path.join(out, "legend.png"), names, colors)

    # Results 요약 복사 + mIoU
    miou, results_files = None, []
    res_dir = os.path.join(ckpt, "Results")
    for fn in ("summary.json", "confusion_matrix.png"):
        src = os.path.join(res_dir, fn)
        if os.path.isfile(src):
            os.makedirs(os.path.join(out, "Results"), exist_ok=True)
            shutil.copy2(src, os.path.join(out, "Results", fn))
            results_files.append(fn)
    if "summary.json" in results_files:
        try:
            with open(os.path.join(res_dir, "summary.json"), "r", encoding="utf-8") as f:
                miou = float(json.load(f)["mIoU"])
        except (OSError, ValueError, KeyError):
            pass
    if not results_files:
        _log("  [알림] Results/ 가 없어 summary.json, confusion_matrix.png 는 빠집니다 (평가 탭 실행 후 다시 내보내세요)")

    from tools.segmentation.utils.onnx_export import _input_hw
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(_readme(cfg, name, fmt, model_file, names, colors, miou, _input_hw(cfg), results_files))
    _log(f"내보내기 완료: {out}")
    return out


def find_exported_model(folder):
    """내보낸 폴더에서 모델 파일을 찾아 (경로, 형식 번호)를 돌려준다. 형식 번호는 추론 탭 콤보와 같다
    (0 PyTorch, 1 ONNX FP32, 2 ONNX INT8). 모델이 없으면 (None, None)."""
    for pattern, idx in (("*_int8.onnx", 2), ("*.onnx", 1), ("*.pth", 0), ("*.pt", 0)):
        for p in sorted(glob.glob(os.path.join(folder, pattern))):
            if idx == 1 and p.endswith("_int8.onnx"):
                continue
            return p, idx
    return None, None
