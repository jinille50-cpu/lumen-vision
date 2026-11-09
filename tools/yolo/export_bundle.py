"""최종 모델 내보내기 (YOLO): 체크포인트 폴더 아래 export/<모델이름>_<YYYYMMDD-HHMM>/ 에
모델 파일 + 모델이름_config.json + 클래스 색 범례 + README + Results 요약을 한 폴더로 모은다.
ONNX/INT8 변환은 Ultralytics의 model.export()를 그대로 쓴다 (직접 변환 코드 없음).
추론 탭은 이 폴더만 골라 바로 불러온다 (find_exported_model). Segmentation의 utils/export_bundle.py와 같은 구성."""
import glob
import importlib.util
import json
import os
import re
import shutil
import tempfile
from datetime import datetime

from tools.yolo.work_state import find_config, load_config

FORMATS = ("PyTorch", "ONNX", "ONNX INT8")
ONNX_META_KEY = "yolo_config"   # ONNX metadata_props 에 config 전체를 JSON 글자로 넣는 키 (Segmentation의 seg_config와 같은 방식)


def checkpoint_dir(model_path):
    """모델이 <체크포인트>/[full|foldN]/weights/ 안에 있으면 체크포인트 폴더까지 올라간다 (평가 탭 results_dir과 같은 규칙)"""
    d = os.path.dirname(os.path.abspath(model_path))
    if os.path.basename(d) == "weights":
        d = os.path.dirname(d)
    if os.path.basename(d) == "full" or re.fullmatch(r"fold\d+", os.path.basename(d)):
        d = os.path.dirname(d)
    return d


def available_formats(data_yaml=""):
    """{형식: 불가능한 이유}. 이유가 빈 글자면 이 PC에서 내보낼 수 있다."""
    def has(mod):
        return importlib.util.find_spec(mod) is not None   # import 없이 설치 여부만 본다

    out = {"PyTorch": ""}
    missing = [m for m in ("onnx", "onnxruntime") if not has(m)]
    out["ONNX"] = f"{', '.join(missing)} 패키지가 없습니다 (pip install {' '.join(missing)})" if missing else ""
    if out["ONNX"]:
        out["ONNX INT8"] = out["ONNX"]
    else:
        # ultralytics를 import하면 torch까지 불러와 화면이 10초쯤 멈춘다 → 설정 파일(default.yaml)에 키가 있는지만 읽는다
        try:
            spec = importlib.util.find_spec("ultralytics")   # 패키지 위치만 찾고 실행은 하지 않는다
            cfg_yaml = os.path.join(spec.submodule_search_locations[0], "cfg", "default.yaml")
            with open(cfg_yaml, encoding="utf-8") as f:
                ok = re.search(r"^(quantize|int8)\s*:", f.read(), re.M) is not None   # re.M: ^가 줄마다 맞음
        except Exception:
            ok = False
        if not ok:
            out["ONNX INT8"] = "설치된 ultralytics가 INT8 내보내기를 지원하지 않습니다"
        elif not data_yaml or not os.path.isfile(data_yaml):
            out["ONNX INT8"] = "캘리브레이션에 쓸 data.yaml이 필요합니다 (평가 탭에서 고르세요)"
        else:
            out["ONNX INT8"] = ""
    return out


def _save_legend(path, names, colors):
    """클래스 색 범례 그림 (colors 는 '#rrggbb' 목록)"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from common import theme
    fig = Figure(figsize=(3.2, 0.4 + 0.32 * len(names)), facecolor=theme.BG0)
    FigureCanvasAgg(fig)   # savefig 가 그릴 수 있게 캔버스를 붙인다 (창은 만들지 않음)
    ax = fig.add_subplot(111); ax.axis("off")
    handles = [Patch(facecolor=c, edgecolor=theme.MUTED, label=f"{i}: {n}")
               for i, (n, c) in enumerate(zip(names, colors))]
    ax.legend(handles=handles, loc="center left", frameon=False, fontsize=10, labelcolor=theme.TEXT)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "평가 기록 없음 (평가 탭을 먼저 실행)"


def _readme(cfg, name, fmt, model_file, names, colors, summary, results_files):
    s = summary or {}
    lines = [
        f"# {name} 내보낸 모델", "",
        f"- 내보낸 시각: {datetime.now():%Y-%m-%d %H:%M}",
        f"- 형식: {fmt} ({model_file})",
        f"- 모델: {cfg.get('model_name', '')}",
        f"- 입력 크기 imgsz: {cfg.get('imgsz', 640)}",
        f"- mAP50: {_fmt(s.get('mAP50'))}",
        f"- mAP50-95: {_fmt(s.get('mAP50-95'))}",
        f"- Precision: {_fmt(s.get('Precision'))}",
        f"- Recall: {_fmt(s.get('Recall'))}",
    ]
    if s.get("split"):
        lines.append(f"- 평가 데이터: {s['split']}  ({s.get('data_yaml', '')})")
    lines += ["", "## 클래스", "", "| 번호 | 이름 | 색 |", "|---|---|---|"]
    lines += [f"| {i} | {n} | {c} |" for i, (n, c) in enumerate(zip(names, colors))]
    lines += ["", "## 파일", "", f"- {model_file}: 모델",
              f"- {name}_config.json: 모델 옵션 (ONNX는 파일 안 메타데이터 '{ONNX_META_KEY}'에도 들어 있음)",
              "- legend.png: 클래스 색 범례"]
    lines += [f"- Results/{f}" for f in results_files]
    lines += ["", "추론 탭의 '내보낸 모델 열기'로 이 폴더를 고르면 모델, config, imgsz가 자동으로 불러와집니다.", ""]
    return "\n".join(lines)


def _embed_onnx_config(onnx_path, cfg):
    """cfg(dict)를 onnx 메타데이터에 저장한다. 같은 키가 있으면 덮어쓴다."""
    import onnx
    model = onnx.load(onnx_path)
    for p in [p for p in model.metadata_props if p.key == ONNX_META_KEY]:
        model.metadata_props.remove(p)
    entry = model.metadata_props.add()
    entry.key, entry.value = ONNX_META_KEY, json.dumps(cfg, ensure_ascii=False)
    onnx.save(model, onnx_path)


def read_onnx_config(onnx_path):
    """onnx 메타데이터의 config(dict). 없거나 읽을 수 없으면 None."""
    try:
        import onnx
        model = onnx.load(onnx_path, load_external_data=False)
        for p in model.metadata_props:
            if p.key == ONNX_META_KEY:
                return json.loads(p.value)
    except Exception:
        pass
    return None


def export_bundle(model_path, fmt, data_yaml="", imgsz=None, log=None):
    """model_path(.pt)를 fmt(FORMATS 중 하나)로 내보내고 폴더 경로를 돌려준다.
    INT8 은 data_yaml(캘리브레이션 데이터)이 필요하다."""
    from common.palette import class_colors
    from tools.yolo.yolo_module_wrap import load_detector

    def _log(msg):
        if log:
            log(msg)

    if fmt not in FORMATS:
        raise ValueError(f"알 수 없는 형식: {fmt}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
    why = available_formats(data_yaml).get(fmt, "")
    if why:
        raise RuntimeError(f"{fmt} 내보내기 불가: {why}")

    model = load_detector(model_path)   # YOLO 또는 RT-DETR
    names_d = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    nc = len(names_d)
    names = [str(names_d[i]) for i in range(nc)]
    colors = class_colors(nc + 1)[1:]   # YOLO에는 배경 클래스가 없어 0번(흰색)을 뺀다

    cfg_file = find_config(model_path)
    cfg = dict(load_config(cfg_file) or {}) if cfg_file else {}
    if imgsz:
        cfg["imgsz"] = int(imgsz)
    cfg.setdefault("imgsz", int(getattr(model, "overrides", {}).get("imgsz") or 640))
    cfg["class_names"], cfg["class_colors"] = names, colors
    cfg.setdefault("model_name", os.path.splitext(os.path.basename(model_path))[0])

    ckpt = checkpoint_dir(model_path)
    # 폴더 이름은 체크포인트 폴더 이름(<모델>_<날짜-시각>)에서 모델 이름 부분을 쓴다. 없으면 config의 model_name
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", str(cfg.get("model_name") or "model"))
    base_out = os.path.join(ckpt, "export", f"{name}_{datetime.now():%Y%m%d-%H%M}")
    out, n = base_out, 1
    while os.path.exists(out):   # 같은 분에 다시 내보내도 다른 형식 파일과 섞이지 않게 _2, _3 을 붙인다
        n += 1
        out = f"{base_out}_{n}"
    os.makedirs(out)
    _log(f"내보내기 폴더: {out}")

    # 모델 파일: export()는 .pt 옆에 결과를 쓰므로 임시 폴더에 복사해서 만든 뒤 옮긴다 (학습 폴더를 어지럽히지 않음)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pt = os.path.join(tmp, f"{name}.pt")
        shutil.copy2(model_path, tmp_pt)
        if fmt == "PyTorch":
            model_file = f"{name}.pt"
            shutil.move(tmp_pt, os.path.join(out, model_file))
        else:
            import torch
            import ultralytics.cfg as ucfg
            args = dict(format="onnx", imgsz=int(cfg["imgsz"]), simplify=True,
                        device=0 if torch.cuda.is_available() else "cpu")
            if fmt == "ONNX INT8":
                args["data"] = data_yaml
                # 8.4 이후는 quantize=8, 그 전 버전은 int8=True
                if "quantize" in ucfg.DEFAULT_CFG_DICT:
                    args["quantize"] = 8
                else:
                    args["int8"] = True
            _log(f"  Ultralytics export 호출: {args}")
            res = type(model)(tmp_pt).export(**args)   # type(model): 위에서 연 것과 같은 클래스(YOLO/RTDETR)로 연다
            res = str(res)
            if not os.path.isfile(res):
                raise RuntimeError(f"export가 파일을 만들지 못했습니다: {res}")
            model_file = f"{name}_int8.onnx" if fmt == "ONNX INT8" else f"{name}.onnx"
            shutil.move(res, os.path.join(out, model_file))

    with open(os.path.join(out, f"{name}_config.json"), "w", encoding="utf-8") as f:   # INT8 도 FP32 와 같은 이름
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    if fmt != "PyTorch":
        _embed_onnx_config(os.path.join(out, model_file), cfg)
        _log("  config를 ONNX 메타데이터에 저장")

    _save_legend(os.path.join(out, "legend.png"), names, colors)

    # Results 요약 복사
    summary, results_files = None, []
    res_dir = os.path.join(ckpt, "Results")
    for fn in ("summary.json", "confusion_matrix.png", "per_class_metrics.png"):
        src = os.path.join(res_dir, fn)
        if os.path.isfile(src):
            os.makedirs(os.path.join(out, "Results"), exist_ok=True)
            shutil.copy2(src, os.path.join(out, "Results", fn))
            results_files.append(fn)
    if "summary.json" in results_files:
        summary = load_config(os.path.join(res_dir, "summary.json"))   # json을 읽는 함수라 그대로 쓴다
    if not results_files:
        _log("  [알림] Results/ 가 없어 summary.json, 그림은 빠집니다 (평가 탭 실행 후 다시 내보내세요)")

    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(_readme(cfg, name, fmt, model_file, names, colors, summary, results_files))
    _log(f"내보내기 완료: {out}")
    return out


def find_exported_model(folder):
    """내보낸 폴더에서 모델 파일을 찾아 (경로, 형식 번호)를 돌려준다. 형식 번호는 PyTorch 0, ONNX 1, ONNX INT8 2.
    모델이 없으면 (None, None)."""
    for pattern, idx in (("*_int8.onnx", 2), ("*.onnx", 1), ("*.pt", 0)):
        for p in sorted(glob.glob(os.path.join(folder, pattern))):
            if idx == 1 and p.endswith("_int8.onnx"):
                continue
            return p, idx
    return None, None
