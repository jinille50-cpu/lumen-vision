"""홈 화면용 작업 환경 정보: GPU, SAM 3 경로, Python 버전, 내 모델 목록.
torch 같은 무거운 라이브러리는 불러오지 않는다 (nvidia-smi 호출만 사용)."""
import ast
import json
import os
import re
import subprocess
import sys

from . import settings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(ROOT, "models", "weights")
MODEL_EXTS = (".pt", ".pth", ".onnx")
BACKBONE_SHORT = {"wide_resnet50_2": "WRN50", "resnet18": "R18", "resnet34": "R34", "resnet50": "R50",
                  "resnext50_32x4d": "RX50"}  # PatchCore 백본 표시 줄임말


def gpu_info():
    """(GPU 이름, VRAM GB) 또는 None. nvidia-smi가 없거나 실패하면 None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        name, mib = [x.strip() for x in out.split(",")]
        return name, int(mib) / 1024
    except Exception:  # noqa: BLE001  -- 없으면 없다고 표시하면 되므로 모든 실패를 같이 처리
        return None


def sam3_path():
    """사용자가 'SAM 3 로드'로 고른 sam3.pt 경로(안 골랐으면 빈 문자열)와 파일이 있는지.
    모든 도구가 이 경로를 같이 쓴다."""
    path = str(settings.get_value("common", "sam3_path", "") or "")
    return path, bool(path) and os.path.isfile(path)


def set_sam3_path(path):
    settings.set_value("common", "sam3_path", str(path))


def python_version():
    return "%d.%d.%d" % sys.version_info[:3]


def _read_literal(path, var):
    """파이썬 파일을 실행하지 않고(ast) 최상위 `var = <리터럴>` 값을 읽는다. 못 읽으면 None.
    각 도구는 PyQt5/torch를 import 하므로 직접 import 하지 않고 소스만 읽는다."""
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == var for t in node.targets):
                return ast.literal_eval(node.value)
    except (OSError, SyntaxError, ValueError):
        pass
    return None


def _group(names):
    """['UNet', 'UNet++', 'SegFormer-B0', ...] -> [(계열, [(변형 또는 None, 원래 이름)])] (등장 순서 유지).
    'X++'/'X+'는 X의 변형, 'X-B0'처럼 끝이 짧은 조각이면 X의 변형. 계열의 기본형은 변형 '기본'."""
    fam = {}
    for n in names:
        if n.endswith("++") or n.endswith("+"):
            base, var = n.rstrip("+"), n[len(n.rstrip("+")):]
        else:
            m = re.fullmatch(r"(.+?)-(\w{1,3})", n)
            base, var = (m.group(1), m.group(2)) if m else (n, None)
        fam.setdefault(base, []).append((var, n))
    out = []
    for base, vs in fam.items():
        if len(vs) > 1:
            vs = [("기본" if v is None else v, n) for v, n in vs]
        out.append((base, vs))
    return out


def _trained_seg_names():
    """models/weights 아래 *_config.json(학습 때 저장되는 구조 정보)의 model_name 모음."""
    found = set()
    for dp, _, fs in os.walk(WEIGHTS_DIR):
        for f in fs:
            if f.endswith("_config.json"):
                try:
                    with open(os.path.join(dp, f), encoding="utf-8") as fh:
                        found.add(json.load(fh).get("model_name", ""))
                except (OSError, ValueError):
                    pass
    return found


def model_catalog():
    """도구가 실제로 지원하는 모델 계열/변형 전체와 학습된 가중치 유무.
    돌려주는 값: {"yolo"|"segmentation"|"patchcore": [(계열 이름, [(변형 이름, 가중치 있음 여부)])]}
    - YOLO: tools/yolo/tabs/train_tab.py 의 YOLO_VERSIONS x YOLO_SIZES(계열별 예외는 MODEL_SIZES), models/weights 에 해당 .pt 가 있으면 있음
    - Segmentation: tools/segmentation/utils/model_builder.py 의 ALL_MODELS, weights 의 *_config.json model_name 과 일치하면 있음
    - PatchCore: tools/patchcore/core.py 의 BACKBONES, 데이터 폴더 옆 checkpoints/patchcore_<백본>_* 에 model.ckpt 가 있으면 있음"""
    tools = os.path.join(ROOT, "tools")
    files = set(os.listdir(WEIGHTS_DIR)) if os.path.isdir(WEIGHTS_DIR) else set()
    cat = {}

    versions = _read_literal(os.path.join(tools, "yolo", "tabs", "train_tab.py"), "YOLO_VERSIONS") or {}
    sizes = _read_literal(os.path.join(tools, "yolo", "tabs", "train_tab.py"), "YOLO_SIZES") or []
    fam_sizes = _read_literal(os.path.join(tools, "yolo", "tabs", "train_tab.py"), "MODEL_SIZES") or {}   # RT-DETR은 l, x
    cat["yolo"] = [(fam, [(sz, pat.format(sz) in files) for sz in fam_sizes.get(fam, sizes)])
                   for fam, pat in versions.items()]

    names = _read_literal(os.path.join(tools, "segmentation", "utils", "model_builder.py"), "ALL_MODELS") or []
    trained = _trained_seg_names()
    cat["segmentation"] = [(fam, [(v, n in trained) for v, n in vs]) for fam, vs in _group(names)]
    # 변형이 하나뿐인 계열(변형 이름 None)은 괄호 없이 표시하도록 변형 이름을 None 그대로 둔다

    backbones = _read_literal(os.path.join(tools, "patchcore", "core.py"), "BACKBONES") or []
    trained = _trained_patchcore_backbones(files)
    cat["patchcore"] = ([("PatchCore", [(BACKBONE_SHORT.get(b, b), b in trained) for b in backbones])]
                        if backbones else [])
    return cat


def _trained_patchcore_backbones(weight_files):
    """학습된 PatchCore 백본 이름 모음. PatchCore는 체크포인트를 <정상 이미지 폴더 상위>/checkpoints/
    patchcore_<백본>_<날짜-시각>/model.ckpt 에 저장하므로, 설정에 남은 PatchCore 데이터 폴더(마지막 폴더,
    최근 목록)의 checkpoints/ 를 본다. models/weights 에 넣어 둔 patchcore_<백본>*.ckpt 도 센다."""
    run_re = re.compile(r"patchcore_(.+)_\d{8}-\d{6}$")   # 백본 이름에도 _ 가 있어 끝의 날짜-시각만 떼어낸다
    found = set()
    for f in weight_files:
        m = re.match(r"patchcore_(.+?)(_\d{8}-\d{6})?\.ckpt$", f, re.I)
        if m:
            found.add(m.group(1))
    dirs = {str(settings.get_value("patchcore", "normal_dir", "") or "")}
    dirs |= {i.get("path", "") for i in settings.recent_items() if i.get("tool") == "patchcore"}
    for d in dirs:
        if not d:
            continue
        ck = os.path.join(os.path.dirname(os.path.abspath(d.rstrip("/\\"))), "checkpoints")
        if not os.path.isdir(ck):
            continue
        for run in os.listdir(ck):
            m = run_re.match(run)
            if m and os.path.isfile(os.path.join(ck, run, "model.ckpt")):
                found.add(m.group(1))
    return found
