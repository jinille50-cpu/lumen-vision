"""현재 작업 상태 + config 이름 규칙 + 선택창 시작 위치 (YOLO용, Segmentation의 utils와 같은 방식).
- WorkState: 모델 경로, config 경로, data.yaml, 이미지/라벨 폴더를 한 곳에 둔다. 한 탭에서 바꾸면 changed 신호로 다른 탭도 따라 바뀐다.
- config: 모델 파일과 같은 이름 + _config.json (best.pt -> best_config.json). 읽을 때 예전 config.json도 찾는다.
- pick_*: 시작 위치는 칸의 기존 경로 -> 데이터 탭 폴더 -> 마지막 연 폴더 -> 기본 위치"""
import json
import os

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog

from common import settings

TOOL = "yolo"
KEYS = ("model_path", "config_path", "data_yaml", "img_dir", "lbl_dir")


# ── config 파일 ─────────────────────────────────────────────
def config_path_for(model_path):
    """새로 저장할 때 쓰는 경로"""
    return os.path.splitext(model_path)[0] + "_config.json"


def find_config(model_path):
    """있는 config 경로 (새 이름 -> 예전 이름들). 없으면 ""."""
    if not model_path:
        return ""
    folder = os.path.dirname(model_path)
    run_dir = os.path.dirname(folder)   # best.pt는 <run>/weights/ 안에 있다
    for p in (config_path_for(model_path),
              config_path_for(model_path[:-len("_int8.onnx")] + ".x") if model_path.endswith("_int8.onnx") else "",   # INT8 onnx는 FP32와 같은 config
              os.path.join(folder, "config.json"),
              os.path.join(run_dir, "config.json"),
              os.path.join(os.path.dirname(run_dir), "config.json")):
        if p and os.path.isfile(p):
            return p
    return ""


def load_config(path):
    """config dict. 없거나 읽을 수 없으면 None."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_config(model_path, cfg):
    """모델 옆에 모델이름_config.json으로 저장하고 경로를 돌려준다."""
    p = config_path_for(model_path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return p


# ── 작업 상태 ───────────────────────────────────────────────
class WorkState(QObject):
    changed = Signal(str)   # 바뀐 키 이름

    def __init__(self):
        super().__init__()
        self._v = {k: "" for k in KEYS}

    def get(self, key):
        return self._v.get(key, "")

    def set(self, key, value):
        value = (value or "").strip()
        if self._v[key] == value:   # 같은 값이면 알리지 않는다 (탭끼리 신호가 돌고 도는 것 방지)
            return
        self._v[key] = value
        self.changed.emit(key)

    def set_model(self, path):
        """모델을 바꾼다. 같은 이름의 config(또는 예전 config.json)가 있으면 config도 함께 바꾼다."""
        path = (path or "").strip()
        cfg = find_config(path)
        cfg_changed = bool(cfg) and cfg != self._v["config_path"]
        if cfg_changed:
            self._v["config_path"] = cfg   # 모델 신호보다 먼저 저장해 받는 쪽이 짝이 맞는 config를 읽게 한다
        self.set("model_path", path)
        if cfg_changed:
            self.changed.emit("config_path")


# ── 폴더/파일 선택창 ────────────────────────────────────────
def _existing_dir(path):
    """경로가 파일이면 그 폴더, 없는 경로면 존재하는 가장 가까운 상위 폴더. 못 찾으면 ''"""
    path = (path or "").strip()
    while path:
        if os.path.isdir(path):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return ""


def start_dir(current="", state=None):
    state = state or {}
    for cand in (current, state.get("train_img_dir"), state.get("train_lbl_dir"),
                 settings.get_value(TOOL, "last_dir", "")):
        d = _existing_dir(cand)
        if d:
            return d
    return ""   # 빈 문자열이면 Qt가 기본 위치를 쓴다


def save_last(key, values):
    """탭의 마지막 설정값(dict)을 JSON 한 덩어리로 저장한다.
    값마다 따로 저장하면 레지스트리에서 True/False가 글자 'true'로 돌아오므로 묶어서 저장"""
    settings.set_value(TOOL, key, json.dumps(values, ensure_ascii=False))


def load_last(key):
    """save_last로 저장한 dict. 없거나 깨졌으면 빈 dict"""
    try:
        return json.loads(settings.get_value(TOOL, key, "") or "{}")
    except (TypeError, ValueError):
        return {}


def _remember(path):
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    if folder:
        settings.set_value(TOOL, "last_dir", folder)


def pick_dir(parent, current="", state=None, title="폴더 선택"):
    path = QFileDialog.getExistingDirectory(parent, title, start_dir(current, state))
    if path:
        _remember(path)
    return path


def pick_open(parent, title, filt, current="", state=None):
    path, _ = QFileDialog.getOpenFileName(parent, title, start_dir(current, state), filt)
    if path:
        _remember(path)
    return path
