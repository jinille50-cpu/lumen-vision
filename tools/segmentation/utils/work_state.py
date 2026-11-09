"""현재 작업 상태: 모델 경로, config 경로, 이미지/라벨 폴더를 한 곳에 둔다.
평가, 추론, 경량화 탭이 이 값으로 칸을 미리 채우고, 한 탭에서 바꾸면 changed 신호로 다른 탭도 따라 바뀐다."""
from PySide6.QtCore import QObject, Signal

from tools.segmentation.utils.config_path import find_config

# onnx_path: 경량화에서 만든 FP32 .onnx. 작업 모델(model_path, .pt)과 따로 저장한다
# img_ext: 데이터 탭에서 고른 이미지 확장자. 평가/추론/경량화가 이 값으로 미리 채운다
KEYS = ("model_path", "onnx_path", "config_path", "img_dir", "lbl_dir", "img_ext")


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
            self._v["config_path"] = cfg   # 모델 신호보다 먼저 저장해 두어야 받는 쪽이 모델과 짝이 맞는 config를 읽는다
        self.set("model_path", path)
        if cfg_changed:
            self.changed.emit("config_path")
