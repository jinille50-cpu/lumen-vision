"""config 파일 위치 규칙 한 곳. 모델 파일과 같은 이름 + _config.json (best.pt -> best_config.json).
읽을 때는 예전 방식(폴더의 config.json)도 같이 찾아서 이미 학습한 모델도 열린다."""
import os


def model_base(model_path):
    """모델 파일 경로에서 확장자를 뗀 기본 경로. ONNX INT8(_int8.onnx)은 FP32와 같은 이름으로 본다."""
    base = os.path.splitext(model_path)[0]
    if model_path.lower().endswith(".onnx") and base.endswith("_int8"):
        base = base[:-5]
    return base


def config_path_for(model_path):
    """새로 저장할 때 쓰는 경로"""
    return model_base(model_path) + "_config.json"


def find_config(model_path):
    """있는 config 경로를 찾는다 (새 이름 -> 예전 폴더 config.json). 없으면 ""."""
    if not model_path:
        return ""
    for p in (config_path_for(model_path),
              os.path.join(os.path.dirname(model_path), "config.json")):
        if os.path.isfile(p):
            return p
    return ""


# ── ONNX 메타데이터 ─────────────────────────────────────────
# ONNX 파일 안(metadata_props)에 config 전체를 JSON 글자로 넣어 파일 하나로 쓸 수 있게 한다.
ONNX_META_KEY = "seg_config"


def embed_onnx_config(onnx_path, cfg):
    """cfg(dict)를 onnx 파일의 메타데이터에 저장한다. 같은 키가 있으면 덮어쓴다."""
    import json
    import onnx
    model = onnx.load(onnx_path)    # 외부 데이터 없이 한 파일에 담긴 모델 기준
    for p in [p for p in model.metadata_props if p.key == ONNX_META_KEY]:
        model.metadata_props.remove(p)
    entry = model.metadata_props.add()
    entry.key, entry.value = ONNX_META_KEY, json.dumps(cfg, ensure_ascii=False)
    onnx.save(model, onnx_path)


def read_onnx_config(onnx_path):
    """onnx 메타데이터의 config(dict). 없거나 읽을 수 없으면 None."""
    import json
    try:
        import onnx
        model = onnx.load(onnx_path, load_external_data=False)
        for p in model.metadata_props:
            if p.key == ONNX_META_KEY:
                return json.loads(p.value)
    except Exception:
        pass
    return None


def load_config(model_path):
    """config dict를 찾는다: ONNX 메타데이터 -> 옆의 _config.json -> 예전 config.json. 없으면 None."""
    import json
    if model_path and model_path.lower().endswith(".onnx") and os.path.isfile(model_path):
        cfg = read_onnx_config(model_path)
        if cfg is not None:
            return cfg
    path = find_config(model_path)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
