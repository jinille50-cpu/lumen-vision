"""
onnx_export.py — 저장된 .pth + _config.json 으로 .onnx 를 만들고 바로 검증한다.

학습 중의 모델 객체를 쓰지 않고 json 으로 모델을 새로 만들어 .pth 를 불러온다.
이렇게 해야 "저장된 파일만으로 모델을 복원할 수 있는지"까지 함께 확인된다.
"""

import json
import os
import warnings

# 출력 차이 허용치: 최대 절대 오차 / max(출력 최댓값, 1)
DIFF_TOLERANCE = 1e-3


def _input_hw(cfg):
    """json 의 orig_size 를 pad_multiple 배수로 올림한 (H, W). 없으면 256 기준."""
    multiple = int(cfg.get("pad_multiple") or 32)
    h, w = cfg.get("orig_size") or (256, 256)
    # -(-a // b) 는 a / b 의 올림 나눗셈 (MATLAB 의 ceil(a/b) 와 같음)
    return -(-int(h) // multiple) * multiple, -(-int(w) // multiple) * multiple


def _compare(session, wrapper, x):
    """같은 입력으로 PyTorch / ONNX 출력을 비교해 상대 오차를 돌려준다."""
    import numpy as np
    import torch
    with torch.no_grad():
        pt = wrapper(x).numpy()
    ox = session.run(None, {"input": x.numpy()})[0]
    if pt.shape != ox.shape:
        raise ValueError(f"출력 모양 다름: PyTorch {pt.shape} / ONNX {ox.shape}")
    return float(np.abs(pt - ox).max() / max(float(np.abs(pt).max()), 1.0))


def check_onnx_packages():
    """ONNX 내보내기에 필요한 패키지를 확인한다. 없으면 설치 명령이 담긴 메시지로 실패시킨다.
    (torch.onnx.export 는 onnx 패키지가 없으면 .onnx 를 만들지 못한다. onnxruntime 만으로는 부족)"""
    import importlib.util
    missing = [n for n in ("onnx", "onnxruntime") if importlib.util.find_spec(n) is None]
    if missing:
        raise RuntimeError(f"{', '.join(missing)} 패키지가 설치되어 있지 않습니다 -> pip install {' '.join(missing)}")


def export_onnx(pth_path, log=None):
    """pth_path 옆의 _config.json 으로 모델을 만들어 같은 이름의 .onnx 로 내보낸다.

    반환: {"path", "diff", "resized_diff", "ok"}
    ok=False 면 파일은 만들어졌지만 PyTorch 와 출력이 달라 믿을 수 없다는 뜻.
    """
    check_onnx_packages()
    import onnxruntime as ort
    import torch
    import torch.nn as nn
    from tools.segmentation.utils.model_builder import build_model, infer_segformer_decoder_hidden

    def _log(msg):
        if log:
            log(msg)

    base = os.path.splitext(pth_path)[0]
    from tools.segmentation.utils.config_path import find_config
    cfg_path = find_config(pth_path)   # best_config.json, 없으면 예전 config.json
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_name = cfg["model_name"]
    if "Mask2Former" in model_name:
        raise ValueError("Mask2Former 는 출력이 텐서 하나가 아니라 ONNX 내보내기를 지원하지 않습니다")

    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    model, is_tr = build_model(
        model_name, cfg.get("encoder", "resnet34"), cfg["in_channels"], cfg["num_classes"],
        pretrained=cfg.get("pretrained"), load_pretrained_weights=False,
        decoder_hidden_size=infer_segformer_decoder_hidden(state),
    )
    # strict=True: 키가 하나라도 안 맞으면 무작위 가중치가 섞인 ONNX 가 되므로 바로 실패시킨다
    model.load_state_dict(state, strict=True)
    model.eval()

    class LogitsOnly(nn.Module):
        """transformers 모델은 출력이 객체이므로 logits 텐서만 꺼낸다."""
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            return out.logits if is_tr else out

    wrapper = LogitsOnly().eval()
    h, w = _input_hw(cfg)
    multiple = int(cfg.get("pad_multiple") or 32)
    in_ch = cfg["in_channels"]
    torch.manual_seed(0)
    x = torch.rand(1, in_ch, h, w)

    onnx_path = base + ".onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # 추적(trace) 경고가 로그를 덮지 않도록
        torch.onnx.export(
            wrapper, (x,), onnx_path,
            input_names=["input"], output_names=["logits"],
            dynamic_axes={
                "input":  {0: "batch", 2: "height", 3: "width"},
                "logits": {0: "batch", 2: "out_height", 3: "out_width"},
            },
            opset_version=17,
            dynamo=False,   # onnxscript 없이 동작하는 TorchScript 방식
        )

    from tools.segmentation.utils.config_path import embed_onnx_config
    embed_onnx_config(onnx_path, cfg)   # config 를 ONNX 안에 넣어 파일 하나로 쓸 수 있게

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    diff = _compare(session, wrapper, x)
    # 동적 축이 실제로 동작하는지: 배치와 이미지 크기를 바꿔 한 번 더 비교
    resized_diff = None
    try:
        resized_diff = _compare(session, wrapper,
                                torch.rand(2, in_ch, h + multiple, w + 2 * multiple))
    except Exception as e:
        _log(f"  [경고] ONNX 동적 크기 입력 실패: {e}")

    ok = (diff <= DIFF_TOLERANCE
          and resized_diff is not None and resized_diff <= DIFF_TOLERANCE)
    if diff > DIFF_TOLERANCE:
        _log(f"  [경고] PyTorch와 ONNX 출력 차이가 큽니다: {diff:.2e} (허용 {DIFF_TOLERANCE:.0e})")
    elif resized_diff is not None and resized_diff > DIFF_TOLERANCE:
        _log(f"  [경고] 다른 입력 크기에서 ONNX 출력이 다릅니다: {resized_diff:.2e} "
             f"→ 학습 크기({h}x{w})로만 쓰세요")
    if ok:
        _log(f"  ONNX 저장: {onnx_path} (출력 차이 {diff:.2e})")
    else:
        _log(f"  ONNX 저장(검증 경고 있음): {onnx_path}")
    return {"path": onnx_path, "diff": diff, "resized_diff": resized_diff, "ok": ok}
