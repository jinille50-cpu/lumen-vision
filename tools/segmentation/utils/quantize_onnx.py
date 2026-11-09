"""
quantize_onnx.py — ONNX Runtime 정적 양자화(INT8)와 FP32/INT8 비교 평가.

평가는 tabs/eval_tab.EvalWorker 를 그대로 쓴다. EvalWorker 는 model(입력) 을 호출할 뿐이므로
모델 자리에 ONNX 세션을 감싼 객체(OnnxModel)를 넣으면 전처리, 패딩, 지표 계산이 평가 탭과 같아진다.
"""

import csv
import glob
import os
import random
import tempfile
import time

# 가중치를 가진 연산. 출력 헤드를 찾을 때 여기서 멈춘다.
WEIGHTED_OPS = {"Conv", "ConvTranspose", "Gemm", "MatMul"}
# 여러 입력이 모두 데이터인 연산. 나머지(Resize 등)는 첫 입력만 데이터로 본다.
MULTI_DATA_OPS = {"Add", "Mul", "Sum", "Concat"}
CALIB_SEED = 42     # 같은 설정이면 같은 캘리브레이션 이미지를 고르도록 고정


def read_config(onnx_path):
    """FP32 .onnx 의 config 를 읽는다 (메타데이터 -> 옆의 _config.json). 전처리와 모델 정보가 여기에 있다."""
    from tools.segmentation.utils.config_path import load_config
    cfg = load_config(onnx_path)
    if cfg is None:
        raise FileNotFoundError(f"config 를 찾을 수 없습니다: {onnx_path}")
    return cfg


def _pad_multiple(cfg):
    from tools.segmentation.utils.pad_utils import get_pad_multiple
    return int(cfg.get("pad_multiple") or get_pad_multiple(cfg.get("model_name", "")))


def list_layers(onnx_path):
    """양자화할 수 있는 레이어를 그래프 순서대로 [(이름, 연산 종류)] 로 돌려준다."""
    import onnx
    from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry
    # quantize_static 은 두 목록의 합집합을 양자화한다. QDQRegistry 만 보면 Add, Mul, Concat,
    # Softmax 가 빠져서, 제외할 수 없는데도 양자화된다 (DINOv3 의 LayerScale gamma 등)
    quantizable = set(QDQRegistry) | set(QLinearOpsRegistry)
    graph = onnx.load(onnx_path, load_external_data=False).graph
    return [(n.name, n.op_type) for n in graph.node if n.op_type in quantizable]


def default_excluded(onnx_path):
    """첫 Conv 와 출력 헤드(출력에서 거슬러 올라가 처음 만나는 가중치 레이어까지)의 이름들."""
    import onnx
    graph = onnx.load(onnx_path, load_external_data=False).graph
    producer = {out: node for node in graph.node for out in node.output}

    excluded = set()
    for node in graph.node:
        if node.op_type == "Conv":
            excluded.add(node.name)
            break

    stack = [o.name for o in graph.output]
    seen = set()
    while stack:     # 깊이 우선 탐색: 출력 → 입력 방향으로 올라간다
        node = producer.get(stack.pop())
        if node is None or node.name in seen:
            continue
        seen.add(node.name)
        excluded.add(node.name)
        if node.op_type in WEIGHTED_OPS:
            continue
        stack.extend(node.input if node.op_type in MULTI_DATA_OPS else node.input[:1])
    return excluded


def _load_input(path, in_scale, multiple, channels=None):
    """eval_tab.EvalWorker._run_eval 과 같은 전처리로 (1, C, H, W) numpy 입력을 만든다."""
    import numpy as np
    import scipy.io as sio
    import torch
    from PIL import Image
    from tools.segmentation.utils.pad_utils import pad_tensor

    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "mat":
        img = sio.loadmat(path)["image"].astype(np.float32)
    elif ext == "npy":
        img = np.load(path).astype(np.float32)
    else:
        img = np.array(Image.open(path)).astype(np.float32)
    from tools.segmentation.utils.pad_utils import select_channels
    img = select_channels(img, channels)   # config의 입력 채널 선택
    t = torch.tensor(img).float() / in_scale
    if t.ndim == 3 and t.shape[-1] < t.shape[0]:
        t = t.permute(2, 0, 1)
    elif t.ndim == 2:
        t = t.unsqueeze(0)
    return pad_tensor(t.unsqueeze(0), multiple=multiple)[0].numpy()


def pick_calibration_files(img_dir, img_ext, count):
    """폴더에서 count 장을 무작위(시드 고정)로 고른다. 모자라면 전부 쓴다."""
    files = sorted(glob.glob(os.path.join(img_dir, f"*.{img_ext}")))
    if len(files) <= count:
        return files
    return sorted(random.Random(CALIB_SEED).sample(files, count))


def quantize_int8(onnx_path, calib_files, excluded, log=None, out_path=None):
    """정적 양자화 후 <이름>_int8.onnx 경로를 돌려준다. excluded 레이어는 FP32 로 남는다.

    out_path: 저장 경로를 따로 지정 (민감도 분석이 사용자의 _int8.onnx 를 덮어쓰지 않게)
    """
    from onnxruntime.quantization import (
        CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
        quant_pre_process, quantize_static,
    )
    from tools.segmentation.utils.pad_utils import read_input_scale

    def _log(msg):
        if log:
            log(msg)

    cfg = read_config(onnx_path)
    in_scale = read_input_scale(onnx_path)
    multiple = _pad_multiple(cfg)

    class Reader(CalibrationDataReader):
        """ONNX Runtime 이 get_next() 를 None 이 나올 때까지 불러 입력을 한 장씩 받아 간다."""
        def __init__(self):
            self.files = iter(calib_files)

        def get_next(self):
            path = next(self.files, None)   # next(반복자, 기본값): 끝나면 None
            return None if path is None else {"input": _load_input(path, in_scale, multiple, cfg.get("channels"))}

    int8_path = out_path or os.path.splitext(onnx_path)[0] + "_int8.onnx"
    with tempfile.TemporaryDirectory() as tmp:
        prepared = os.path.join(tmp, "prepared.onnx")
        # skip_optimization: 최적화를 하면 레이어 이름이 바뀌어 제외 목록이 안 맞게 된다
        quant_pre_process(onnx_path, prepared, skip_optimization=True, skip_symbolic_shape=True)
        _log(f"  캘리브레이션 {len(calib_files)}장, 제외 레이어 {len(excluded)}개")
        quantize_static(
            prepared, int8_path, Reader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_exclude=sorted(excluded),
        )
    from tools.segmentation.utils.config_path import embed_onnx_config
    embed_onnx_config(int8_path, cfg)   # INT8 도 파일 하나로 쓸 수 있게 config 를 넣는다
    _log(f"  INT8 저장: {int8_path}")
    return int8_path


class OnnxModel:
    """EvalWorker 가 PyTorch 모델처럼 호출하도록 ONNX 세션을 감싸고 추론 시간을 잰다."""

    def __init__(self, onnx_path, providers=None, threads=1):
        import onnxruntime as ort
        # FP32 와 INT8 을 같은 조건(CPU 1스레드)에서 비교한다. 기본 다중 스레드는
        # torch 스레드와 경합해 같은 모델도 5 ms ~ 120 ms 로 흔들려 비교가 안 된다.
        # (추론 탭처럼 시간 비교가 목적이 아니면 threads=0 → onnxruntime 기본값)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(onnx_path, options,
                                            providers=providers or ["CPUExecutionProvider"])
        self.init_s = time.perf_counter() - t0      # 세션 생성 시간 (TensorRT 는 여기서 엔진을 빌드하기도 함)
        self.input_name = self.session.get_inputs()[0].name
        # 실제로 쓰인 실행 장치. providers 로 요청한 것과 다를 수 있어 세션에서 다시 읽는다
        self.provider = self.session.get_providers()[0]
        self.times = []

    def __call__(self, t):
        import torch
        x = t.detach().cpu().numpy()
        start = time.perf_counter()
        out = self.session.run(None, {self.input_name: x})[0]
        self.times.append(time.perf_counter() - start)
        return torch.from_numpy(out).to(t.device)

    def first_ms(self):
        """첫 추론 시간. TensorRT 는 엔진 빌드·로드가 여기에 섞이므로 추론 시간과 따로 본다."""
        return 1000.0 * self.times[0] if self.times else 0.0

    def median_ms(self):
        """장당 추론 시간의 중앙값. 첫 장은 초기화 시간이 섞이므로 뺀다."""
        import numpy as np
        return 1000.0 * float(np.median(self.times[1:] or self.times))


def benchmark_onnx(run_path, config_onnx_path, sample_path, providers=None, threads=0,
                   warmup=3, runs=20):
    """고정 입력 한 장으로 추론 시간을 잰다: 워밍업 warmup 회 버리고 runs 회 측정해 (평균, 표준편차, 실제 장치) ms.

    기본값은 워밍업 3회 + 20회. 정확도 평가 루프의 시간(이미지마다 크기·디스크 읽기 섞임)과 분리했다.
    """
    import numpy as np
    from tools.segmentation.utils.pad_utils import read_input_scale
    cfg = read_config(config_onnx_path)
    x = _load_input(sample_path, read_input_scale(config_onnx_path), _pad_multiple(cfg), cfg.get("channels"))
    model = OnnxModel(run_path, providers=providers, threads=threads)
    feed = {model.input_name: x}
    for _ in range(warmup):
        model.session.run(None, feed)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.session.run(None, feed)
        times.append(1000.0 * (time.perf_counter() - t0))
    return float(np.mean(times)), float(np.std(times)), model.provider


TRT = "TensorrtExecutionProvider"


def available_providers():
    """이 PC의 onnxruntime 이 실제로 쓸 수 있는 실행 방식 목록 (없으면 CPU 만)."""
    try:
        import onnxruntime as ort
        # Azure 는 클라우드 호출용이라 모델 실행 방식이 아니다. CPU 를 맨 앞(기본값)에 둔다
        found = [p for p in ort.get_available_providers() if p != "AzureExecutionProvider"]
        return sorted(found, key=lambda p: p != "CPUExecutionProvider")   # False(CPU)가 먼저
    except Exception:
        return ["CPUExecutionProvider"]


def tensorrt_status():
    """TensorRT 로 평가할 수 있는지 (가능 여부, 불가능한 이유) 를 돌려준다."""
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if TRT in avail:
        return True, ""
    if "CUDAExecutionProvider" not in avail:
        return False, ("TensorRT 라이브러리 설치 필요 — onnxruntime-gpu 가 아니라서 TensorRT 공급자가 없습니다 "
                       f"(사용 가능: {', '.join(avail)})")
    return False, "TensorRT 라이브러리 설치 필요 — 이 onnxruntime 에 TensorRT 공급자가 없습니다 (TensorRT 설치와 버전을 확인하세요)"


def cuda_available():
    """CUDA 공급자를 쓸 수 있는지 (GPU 속도 측정 줄을 넣을지 결정)."""
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def trt_fail_reason(err):
    """TensorRT 세션/평가 실패 예외를 한 줄 원인으로 줄인다. 라이브러리를 못 찾는 경우를 구분한다."""
    msg = str(err).splitlines()[0][:200] if str(err) else type(err).__name__
    if any(k in str(err).lower() for k in ("nvinfer", "tensorrt", "cudnn", "cublas", "loadlibrary", "dll")):
        return f"TensorRT 라이브러리 설치 필요 — {msg}"
    return msg


def trt_providers(onnx_path, int8):
    """TensorRT 평가용 providers 와 엔진 캐시 폴더. 캐시는 .onnx 옆 trt_cache 에 둔다.

    int8=False 는 FP16 기준(FP32 모델), True 는 INT8 (QDQ 모델, 제외한 레이어는 FP16 로 폴백).
    """
    import onnxruntime as ort
    cache = os.path.join(os.path.dirname(os.path.abspath(onnx_path)), "trt_cache")
    options = {
        "trt_fp16_enable": True,
        "trt_int8_enable": bool(int8),
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache,
    }
    # TensorRT 가 못 받는 노드는 CUDA, 그것도 안 되면 CPU 로 넘어간다
    avail = ort.get_available_providers()
    rest = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    return [(TRT, options)] + rest, cache


def count_engines(cache_dir):
    """캐시 폴더의 엔진 파일 수. 평가 전후로 비교해 새로 빌드했는지 알아낸다."""
    if not os.path.isdir(cache_dir):
        return 0
    return sum(1 for f in os.listdir(cache_dir) if f.endswith(".engine"))


def evaluate_onnx(run_path, config_onnx_path, img_dir, lbl_dir, img_ext, lbl_ext, log=None,
                  trt_int8=None, threads=1):
    """평가 탭의 EvalWorker 로 ONNX 모델을 평가해 (지표 dict, 중앙값 ms) 를 돌려준다.

    config_onnx_path: 전처리 정보(_config.json)를 읽을 FP32 .onnx 경로.
    run_path 가 _int8.onnx 여도 FP32 와 같은 전처리를 쓰게 하려고 따로 받는다.
    trt_int8: None 이면 CPU, False/True 면 TensorRT FP16 / INT8 로 평가.
    threads: CPU 스레드 수. 시간 비교는 1 (기본), 정확도만 볼 때는 0 (onnxruntime 기본값)
    """
    from tools.segmentation.tabs.eval_tab import EvalWorker

    cfg = read_config(config_onnx_path)
    if trt_int8 is None:
        model = OnnxModel(run_path, threads=threads)
    else:
        providers, cache = trt_providers(run_path, trt_int8)
        engines_before = count_engines(cache)
        if log:
            log(f"  TensorRT {'INT8' if trt_int8 else 'FP16'} 세션 생성 중 "
                f"(엔진 캐시: {cache}) — 첫 실행은 엔진 빌드로 오래 걸립니다")
        model = OnnxModel(run_path, providers=providers, threads=0)
        if model.provider != TRT:       # 조용히 CPU 로 평가되어 TensorRT 라고 표시되는 것을 막는다
            raise RuntimeError(f"TensorRT 공급자를 불러오지 못했습니다 (실제: {model.provider})")
    if log:
        log(f"  실행 장치: {model.provider}")

    class OnnxEvalWorker(EvalWorker):
        def _load_model(self, *args):
            return model, False     # is_tr=False: 출력이 logits 텐서

    worker = OnnxEvalWorker({
        "model_path": config_onnx_path, "model_name": cfg["model_name"],
        "img_dir": img_dir, "lbl_dir": lbl_dir, "img_ext": img_ext, "lbl_ext": lbl_ext,
        "in_channels": cfg["in_channels"], "channels": cfg.get("channels"), "num_classes": cfg["num_classes"],
        "encoder": cfg.get("encoder", "resnet34"), "pad_multiple": cfg.get("pad_multiple"),
    })
    results = {}
    worker.result_ready.connect(results.update)
    if log:
        worker.log_msg.connect(log)
    worker._run_eval()      # 이미 작업 스레드 안이므로 start() 없이 바로 실행
    if not results:
        raise RuntimeError(f"평가 결과가 없습니다: {os.path.basename(run_path)}")
    results["실행 장치"] = model.provider
    ms = model.median_ms()
    if log:
        log(f"  추론 시간 {ms:.2f} ms/장 ({model.provider})")
        if trt_int8 is not None:
            # 엔진 빌드 시간은 추론 시간(중앙값, 첫 장 제외)에 섞이지 않게 따로 적는다
            built = count_engines(cache) - engines_before
            state = f"엔진 {built}개 새로 빌드" if built > 0 else "캐시된 엔진 사용 (빌드 건너뜀)"
            log(f"  엔진 준비 시간 (추론 시간에서 제외): 세션 생성 {model.init_s:.1f}초 + "
                f"첫 추론 {model.first_ms() / 1000:.1f}초 — {state}")
    return results, ms


def file_mb(path):
    """모델 크기(MB). 2 GB 넘는 모델의 외부 가중치 파일도 합산한다."""
    size = os.path.getsize(path)
    if os.path.isfile(path + ".data"):
        size += os.path.getsize(path + ".data")
    return size / (1024 * 1024)


COLUMNS = ["모델", "Pixel Accuracy", "mIoU", "F1 Score (Dice)", "추론 시간 (ms/장, 평균±표준편차)", "크기 (MB)", "실행 장치"]
TIME_COL = COLUMNS[4]
NUMERIC = [c for c in COLUMNS[1:-1] if c != TIME_COL]    # 변화(차이)를 계산하는 숫자 열 (시간은 문자열이라 따로)


def _row(name, path, res, speed, device):
    """speed: (평균 ms, 표준편차 ms). 평균은 _ms 에 숫자로 따로 둔다 (변화 계산용, 표에는 안 나옴)."""
    mean, std = speed
    return {"모델": name, "Pixel Accuracy": res["Pixel Accuracy"], "mIoU": res["mIoU"],
            "F1 Score (Dice)": res["F1 Score (Dice)"], TIME_COL: f"{mean:.2f} ± {std:.2f}", "_ms": mean,
            "크기 (MB)": file_mb(path), "실행 장치": device}


def failed_row(name, reason):
    """측정하지 못한 줄: 원인을 실행 장치 칸에 적는다."""
    row = {c: "-" for c in COLUMNS}
    row["모델"] = name
    row["실행 장치"] = reason
    return row


def compare_rows(fp32_path, int8_path, fp32, int8, speeds, threads, gpu_speeds=None):
    """비교표 행: CPU FP32, INT8, 변화(INT8 - FP32), (CUDA 가 있으면) GPU FP32, INT8.

    fp32/int8: 정확도 평가 결과 dict. speeds/gpu_speeds: {"FP32": (평균, 표준편차), "INT8": ...}.
    GPU 줄의 정확도는 CPU 평가와 같은 값을 쓰고 시간만 GPU 로 잰다.
    """
    dev = f"CPU ({threads}스레드)"
    a = _row("FP32 (CPU)", fp32_path, fp32, speeds["FP32"], dev)
    b = _row("INT8 (CPU)", int8_path, int8, speeds["INT8"], dev)
    delta = {k: b[k] - a[k] for k in NUMERIC}
    delta[TIME_COL] = f"{b['_ms'] - a['_ms']:+.2f}"
    delta["모델"] = "변화 (INT8 - FP32)"
    delta["실행 장치"] = ""
    rows = [a, b, delta]
    if gpu_speeds:
        rows.append(_row("FP32 (GPU)", fp32_path, fp32, gpu_speeds["FP32"][:2], gpu_speeds["FP32"][2]))
        rows.append(_row("INT8 (GPU)", int8_path, int8, gpu_speeds["INT8"][:2], gpu_speeds["INT8"][2]))
    return rows


def trt_rows(fp32_path, int8_path, fp16, int8):
    """TensorRT 행: FP32 기준을 FP16 으로 돌린 줄과 INT8 줄. fp16/int8 은 (지표 dict, 평균, 표준편차, 장치)."""
    return [_row("TensorRT FP16", fp32_path, fp16[0], fp16[1:3], "TensorRT (FP16)"),
            _row("TensorRT INT8", int8_path, int8[0], int8[1:3], "TensorRT (INT8, FP16 폴백)")]


def save_csv(path, rows):
    # utf-8-sig: 엑셀에서 한글이 깨지지 않게 BOM 을 붙인다
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")   # _ms 는 저장 안 함
        writer.writeheader()
        writer.writerows(rows)
