"""
sensitivity.py — INT8 양자화 민감도 분석 (B-015).

1) layer_sqnr    : FP32 와 전체 INT8 모델의 레이어별 출력 오차(SQNR, dB). 낮을수록 오차가 크다.
2) run_blocks    : 레이어를 블록으로 묶어 블록 하나만 INT8 로 바꿨을 때의 mIoU 하락.
3) run_auto      : 전부 INT8 → 첫 Conv·출력 헤드만 FP32 → 민감도 순서에서 FP32 개수를 이분 탐색해 목표 안에 들어오는 최소 개수를 찾고,
                   그 뒤 '줄이기' 단계에서 제외한 레이어를 하나씩 INT8 로 되돌려 목표가 유지되면 INT8 로 남긴다.

평가는 quantize_onnx.evaluate_onnx 를 그대로 써서 평가 탭과 같은 전처리·지표를 쓴다.
"""

import csv
import math
import os
import random
import shutil
import tempfile
from collections import OrderedDict

from tools.segmentation.utils.quantize_onnx import (
    CALIB_SEED, _load_input, _pad_multiple, evaluate_onnx, list_layers,
    pick_calibration_files, quantize_int8, read_config,
)

SQNR_CAP = 100.0        # 오차가 0 이면 SQNR 이 무한대이므로 이 값으로 자른다
TOP_N = 30              # 표와 그래프에 보여줄 레이어 수
CALIB_N = 20            # 민감도 분석에 쓰는 캘리브레이션·SQNR 이미지 수
VAL_N = 50              # mIoU 측정에 쓰는 검증 이미지 수


class Stopped(Exception):
    """사용자가 중지를 눌렀을 때 분석을 빠져나오는 데 쓴다."""


def sqnr_db(signal, noise):
    """10*log10(신호 에너지 / 오차 에너지)."""
    if noise <= 0 or signal <= 0:
        return SQNR_CAP
    return min(SQNR_CAP, 10.0 * math.log10(signal / noise))


# ═══════════════════════════════════════════════════════════════
#  블록 묶기
# ═══════════════════════════════════════════════════════════════
def group_blocks(names, min_blocks=8, max_blocks=60):
    """레이어 이름의 앞부분(경로 '/' 기준)으로 블록을 묶는다. {블록: [레이어 이름]} (그래프 순서).

    앞에서 몇 단계까지 볼지는 블록이 min_blocks 개 이상이 되는 가장 얕은 깊이로 자동 결정한다
    (단, max_blocks 를 넘으면 한 단계 덜 자른다). 이름에 '/' 가 없으면 순서대로 20구간으로 나눈다.
    """
    parts = []
    for n in names:
        p = [x for x in n.strip("/").split("/") if x]
        parts.append(p[:-1] or p)       # 마지막 조각은 Conv, MatMul 같은 연산 이름이므로 뺀다

    max_depth = max((len(p) for p in parts), default=1)
    if max_depth <= 1 and all("/" not in n for n in names):
        size = max(1, math.ceil(len(names) / 20))
        blocks = OrderedDict()
        for i, n in enumerate(names):
            blocks.setdefault(f"구간 {i // size + 1}", []).append(n)
        return blocks

    def build(depth):
        blocks = OrderedDict()
        for n, p in zip(names, parts):
            blocks.setdefault("/".join(p[:depth]), []).append(n)
        return blocks

    chosen = build(1)
    for depth in range(1, max_depth + 1):
        blocks = build(depth)
        if len(blocks) > max_blocks and depth > 1:
            break
        chosen = blocks
        if len(blocks) >= min_blocks:
            break
    return chosen


# ═══════════════════════════════════════════════════════════════
#  검증 부분집합
# ═══════════════════════════════════════════════════════════════
def _link_or_copy(src, dst):
    try:
        os.link(src, dst)           # 하드링크: 복사 없이 같은 파일 (다른 드라이브면 실패)
    except OSError:
        shutil.copy2(src, dst)


def make_val_subset(img_dir, lbl_dir, img_ext, lbl_ext, count, dest):
    """검증 폴더에서 count 쌍을 (시드 고정) 골라 dest 아래 img/lbl 폴더에 모은다. EvalWorker 가 폴더 단위라서."""
    from tools.segmentation.utils.pair_utils import pair_dirs
    imgs, lbls, _ = pair_dirs(img_dir, lbl_dir, img_ext, lbl_ext)
    if not imgs:
        raise RuntimeError(f"검증 이미지 ↔ 라벨 짝이 없습니다: {img_dir} / {lbl_dir}")
    picked = sorted(random.Random(CALIB_SEED).sample(range(len(imgs)), min(count, len(imgs))))
    di, dl = os.path.join(dest, "img"), os.path.join(dest, "lbl")
    os.makedirs(di); os.makedirs(dl)
    for i in picked:
        _link_or_copy(imgs[i], os.path.join(di, os.path.basename(imgs[i])))
        _link_or_copy(lbls[i], os.path.join(dl, os.path.basename(lbls[i])))
    return di, dl, len(picked)


# ═══════════════════════════════════════════════════════════════
#  분석 실행기
# ═══════════════════════════════════════════════════════════════
class SensitivityRunner:
    """분석 한 번에 쓰는 설정과 임시 폴더를 들고 있다. with 문으로 사용한다.

    c: onnx_path, calib_dir, val_img_dir, val_lbl_dir, img_ext, lbl_ext
    """

    def __init__(self, c, log=None, progress=None, stop=None):
        self.c = c
        self._log = log or (lambda msg: None)
        self._progress = progress or (lambda cur, total: None)
        self._stop = stop or (lambda: False)
        self._tmp = None
        self._fp32_miou = None
        self._n_tmp = 0

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        return self

    def __exit__(self, *exc):
        self._tmp.cleanup()

    def _check_stop(self):
        if self._stop():
            raise Stopped()

    # ── 공통 ────────────────────────────────────────────────────
    def calib_files(self):
        files = pick_calibration_files(self.c["calib_dir"], self.c["img_ext"], CALIB_N)
        if not files:
            raise RuntimeError(f"캘리브레이션 이미지 없음: {self.c['calib_dir']}")
        return files

    def _subset(self):
        if not hasattr(self, "_val"):
            dest = os.path.join(self._tmp.name, "val")
            os.makedirs(dest)
            di, dl, n = make_val_subset(self.c["val_img_dir"], self.c["val_lbl_dir"],
                                        self.c["img_ext"], self.c["lbl_ext"], VAL_N, dest)
            self._log(f"  검증 이미지 {n}장으로 mIoU 측정")
            self._val = (di, dl)
        return self._val

    def miou(self, onnx_file):
        di, dl = self._subset()
        res, _ = evaluate_onnx(onnx_file, self.c["onnx_path"], di, dl,
                               self.c["img_ext"], self.c["lbl_ext"], threads=0)
        return res["mIoU"]

    def fp32_miou(self):
        if self._fp32_miou is None:
            self._fp32_miou = self.miou(self.c["onnx_path"])
            self._log(f"  FP32 기준 mIoU {self._fp32_miou:.4f}")
        return self._fp32_miou

    def quant_miou(self, excluded):
        """excluded 레이어만 FP32 로 두고 나머지를 INT8 로 양자화해 mIoU 를 잰다."""
        self._n_tmp += 1
        path = os.path.join(self._tmp.name, f"trial_{self._n_tmp}.onnx")
        quantize_int8(self.c["onnx_path"], self.calib_files(), excluded, out_path=path)
        value = self.miou(path)
        os.remove(path)     # 시도마다 모델이 쌓이지 않게
        return value

    # ── 1) 레이어별 SQNR ────────────────────────────────────────
    def layer_sqnr(self):
        """[{레이어, 연산, SQNR (dB)}] 를 SQNR 이 낮은(오차가 큰) 순으로 돌려준다."""
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization.qdq_loss_debug import (
            _TENSOR_SAVE_POSTFIX as POST, modify_model_output_intermediate_tensors,
        )
        from tools.segmentation.utils.pad_utils import read_input_scale

        onnx_path = self.c["onnx_path"]
        files = self.calib_files()
        cfg = read_config(onnx_path)
        in_scale = read_input_scale(onnx_path)
        multiple = _pad_multiple(cfg)

        self._log(f"[1] 전체 INT8 모델 생성 (캘리브레이션 {len(files)}장)")
        full = os.path.join(self._tmp.name, "full_int8.onnx")
        quantize_int8(onnx_path, files, [], out_path=full)

        # 모든 중간 텐서를 출력으로 내보내도록 두 모델을 고친다 (onnxruntime 의 qdq_loss_debug 도구).
        # 양자화 모델에도 원래 텐서 이름이 남아 있어서 이름으로 짝을 짓는다.
        aug_f = os.path.join(self._tmp.name, "float_aug.onnx")
        aug_q = os.path.join(self._tmp.name, "qdq_aug.onnx")
        modify_model_output_intermediate_tensors(onnx_path, aug_f)
        modify_model_output_intermediate_tensors(full, aug_q)
        # 그래프 최적화를 끈다: 켜 두면 ONNX Runtime 이 중간 출력을 끼워 넣은 양자화 그래프를 다시 고치다가
        # 잘못된 그래프(INVALID_GRAPH)를 만든다 (DINOv3 에서 확인). 오차 비교에는 최적화가 필요 없다.
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_f = ort.InferenceSession(aug_f, opts, providers=["CPUExecutionProvider"])
        sess_q = ort.InferenceSession(aug_q, opts, providers=["CPUExecutionProvider"])

        ops = dict(list_layers(onnx_path))
        # 양자화 모델에서 같은 이름의 노드를 찾아 그 출력 텐서를 쓴다. ONNX Runtime 이 Conv→Relu 를
        # 하나로 합치면 Conv 의 출력 이름이 Relu 출력 이름으로 바뀌는데, FP32 모델에도 그 이름의
        # 텐서(Relu 출력)가 있으므로 합쳐진 한 쌍이 Conv 레이어 한 줄로 측정된다.
        qgraph = onnx.load(full, load_external_data=False).graph
        float_tensors = {o.name[:-len(POST)] for o in sess_f.get_outputs() if o.name.endswith(POST)}
        consumers = {}
        for n in qgraph.node:
            for t in n.input:
                consumers.setdefault(t, []).append(n)

        def resolve(tensor):
            """tensor 가 FP32 모델에 없으면 뒤따르는 Quantize → Dequantize 의 출력 이름을 찾는다.
            (그래프 출력을 만드는 레이어는 Conv → Quantize → Dequantize → 출력 으로 바뀐다)"""
            if tensor in float_tensors:
                return tensor
            for q in consumers.get(tensor, []):
                if q.op_type == "QuantizeLinear":
                    for dq in consumers.get(q.output[0], []):
                        if dq.op_type == "DequantizeLinear" and dq.output[0] in float_tensors:
                            return dq.output[0]
            return None

        owner = {}
        for n in qgraph.node:
            if n.name in ops and n.output:
                t = resolve(n.output[0])
                if t is not None:
                    owner[t + POST] = n.name
        names_q = {o.name for o in sess_q.get_outputs()}
        common = [o.name for o in sess_f.get_outputs() if o.name in owner and o.name in names_q]
        self._log(f"  비교할 레이어 출력 {len(common)}개 (레이어 {len(ops)}개 중, "
                  f"나머지는 형태 계산 등 실수 출력이 없는 노드)")

        signal = dict.fromkeys(common, 0.0)
        noise = dict.fromkeys(common, 0.0)
        for k, path in enumerate(files, 1):
            self._check_stop()
            x = {"input": _load_input(path, in_scale, multiple, cfg.get("channels"))}
            # 필요한 출력만 계산해서 받는다. 이미지마다 더해 가므로 모든 이미지의 활성값을 쌓아 두지 않는다
            for name, a, b in zip(common, sess_f.run(common, x), sess_q.run(common, x)):
                a = a.astype(np.float64); b = b.astype(np.float64)
                signal[name] += float(np.sum(a * a))
                noise[name] += float(np.sum((a - b) ** 2))
            self._progress(k, len(files))

        rows = [{"레이어": owner[n], "연산": ops[owner[n]], "SQNR (dB)": sqnr_db(signal[n], noise[n])}
                for n in common]
        rows.sort(key=lambda r: r["SQNR (dB)"])
        return rows

    # ── 2) 블록별 민감도 ────────────────────────────────────────
    def run_blocks(self):
        """[{블록, 레이어 수, mIoU, 하락}]. 하락은 FP32 mIoU - 블록만 INT8 일 때 mIoU (그래프 순서)."""
        all_names = [n for n, _ in list_layers(self.c["onnx_path"])]
        blocks = group_blocks(all_names)
        base = self.fp32_miou()
        self._log(f"[2] 블록 {len(blocks)}개를 하나씩 INT8 로 바꿔 평가")
        rows = []
        for i, (key, names) in enumerate(blocks.items(), 1):
            self._check_stop()
            inside = set(names)
            m = self.quant_miou([n for n in all_names if n not in inside])
            rows.append({"블록": key, "레이어 수": len(names), "mIoU": m, "하락": base - m})
            self._log(f"  [{i}/{len(blocks)}] {key}  레이어 {len(names)}개  mIoU {m:.4f}  하락 {100 * (base - m):+.2f}%p")
            self._progress(i, len(blocks))
        return rows

    # ── 3) 자동 제외 ────────────────────────────────────────────
    def order_candidates(self, ranking, blocks=None):
        """FP32 로 추가할 후보 순서. 블록 민감도가 있으면 하락이 큰 블록부터(블록 안은 SQNR 순위),
        없으면 SQNR 순위(오차 큰 순) 그대로."""
        if not blocks:
            return list(ranking), "SQNR 순위"
        names = [n for n, _ in list_layers(self.c["onnx_path"])]
        groups = group_blocks(names)
        drop = {r["블록"]: r["하락"] for r in blocks}
        pos = {n: i for i, n in enumerate(ranking)}
        order = []
        for key in sorted(groups, key=lambda k: -drop.get(k, 0.0)):
            order += sorted((n for n in groups[key] if n in pos), key=pos.get)
        order += [n for n in ranking if n not in set(order)]     # 블록에 없는 레이어는 뒤에
        return order, "블록 민감도 순서"

    def run_auto(self, ranking, target, reduce=True, blocks=None):
        """1) 전부 INT8 평가 → 2) 첫 Conv 와 출력 헤드만 FP32 → 3) 민감도 순서에서 FP32 개수를 이분 탐색
        (FP32 대비 mIoU 차이가 target(0.01 = 1%p) 이내가 되면 멈춤) → 4) 줄이기."""
        from tools.segmentation.utils.quantize_onnx import default_excluded
        base = self.fp32_miou()
        quantizable = {n for n, _ in list_layers(self.c["onnx_path"])}
        trials = {}

        def gap(excluded, stage):
            self._check_stop()
            k = len(excluded)
            m = self.quant_miou(excluded)
            trials[k] = m
            self._log(f"  [{stage}] FP32 {k}개 → mIoU {m:.4f}  (FP32 대비 {100 * (base - m):+.2f}%p)")
            if k:
                self._log(f"    FP32 레이어: {self._names(excluded)}")
            return base - m

        self._log(f"[3] 자동 제외: 목표 mIoU 차이 {100 * target:.2f}%p 이내")
        excluded, best, reached = [], [], False
        # 단계 1: 전부 INT8
        self._log("[3-1] 전부 INT8 평가")
        if gap([], "1단계") <= target:
            reached = True
        else:
            # 단계 2: 첫 Conv 와 출력 헤드만 FP32
            head = [n for n in default_excluded(self.c["onnx_path"]) if n in quantizable]
            self._log(f"[3-2] 첫 Conv 와 출력 헤드만 FP32 ({len(head)}개)")
            excluded = list(head)
            if gap(excluded, "2단계") <= target:
                reached = True
            else:
                # 단계 3: 민감도 순서에서 FP32 개수 이분 탐색
                order, how = self.order_candidates(ranking, blocks)
                order = [n for n in order if n not in set(head)]
                self._log(f"[3-3] {how}로 FP32 개수 이분 탐색 (후보 {len(order)}개)")
                # 이분 탐색: 앞에서 n개를 FP32 로 둘 때 목표를 만족하는 최소 n 을 찾는다 (n 이 클수록 좋아진다고 가정).
                # lo: 목표 못 미침(2단계 결과 = 0개), hi: 목표 만족. 먼저 후보를 전부 넣어 가능한지 본다.
                lo, hi = 0, len(order)
                excluded = head + order
                self._log(f"  후보 전부 추가: {self._names(order)}")
                if gap(excluded, "3단계 전부") <= target:
                    reached = True
                    steps = max(1, (hi - lo).bit_length())
                    done = 0
                    while hi - lo > 1:
                        mid = (lo + hi) // 2
                        self._log(f"  추가 {mid}개: {self._names(order[:mid])}")
                        if gap(head + order[:mid], f"3단계 {mid}개") <= target:
                            hi = mid
                        else:
                            lo = mid
                        done += 1
                        self._progress(done, steps)
                    excluded = head + order[:hi]
        best = len(excluded)
        final_miou = trials[best]
        rtrials = []
        if reduce and reached and excluded:
            excluded, final_miou, rtrials = self._reduce(excluded, base, target, final_miou)
        elif reduce and excluded:
            self._log("[4] 줄이기 건너뜀: 목표에 못 미쳐 기준이 없습니다")
        self._log(f"최종 FP32 유지 레이어 {len(excluded)}개 (INT8 로 되돌린 레이어 {best - len(excluded)}개)")
        for i, name in enumerate(excluded, 1):
            self._log(f"  {i}. {name}")
        return {
            "k": len(excluded), "k_found": best, "excluded": excluded, "miou": final_miou, "base": base,
            "target": target, "reached": reached, "reduce_trials": rtrials,
            "trials": [{"제외 개수": k, "mIoU": m, "차이": base - m} for k, m in sorted(trials.items())],
        }

    @staticmethod
    def _names(names, limit=8):
        """로그용: 레이어 이름 목록. 많으면 앞 limit 개만."""
        text = ", ".join(names[:limit])
        return text + (f" 외 {len(names) - limit}개" if len(names) > limit else "")

    # ── 4) 줄이기 ───────────────────────────────────────────────
    def _reduce(self, excluded, base, target, miou):
        """제외 레이어를 오차가 작은 쪽(순위 뒤)부터 하나씩 INT8 로 되돌려 본다.
        목표가 유지되면 INT8 로 남기고(제외 목록에서 뺌), 깨지면 FP32 로 되돌린다."""
        keep = list(excluded)
        self._log(f"[4] 줄이기: 제외 {len(keep)}개를 하나씩 INT8 로 되돌려 확인")
        rtrials = []
        for i, name in enumerate(reversed(excluded), 1):
            self._check_stop()
            trial = [x for x in keep if x != name]
            m = self.quant_miou(trial)
            ok = (base - m) <= target
            if ok:
                keep, miou = trial, m
            rtrials.append({"레이어": name, "mIoU": m, "차이": base - m, "INT8 유지": ok})
            self._log(f"  [{i}/{len(excluded)}] {name} → INT8  mIoU {m:.4f} "
                      f"(FP32 대비 {100 * (base - m):+.2f}%p)  {'INT8 로 남김' if ok else 'FP32 로 되돌림'}")
            self._progress(i, len(excluded))
        return keep, miou, rtrials


# ═══════════════════════════════════════════════════════════════
#  CSV
# ═══════════════════════════════════════════════════════════════
CSV_COLUMNS = ["단계", "이름", "연산 / 레이어 수", "값", "비고"]


def save_csv(path, layers=None, blocks=None, auto=None):
    """있는 결과만 한 파일에 이어서 쓴다. utf-8-sig: 엑셀에서 한글이 깨지지 않게 BOM 을 붙인다."""
    rows = []
    for i, r in enumerate(layers or [], 1):
        rows.append(["1 레이어 SQNR", r["레이어"], r["연산"], f"{r['SQNR (dB)']:.3f}", f"순위 {i} (dB, 낮을수록 오차 큼)"])
    for r in blocks or []:
        rows.append(["2 블록 민감도", r["블록"], r["레이어 수"], f"{100 * r['하락']:.3f}",
                     f"mIoU 하락 %p (INT8 일 때 mIoU {r['mIoU']:.4f})"])
    if auto:
        for t in auto["trials"]:
            rows.append(["3 자동 제외 시도", f"제외 {t['제외 개수']}개", "", f"{t['mIoU']:.4f}",
                         f"FP32 대비 {100 * t['차이']:+.2f}%p"])
        for t in auto.get("reduce_trials", []):
            rows.append(["3 줄이기 시도", t["레이어"], "", f"{t['mIoU']:.4f}",
                         f"FP32 대비 {100 * t['차이']:+.2f}%p, {'INT8 로 남김' if t['INT8 유지'] else 'FP32 유지'}"])
        for i, name in enumerate(auto["excluded"], 1):
            rows.append(["3 자동 제외 결과", name, "", i, f"목표 {100 * auto['target']:.2f}%p "
                         f"{'달성' if auto['reached'] else '미달'}"])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)
