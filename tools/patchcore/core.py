"""PatchCore 도구의 화면과 관계없는 계산 모음: 폴더 읽기, 학습, 추론, 평가, 내보내기.
anomalib, torch는 함수가 불려갈 때 처음 불러온다 (홈 화면이 빨리 뜨게)."""
import json
import os
import warnings
from datetime import datetime

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
BACKBONES = ["wide_resnet50_2", "resnet18", "resnet34", "resnet50", "resnext50_32x4d"]
GOOD_NAMES = ("good", "normal", "ok")   # 이 이름의 하위 폴더는 정상으로 본다
MASK_DIR_NAMES = ("ground_truth", "mask", "masks", "gt")


def list_images(folder):
    """폴더 바로 아래의 이미지 파일 경로를 이름순으로 돌려준다."""
    if not folder or not os.path.isdir(folder):
        return []
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))
    return [os.path.join(folder, f) for f in names]


def scan_test_folder(test_dir):
    """테스트 폴더를 읽어 정상/불량과 마스크를 자동 인식한다.
    - good(또는 normal, ok) 하위 폴더 = 정상, 그 외 하위 폴더 = 불량 (폴더 이름이 불량 종류)
    - 마스크: test/../ground_truth/<불량>, test/ground_truth/<불량>, test/<불량>_mask 중 이미지 수가 같은 것
    돌려주는 값: {"good": [경로], "defects": {이름: [경로]}, "masks": {이름: [마스크 경로]}}"""
    out = {"good": [], "defects": {}, "masks": {}}
    if not test_dir or not os.path.isdir(test_dir):
        return out
    subs = sorted(d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)))
    mask_roots = [os.path.join(os.path.dirname(test_dir.rstrip("/\\")), n) for n in MASK_DIR_NAMES]
    mask_roots += [os.path.join(test_dir, n) for n in MASK_DIR_NAMES]
    for name in subs:
        if name.lower() in MASK_DIR_NAMES or name.lower().endswith("_mask"):
            continue   # 마스크 폴더는 이미지 폴더가 아니다
        imgs = list_images(os.path.join(test_dir, name))
        if not imgs:
            continue
        if name.lower() in GOOD_NAMES:
            out["good"] += imgs
            continue
        out["defects"][name] = imgs
        cands = [os.path.join(r, name) for r in mask_roots] + [os.path.join(test_dir, name + "_mask")]
        for c in cands:
            m = list_images(c)
            if len(m) == len(imgs):   # anomalib은 이름순으로 이미지와 마스크를 짝짓는다 -> 수가 같아야 한다
                out["masks"][name] = m
                break
    if not subs:   # 하위 폴더가 없으면 폴더 안 이미지를 모두 정상으로 본다
        out["good"] = list_images(test_dir)
    return out


def has_all_masks(scan):
    return bool(scan["defects"]) and all(n in scan["masks"] for n in scan["defects"])


# ── 학습 ─────────────────────────────────────────────────────
def _quiet():
    warnings.filterwarnings("ignore")


class TrainStopped(Exception):
    """사용자가 학습을 중단했다"""


def _stop_callback(stop_event):
    """배치가 시작될 때마다 중단 신호를 확인해 학습을 멈추는 Lightning 콜백"""
    from lightning.pytorch.callbacks import Callback

    class _Stop(Callback):
        def _check(self):
            if stop_event.is_set():
                raise TrainStopped("사용자가 학습을 중단했습니다")

        def on_train_batch_start(self, *a, **k):
            self._check()

        def on_validation_batch_start(self, *a, **k):
            self._check()

    return _Stop()


def train(cfg, log, stop_event=None):
    """cfg: {"normal_dir","test_dir","backbone","layers","coreset","size":(h,w),"out_dir"} 로 학습하고 ckpt 경로를 돌려준다.
    테스트 폴더가 있으면 정상/불량으로 나눠 anomalib에 넘기고(검증용 임계값 계산), 없으면 학습 폴더의 20%를 정상 검증에 쓴다."""
    _quiet()
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    kw = dict(name="patchcore", normal_dir=cfg["normal_dir"], train_batch_size=8, eval_batch_size=8,
              num_workers=0, seed=42)   # Windows에서 데이터 로더 에러 방지: num_workers=0
    scan = scan_test_folder(cfg.get("test_dir", ""))
    if scan["good"] and scan["defects"]:
        good_dirs = sorted({os.path.dirname(p) for p in scan["good"]})
        kw["normal_test_dir"] = good_dirs
        kw["abnormal_dir"] = [os.path.dirname(v[0]) for v in scan["defects"].values()]
        if has_all_masks(scan):
            kw["mask_dir"] = [os.path.dirname(scan["masks"][n][0]) for n in scan["defects"]]
        log(f"테스트: 정상 {len(scan['good'])}장, 불량 {sum(len(v) for v in scan['defects'].values())}장"
            f"{', 마스크 사용' if 'mask_dir' in kw else ''}")
    else:
        log("테스트 폴더에 정상/불량 구분이 없어 학습 이미지의 20%를 검증에 쓴다")
    dm = Folder(**kw)
    model = Patchcore(backbone=cfg["backbone"], layers=list(cfg["layers"]), coreset_sampling_ratio=cfg["coreset"],
                      pre_processor=Patchcore.configure_pre_processor(image_size=tuple(cfg["size"])),
                      visualizer=False)   # 결과 그림 파일은 이 도구가 따로 그린다
    os.makedirs(cfg["out_dir"], exist_ok=True)
    eng = Engine(default_root_dir=cfg["out_dir"], enable_progress_bar=False, logger=False,
                 callbacks=[_stop_callback(stop_event)] if stop_event is not None else None)
    log(f"학습 시작: {cfg['backbone']}, 층 {','.join(cfg['layers'])}, coreset {cfg['coreset']}, 입력 {cfg['size'][0]}x{cfg['size'][1]}")
    eng.fit(model=model, datamodule=dm)
    ckpt = os.path.join(cfg["out_dir"], "model.ckpt")
    eng.trainer.save_checkpoint(ckpt)
    import shutil
    shutil.rmtree(os.path.join(cfg["out_dir"], "Patchcore"), ignore_errors=True)   # Lightning이 따로 만든 중복 체크포인트 폴더
    log(f"모델 저장: {ckpt}")
    return ckpt


# ── 추론 ─────────────────────────────────────────────────────
_model_cache = {}


def load_model(ckpt):
    """체크포인트를 읽어 평가 모드 모델을 돌려준다 (같은 파일은 다시 읽지 않는다)."""
    _quiet()
    if ckpt not in _model_cache:
        from anomalib.models import Patchcore
        m = Patchcore.load_from_checkpoint(ckpt)
        m.eval()
        _model_cache.clear()
        _model_cache[ckpt] = m
    return _model_cache[ckpt]


def predict_folder(ckpt, folder, size):
    """폴더의 이미지를 추론한다. 돌려주는 값: {이미지 경로: (점수 0~1, anomaly_map 2차원 numpy)}
    점수는 anomalib 후처리로 0~1로 맞춰진 값이고 0.5가 학습 중 자동으로 정한 기준이다."""
    _quiet()
    from anomalib.data import PredictDataset
    from anomalib.engine import Engine
    from torch.utils.data import DataLoader

    model = load_model(ckpt)
    ds = PredictDataset(path=folder, image_size=tuple(size))
    eng = Engine(default_root_dir=os.path.dirname(ckpt), enable_progress_bar=False, logger=False)
    res = eng.predict(model=model, dataloaders=DataLoader(ds, batch_size=8, collate_fn=ds.collate_fn))
    out = {}
    for b in res or []:
        for i, p in enumerate(b.image_path):
            out[os.path.normpath(p)] = (float(b.pred_score[i]), b.anomaly_map[i].cpu().numpy())
    return out


def predict_paths(ckpt, paths, size):
    """경로 목록을 폴더별로 묶어 추론한다 (PredictDataset이 폴더를 받기 때문)."""
    folders = sorted({os.path.dirname(p) for p in paths})
    out = {}
    for f in folders:
        out.update(predict_folder(ckpt, f, size))
    return {os.path.normpath(p): out[os.path.normpath(p)] for p in paths if os.path.normpath(p) in out}


# ── 평가 ─────────────────────────────────────────────────────
def suggest_threshold(scores, labels):
    """F1이 가장 큰 임계값 (이웃한 두 점수의 가운데). 정상/불량이 한쪽뿐이면 None"""
    import numpy as np
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    if y.min() == y.max():
        return None
    uniq = np.unique(s)
    cands = np.concatenate([[uniq[0] - 1e-6], (uniq[:-1] + uniq[1:]) / 2, [uniq[-1] + 1e-6]])
    best, best_f1 = None, -1
    for t in cands:
        pred = s >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best, best_f1 = float(t), f1
    return best


def evaluate(ckpt, scan, size, log):
    """테스트 폴더 전체를 추론해 이미지 AUROC, 픽셀 AUROC(마스크 있을 때), 점수 목록, 임계값 제안을 돌려준다."""
    import numpy as np
    from PIL import Image
    from sklearn.metrics import roc_auc_score

    defect_paths = [p for v in scan["defects"].values() for p in v]
    paths = scan["good"] + defect_paths
    log(f"평가: 정상 {len(scan['good'])}장, 불량 {len(defect_paths)}장 추론")
    pred = predict_paths(ckpt, paths, size)
    rows = []   # (경로, 정상0/불량1, 점수)
    for p in paths:
        key = os.path.normpath(p)
        if key in pred:
            rows.append((p, int(p in defect_paths), pred[key][0]))
    y = [r[1] for r in rows]
    s = [r[2] for r in rows]
    res = {"n_good": y.count(0), "n_defect": y.count(1), "rows": rows,
           "image_auroc": None, "pixel_auroc": None, "threshold": suggest_threshold(s, y) if rows else None}
    if len(set(y)) == 2:
        res["image_auroc"] = float(roc_auc_score(y, s))
    if has_all_masks(scan) and res["n_defect"]:
        res["pixel_auroc"] = _pixel_auroc(pred, scan, size)
    return res


def _pixel_auroc(pred, scan, size):
    import numpy as np
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    ys, ss = [], []
    items = [(p, None) for p in scan["good"]]
    for name, imgs in scan["defects"].items():
        items += list(zip(imgs, scan["masks"][name]))
    for p, mp in items:
        key = os.path.normpath(p)
        if key not in pred:
            continue
        amap = pred[key][1]
        if mp:
            m = np.array(Image.open(mp).convert("L").resize((amap.shape[1], amap.shape[0]), Image.NEAREST)) > 0
        else:
            m = np.zeros(amap.shape, bool)
        idx = rng.choice(m.size, size=min(m.size, 20000), replace=False)   # 픽셀 수가 너무 많아지지 않게 이미지마다 표본 추출
        ys.append(m.ravel()[idx])
        ss.append(amap.ravel()[idx])
    y = np.concatenate(ys)
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, np.concatenate(ss)))


# ── config / 내보내기 ────────────────────────────────────────
def config_path_for(ckpt):
    return os.path.splitext(ckpt)[0] + "_config.json"


def save_config(ckpt, cfg):
    with open(config_path_for(ckpt), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_config(ckpt):
    try:
        with open(config_path_for(ckpt), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def new_run_dir(normal_dir, backbone):
    """checkpoints/<모델>_<날짜-시각>/ (이미지 폴더의 상위 폴더 아래, Segmentation과 같은 규칙)"""
    base = os.path.join(os.path.dirname(os.path.abspath(normal_dir.rstrip("/\\"))), "checkpoints")
    return os.path.join(base, f"patchcore_{backbone}_{datetime.now():%Y%m%d-%H%M%S}")


def export_onnx(ckpt, size, threshold, log):
    """ONNX + config를 <체크포인트 폴더>/export/<모델>_<날짜-시각>/ 에 만든다. 폴더 경로를 돌려준다."""
    _quiet()
    import shutil
    from anomalib.engine import Engine

    cfg = load_config(ckpt)
    name = os.path.basename(os.path.dirname(ckpt))
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = os.path.join(os.path.dirname(ckpt), "export", f"{name.rsplit('_', 2)[0]}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    model = load_model(ckpt)
    eng = Engine(default_root_dir=out_dir, enable_progress_bar=False, logger=False)
    log("ONNX 변환 중")
    tmp = eng.export(model=model, export_type="onnx", export_root=out_dir, model_file_name="model", input_size=tuple(size))
    onnx_path = os.path.join(out_dir, "model.onnx")
    src = os.path.dirname(str(tmp))   # 큰 모델은 가중치가 model.onnx.data로 따로 나뉘므로 폴더째 옮긴다
    for fn in os.listdir(src):
        shutil.move(os.path.join(src, fn), os.path.join(out_dir, fn))
    shutil.rmtree(os.path.join(out_dir, "weights"), ignore_errors=True)   # anomalib이 만드는 빈 중간 폴더
    cfg = dict(cfg, input_size=list(size), onnx_batch_size=2, threshold=threshold, score_note="점수는 0~1, 임계값 이상이면 불량")
    with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    try:   # ONNX 파일 안에도 config를 넣는다 (Segmentation과 같은 방식)
        import onnx
        m = onnx.load(onnx_path)   # 외부 데이터가 있으면 같은 폴더에서 함께 읽는다
        p = m.metadata_props.add()
        p.key, p.value = "config", json.dumps(cfg, ensure_ascii=False)
        if os.path.isfile(onnx_path + ".data"):   # 외부 데이터 파일을 쓰는 모델은 같은 방식으로 다시 저장
            onnx.save(m, onnx_path, save_as_external_data=True, all_tensors_to_one_file=True, location="model.onnx.data")
        else:
            onnx.save(m, onnx_path)
    except Exception as e:   # 메타데이터만 실패한 것이므로 내보내기는 계속
        log(f"ONNX 메타데이터 저장 실패: {e}")
    results = os.path.join(os.path.dirname(ckpt), "Results")
    if os.path.isdir(results):
        shutil.copytree(results, os.path.join(out_dir, "Results"), dirs_exist_ok=True)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# {name}\n\n- 모델: PatchCore ({cfg.get('backbone', '')})\n- 입력 크기: {size[0]}x{size[1]}\n"
                f"- 임계값: {threshold}\n- 파일: model.onnx, model_config.json\n")
    return out_dir
