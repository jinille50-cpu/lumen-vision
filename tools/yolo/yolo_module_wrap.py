"""
YOLO Detection 학습/데이터 래퍼

기존 YOLO_Module.py(YOLOproject)를 기반으로,
GUI에서 쓰기 좋게 다음을 추가:
  - scan_dataset()  : 이미지/라벨 검증 (개수·짝·클래스 ID)
  - build_yaml()    : data.yaml 생성 (manual 폴더 분리 / hold-out 랜덤 분할)
  - train()         : Ultralytics 콜백으로 epoch별 metric 추출 + 중단 지원

라벨은 YOLO Detection txt 형식(class cx cy w h)만 사용.
이미지는 png / jpg / bmp / tiff  ("tiff" 선택 시 .tif, "jpg" 선택 시 .jpeg 파일도 함께 매칭).
"""

import os
import glob
import shutil
import random


IMG_EXTS = ("png", "jpg", "bmp", "tiff")

# 가중치 폴더: Vision Toolkit/models/weights (이 파일은 tools/yolo/ 안에 있음)
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "models", "weights")


def local_weight_path(model_name):
    """models/weights 안에 가중치가 있으면 그 경로, 없으면 None"""
    p = os.path.join(WEIGHTS_DIR, model_name)
    return p if os.path.isfile(p) else None


def resolve_model(model_name, log=print):
    """models/weights의 가중치를 우선 사용. 없으면 그 폴더로 받아 온다(처음 학습 때 한 번).
    받기에 실패하면 이름 그대로 돌려줘 Ultralytics가 알아서 받게 한다."""
    p = local_weight_path(model_name)
    if p:
        return p
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        target = os.path.join(WEIGHTS_DIR, model_name)
        log(f"  {model_name}: models/weights에 없어 내려받습니다...")
        attempt_download_asset(target)
        if os.path.isfile(target):
            return target
    except Exception as e:
        log(f"  [안내] models/weights로 받기 실패({e}) — Ultralytics 기본 경로로 시도합니다")
    return model_name


def _rtdetr_by_name(model_path):
    """파일 이름, 같은 이름 config의 모델 이름, ONNX 메타데이터 중 하나라도 rtdetr로 시작하면 True"""
    names = [os.path.basename(model_path)]
    from tools.yolo.work_state import find_config, load_config
    cfg = load_config(find_config(model_path)) or {}
    if model_path.lower().endswith(".onnx"):
        from tools.yolo.export_bundle import read_onnx_config
        cfg = {**(read_onnx_config(model_path) or {}), **cfg}
    names += [str(cfg.get("model", "")), str(cfg.get("model_name", ""))]
    return any(n.lower().startswith("rtdetr") for n in names)


def load_detector(model_path):
    """객체 인식 모델을 연다. RT-DETR은 결과 처리 방식이 달라(NMS 없음) YOLO가 아니라 RTDETR 클래스로 열어야 한다."""
    from ultralytics import RTDETR, YOLO
    if _rtdetr_by_name(model_path):
        return RTDETR(model_path)
    model = YOLO(model_path)
    # 이름으로 모르는 .pt(예: 다른 곳에서 가져온 best.pt)는 불러온 구조를 보고 다시 연다
    if type(getattr(model, "model", None)).__name__ == "RTDETRDetectionModel":
        return RTDETR(model_path)
    return model


# ──────────────────────────────────────────────────────────────────
#  데이터 검증
# ──────────────────────────────────────────────────────────────────
def _ext_patterns(img_ext):
    """확장자 문자열 → 실제 매칭할 확장자 목록 (tiff는 .tif, jpg는 .jpeg 포함)"""
    e = img_ext.lower().lstrip(".")
    if e in ("tiff", "tif"):
        return ("tiff", "tif")
    if e in ("jpg", "jpeg"):
        return ("jpg", "jpeg")
    return (e,)


def _list_images(img_dir, img_ext):
    files = []
    for e in _ext_patterns(img_ext):
        files += glob.glob(os.path.join(img_dir, f"*.{e}"))
    return sorted(set(files))


def _label_for(img_path, lbl_dir):
    """이미지 경로 → 같은 stem의 .txt 라벨 경로"""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(lbl_dir, stem + ".txt")


# ──────────────────────────────────────────────────────────────────
#  다채널 TIFF 처리
# ──────────────────────────────────────────────────────────────────
def _load_array(img_path):
    """이미지를 numpy 배열로 로드 (tifffile 우선, 없으면 PIL 멀티프레임 지원)"""
    import numpy as np
    try:
        import tifffile
        return tifffile.imread(img_path)
    except ImportError:
        pass
    from PIL import Image
    im = Image.open(img_path)
    frames = []
    try:
        i = 0
        while True:
            im.seek(i)
            frames.append(np.array(im))
            i += 1
    except EOFError:
        pass
    if len(frames) > 1:
        return np.stack(frames, axis=0)   # (C, H, W)
    return frames[0]


def _split_channels(arr):
    """(H,W) / (H,W,C) / (C,H,W) 배열 → (채널 수, 채널 추출 함수)"""
    if arr.ndim == 2:
        return 1, lambda c: arr
    if arr.ndim == 3:
        # 작은 쪽 축을 채널로 간주 (채널 수 << 이미지 크기)
        if arr.shape[0] <= arr.shape[2]:
            return arr.shape[0], lambda c: arr[c]
        return arr.shape[2], lambda c: arr[..., c]
    raise ValueError(f"지원하지 않는 이미지 차원: {arr.shape}")


def _to_uint8(ch):
    """채널 배열을 min-max 정규화해 uint8로 변환 (16bit 현미경 이미지 대응)"""
    import numpy as np
    if ch.dtype == np.uint8:
        return ch
    ch = ch.astype(np.float64)
    lo, hi = float(ch.min()), float(ch.max())
    if hi <= lo:
        return np.zeros(ch.shape, dtype=np.uint8)
    return ((ch - lo) / (hi - lo) * 255.0).round().astype(np.uint8)


def get_channel_count(img_path):
    """이미지의 채널 수 반환 (판별 실패 시 1)"""
    try:
        arr = _load_array(img_path)
        n, _ = _split_channels(arr)
        return n
    except Exception:
        return 1


def _detect_yaml_channels(img_paths, img_ext, log=print):
    """channel 미지정 시 data.yaml에 넣을 channels 값 감지.
    Ultralytics는 TIFF를 페이지=채널 그대로 로드하므로 원본 TIFF가 3채널이
    아니면 channels 키를 명시해야 모델 입력 채널이 맞는다.
    (PNG 등은 항상 3채널(BGR)로 로드되므로 해당 없음 → None)"""
    if not img_paths or _ext_patterns(img_ext)[0] not in ("tiff", "tif"):
        return None
    n = get_channel_count(img_paths[0])
    if n != 3:
        log(f"  [yaml] TIFF 원본 {n}채널 감지 → channels: {n} 기록 ({n}채널 입력 모델)")
        return n
    return None


def _normalize_channels(channel):
    """channel 인자 정규화: int | list | None → 채널 인덱스 리스트 or None"""
    if channel is None:
        return None
    if isinstance(channel, (list, tuple)):
        chs = [int(c) for c in channel]
        return chs or None
    return [int(channel)]


def extract_channels(img_path, channels):
    """다채널 이미지에서 지정 채널들을 uint8 (C,H,W) 배열로 추출.
    채널 순서는 리스트 순서 그대로. 인덱스가 범위를 넘으면 마지막 채널로 클램프."""
    import numpy as np
    arr = _load_array(img_path)
    n, pick = _split_channels(arr)
    planes = [_to_uint8(pick(min(max(0, int(c)), n - 1))) for c in channels]
    return np.stack(planes, axis=0)


def extract_channel(img_path, channel):
    """다채널 이미지에서 지정 채널 1개를 uint8 (H,W) 배열로 추출."""
    return extract_channels(img_path, [channel])[0]


def false_color_image(img_path, bands, fc):
    """다채널 이미지에서 밴드 3개를 골라 false color RGB (H,W,3) uint8로. fc: {"lo": 하한 %, "hi": 상한 %}"""
    import numpy as np
    from common.falsecolor import false_color
    arr = _load_array(img_path)
    n, pick = _split_channels(arr)
    cube = np.stack([pick(min(max(0, int(c)), n - 1)) for c in bands], axis=-1)   # 고른 3밴드만 (H,W,3)
    return false_color(cube, [0, 1, 2], float(fc.get("lo", 1.0)), float(fc.get("hi", 99.0)))


def scan_dataset(img_dir, lbl_dir, img_ext="png"):
    """
    이미지/라벨 폴더를 스캔하고 검증 결과를 dict로 반환.

    반환:
      {
        "images": [paths], "labels": [paths],
        "n_img": int, "n_lbl": int,
        "missing_label": [img stem...],   # 라벨 없는 이미지
        "empty_label":   [stem...],       # 빈 txt (배경 이미지일 수도 있음 — 경고만)
        "orphan_label":  [stem...],       # 이미지 없는 라벨
        "class_ids":     sorted set,      # txt에서 발견된 클래스 ID
        "max_class_id":  int or None,
        "bad_lines":     [(file, lineno, reason)...],  # 형식 오류
        "ok": bool, "messages": [str...]
      }
    """
    messages = []
    result = {
        "images": [], "labels": [], "n_img": 0, "n_lbl": 0,
        "missing_label": [], "empty_label": [], "orphan_label": [],
        "class_ids": [], "max_class_id": None, "bad_lines": [],
        "ok": False, "messages": messages,
    }

    if not img_dir or not os.path.isdir(img_dir):
        messages.append(f"[오류] 이미지 폴더가 없습니다: {img_dir}")
        return result
    if not lbl_dir or not os.path.isdir(lbl_dir):
        messages.append(f"[오류] 라벨 폴더가 없습니다: {lbl_dir}")
        return result

    images = _list_images(img_dir, img_ext)
    all_labels = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
    result["images"] = images
    result["labels"] = all_labels
    result["n_img"] = len(images)
    result["n_lbl"] = len(all_labels)

    if not images:
        messages.append(f"[오류] .{img_ext} 이미지가 없습니다: {img_dir}")
        return result

    img_stems = {os.path.splitext(os.path.basename(p))[0] for p in images}
    lbl_stems = {os.path.splitext(os.path.basename(p))[0] for p in all_labels}

    class_ids = set()

    for img in images:
        lp = _label_for(img, lbl_dir)
        stem = os.path.splitext(os.path.basename(img))[0]
        if not os.path.isfile(lp):
            result["missing_label"].append(stem)
            continue
        # 라벨 파싱 + 검증
        try:
            with open(lp, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except Exception as e:
            result["bad_lines"].append((os.path.basename(lp), 0, f"읽기 실패: {e}"))
            continue
        if not lines:
            result["empty_label"].append(stem)
            continue
        for i, ln in enumerate(lines, 1):
            parts = ln.split()
            # Detection: class cx cy w h  (5개)
            if len(parts) != 5:
                result["bad_lines"].append(
                    (os.path.basename(lp), i, f"열 개수 {len(parts)} (5개 필요)"))
                continue
            try:
                cid = int(float(parts[0]))
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                result["bad_lines"].append(
                    (os.path.basename(lp), i, "숫자 변환 실패"))
                continue
            class_ids.add(cid)
            if any(c < 0 or c > 1 for c in coords):
                result["bad_lines"].append(
                    (os.path.basename(lp), i, "좌표가 0~1 범위를 벗어남 (정규화 필요)"))

    # 이미지 없는 라벨(고아)
    result["orphan_label"] = sorted(lbl_stems - img_stems)

    result["class_ids"] = sorted(class_ids)
    result["max_class_id"] = max(class_ids) if class_ids else None

    # 요약 메시지
    messages.append(f"이미지 {result['n_img']}개 / 라벨(txt) {result['n_lbl']}개")
    if result["class_ids"]:
        messages.append(f"발견된 클래스 ID: {result['class_ids']}  (최대 {result['max_class_id']})")
    if result["missing_label"]:
        messages.append(f"[경고] 라벨 없는 이미지 {len(result['missing_label'])}개 "
                        f"(예: {result['missing_label'][:5]})")
    if result["empty_label"]:
        messages.append(f"[정보] 빈 라벨 {len(result['empty_label'])}개 "
                        f"(배경 이미지로 처리됨)")
    if result["orphan_label"]:
        messages.append(f"[경고] 이미지 없는 라벨 {len(result['orphan_label'])}개 "
                        f"(예: {result['orphan_label'][:5]})")
    if result["bad_lines"]:
        messages.append(f"[오류] 형식 오류 {len(result['bad_lines'])}줄 발견:")
        for fn, ln, why in result["bad_lines"][:10]:
            messages.append(f"    {fn}:{ln}  — {why}")
        if len(result["bad_lines"]) > 10:
            messages.append(f"    ... 외 {len(result['bad_lines'])-10}줄")

    # ok 판정: 형식 오류 없고, 라벨 달린 이미지가 1개 이상
    n_labeled = result["n_img"] - len(result["missing_label"])
    result["ok"] = (not result["bad_lines"]) and (n_labeled > 0)
    if result["ok"]:
        messages.append(f"✔ 검증 통과 (학습 가능 이미지 {n_labeled}개)")
    else:
        messages.append("✘ 검증 실패 — 위 문제를 해결하세요")
    return result


# ──────────────────────────────────────────────────────────────────
#  data.yaml 생성
# ──────────────────────────────────────────────────────────────────
def _copy_pair(img_path, src_lbl_dir, dst_img_dir, dst_lbl_dir, channel=None, false_color=None):
    """이미지 + 짝 라벨(txt)을 대상 폴더로 복사.
    channel 지정 시(정수 또는 리스트) 해당 채널들만 추출해
    멀티페이지 TIFF(페이지=채널)로 저장 — Ultralytics가 (H,W,C)로 로드.
    false_color({"lo", "hi"})가 있고 채널이 3개면 퍼센타일로 늘린 RGB PNG로 저장 (일반 3채널 입력)."""
    chs = _normalize_channels(channel)
    if chs is None:
        shutil.copy(img_path, dst_img_dir)
    elif false_color and len(chs) == 3:
        from PIL import Image
        stem = os.path.splitext(os.path.basename(img_path))[0]
        Image.fromarray(false_color_image(img_path, chs, false_color)).save(os.path.join(dst_img_dir, stem + ".png"))
    else:
        try:
            import tifffile
        except ImportError:
            raise ImportError("채널 추출 저장에는 tifffile이 필요합니다 "
                              "→ pip install tifffile")
        stem = os.path.splitext(os.path.basename(img_path))[0]
        # photometric='minisblack' 로 항상 멀티페이지(페이지=채널) TIFF 강제.
        # (미지정 시 tifffile이 3·4채널을 RGB/RGBA 인터리브로 저장 → Ultralytics가
        #  단일 프레임으로 읽어 채널 수가 어긋남. minisblack이면 1~N채널 모두
        #  cv2.imdecodemulti가 페이지 수 = 채널 수로 읽어 (H,W,C)가 일관되게 나옴)
        tifffile.imwrite(os.path.join(dst_img_dir, stem + ".tiff"),
                         extract_channels(img_path, chs),
                         photometric="minisblack")
    lp = _label_for(img_path, src_lbl_dir)
    if os.path.isfile(lp):
        shutil.copy(lp, dst_lbl_dir)


def build_yaml(out_dir, class_names,
               split_mode="manual",
               train_img_dir=None, train_lbl_dir=None,
               val_img_dir=None,   val_lbl_dir=None,
               img_ext="png", holdout_ratio=0.2,
               aug=None, seed=42, channel=None, log=print, false_color=None):
    """
    Ultralytics용 data.yaml 생성.

    split_mode:
      - "manual" : train/val 폴더가 이미 분리됨 → 경로만 yaml에 기록
                   (channel 지정 시 채널 추출본을 out_dir/dataset에 생성)
      - "holdout": train 폴더 데이터를 랜덤 분할 → out_dir/dataset/{train,val}로 복사

    channel: 다채널 TIFF에서 사용할 채널 인덱스 — 정수 1개 또는 리스트
             (None이면 원본 그대로, 리스트면 순서대로 스택해 N채널 입력)

    반환: data.yaml 절대경로
    """
    import yaml
    os.makedirs(out_dir, exist_ok=True)
    yaml_path = os.path.join(out_dir, "data.yaml")
    names = {i: n for i, n in enumerate(class_names)}
    channel = _normalize_channels(channel)

    if split_mode == "manual" and channel is None:
        content = {
            "train": os.path.abspath(train_img_dir),
            "val":   os.path.abspath(val_img_dir),
            "nc":    len(class_names),
            "names": names,
        }
        n_ch = _detect_yaml_channels(_list_images(train_img_dir, img_ext), img_ext, log)
        if n_ch:
            content["channels"] = n_ch
        log(f"  [yaml] manual — train: {content['train']}")
        log(f"  [yaml] manual — val:   {content['val']}")

    elif split_mode in ("manual", "holdout"):
        ds = os.path.join(out_dir, "dataset")
        # 이전 실행 잔여 파일(채널 수가 다른 TIFF 등)이 섞이지 않도록 비우고 시작
        shutil.rmtree(ds, ignore_errors=True)
        ti = os.path.join(ds, "train", "images"); tl = os.path.join(ds, "train", "labels")
        vi = os.path.join(ds, "val",   "images"); vl = os.path.join(ds, "val",   "labels")
        for d in (ti, tl, vi, vl):
            os.makedirs(d, exist_ok=True)

        if split_mode == "manual":
            # 채널 추출본을 dataset 폴더에 생성 (train/val 각각)
            train_set = _list_images(train_img_dir, img_ext)
            val_set   = _list_images(val_img_dir, img_ext)
            for p in train_set:
                _copy_pair(p, train_lbl_dir, ti, tl, channel, false_color)
            for p in val_set:
                _copy_pair(p, val_lbl_dir, vi, vl, channel, false_color)
            log(f"  [yaml] manual — TIFF 채널 {channel} 추출 "
                f"({len(channel)}채널 입력) → {ds}")
        else:
            # 분할은 채널 추출 전 원본 파일 목록(정렬됨)에서 시드 고정으로 수행
            # → 같은 seed면 채널을 몇 개 선택하든 항상 동일한 hold-out 분할
            imgs = _list_images(train_img_dir, img_ext)
            random.Random(seed).shuffle(imgs)
            n_val = max(1, int(len(imgs) * holdout_ratio))
            val_set, train_set = imgs[:n_val], imgs[n_val:]

            for p in train_set:
                _copy_pair(p, train_lbl_dir, ti, tl, channel, false_color)
            for p in val_set:
                _copy_pair(p, train_lbl_dir, vi, vl, channel, false_color)
            if channel is not None:
                log(f"  [yaml] TIFF 채널 {channel} 추출 사용 ({len(channel)}채널 입력)")
            log(f"  [yaml] hold-out — Train {len(train_set)} / Val {len(val_set)} "
                f"(ratio={holdout_ratio}, seed={seed})")
            val_names = [os.path.splitext(os.path.basename(p))[0] for p in val_set]
            shown = ", ".join(val_names[:8])
            more = f" 외 {len(val_names)-8}개" if len(val_names) > 8 else ""
            log(f"  [yaml] Val 파일: {shown}{more}")

        content = {
            "train": os.path.abspath(ti),
            "val":   os.path.abspath(vi),
            "nc":    len(class_names),
            "names": names,
        }
        if channel is not None:
            content["channels"] = len(channel)
        else:
            n_ch = _detect_yaml_channels(train_set, img_ext, log)
            if n_ch:
                content["channels"] = n_ch
    else:
        raise ValueError(f"알 수 없는 split_mode: {split_mode}")

    # Augmentation 키 병합 (있을 때만)
    if aug:
        content.update(aug)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, allow_unicode=True, sort_keys=False)
    log(f"  [yaml] 저장: {yaml_path}")
    return yaml_path


# ──────────────────────────────────────────────────────────────────
#  학습
# ──────────────────────────────────────────────────────────────────
def _extract_metrics(trainer):
    """
    trainer에서 epoch별 지표를 안전하게 추출.
    Ultralytics 버전마다 키가 조금씩 달라 여러 경로를 시도.
    반환: dict(box_loss, cls_loss, dfl_loss, mAP50, mAP50_95, precision, recall)
    """
    out = {"box_loss": None, "cls_loss": None, "dfl_loss": None,
           "mAP50": None, "mAP50_95": None, "precision": None, "recall": None}

    # 손실 — trainer.loss_items / trainer.tloss
    label_loss = getattr(trainer, "loss_names", None)  # ('box_loss','cls_loss','dfl_loss')
    loss_items = getattr(trainer, "tloss", None)
    if loss_items is None:
        loss_items = getattr(trainer, "loss_items", None)
    try:
        if label_loss is not None and loss_items is not None:
            if isinstance(loss_items, dict):   # RT-DETR은 {이름: 값} dict로 준다
                loss_items = [loss_items.get(n, 0.0) for n in label_loss]
            vals = [float(v) for v in loss_items]
            for name, v in zip(label_loss, vals):
                if "box" in name or "giou" in name: out["box_loss"] = v   # RT-DETR은 giou_loss
                elif "cls" in name: out["cls_loss"] = v
                elif "dfl" in name: out["dfl_loss"] = v
    except Exception:
        pass

    # 검증 지표 — trainer.metrics dict
    metrics = getattr(trainer, "metrics", None)
    if isinstance(metrics, dict):
        def pick(*keys):
            for k in keys:
                if k in metrics:
                    try:
                        return float(metrics[k])
                    except Exception:
                        pass
            return None
        out["mAP50"]     = pick("metrics/mAP50(B)", "metrics/mAP50", "mAP50")
        out["mAP50_95"]  = pick("metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95")
        out["precision"] = pick("metrics/precision(B)", "metrics/precision")
        out["recall"]    = pick("metrics/recall(B)", "metrics/recall")
    return out


def train(config, on_epoch=None, on_log=None, should_stop=None):
    """
    Ultralytics YOLO Detection 학습.

    config 키:
      model      : "yolov8n.pt" 등
      data_yaml  : data.yaml 경로
      epochs, batch, imgsz, lr0, optimizer, patience, seed
      project, name  : 결과 저장 경로 (project/name/)
    콜백:
      on_epoch(epoch, total, metrics_dict)
      on_log(str)
      should_stop() -> bool
    반환: dict(success, best_pt, save_dir, error)
    """
    def log(msg):
        if on_log: on_log(msg)

    # GUI 내장 실행이므로 wandb 등 외부 로거 자동 연동을 끈다 (예기치 않은 로그인/업로드 방지)
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("YOLO_VERBOSE", "True")

    try:
        from ultralytics import settings as yolo_settings
        import torch
    except ImportError as e:
        return {"success": False, "error":
                f"ultralytics/torch 미설치: {e}\n→ pip install ultralytics"}

    # Ultralytics 내장 로거(wandb/comet 등) 비활성화
    try:
        yolo_settings.update({"wandb": False, "comet": False, "tensorboard": False,
                              "mlflow": False, "dvc": False, "neptune": False})
    except Exception:
        pass

    model_name = config.get("model", "yolov8n.pt")
    data_yaml  = config.get("data_yaml")
    epochs     = int(config.get("epochs", 100))
    device     = 0 if torch.cuda.is_available() else "cpu"
    # project를 절대경로로 만들어 Ultralytics가 runs/detect/ 아래로 중첩 저장하지 않게 함
    project = config.get("project", "YOLO_Project")
    project = os.path.abspath(project)

    log(f"[학습 시작] 디바이스: {'GPU(cuda:0)' if device == 0 else 'CPU'}")
    log(f"  모델: {model_name} | Epochs: {epochs} | "
        f"Batch: {config.get('batch', 16)} | imgsz: {config.get('imgsz', 640)}")
    log(f"  data.yaml: {data_yaml}")

    try:
        model = load_detector(resolve_model(model_name, log))
    except Exception as e:
        return {"success": False, "error": f"모델 로드 실패 ({model_name}): {e}"}

    # epoch별 mAP를 추적해 best를 기록 (K-Fold 집계용)
    best_map50 = {"v": 0.0}
    best_map5095 = {"v": 0.0}
    best_epoch = {"v": 0}      # best mAP50이 나온 epoch (1-indexed)
    last_epoch = {"v": 0}      # 실제로 돈 마지막 epoch (early stop 대응)

    # ── epoch 종료 콜백: 지표 추출 → GUI로 전달 + 중단 처리 ──────────
    def _on_fit_epoch_end(trainer):
        ep = int(getattr(trainer, "epoch", 0)) + 1   # 0-indexed → 1-indexed
        last_epoch["v"] = max(last_epoch["v"], ep)
        metrics = _extract_metrics(trainer)
        if metrics.get("mAP50") is not None and metrics["mAP50"] > best_map50["v"]:
            best_map50["v"] = metrics["mAP50"]
            best_map5095["v"] = metrics.get("mAP50_95") or best_map5095["v"]
            best_epoch["v"] = ep
        if on_epoch:
            on_epoch(ep, epochs, metrics)
        if should_stop and should_stop():
            log("  [중단 요청] 다음 epoch에서 학습을 멈춥니다...")
            trainer.stop = True            # Ultralytics가 인식하는 중단 플래그
            try:
                trainer.stop_training = True
            except Exception:
                pass

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    # 시드: 켜면 고정(재현 가능), 끄면 매번 다른 시드 + 비결정 모드
    seed_on = bool(config.get("seed_on", True))
    seed = int(config.get("seed", 42)) if seed_on else random.randrange(2**31)

    train_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        batch=int(config.get("batch", 16)),
        imgsz=int(config.get("imgsz", 640)),
        device=device,
        project=project,
        name=config.get("name", "train"),
        exist_ok=True,
        optimizer=config.get("optimizer", "auto"),
        lr0=float(config.get("lr0", 0.01)),
        patience=int(config.get("patience", 50)),
        seed=seed,
        deterministic=seed_on,
        # Windows + GUI(QThread)에서 workers>0이면 DataLoader spawn 교착으로 멈춤
        workers=int(config.get("workers", 0 if os.name == "nt" else 4)),
        plots=True,
        verbose=True,
    )
    # augmentation 등 추가 키 병합 (있을 때만 전달)
    for k in ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
              "shear", "perspective", "flipud", "fliplr", "mosaic",
              "mixup", "copy_paste", "erasing"):
        if k in config and config[k] is not None:
            train_kwargs[k] = config[k]
    # GUI에서 넘어온 aug dict — data.yaml이 아니라 학습 인자로 줘야 실제 적용됨
    for k, v in (config.get("aug") or {}).items():
        if v is not None:
            train_kwargs[k] = v

    try:
        results = model.train(**train_kwargs)
    except Exception as e:
        import traceback
        return {"success": False, "error": f"{e}\n{traceback.format_exc()}"}

    save_dir = str(getattr(results, "save_dir", "") or
                   getattr(getattr(model, "trainer", None), "save_dir", ""))
    best_pt = os.path.join(save_dir, "weights", "best.pt") if save_dir else ""
    # 중단 등으로 best가 없으면 last를 best로 복사해 둔다
    last_pt = os.path.join(save_dir, "weights", "last.pt") if save_dir else ""
    if best_pt and not os.path.isfile(best_pt) and os.path.isfile(last_pt):
        shutil.copy(last_pt, best_pt)
        log("  best.pt가 없어 last.pt를 best.pt로 저장했습니다")
    stopped = bool(should_stop and should_stop())

    # best epoch — Ultralytics trainer.best_epoch(0-indexed) 우선, 없으면 콜백 추적값
    n_best = best_epoch["v"]
    try:
        tb = getattr(getattr(model, "trainer", None), "best_epoch", None)
        if tb is not None:
            n_best = int(tb) + 1
    except Exception:
        pass
    n_best = max(1, min(int(n_best or 1), max(1, last_epoch["v"])))

    log(f"\n✔ 학습 완료")
    if save_dir:
        log(f"  결과 폴더: {save_dir}")
    if best_pt and os.path.isfile(best_pt):
        log(f"  Best 모델: {best_pt}")
    log(f"  Best epoch: {n_best} / 실행 {last_epoch['v']} epoch")

    return {"success": True, "best_pt": best_pt, "save_dir": save_dir, "error": None,
            "best_map50": best_map50["v"], "best_map5095": best_map5095["v"],
            "best_epoch": n_best, "last_epoch": last_epoch["v"],
            "stopped": stopped, "seed": seed}


# ──────────────────────────────────────────────────────────────────
#  K-Fold
# ──────────────────────────────────────────────────────────────────
def _stratified_folds(imgs, lbl_dir, k, seed):
    """이미지 안에 있는 클래스 기준으로 k개 fold에 고르게 배분 (희귀 클래스 우선 탐욕 배정).
    imgs는 이미 섞인 목록. 반환: k개 리스트"""
    img_cls = {}
    cls_total = {}
    for p in imgs:
        ids = set()
        lp = _label_for(p, lbl_dir)
        if os.path.isfile(lp):
            with open(lp, encoding="utf-8") as f:
                for line in f:
                    t = line.split()
                    if t:
                        ids.add(t[0])
        img_cls[p] = ids
        for c in ids:
            cls_total[c] = cls_total.get(c, 0) + 1
    # 희귀 클래스를 가진 이미지부터 (sorted는 안정 정렬 → 같은 순위는 섞인 순서 유지)
    order = sorted(imgs, key=lambda p: min((cls_total[c] for c in img_cls[p]), default=10**9))
    folds = [[] for _ in range(k)]
    fold_cls = [dict() for _ in range(k)]   # fold별 클래스 이미지 수
    rng = random.Random(seed)
    for p in order:
        # 이 이미지의 클래스가 이미 적게 들어간 fold, 같으면 이미지 수가 적은 fold 우선
        def score(i):
            return (sum(fold_cls[i].get(c, 0) / cls_total[c] for c in img_cls[p]),
                    len(folds[i]), rng.random())
        best = min(range(k), key=score)
        folds[best].append(p)
        for c in img_cls[p]:
            fold_cls[best][c] = fold_cls[best].get(c, 0) + 1
    return folds


def build_kfold_yamls(out_dir, class_names, train_img_dir, train_lbl_dir,
                      k=5, img_ext="png", aug=None, seed=42, channel=None, log=print, false_color=None,
                      stratified=False, inner_val=False):
    """
    Train 데이터를 K등분해 fold별 dataset 폴더 + data.yaml 생성.
    기본: fold i 는 i번째 조각을 val, 나머지를 train 으로 사용.
    inner_val=True: i번째 조각은 test(평가 전용), (i+1)번째 조각은 val(best 선택용), 나머지 train.
    stratified=True: 이미지 안 클래스 기준으로 fold에 고르게 배분.

    반환: [yaml_path_fold1, ..., yaml_path_foldK]
    """
    import yaml
    os.makedirs(out_dir, exist_ok=True)
    names = {i: n for i, n in enumerate(class_names)}
    channel = _normalize_channels(channel)
    if channel is not None:
        log(f"  [K-Fold] TIFF 채널 {channel} 추출 사용 ({len(channel)}채널 입력)")

    imgs = _list_images(train_img_dir, img_ext)
    if len(imgs) < k:
        raise ValueError(f"이미지 {len(imgs)}개 < Fold 수 {k} — K를 줄이거나 데이터를 늘리세요")
    # 채널 추출 전 원본 파일 목록에서 시드 고정 분할 → 채널 선택과 무관하게 동일 fold
    random.Random(seed).shuffle(imgs)
    log(f"  [K-Fold] 분할 시드 seed={seed}")

    if inner_val and k < 3:
        raise ValueError("fold 안에서 검증 따로 두기는 K가 3 이상이어야 합니다")
    if stratified:
        folds = _stratified_folds(imgs, train_lbl_dir, k, seed)
        log("  [K-Fold] Stratified 분할 (이미지 안 클래스 기준)")
    else:
        # K등분 (앞쪽 fold가 1개 더 가질 수 있음)
        folds = [imgs[i::k] for i in range(k)]

    yaml_paths = []
    for i in range(k):
        if inner_val:
            test_set = folds[i]
            val_set = folds[(i + 1) % k]
            train_set = [p for j in range(k) if j not in (i, (i + 1) % k) for p in folds[j]]
        else:
            test_set = []
            val_set = folds[i]
            train_set = [p for j in range(k) if j != i for p in folds[j]]

        ds = os.path.join(out_dir, f"fold{i+1}", "dataset")
        # 이전 실행 잔여 파일이 섞이지 않도록 비우고 시작
        shutil.rmtree(ds, ignore_errors=True)
        ti = os.path.join(ds, "train", "images"); tl = os.path.join(ds, "train", "labels")
        vi = os.path.join(ds, "val",   "images"); vl = os.path.join(ds, "val",   "labels")
        for d in (ti, tl, vi, vl):
            os.makedirs(d, exist_ok=True)

        for p in train_set:
            _copy_pair(p, train_lbl_dir, ti, tl, channel, false_color)
        for p in val_set:
            _copy_pair(p, train_lbl_dir, vi, vl, channel, false_color)
        if inner_val:
            si = os.path.join(ds, "test", "images"); sl = os.path.join(ds, "test", "labels")
            for d in (si, sl):
                os.makedirs(d, exist_ok=True)
            for p in test_set:
                _copy_pair(p, train_lbl_dir, si, sl, channel, false_color)

        content = {
            "train": os.path.abspath(ti),
            "val":   os.path.abspath(vi),
            "nc":    len(class_names),
            "names": names,
        }
        if inner_val:
            content["test"] = os.path.abspath(si)
        if channel is not None:
            content["channels"] = len(channel)
        else:
            n_ch = _detect_yaml_channels(train_set, img_ext, log if i == 0 else (lambda *_: None))
            if n_ch:
                content["channels"] = n_ch
        if aug:
            content.update(aug)

        yp = os.path.join(out_dir, f"fold{i+1}", "data.yaml")
        with open(yp, "w", encoding="utf-8") as f:
            yaml.dump(content, f, allow_unicode=True, sort_keys=False)
        yaml_paths.append(yp)
        if inner_val:
            log(f"  [K-Fold] Fold {i+1}/{k}: Train {len(train_set)} / Val {len(val_set)} / Test {len(test_set)}")
        else:
            log(f"  [K-Fold] Fold {i+1}/{k}: Train {len(train_set)} / Val {len(val_set)}")

    return yaml_paths


def build_full_yaml(out_dir, class_names, train_img_dir, train_lbl_dir,
                    img_ext="png", aug=None, channel=None, log=print, false_color=None):
    """전체 데이터를 train=val로 쓰는 data.yaml (K-Fold 후 Full 재학습용)"""
    import yaml
    os.makedirs(out_dir, exist_ok=True)
    names = {i: n for i, n in enumerate(class_names)}
    channel = _normalize_channels(channel)

    if channel is not None:
        # 채널 추출본을 dataset_full 폴더에 생성
        ds = os.path.join(out_dir, "dataset_full")
        # 이전 실행 잔여 파일이 섞이지 않도록 비우고 시작
        shutil.rmtree(ds, ignore_errors=True)
        fi = os.path.join(ds, "images"); fl = os.path.join(ds, "labels")
        for d in (fi, fl):
            os.makedirs(d, exist_ok=True)
        for p in _list_images(train_img_dir, img_ext):
            _copy_pair(p, train_lbl_dir, fi, fl, channel, false_color)
        log(f"  [Full] TIFF 채널 {channel} 추출 ({len(channel)}채널 입력) → {ds}")
        train_img_dir = fi

    content = {
        "train": os.path.abspath(train_img_dir),
        "val":   os.path.abspath(train_img_dir),  # Full 학습: 전체를 val로도 사용
        "nc":    len(class_names),
        "names": names,
    }
    if channel is not None:
        content["channels"] = len(channel)
    else:
        n_ch = _detect_yaml_channels(_list_images(train_img_dir, img_ext), img_ext, log)
        if n_ch:
            content["channels"] = n_ch
    if aug:
        content.update(aug)
    yp = os.path.join(out_dir, "data_full.yaml")
    with open(yp, "w", encoding="utf-8") as f:
        yaml.dump(content, f, allow_unicode=True, sort_keys=False)
    log(f"  [Full] data.yaml 저장: {yp}")
    return yp


def train_kfold(config, on_epoch=None, on_log=None, should_stop=None, on_fold=None):
    """
    K-Fold 학습: fold마다 train() 호출 → 평균±std 집계 → Full 데이터 재학습.

    config 키 (train()과 동일) + kfold_k, train_img_dir, train_lbl_dir, img_ext, class_names, aug
    추가 콜백:
      on_fold(fold_idx, total_folds)  # fold_idx=0,total=0 → Full 학습 시작 신호
    반환: dict(success, best_pt, save_dir, fold_results, mean_map50, std_map50, error)
    """
    import statistics
    def log(msg):
        if on_log: on_log(msg)

    k          = int(config.get("kfold_k", 5))
    project    = os.path.abspath(config.get("project", "YOLO_Project"))
    class_names = config["class_names"]
    aug        = config.get("aug")
    # 분할 시드: split_seed 우선 (학습 시드와 분리 — 채널 조합 실험 시 동일 분할 보장)
    split_seed = config.get("split_seed")
    if split_seed is None:
        split_seed = config.get("seed", 42)

    log(f"[K-Fold 시작] K={k}")
    try:
        yaml_paths = build_kfold_yamls(
            out_dir=project, class_names=class_names,
            train_img_dir=config["train_img_dir"], train_lbl_dir=config["train_lbl_dir"],
            k=k, img_ext=config.get("img_ext", "png"), aug=aug,
            seed=int(split_seed), channel=config.get("channel"), log=log, false_color=config.get("false_color"),
            stratified=bool(config.get("kfold_stratified", False)),
            inner_val=bool(config.get("kfold_inner_val", False)))
    except Exception as e:
        return {"success": False, "error": f"K-Fold 분할 실패: {e}"}
    inner_val = bool(config.get("kfold_inner_val", False))

    fold_results = []
    stopped_fold = False
    for i, yp in enumerate(yaml_paths):
        if should_stop and should_stop():
            log("[중단] K-Fold 중지")
            break
        log(f"\n{'='*46}\n── Fold {i+1}/{k} 학습 ──")
        if on_fold:
            on_fold(i + 1, k)   # 그래프 초기화 신호
        fold_cfg = dict(config)
        fold_cfg["data_yaml"] = yp
        fold_cfg["name"] = f"fold{i+1}"
        res = train(fold_cfg, on_epoch=on_epoch, on_log=on_log, should_stop=should_stop)
        if not res.get("success"):
            return {"success": False, "error": f"Fold {i+1} 학습 실패: {res.get('error')}"}
        if res.get("stopped"):
            stopped_fold = True
        m50, m5095 = res.get("best_map50", 0.0), res.get("best_map5095", 0.0)
        if inner_val and res.get("best_pt") and os.path.isfile(res["best_pt"]):
            # val로 best를 골랐으므로 점수는 빠진 test fold에서 따로 잰다
            try:
                mt = load_detector(res["best_pt"]).val(
                    data=yp, split="test", imgsz=int(config.get("imgsz", 640)),
                    workers=0, plots=False, verbose=False,
                    project=os.path.join(project, f"fold{i+1}"), name="test", exist_ok=True)
                m50, m5095 = float(mt.box.map50), float(mt.box.map)
                log(f"  Fold {i+1} test: mAP50={m50:.4f}  mAP50-95={m5095:.4f}")
            except Exception as e:
                log(f"  [경고] Fold {i+1} test 평가 실패, val 점수를 사용: {e}")
        fold_results.append({
            "fold": i + 1,
            "best_map50": round(m50, 6),
            "best_map5095": round(m5095, 6),
            "best_epoch": int(res.get("best_epoch", 0) or 0),
            "best_pt": res.get("best_pt", ""),
        })

    # ── 집계 ──────────────────────────────────────────────────────
    summary = {"success": True, "fold_results": fold_results, "error": None,
               "stopped": bool(stopped_fold or (should_stop and should_stop()))}
    if fold_results:
        m50 = [r["best_map50"] for r in fold_results]
        m5095 = [r["best_map5095"] for r in fold_results]
        summary["mean_map50"] = round(statistics.mean(m50), 6)
        summary["std_map50"]  = round(statistics.pstdev(m50), 6) if len(m50) > 1 else 0.0
        summary["mean_map5095"] = round(statistics.mean(m5095), 6)
        summary["std_map5095"]  = round(statistics.pstdev(m5095), 6) if len(m5095) > 1 else 0.0
        # Full 재학습 epoch 결정용 — fold별 best epoch 평균 (올림)
        beps = [r["best_epoch"] for r in fold_results if r.get("best_epoch")]
        if beps:
            summary["mean_best_epoch"] = round(statistics.mean(beps), 2)
            summary["full_epochs"] = max(1, int(round(statistics.mean(beps))))
        log(f"\n{'='*46}\n[K-Fold 결과 요약]")
        for r in fold_results:
            log(f"  Fold {r['fold']}: mAP50={r['best_map50']:.4f}  "
                f"mAP50-95={r['best_map5095']:.4f}  best_epoch={r['best_epoch']}")
        log(f"  평균 mAP50    : {summary['mean_map50']:.4f} ± {summary['std_map50']:.4f}")
        log(f"  평균 mAP50-95 : {summary['mean_map5095']:.4f} ± {summary['std_map5095']:.4f}")
        if beps:
            log(f"  평균 Best Epoch: {summary['mean_best_epoch']} "
                f"→ Full 학습 epochs = {summary['full_epochs']}")

        # JSON 저장
        import json
        out = {kk: vv for kk, vv in summary.items() if kk != "success"}
        # kfold_summary.json: 'Full 학습에 K-Fold 평균 epoch 쓰기'가 읽는 파일
        with open(os.path.join(project, "kfold_summary.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        # oof_metrics.json: fold별 검증 지표와 평균 (fold마다 빠진 데이터로 잰 값)
        with open(os.path.join(project, "oof_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    # ── Full 데이터 재학습 (최종 모델) ────────────────────────────
    if config.get("full_retrain", True) and not (should_stop and should_stop()):
        # Full 학습 epoch = fold별 best epoch의 평균 (없으면 설정값 그대로)
        full_epochs = summary.get("full_epochs") or int(config.get("epochs", 100))
        log(f"\n{'='*46}\n[Full Data 학습]  전체 데이터로 최종 모델 생성")
        if summary.get("full_epochs"):
            log(f"  Epochs = {full_epochs} (fold별 best epoch 평균 "
                f"{summary.get('mean_best_epoch')} 기준, 설정값 {config.get('epochs')}에서 조정)")
        if on_fold:
            on_fold(0, 0)   # Full 학습 신호 (그래프 초기화)
        try:
            full_yaml = build_full_yaml(
                out_dir=project, class_names=class_names,
                train_img_dir=config["train_img_dir"], train_lbl_dir=config["train_lbl_dir"],
                img_ext=config.get("img_ext", "png"), aug=aug,
                channel=config.get("channel"), log=log, false_color=config.get("false_color"))
            full_cfg = dict(config)
            full_cfg["data_yaml"] = full_yaml
            full_cfg["name"] = "full"
            full_cfg["epochs"] = full_epochs
            # 전체 데이터를 val로 쓰므로 early stop은 의미 없음 — 지정 epoch를 끝까지
            full_cfg["patience"] = 0
            res_full = train(full_cfg, on_epoch=on_epoch, on_log=on_log, should_stop=should_stop)
            if res_full.get("success"):
                summary["best_pt"]  = res_full.get("best_pt", "")
                summary["save_dir"] = res_full.get("save_dir", "")
                summary["data_yaml"] = full_yaml   # 평가탭 연동용
                log(f"  ✔ Full 최종 모델: {summary['best_pt']}")
        except Exception as e:
            log(f"  [Full 학습 오류] {e}")

    # Full 학습을 안 했거나 실패하면 마지막 fold best를 대표 모델로
    if "best_pt" not in summary and fold_results:
        last_fold = fold_results[-1]["fold"]
        summary["best_pt"]  = fold_results[-1]["best_pt"]
        summary["save_dir"] = os.path.dirname(os.path.dirname(fold_results[-1]["best_pt"]))
        # 대표 모델(fold)과 채널 수가 일치하는 해당 fold의 data.yaml을 평가탭에 연동
        fold_yaml = os.path.join(project, f"fold{last_fold}", "data.yaml")
        if os.path.isfile(fold_yaml):
            summary["data_yaml"] = fold_yaml

    return summary


# ──────────────────────────────────────────────────────────────────
#  Full Data 단독 학습
# ──────────────────────────────────────────────────────────────────
def _best_epoch_from_csv(csv_path):
    """fold의 results.csv에서 mAP50이 가장 높은 epoch 번호를 읽는다 (실패 시 None)"""
    import csv as _csv
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
    except Exception:
        return None
    if not rows:
        return None
    # 헤더 공백 제거 (Ultralytics 버전에 따라 ' metrics/mAP50(B)' 처럼 앞 공백이 붙음)
    key_map = {k.strip(): k for k in rows[0].keys() if k}
    mk = next((key_map[k] for k in ("metrics/mAP50(B)", "metrics/mAP50")
               if k in key_map), None)
    ek = key_map.get("epoch")
    if mk is None or ek is None:
        return None
    best_v, best_e = -1.0, None
    for r in rows:
        try:
            v = float(r[mk]); e = int(float(r[ek]))
        except (TypeError, ValueError):
            continue
        if v > best_v:
            best_v, best_e = v, e
    return best_e


def read_kfold_best_epoch(project):
    """이전 K-Fold 실행에서 fold별 best epoch의 평균을 구한다.

    1) kfold_summary.json의 full_epochs / mean_best_epoch / fold_results[].best_epoch
    2) 없으면(구버전 실행) project/fold_*/results.csv 를 직접 스캔해 mAP50 최고 epoch

    반환: (full_epochs:int|None, mean_best_epoch:float|None)
    """
    import json
    project = os.path.abspath(project)
    path = os.path.join(project, "kfold_summary.json")
    fe = me = None
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        fe = data.get("full_epochs")
        me = data.get("mean_best_epoch")
        if fe is None and me is not None:
            fe = max(1, int(round(float(me))))
        # 구버전 summary(=best_epoch 미기록) 대비 — fold_results에서 직접 평균
        if fe is None:
            beps = [r.get("best_epoch") for r in (data.get("fold_results") or [])
                    if r.get("best_epoch")]
            if beps:
                me = round(sum(beps) / len(beps), 2)
                fe = max(1, int(round(me)))

    # 최후 수단 — fold 폴더의 results.csv에서 직접 계산
    if fe is None:
        beps = []
        for d in sorted(glob.glob(os.path.join(project, "fold[0-9]*")) + glob.glob(os.path.join(project, "fold_*"))):
            e = _best_epoch_from_csv(os.path.join(d, "results.csv"))
            if e:
                beps.append(e)
        if beps:
            me = round(sum(beps) / len(beps), 2)
            fe = max(1, int(round(me)))

    return (int(fe) if fe else None), me


def latest_kfold_run(ckpt_root):
    """checkpoints 폴더 안에서 kfold_summary.json이 있는 가장 최근 실행 폴더 (없으면 None)"""
    cands = glob.glob(os.path.join(ckpt_root, "*", "kfold_summary.json"))
    if not cands:
        return None
    return os.path.dirname(max(cands, key=os.path.getmtime))


def train_full(config, on_epoch=None, on_log=None, should_stop=None, on_fold=None):
    """
    Full Data 단독 학습 — K-Fold 없이 전체 데이터로만 학습 (train=val).

    config 키 (train()과 동일) + train_img_dir, train_lbl_dir, img_ext,
                                 class_names, aug, channel
      epochs      : 사용할 epoch 수 (아래 use_kfold_epochs가 우선)
      use_kfold_epochs : True면 project/kfold_summary.json의 평균 best epoch 사용
    반환: dict(success, best_pt, save_dir, data_yaml, error)
    """
    def log(msg):
        if on_log: on_log(msg)

    project = os.path.abspath(config.get("project", "YOLO_Project"))
    epochs = int(config.get("epochs", 100))

    if config.get("use_kfold_epochs"):
        fe, me = read_kfold_best_epoch(config.get("kfold_run_dir") or project)
        if fe:
            log(f"[Full] 이전 K-Fold 평균 best epoch {me} → epochs={fe} 적용")
            epochs = fe
        else:
            log(f"[Full] kfold_summary.json에서 best epoch를 찾지 못함 "
                f"→ 설정값 epochs={epochs} 사용")

    log(f"\n{'='*46}\n[Full Data 학습]  전체 데이터로 최종 모델 생성  (epochs={epochs})")
    if on_fold:
        on_fold(0, 0)   # Full 학습 신호 (그래프 초기화)
    try:
        full_yaml = build_full_yaml(
            out_dir=project, class_names=config["class_names"],
            train_img_dir=config["train_img_dir"], train_lbl_dir=config["train_lbl_dir"],
            img_ext=config.get("img_ext", "png"), aug=config.get("aug"),
            channel=config.get("channel"), log=log, false_color=config.get("false_color"))
    except Exception as e:
        return {"success": False, "error": f"Full data.yaml 생성 실패: {e}"}

    full_cfg = dict(config)
    full_cfg["data_yaml"] = full_yaml
    full_cfg["name"] = config.get("name") or "full"
    full_cfg["epochs"] = epochs
    # 전체 데이터를 val로 쓰므로 early stop은 의미 없음 — 지정 epoch를 끝까지
    full_cfg["patience"] = 0

    res = train(full_cfg, on_epoch=on_epoch, on_log=on_log, should_stop=should_stop)
    if res.get("success"):
        res["data_yaml"] = full_yaml
        log(f"  ✔ Full 최종 모델: {res.get('best_pt')}")
    return res
