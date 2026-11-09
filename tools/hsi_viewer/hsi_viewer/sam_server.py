"""SAM3 inference server that runs in a separate *process*.

Why a process and not a thread: loading sam3.pt (3.4 GB) and building the model
hold the Python GIL for seconds at a time, which freezes the Qt event loop and makes
Windows show "응답 없음". In its own process the GUI never blocks, and CUDA state is
isolated from the viewer.

Protocol (multiprocessing.Pipe, pickled dicts):
    -> {"cmd": "load", "model": path}
    <- {"ev": "status", "text": ...} ... {"ev": "ready", "ok": bool, "error": str|None}
    -> {"cmd": "segment", "rgb": uint8 HxWx3 RGB, "key": any, "prompts": [...], "conf": f,
        "nms_iou": f}
       (점/박스로 물체 하나 지정: "points": [[x, y]] 또는 "box": [x0, y0, x1, y1] 를 함께 보내면
        텍스트 대신 대화형 SAM3 모델로 그 물체의 마스크 하나만 만든다. 이 모델은 처음 쓸 때 따로 로딩)
    <- {"ev": "status", ...} ... {"ev": "result", "masks": packed bits (N, H*W/8), "shape": (H, W),
        "scores": (N,), "cls": (N,), "boxes": (N,4), "names": [...], "elapsed": f}
       or {"ev": "error", "error": str}
    -> {"cmd": "quit"}
"""
from __future__ import annotations

import time
import traceback


def serve(conn, model_path: str):
    import numpy as np

    predictor = None
    image_key = None
    inter = None              # 점/박스용 대화형 SAM3 (처음 쓸 때 로딩)
    inter_key = None

    def send(**kw):
        try:
            conn.send(kw)
        except Exception:
            pass

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        cmd = msg.get("cmd")
        if cmd == "quit":
            break

        if cmd == "load":
            if predictor is not None:
                send(ev="ready", ok=True, error=None)
                continue
            try:
                t0 = time.time()
                send(ev="status", text="SAM3 모델 로딩 중… (최초 1회, 약 10~20초)")
                from ultralytics.models.sam import SAM3SemanticPredictor

                predictor = SAM3SemanticPredictor(
                    overrides={
                        "conf": 0.2,
                        "imgsz": 1008,      # SAM3 기본 입력 크기 (지정 안 하면 640→644로 돌아 작은 물체 경계가 거칠어짐)
                        "task": "segment",
                        "mode": "predict",
                        "model": msg.get("model", model_path),
                        "save": False,
                        "verbose": False,
                    }
                )
                predictor.setup_model()
                send(ev="status", text=f"SAM3 준비 완료 ({time.time() - t0:.1f}s)")
                send(ev="ready", ok=True, error=None)
            except Exception as e:  # noqa: BLE001
                predictor = None
                send(ev="ready", ok=False, error=f"{e}\n{traceback.format_exc()}")
            continue

        if cmd == "segment":
            try:
                if predictor is None:
                    conn.send({"ev": "error", "error": "model not loaded"})
                    continue
                t0 = time.time()
                rgb = msg["rgb"]
                prompts = list(msg.get("prompts") or [])
                predictor.args.conf = float(msg.get("conf", 0.2))
                predictor.args.iou = float(msg.get("nms_iou", 0.7))
                key = msg.get("key")
                h, w = rgb.shape[:2]
                # SAM3는 긴 변을 1008px로 맞추므로 2000x50 같은 라인스캔 큐브는 높이가 25px로 줄어 아무것도 못 잡는다.
                # 비율이 4:1을 넘으면 짧은 변을 정수배로 늘려 넣고, 결과 마스크/박스를 원래 크기로 되돌린다.
                sy = sx = 1
                if w / max(h, 1) > 4:
                    sy = int(np.ceil(w / h / 4))
                elif h / max(w, 1) > 4:
                    sx = int(np.ceil(h / w / 4))
                if sy > 1 or sx > 1:
                    import cv2
                    rgb_in = cv2.resize(rgb, (w * sx, h * sy), interpolation=cv2.INTER_LINEAR)
                else:
                    rgb_in = rgb
                points, box = msg.get("points"), msg.get("box")
                bboxes = msg.get("bboxes")
                if points or box:
                    if inter is None:
                        send(ev="status", text="점/박스용 SAM3 모델 로딩 중… (최초 1회, 약 15초)")
                        from ultralytics.models.sam import SAM3Predictor

                        inter = SAM3Predictor(overrides={"task": "segment", "mode": "predict", "imgsz": 1008,
                                                         "model": model_path, "save": False, "verbose": False})
                        inter.setup_model()
                    if key is None or key != inter_key:
                        send(ev="status", text="이미지 인코딩 중…")
                        inter.set_image(np.ascontiguousarray(rgb_in[:, :, ::-1]))
                        inter_key = key
                    send(ev="status", text="지정한 물체 분할 중…")
                    if points:
                        # 점 여러 개를 한 물체의 힌트로 쓰려면 (물체 수, 점 수, 2) 모양으로 넣는다 (labels: 1 포함, 0 제외)
                        pt = np.asarray(points, dtype=np.float32).reshape(1, -1, 2) * (sx, sy)
                        lab = np.asarray(msg.get("point_labels") or np.ones(pt.shape[1]), dtype=np.int32).reshape(1, -1)
                        r = inter(points=pt, labels=lab)[0]
                    else:
                        bb = np.asarray(box, dtype=np.float32).reshape(-1, 4) * (sx, sy, sx, sy)
                        r = inter(bboxes=bb)[0]
                    r.names = prompts      # 앱이 보낸 ["직접 지정"]
                else:
                    if key is None or key != image_key:
                        send(ev="status", text="이미지 인코딩 중…" + (f" (비율 보정 ×{max(sx, sy)})" if sy > 1 or sx > 1 else ""))
                        predictor.set_image(np.ascontiguousarray(rgb_in[:, :, ::-1]))
                        image_key = key
                    if bboxes:
                        send(ev="status", text=f"박스 예시 {len(bboxes)}개로 탐지 중…")
                        bb = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
                        bb[:, [0, 2]] *= sx
                        bb[:, [1, 3]] *= sy
                        r = predictor(bboxes=bb)[0]
                    else:
                        send(ev="status", text="탐지 중: " + ", ".join(prompts))
                        r = predictor(text=prompts)[0]
                if r.masks is not None and len(r.boxes):
                    masks = r.masks.data.cpu().numpy().astype(bool)
                    boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                    if sy > 1 or sx > 1:
                        import cv2
                        masks = np.stack([cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                                          for m in masks]) if len(masks) else masks.reshape(0, h, w)
                        boxes[:, [0, 2]] /= sx
                        boxes[:, [1, 3]] /= sy
                    out = {
                        "ev": "result",
                        "masks": np.packbits(masks.reshape(masks.shape[0], -1), axis=1),
                        "shape": (h, w),
                        "scores": r.boxes.conf.cpu().numpy().astype(np.float32),
                        "cls": r.boxes.cls.cpu().numpy().astype(np.int32),
                        "boxes": boxes.round().astype(np.int32),
                        "names": list(r.names) if isinstance(r.names, (list, tuple)) else [r.names[i] for i in sorted(r.names)],
                        "elapsed": time.time() - t0,
                    }
                else:
                    out = {
                        "ev": "result",
                        "masks": np.zeros((0, (h * w + 7) // 8), np.uint8),
                        "shape": (h, w),
                        "scores": np.zeros(0, np.float32),
                        "cls": np.zeros(0, np.int32),
                        "boxes": np.zeros((0, 4), np.int32),
                        "names": prompts,
                        "elapsed": time.time() - t0,
                    }
                conn.send(out)
            except Exception as e:  # noqa: BLE001
                send(ev="error", error=f"{e}\n{traceback.format_exc()}")
            continue


def _entry(conn, model_path):
    try:
        serve(conn, model_path)
    finally:
        try:
            conn.close()
        except Exception:
            pass
