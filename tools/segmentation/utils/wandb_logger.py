"""
wandb_logger.py — W&B 기록. 학습 지표, 설정 json, 모델 파일, 양자화 비교표만 올린다.

이미지와 원본 데이터는 올리지 않는다. 폴더 경로, 콘솔 로그, 코드, 실행 환경 정보도 올리지 않는다.
W&B 쪽 오류(미설치, 네트워크 등)가 나도 학습과 평가는 멈추지 않는다. 경고만 남기고 기록을 끈다.
"""

import json
import os
import re

DEFAULT_PROJECT = "segtoolkit"


def _settings(wandb):
    return wandb.Settings(
        console="off",              # 콘솔 로그에는 데이터 경로가 들어 있다
        save_code=False,
        disable_git=True,
        x_disable_meta=True,        # 실행 경로, 사용자, 하드웨어 정보
        x_disable_stats=True,       # 시스템 사용량 기록
        x_save_requirements=False,  # 설치 패키지 목록
    )


def _mode(wandb):
    """저장된 API 키가 없으면 오프라인으로 기록한다.

    GUI 앱이라 키가 없을 때 wandb 가 터미널에서 로그인을 물으면 학습 스레드가 멈춘다.
    """
    if os.environ.get("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    return "online" if wandb.api.api_key else "offline"


class WandbRun:
    """학습 한 번 = W&B 실험(run) 하나. enabled=False 면 아무것도 하지 않는다."""

    def __init__(self, config, log=None):
        self.run = None
        self._log = log or (lambda msg: None)
        self.project = config.get("wandb_project") or DEFAULT_PROJECT
        if not config.get("wandb_enabled"):
            return
        self._guard(self._start, config)

    def _guard(self, func, *args):
        """W&B 호출은 전부 여기를 거친다. 실패하면 경고 후 이후 기록을 끈다."""
        try:
            return func(*args)
        except Exception as e:
            first_line = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
            self._log(f"  [경고] W&B 기록 중단 (학습은 계속됨): {first_line}")
            self.run = None

    def _start(self, config):
        import wandb
        # dict 컴프리헨션: 경로 항목(*_dir)을 뺀 설정만 남긴다
        safe = {k: v for k, v in config.items()
                if not k.endswith("_dir") and not k.startswith("wandb_")}
        mode = _mode(wandb)
        self.run = wandb.init(
            project=self.project, config=safe, mode=mode,
            dir=config.get("ckpt_dir"), settings=_settings(wandb),
        )
        self._log(f"  W&B 기록 시작: {self.project}/{self.run.id} ({mode})")
        if mode == "offline":
            self._log("  [주의] W&B 로그인 정보가 없어 오프라인으로 기록합니다. "
                      "터미널에서 wandb login 후 wandb sync 로 올릴 수 있습니다")

    def log(self, metrics):
        if self.run:
            self._guard(self.run.log, metrics)

    def log_model(self, pth_path, alias=None):
        """.pth, json, .onnx 를 아티팩트 하나로 올린다. 같은 모델 이름이면 새 버전이 된다."""
        if self.run:
            self._guard(self._log_model, pth_path, alias)

    def _log_model(self, pth_path, alias):
        import wandb
        base = os.path.splitext(pth_path)[0]
        from tools.segmentation.utils.config_path import find_config, config_path_for
        cfg_file = find_config(pth_path) or config_path_for(pth_path)
        with open(cfg_file, "r", encoding="utf-8") as f:
            model_cfg = json.load(f)
        # 아티팩트 이름에 쓸 수 없는 문자(+, 공백, 괄호 등)는 - 로 바꾼다
        name = re.sub(r"[^A-Za-z0-9._-]", "-", model_cfg["model_name"])
        artifact = wandb.Artifact(name, type="model", metadata=model_cfg)
        files = [pth_path, cfg_file, base + ".onnx"]
        missing = [os.path.basename(p) for p in files if not os.path.isfile(p)]
        for path in files:
            if os.path.isfile(path):
                artifact.add_file(path)
        self.run.log_artifact(artifact, aliases=["latest"] + ([alias] if alias else []))
        self.run.config.update({os.path.basename(base): model_cfg}, allow_val_change=True)
        # 양자화 탭이 같은 실험에 이어서 기록하도록 실험 정보를 모델 옆에 남긴다
        with open(base + "_wandb.json", "w", encoding="utf-8") as f:
            json.dump({"project": self.project, "run_id": self.run.id}, f, indent=2)
        if missing:
            self._log(f"  [주의] W&B 아티팩트에서 빠진 파일: {', '.join(missing)}")
        self._log(f"  W&B 아티팩트 기록: {name}")

    def finish(self):
        if self.run:
            self._guard(self.run.finish)
            self.run = None


def log_quant_table(onnx_path, rows, columns, log=None):
    """양자화 비교표를 학습 때의 W&B 실험에 이어서 기록한다."""
    _log = log or (lambda msg: None)
    info_path = os.path.splitext(onnx_path)[0] + "_wandb.json"
    if not os.path.isfile(info_path):
        _log("  [주의] 이 모델의 W&B 학습 실험 정보가 없어 비교표를 기록하지 않습니다")
        return
    try:
        import wandb
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        run = wandb.init(
            project=info["project"], id=info["run_id"], resume="allow",
            mode=_mode(wandb), dir=os.path.dirname(onnx_path) or None,
            settings=_settings(wandb),
        )
        table = wandb.Table(columns=columns, data=[[row[c] for c in columns] for row in rows])
        run.log({"quantization/compare": table})
        run.finish()
        _log(f"  W&B 비교표 기록: {info['project']}/{info['run_id']}")
    except Exception as e:
        first_line = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
        _log(f"  [경고] W&B 비교표 기록 실패: {first_line}")
