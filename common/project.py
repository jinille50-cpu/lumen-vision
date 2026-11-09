"""프로젝트 파일(project.json) 읽기/쓰기. 도구 하나의 작업 묶음(데이터 경로, 설정값, 학습 결과 폴더)을 파일 하나에 둔다.
경로는 프로젝트 파일이 있는 폴더 기준 상대경로로 저장해서, 폴더째 옮겨도 열리게 한다.

도구 쪽 형식 (dict):
    {"paths": {이름: 경로}, "runs": [학습 결과 폴더, ...], "values": {이름: 값}}
paths와 runs만 상대경로로 바뀌고, values는 그대로 저장된다."""
import json
import os
from datetime import datetime

FORMAT = "vision-toolkit-project"
VERSION = 1


def _to_rel(base, path):
    if not path:
        return ""
    try:
        return os.path.relpath(path, base)
    except ValueError:  # 드라이브가 다르면(C: / D:) 상대경로를 만들 수 없어 절대경로 그대로
        return path


def _to_abs(base, path):
    if not path:
        return ""
    return os.path.normpath(os.path.join(base, path))  # path가 절대경로면 join 결과는 path 그대로


def save(file, tool, state):
    """state를 file에 저장한다."""
    base = os.path.dirname(os.path.abspath(file))
    data = {
        "format": FORMAT,
        "version": VERSION,
        "tool": tool,
        "saved": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "paths": {k: _to_rel(base, v) for k, v in state.get("paths", {}).items()},
        "runs": [_to_rel(base, r) for r in state.get("runs", [])],
        "values": state.get("values", {}),
    }
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load(file):
    """(도구 키, state)를 돌려준다. 경로는 절대경로로 바꿔서 준다. 프로젝트 파일이 아니면 ValueError."""
    with open(file, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("Vision Toolkit 프로젝트 파일이 아닙니다")
    base = os.path.dirname(os.path.abspath(file))
    state = {
        "paths": {k: _to_abs(base, v) for k, v in data.get("paths", {}).items()},
        "runs": [_to_abs(base, r) for r in data.get("runs", [])],
        "values": data.get("values", {}),
    }
    return data.get("tool", ""), state


def is_project(path):
    """최근 목록 항목이 프로젝트 파일인지 (.json이고 형식 표시가 맞는지)."""
    if not (path.lower().endswith(".json") and os.path.isfile(path)):
        return False
    try:
        load(path)
        return True
    except (OSError, ValueError):
        return False
