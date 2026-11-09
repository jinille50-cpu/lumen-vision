"""폴더/파일 선택창 한 곳 (PySide6 QFileDialog).
시작 위치: 칸에 적힌 경로 → 데이터 탭에서 고른 폴더 → 마지막으로 연 폴더 → 기본 위치"""
import os

from PySide6.QtWidgets import QFileDialog

from common import settings

TOOL = "segmentation"


def _existing_dir(path):
    """경로가 파일이면 그 폴더, 없는 경로면 존재하는 가장 가까운 상위 폴더. 못 찾으면 ''"""
    path = (path or "").strip()
    while path:
        if os.path.isdir(path):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return ""


def start_dir(current="", state=None):
    state = state or {}
    for cand in (current, state.get("train_img_dir"), state.get("train_lbl_dir"),
                 settings.get_value(TOOL, "last_dir", "")):
        d = _existing_dir(cand)
        if d:
            return d
    return ""   # 빈 문자열이면 Qt가 기본 위치를 쓴다


def _remember(path):
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    if folder:
        settings.set_value(TOOL, "last_dir", folder)


def pick_dir(parent, current="", state=None, title="폴더 선택"):
    path = QFileDialog.getExistingDirectory(parent, title, start_dir(current, state))
    if path:
        _remember(path)
    return path


def pick_open(parent, title, filt, current="", state=None):
    path, _ = QFileDialog.getOpenFileName(parent, title, start_dir(current, state), filt)
    if path:
        _remember(path)
    return path


def pick_save(parent, title, filt, default="", state=None):
    """default는 저장할 파일 경로 제안. 그 폴더가 있으면 거기서 시작한다."""
    start = default if _existing_dir(default) else os.path.join(start_dir("", state), os.path.basename(default))
    path, _ = QFileDialog.getSaveFileName(parent, title, start, filt)
    if path:
        _remember(path)
    return path
