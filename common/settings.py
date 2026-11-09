"""최근 연 폴더/파일과 마지막 설정값 저장. QSettings 사용 (HSI Viewer와 같은 방식).
Windows에서는 레지스트리에 저장된다."""
import json
from datetime import datetime

from PySide6.QtCore import QSettings
from PySide6.QtGui import QGuiApplication

ORG = "VisionToolkit"
APP = "VisionToolkit"
MAX_RECENT = 10


def _settings():
    return QSettings(ORG, APP)


# ---- 최근 열기 목록: 각 항목은 {"tool": 도구 키, "path": 경로} ----
def recent_items():
    """전체 도구의 최근 항목을 최신 순으로 돌려준다."""
    out = []
    for s in _settings().value("recent", []) or []:
        try:
            out.append(json.loads(s))
        except (TypeError, ValueError):
            pass  # 깨진 항목은 무시
    return out


def push_recent(tool, path):
    """최근 항목에 추가 (같은 도구+경로가 있으면 맨 앞으로). 마지막 폴더도 같이 기억."""
    path = str(path)
    items = [i for i in recent_items() if not (i.get("tool") == tool and i.get("path") == path)]
    items.insert(0, {"tool": tool, "path": path, "time": datetime.now().strftime("%Y-%m-%d")})
    _settings().setValue("recent", [json.dumps(i, ensure_ascii=False) for i in items[:MAX_RECENT]])


def clear_recent():
    _settings().remove("recent")


# ---- 마지막 설정값 (도구별로 이름 붙여 저장) ----
def get_value(tool, key, default=None):
    return _settings().value(f"{tool}/{key}", default)


def set_value(tool, key, value):
    _settings().setValue(f"{tool}/{key}", value)


# ---- 창 위치/크기: 저장하고, 복원할 때 지금 연결된 모니터 안에 있는지 확인 ----
def save_window(tool, win):
    set_value(tool, "geometry", win.saveGeometry())


def restore_window(tool, win):
    """저장된 위치로 복원. 어느 모니터 화면과도 충분히 겹치지 않으면 주 모니터 가운데로 옮긴다."""
    g = get_value(tool, "geometry")
    if g:
        win.restoreGeometry(g)
    if not _on_screen(win):
        screen = QGuiApplication.primaryScreen().availableGeometry()
        win.resize(min(win.width(), screen.width()), min(win.height(), screen.height()))
        frame = win.frameGeometry()
        frame.moveCenter(screen.center())  # 주 모니터 가운데
        win.move(frame.topLeft())


def _on_screen(win):
    """창 중심이 어느 모니터 안에 있고, 그 모니터와 100x30 이상 겹치는지"""
    frame = win.frameGeometry()
    for scr in QGuiApplication.screens():
        area = scr.availableGeometry()
        part = frame.intersected(area)
        if area.contains(frame.center()) and part.width() >= 100 and part.height() >= 30:
            return True
    return False
