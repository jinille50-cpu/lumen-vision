"""HSI Viewer 테마: 색, 글꼴, 스타일시트는 모두 common/theme.py 에서 가져온다 (B-022).
이 파일은 기존 코드의 `theme.XXX` 이름을 그대로 쓸 수 있게 이어주기만 한다."""
from __future__ import annotations

import os
import sys

# 이 폴더의 위치(tools/hsi_viewer/hsi_viewer)에서 세 단계 위가 Vision Toolkit 루트
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from common.theme import *  # noqa: E402,F401,F403  (색 상수, QSS, hex_to_rgb 등)
from common import theme as _ct  # noqa: E402

ACCENT_DIM = _ct.ACCENT_LIGHT  # 선택 표시 배경 (원래 코드에서 쓰던 이름)


def apply_theme(app) -> None:
    """단독 실행용: 공통 테마 + 그래프 기본값을 적용한다."""
    _ct.apply_theme(app)
    _ct.apply_pyqtgraph_defaults()
