"""Labeler 다파장(큐브) 데이터: 읽기는 tools/hsi_viewer의 load_cube, 보기 변환은 common/falsecolor (Object Detection과 같은 계산).
보기 설정 view = {"bands": [R, G, B 밴드 번호], "lo": 하한 %, "hi": 상한 %, "single": 한 밴드 번호 또는 None}"""
import os
import sys

import numpy as np

from common.falsecolor import DEFAULT_HI, DEFAULT_LO, false_color, stretch

CUBE_EXT = (".mat", ".npy", ".hdr", ".h5", ".hdf5")   # 늘 큐브로 여는 확장자 (.tif는 밴드 수를 보고 정한다)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def is_cube_file(path):
    """큐브로 열 파일인가: 큐브 확장자이거나, 밴드가 RGB(3)/RGBA(4)가 아닌 다채널 tif (Qt는 그런 tif의 첫 장만 읽는다)"""
    low = path.lower()
    if low.endswith(CUBE_EXT):
        return True
    if low.endswith((".tif", ".tiff")) and os.path.isfile(path):
        try:
            import tifffile
            with tifffile.TiffFile(path) as tf:
                shape = tf.series[0].shape
        except Exception:  # noqa: BLE001  -- 못 읽으면 일반 그림으로 시도하게 둔다
            return False
        if len(shape) == 3:
            bands = min(shape[0], shape[-1])   # (밴드, H, W) 또는 (H, W, 밴드)
            return bands not in (3, 4)
    return False


def load(path):
    """Cube(data (H, W, 밴드) float32, wavelengths) 를 돌려준다. 못 읽으면 예외"""
    tool_dir = os.path.join(ROOT, "tools", "hsi_viewer")
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)   # hsi_viewer 패키지를 import 하려고 (main.build_hsi_viewer와 같은 방식)
    from hsi_viewer.io_utils import load_cube
    return load_cube(path)


def default_bands(n):
    """처음 보는 큐브의 R, G, B 밴드: 긴 파장 쪽 3/4, 가운데, 1/4 지점"""
    return [min(n - 1, n * 3 // 4), n // 2, n // 4]


def fit_view(view, n):
    """밴드 번호가 이 큐브 밴드 수를 넘으면 기본값으로 바꾼 보기 설정"""
    v = dict(view or {})
    bands = v.get("bands")
    if not bands or len(bands) != 3 or max(bands) >= n or min(bands) < 0:
        v["bands"] = default_bands(n)
    if v.get("single") is not None and not 0 <= v["single"] < n:
        v["single"] = None
    v.setdefault("lo", DEFAULT_LO)
    v.setdefault("hi", DEFAULT_HI)
    return v


def view_rgb(data, view):
    """큐브 (H, W, 밴드) → 화면에 보일 (H, W, 3) uint8 RGB. 한 밴드 보기면 그 밴드를 회색으로"""
    if data.shape[2] == 1:
        g = stretch(data[..., 0], view["lo"], view["hi"])
        return np.stack([g] * 3, axis=-1)
    if view.get("single") is not None:
        g = stretch(data[..., int(view["single"])], view["lo"], view["hi"])
        return np.stack([g] * 3, axis=-1)
    return false_color(data, view["bands"], view["lo"], view["hi"])


def band_label(cube, b):
    """밴드 번호 옆에 붙일 글자: 파장을 알면 '650.2 nm'"""
    return f"{cube.wavelengths[b]:.1f} nm" if cube.has_wavelengths else f"밴드 {b}"


def class_means(data, label_arr):
    """클래스별 평균/표준편차 스펙트럼 {마스크 값: (평균, 표준편차, 픽셀 수)} (0=배경은 뺌)"""
    out = {}
    for v in np.unique(label_arr):
        if v == 0:
            continue
        px = data[label_arr == v]   # (픽셀 수, 밴드)
        out[int(v)] = (px.mean(0), px.std(0), len(px))
    return out
