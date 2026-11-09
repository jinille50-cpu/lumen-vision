"""
pair_utils.py — 이미지 ↔ 라벨 파일을 파일명(stem) 기준으로 짝지어 주는 공통 유틸리티

정렬 순서(인덱스)로 짝지으면 한쪽에 파일이 하나만 빠져도 이후가 전부 어긋나므로,
다음 순서로 매칭합니다.
  1) 같은 stem              : 0001.png ↔ 0001.png
  2) stem + 접미사(자동 감지) : 10000.png ↔ 10000_GT.png  (_GT, _mask, _label ...)
  3) 이름으로 거의 매칭이 안 되면(절반 미만) 기존처럼 정렬 순서로 매칭 (경고)

사용 예:
    from tools.segmentation.utils.pair_utils import pair_dirs
    img_files, lbl_files, info = pair_dirs(img_dir, lbl_dir, "png", "png")
    for m in info["messages"]: log(m)
"""

import os
import re
import glob

# 마스크 파일명 접미사 후보
_SUFFIXES = ("", "_GT", "_gt", "_mask", "_Mask", "_label", "_Label",
             "_seg", "_lbl", "-mask", "-label")


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def detect_suffix(img_stems, lbl_stems):
    """가장 많이 매칭되는 라벨 접미사와 매칭 수 반환"""
    img_stems, lbl_stems = set(img_stems), set(lbl_stems)
    cands = list(_SUFFIXES)
    # 라벨 이름 끝의 "_xxx" 패턴도 후보에 추가 (샘플 일부만)
    for s in sorted(lbl_stems)[:50]:
        m = re.match(r"^.+?([_\-][A-Za-z]+)$", s)
        if m:
            cands.append(m.group(1))
    best, best_n = "", 0
    for suf in dict.fromkeys(cands):
        n = sum(1 for st in img_stems if st + suf in lbl_stems)
        if n > best_n:
            best, best_n = suf, n
    return best, best_n


def pair_lists(img_files, lbl_files):
    """
    이미지/라벨 경로 목록을 짝지음.
    반환: (pairs, info)
      pairs : [(img_path, lbl_path 또는 None), ...]  — img_files 순서 유지
      info  : {"mode": "name"|"index", "suffix": str, "n_paired": int,
               "missing": [img stem...], "orphan": [lbl stem...], "messages": [str...]}
    """
    info = {"mode": "name", "suffix": "", "n_paired": 0,
            "missing": [], "orphan": [], "messages": []}
    msgs = info["messages"]
    if not img_files:
        return [], info

    lbl_by_stem = {_stem(p): p for p in lbl_files}
    suffix, n_match = detect_suffix([_stem(p) for p in img_files], lbl_by_stem)

    # 이름 매칭이 절반도 안 되면 기존 방식(정렬 순서)으로 fallback
    if lbl_files and n_match < min(len(img_files), len(lbl_files)) / 2:
        info["mode"] = "index"
        imgs_s, lbls_s = list(img_files), sorted(lbl_files)
        pairs = [(ip, lbls_s[i] if i < len(lbls_s) else None)
                 for i, ip in enumerate(imgs_s)]
        info["n_paired"] = min(len(imgs_s), len(lbls_s))
        msgs.append("  [경고] 파일명으로 짝을 찾지 못해 정렬 순서로 매칭합니다 "
                    "(이미지/라벨 이름 규칙을 맞추는 것을 권장)")
        if len(imgs_s) != len(lbls_s):
            msgs.append(f"  [경고] 이미지 {len(imgs_s)}개 / 라벨 {len(lbls_s)}개 — 수 불일치")
        return pairs, info

    info["suffix"] = suffix
    pairs, used = [], set()
    for ip in img_files:
        key = _stem(ip) + suffix
        lp = lbl_by_stem.get(key)
        if lp is None:
            info["missing"].append(_stem(ip))
        else:
            used.add(key)
        pairs.append((ip, lp))
    info["n_paired"] = len(used)
    info["orphan"] = sorted(set(lbl_by_stem) - used)

    if suffix:
        msgs.append(f"  라벨 접미사 자동 감지: '{suffix}'  "
                    f"(예: {_stem(img_files[0])} ↔ {_stem(img_files[0])}{suffix})")
    msgs.append(f"  파일명 매칭: {info['n_paired']}쌍")
    if info["missing"]:
        msgs.append(f"  [경고] 라벨 없는 이미지 {len(info['missing'])}개 — 제외 "
                    f"(예: {info['missing'][:5]})")
    if info["orphan"]:
        msgs.append(f"  [경고] 이미지 없는 라벨 {len(info['orphan'])}개 — 무시 "
                    f"(예: {info['orphan'][:5]})")
    return pairs, info


def pair_dirs(img_dir, lbl_dir, img_ext, lbl_ext):
    """
    폴더 기준 매칭. 라벨이 있는 쌍만 반환.
    반환: (img_files, lbl_files, info)  — 두 목록의 길이/순서가 일치
    """
    img_files = sorted(glob.glob(os.path.join(img_dir, f"*.{img_ext}")))
    lbl_files = sorted(glob.glob(os.path.join(lbl_dir, f"*.{lbl_ext}")))
    pairs, info = pair_lists(img_files, lbl_files)
    pairs = [(ip, lp) for ip, lp in pairs if lp is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs], info
