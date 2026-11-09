"""레퍼런스 스펙트럼 라이브러리 + 스펙트럼 각도(SAM) 기반 재질 분류."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .processing import spectral_angle


@dataclass
class Reference:
    name: str
    spectrum: np.ndarray          # (B,) 정규화된 특징 공간 (SNV / 배경정규화 플래그는 라이브러리에 저장)
    n: int = 1                    # 평균에 쓰인 스펙트럼 수
    source: str = ""              # 출처 (파일명, "cluster C0" 등)


@dataclass
class Classification:
    name: str
    second: str
    angle: float
    margin: float                 # 2순위 각도 - 1순위 각도 (작을수록 애매)
    uncertain: bool
    ranked: list[tuple[float, str]] = field(default_factory=list)


class ReferenceLibrary:
    def __init__(self, use_snv: bool = True, bg_norm: bool = True):
        self.refs: dict[str, Reference] = {}
        self.use_snv = use_snv
        self.bg_norm = bg_norm
        self.margin = 0.02
        self.path: str = ""

    # ------------------------------------------------------------------ 편집
    def add(self, name: str, spectrum: np.ndarray, source: str = "", merge: bool = True) -> Reference:
        spectrum = np.asarray(spectrum, np.float32).ravel()
        name = name.strip()
        if merge and name in self.refs:
            r = self.refs[name]
            if len(r.spectrum) == len(spectrum):
                r.spectrum = (r.spectrum * r.n + spectrum) / (r.n + 1)
                r.n += 1
                r.source = (r.source + ", " + source).strip(", ") if source else r.source
                return r
        r = Reference(name=name, spectrum=spectrum, n=1, source=source)
        self.refs[name] = r
        return r

    def remove(self, name: str):
        self.refs.pop(name, None)

    def rename(self, old: str, new: str):
        if old in self.refs and new.strip():
            r = self.refs.pop(old)
            r.name = new.strip()
            self.refs[r.name] = r

    def clear(self):
        self.refs.clear()

    def names(self) -> list[str]:
        return list(self.refs.keys())

    def __len__(self):
        return len(self.refs)

    @property
    def n_bands(self) -> int | None:
        for r in self.refs.values():
            return len(r.spectrum)
        return None

    # ------------------------------------------------------------------ 분류
    def classify(self, spectrum: np.ndarray) -> Classification | None:
        if not self.refs:
            return None
        ranked = sorted((spectral_angle(spectrum, r.spectrum), name) for name, r in self.refs.items())
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else (float("inf"), "")
        margin = second[0] - best[0]
        return Classification(name=best[1], second=second[1], angle=best[0], margin=margin,
                              uncertain=margin < self.margin, ranked=ranked)

    def classify_many(self, features: np.ndarray) -> list[Classification | None]:
        return [self.classify(f) for f in features]

    # ------------------------------------------------------------------ 저장
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "use_snv": self.use_snv,
            "bg_norm": self.bg_norm,
            "margin": self.margin,
            "references": [
                {"name": r.name, "spectrum": [float(v) for v in r.spectrum], "n": r.n, "source": r.source}
                for r in self.refs.values()
            ],
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        self.path = path

    @classmethod
    def load(cls, path: str) -> "ReferenceLibrary":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        lib = cls(bool(d.get("use_snv", True)), bool(d.get("bg_norm", True)))
        lib.margin = float(d.get("margin", 0.02))
        for r in d.get("references", []):
            lib.refs[r["name"]] = Reference(r["name"], np.asarray(r["spectrum"], np.float32), int(r.get("n", 1)), r.get("source", ""))
        lib.path = path
        return lib

    def export_csv(self, path: str, x: np.ndarray | None = None):
        names = self.names()
        if not names:
            return
        b = self.n_bands or 0
        x = np.arange(b) if x is None or len(x) != b else x
        arr = np.column_stack([x] + [self.refs[n].spectrum for n in names])
        np.savetxt(path, arr, fmt="%.6f", delimiter=",", header=",".join(["x"] + names), comments="")


def name_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
