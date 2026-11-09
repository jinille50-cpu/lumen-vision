"""Object Detection 도구 창(YOLO, RT-DETR, 폴더 이름은 yolo 그대로): 위쪽 단계 표시줄(데이터/학습/평가/추론/내보내기) + 아래 탭 페이지. Vision Toolkit 홈에서 열린다.
(원래 yolo_main.py의 MainWindow를 PySide6용으로 옮긴 것. 색과 글꼴은 common 테마가 정한다.)"""
import os

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from tools.yolo.work_state import WorkState
from tools.yolo.tabs.data_tab import DataTab
from tools.yolo.tabs.train_tab import TrainTab
from tools.yolo.tabs.eval_tab import EvalTab
from tools.yolo.tabs.infer_tab import InferTab
from tools.yolo.tabs.export_tab import ExportTab

STEPS = ["데이터", "학습", "평가", "추론", "내보내기"]
PROJECT_PATHS = ["train_img_dir", "train_lbl_dir", "val_img_dir", "val_lbl_dir"]   # 프로젝트에 상대경로로 저장할 데이터 폴더


def _setup_matplotlib_font():
    """matplotlib 한글/영문 폰트 깨짐 방지"""
    try:
        import matplotlib
        for font in ["Malgun Gothic", "AppleGothic", "NanumGothic",
                     "DejaVu Sans", "sans-serif"]:
            try:
                matplotlib.rcParams['font.family'] = font
                matplotlib.rcParams['axes.unicode_minus'] = False
                break
            except Exception:
                continue
    except Exception:
        pass


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        _setup_matplotlib_font()
        self.shared_state = {}  # 탭 간 공유 데이터
        self.shared_state["work"] = WorkState()   # 모델/config/데이터 경로 (평가/추론이 미리 채움)
        self.done = [False] * len(STEPS)
        self.current = 0
        self.runs = []   # 이번 프로젝트에서 학습한 best.pt 목록 (프로젝트 저장 때 같이 저장)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 단계 표시줄: 데이터 → 학습 → 평가 → 추론 → 내보내기 (눌러서 이동)
        bar = QFrame()
        bar.setObjectName("StepBar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(16, 8, 16, 8)
        self.step_btns = []
        for i, name in enumerate(STEPS):
            if i:
                arrow = QLabel("→")
                arrow.setObjectName("StepArrow")
                bl.addWidget(arrow)
            btn = QPushButton(name)
            btn.setObjectName("Step")
            btn.clicked.connect(lambda _=False, k=i: self.go_step(k))
            bl.addWidget(btn)
            self.step_btns.append(btn)
        bl.addStretch(1)
        layout.addWidget(bar)

        # 탭 페이지
        self.pages = QStackedWidget()
        self.tab_data  = DataTab(self.shared_state)
        self.tab_train = TrainTab(self.shared_state)
        self.tab_eval  = EvalTab(self.shared_state)
        self.tab_infer = InferTab(self.shared_state)
        self.tab_export = ExportTab(self.shared_state)

        # 학습탭 → 데이터탭 직접 참조 (저장 버튼 없이도 경로 읽기)
        self.tab_train.set_data_tab(self.tab_data)

        tabs = [self.tab_data, self.tab_train, self.tab_eval, self.tab_infer, self.tab_export]
        for i, tab in enumerate(tabs):
            self.pages.addWidget(tab)
            tab.step_done.connect(lambda k=i: self._step_done(k))
        layout.addWidget(self.pages, 1)
        self.go_step(0)

    # ── 단계 이동/표시 ───────────────────────────────────────────
    def go_step(self, k):
        self.pages.setCurrentIndex(k)
        self.current = k
        self._refresh_steps()

    def _step_done(self, k):
        """k번 단계가 끝났다. 데이터는 '저장하고 다음'이라 학습 탭으로 넘어간다."""
        self.done[k] = True
        if k == 0:
            self.go_step(1)
        else:
            self._refresh_steps()
        if k == 1:
            # 학습 탭은 끝남 신호를 보낸 '뒤에' best.pt를 적으므로, 지금 처리가 끝난 다음에 읽는다
            QTimer.singleShot(0, self._remember_run)

    def _remember_run(self):
        best = self.shared_state.get("best_pt", "")
        if best and best not in self.runs:
            self.runs.append(best)

    # ── 프로젝트 (main.py 제목줄의 프로젝트 저장/불러오기가 부른다, 파일 읽기/쓰기는 common/project.py) ──
    def project_state(self):
        data = self.tab_data.get_values()
        paths = {k: data.pop(k) for k in PROJECT_PATHS}   # pop: dict에서 꺼내면서 지운다 (경로는 상대경로로 따로 저장)
        paths["data_yaml"] = self.shared_state["work"].get("data_yaml")   # 평가/내보내기에 쓰는 data.yaml
        return {"paths": paths, "runs": list(self.runs),
                "values": {"data": data, "train": self.tab_train.get_values()}}

    def apply_project(self, state):
        """프로젝트 값을 칸에 채운다. 돌려주는 값: 상태 표시줄에 붙일 짧은 안내 (없으면 "")."""
        d = self.tab_data
        paths, values = state.get("paths", {}), state.get("values", {})
        for edit in (d.train_img_edit, d.train_lbl_edit, d.val_img_edit, d.val_lbl_edit):
            edit.clear()   # 이전 작업 값이 남아 섞이지 않게
        d.class_table.setRowCount(0)
        d.file_list.clear(); d.label_list.clear()
        d.set_values({**values.get("data", {}), **{k: paths.get(k, "") for k in PROJECT_PATHS}})
        self.tab_train.set_values(values.get("train", {}))
        ws = self.shared_state["work"]
        if os.path.isfile(paths.get("data_yaml", "")):
            ws.set("data_yaml", paths["data_yaml"])
        self.runs = list(state.get("runs", []))
        for best in reversed(self.runs):   # 가장 최근 학습 결과를 평가/추론/내보내기 탭의 작업 모델로
            if os.path.isfile(best):
                ws.set_model(best)
                break
        missing = [k for k in PROJECT_PATHS if paths.get(k) and not os.path.isdir(paths[k])]
        return f"찾지 못한 데이터 폴더 {len(missing)}개" if missing else ""

    def _refresh_steps(self):
        for i, btn in enumerate(self.step_btns):
            btn.setProperty("done", self.done[i])
            btn.setProperty("current", i == self.current)
            btn.style().unpolish(btn)  # 속성이 바뀌면 스타일을 다시 입혀야 색이 바뀐다
            btn.style().polish(btn)
