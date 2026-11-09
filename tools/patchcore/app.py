"""PatchCore 도구 창: 위쪽 단계 표시줄(데이터/모델/학습/평가/추론/내보내기) + 아래 탭 페이지. Vision Toolkit 홈에서 열린다.
anomalib, torch는 core.py 안에서 학습/추론을 누를 때 처음 불러온다."""
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from tools.patchcore.tabs.data_tab import DataTab
from tools.patchcore.tabs.eval_tab import EvalTab
from tools.patchcore.tabs.export_tab import ExportTab
from tools.patchcore.tabs.infer_tab import InferTab
from tools.patchcore.tabs.model_tab import ModelTab
from tools.patchcore.tabs.train_tab import TrainTab
from tools.patchcore.ui_common import WorkState

STEPS = ["데이터", "모델", "학습", "평가", "추론", "내보내기"]


def _setup_matplotlib_font():
    """matplotlib 한글 폰트 깨짐 방지"""
    try:
        import matplotlib
        matplotlib.rcParams['font.family'] = "Malgun Gothic"
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        _setup_matplotlib_font()
        self.work = WorkState()
        self.done = [False] * len(STEPS)
        self.current = 0
        self._build_ui()
        self.tab_data.load_last()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

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

        self.pages = QStackedWidget()
        self.tab_data = DataTab(self.work)
        self.tab_model = ModelTab(self.work)
        self.tab_train = TrainTab(self.work, self.tab_model)
        self.tab_eval = EvalTab(self.work)
        self.tab_infer = InferTab(self.work)
        self.tab_export = ExportTab(self.work)
        tabs = [self.tab_data, self.tab_model, self.tab_train, self.tab_eval, self.tab_infer, self.tab_export]
        for i, tab in enumerate(tabs):
            self.pages.addWidget(tab)
            tab.step_done.connect(lambda k=i: self._step_done(k))
        layout.addWidget(self.pages, 1)
        self.go_step(0)

    def go_step(self, k):
        self.pages.setCurrentIndex(k)
        self.current = k
        self._refresh_steps()

    def _step_done(self, k):
        """k번 단계가 끝났다. 학습이 끝나면 평가 탭에서 모델이 자동으로 로드되고 평가가 시작된다."""
        self.done[k] = True
        if k == 2:
            self.go_step(3)
            self.tab_eval.load_trained()
        elif k < len(STEPS) - 1:
            self.go_step(k + 1)
        else:
            self._refresh_steps()

    def _refresh_steps(self):
        for i, btn in enumerate(self.step_btns):
            btn.setProperty("done", self.done[i])
            btn.setProperty("current", i == self.current)
            btn.style().unpolish(btn)   # 속성이 바뀌면 스타일을 다시 입혀야 색이 바뀐다
            btn.style().polish(btn)
