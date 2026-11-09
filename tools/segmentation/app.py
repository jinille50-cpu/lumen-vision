"""Segmentation 도구 창: 위쪽 단계 표시줄 + 아래 탭 페이지. Vision Toolkit 홈에서 열린다.
(원래 main.py의 MainWindow를 PySide6용으로 옮긴 것. 색과 글꼴은 common 테마가 정한다.)"""
import os

# 반드시 다른 Qt 관련 import보다 먼저 불러온다. 구버전 msvcp140.dll이 먼저 올라가면
# onnxruntime 초기화가 "DLL 초기화 루틴을 실행할 수 없습니다" 로 실패한다 (Windows).
try:
    import onnxruntime  # noqa: F401
except ImportError:
    pass    # 없어도 앱은 동작. ONNX 내보내기와 양자화 평가만 경고가 난다

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSpinBox, QStackedWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from common import settings

from tools.segmentation.utils.work_state import WorkState
from tools.segmentation.tabs.data_tab import DataTab
from tools.segmentation.tabs.model_tab import ModelTab
from tools.segmentation.tabs.train_tab import TrainTab
from tools.segmentation.tabs.eval_tab import EvalTab
from tools.segmentation.tabs.infer_tab import InferTab
from tools.segmentation.tabs.quant_tab import QuantTab

TOOL = "segmentation"  # common 설정에 저장할 때 쓰는 이름
STEPS = ["데이터", "모델", "학습", "평가", "추론", "경량화"]

# 프로젝트에 저장할 칸: (탭 속성 이름, 위젯 속성 이름 목록). 모델 종류를 가중치보다 먼저 (모델이 바뀌면 가중치 목록이 바뀜)
PROJECT_FIELDS = [
    ("tab_data", ["img_ext", "lbl_ext", "holdout_spin", "kfold_spin", "kfold_strat", "kfold_inner"]),
    ("tab_model", ["model_combo", "weight_combo", "encoder_combo", "input_h", "input_w", "onnx_combo", "num_classes"]),
    ("tab_train", ["epochs", "batch", "lr", "optimizer", "loss_fn", "scheduler", "sch_p1", "sch_p2",
                   "use_es", "es_patience", "aug_preset", "seed_check", "seed_spin"]),
]
PROJECT_PATHS = ["train_img", "train_lbl", "val_img", "val_lbl"]  # 데이터 탭 <이름>_edit 칸


def _widget_value(w):
    if isinstance(w, QComboBox):
        return w.currentText()
    if isinstance(w, (QSpinBox, QDoubleSpinBox)):  # QDoubleSpinBox는 QSpinBox의 자식이 아니라 둘 다 적는다
        return w.value()
    if isinstance(w, QCheckBox):
        return w.isChecked()
    return None


def _set_widget_value(w, v):
    if isinstance(w, QComboBox):
        if w.findText(str(v)) >= 0:  # 목록에 없는 값(지금 버전에 없는 모델 등)은 건너뛴다
            w.setCurrentText(str(v))
    elif isinstance(w, QSpinBox):
        w.setValue(int(v))
    elif isinstance(w, QDoubleSpinBox):
        w.setValue(float(v))
    elif isinstance(w, QCheckBox):
        w.setChecked(bool(v))


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
        self.shared_state = {}  # 탭 간 공유 데이터
        self.shared_state["work"] = WorkState()   # 모델/config/데이터 경로 (평가·추론·경량화가 이어받음)
        self.done = [False] * len(STEPS)
        self.runs = []  # 이번 프로젝트에서 학습한 결과 폴더 (프로젝트 저장 때 같이 저장)
        self._build_ui()
        self._load_last_settings()

    # ── 화면 ─────────────────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 단계 표시줄: 데이터 → 모델 → 학습 → 평가 → 추론 → 경량화 (눌러서 이동)
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
        self.tab_data = DataTab(self.shared_state)
        self.tab_model = ModelTab(self.shared_state)
        self.tab_train = TrainTab(self.shared_state)
        self.tab_eval = EvalTab(self.shared_state)
        self.tab_infer = InferTab(self.shared_state)
        self.tab_quant = QuantTab(self.shared_state)
        # 평가탭 → 추론탭 모델/데이터 공유 연결
        self.tab_eval.set_infer_tab(self.tab_infer)
        # 학습탭 · 모델탭 · 경량화탭 → 데이터탭 직접 참조 (저장 버튼 없이도 경로 읽기)
        self.tab_train.set_data_tab(self.tab_data)
        self.tab_model.set_data_tab(self.tab_data)
        self.tab_quant.set_data_tab(self.tab_data)
        tabs = [self.tab_data, self.tab_model, self.tab_train, self.tab_eval, self.tab_infer, self.tab_quant]
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
        """k번 단계가 끝났다. 데이터/모델은 저장 뒤 다음 단계로 넘어간다 ('저장하고 다음')."""
        self.done[k] = True
        if k == 0:
            self._save_last_settings()
            st = self.shared_state   # 데이터 탭에서 감지한 채널/클래스를 모델 탭에 채움
            self.tab_model.apply_detected(st.get("detected_in_channels"),
                                          st.get("detected_class_names"))
        if k == 2:   # 학습 끝남 또는 중단: 평가 탭으로 이동하고 결과 모델 자동 로드
            run_dir = getattr(self.tab_train, "run_dir", None)
            if run_dir and run_dir not in self.runs:
                self.runs.append(run_dir)
            d = self.tab_data
            val = None   # '폴더로 분리'(0번)였으면 Validation 폴더를 평가 탭에 넘긴다
            if d.split_btns.checkedId() == 0:
                val = (d.val_img_edit.text().strip(), d.val_lbl_edit.text().strip())
            self.tab_eval.load_trained(self.tab_train.result_model_path(), val)
            onnx = os.path.splitext(self.tab_train.result_model_path())[0] + ".onnx"   # 학습 후 자동으로 만든 ONNX
            if os.path.isfile(onnx):
                self.shared_state["work"].set("onnx_path", onnx)
            self.go_step(3)
        elif k in (0, 1):
            self.go_step(k + 1)
        else:
            self._refresh_steps()

    def _refresh_steps(self):
        for i, btn in enumerate(self.step_btns):
            btn.setProperty("done", self.done[i])
            btn.setProperty("current", i == self.current)
            btn.style().unpolish(btn)  # 속성이 바뀌면 스타일을 다시 입혀야 색이 바뀐다
            btn.style().polish(btn)

    # ── 마지막 설정 ──────────────────────────────────────────────
    def _load_last_settings(self):
        self.tab_model.load_wandb_settings()
        d = self.tab_data
        for key, edit in (("train_img", d.train_img_edit), ("train_lbl", d.train_lbl_edit),
                          ("val_img", d.val_img_edit), ("val_lbl", d.val_lbl_edit)):
            edit.setText(str(settings.get_value(TOOL, key, "")))
        for key, combo in (("img_ext", d.img_ext), ("lbl_ext", d.lbl_ext)):
            last = settings.get_value(TOOL, key)
            if last and combo.findText(str(last)) >= 0:
                combo.setCurrentText(str(last))

    def _save_last_settings(self):
        d = self.tab_data
        for key, edit in (("train_img", d.train_img_edit), ("train_lbl", d.train_lbl_edit),
                          ("val_img", d.val_img_edit), ("val_lbl", d.val_lbl_edit)):
            settings.set_value(TOOL, key, edit.text().strip())
        settings.set_value(TOOL, "img_ext", d.img_ext.currentText())
        settings.set_value(TOOL, "lbl_ext", d.lbl_ext.currentText())
        if d.train_img_edit.text().strip():
            settings.push_recent(TOOL, d.train_img_edit.text().strip())

    # ── 프로젝트 (main.py 제목줄의 프로젝트 저장/불러오기가 부른다, 파일 읽기/쓰기는 common/project.py) ──
    def project_state(self):
        d = self.tab_data
        values = {}
        for tab_name, names in PROJECT_FIELDS:
            tab = getattr(self, tab_name)
            for n in names:
                values[f"{tab_name}.{n}"] = _widget_value(getattr(tab, n))
        values["split_mode"] = d.split_btns.checkedId()  # 0=폴더, 1=비율, 2=K-fold
        names, weights = [], []
        for i in range(d.class_table.rowCount()):
            names.append(d.class_table.item(i, 1).text())
            weights.append(d.class_table.item(i, 3).text())
        values["class_names"] = names
        values["class_weights"] = weights
        values["channels"] = self.tab_model.selected_channels()  # None이면 전부 사용
        paths = {n: getattr(d, f"{n}_edit").text().strip() for n in PROJECT_PATHS}
        return {"paths": paths, "runs": list(self.runs), "values": values}

    def apply_project(self, state):
        """프로젝트 값을 칸에 채운다. 데이터 폴더가 있으면 파일 목록/클래스를 다시 읽고 저장된 클래스 이름, 가중치를 덮어쓴다.
        돌려주는 값: 상태 표시줄에 붙일 짧은 안내 (없으면 "")."""
        d, m = self.tab_data, self.tab_model
        values = state.get("values", {})
        for n in PROJECT_PATHS:
            getattr(d, f"{n}_edit").setText(state.get("paths", {}).get(n, ""))
        btn = d.split_btns.button(int(values.get("split_mode", 0)))
        if btn is not None:
            btn.setChecked(True)
            d._on_split_changed()
        for tab_name, names in PROJECT_FIELDS:
            tab = getattr(self, tab_name)
            for n in names:
                key = f"{tab_name}.{n}"
                if key in values and values[key] is not None:
                    _set_widget_value(getattr(tab, n), values[key])
        d._update_kfold_note()
        note = ""
        if os.path.isdir(d.train_img_edit.text().strip()):
            d._scan_files()  # 파일 목록, 입력 채널, 클래스 표를 다시 채운다
            for i, (name, w) in enumerate(zip(values.get("class_names", []), values.get("class_weights", []))):
                if i < d.class_table.rowCount():
                    d.class_table.setItem(i, 1, QTableWidgetItem(name))
                    d.class_table.setItem(i, 3, QTableWidgetItem(w))
            found = getattr(d, "in_channels_found", None)
            if found:
                m._set_total_channels(found, values.get("channels"))
        else:
            note = "데이터 폴더를 찾지 못했습니다"
        self.runs = list(state.get("runs", []))
        for run in reversed(self.runs):  # 가장 최근 학습 결과를 평가/추론 탭의 작업 모델로
            self.tab_train.run_dir = run
            model = self.tab_train.result_model_path()
            if model:
                self.shared_state["work"].set_model(model)
                break
        return note
