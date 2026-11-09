"""Vision Toolkit 메인 창: 홈 화면(대표 이미지 + 도구 버튼 5개)과 도구 페이지 전환."""
import html
import os
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
    QPushButton, QStackedWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from common import project, settings, sysinfo, theme

ROOT = os.path.dirname(os.path.abspath(__file__))
HOME_IMAGE = os.path.join(ROOT, "assets", "home.png")  # 이 파일을 넣으면 홈 위쪽에 표시된다

# (키, 버튼 이름, 한 줄 설명, 사용법 3줄) -- 홈 화면에 이 순서대로 표시
TOOLS = [
    ("labeler", "Labeler", "박스/분할 라벨링",
     ["이미지 폴더를 열고 박스나 분할 영역을 그려 라벨을 만든다.",
      "SAM 3 글자 프롬프트로 자동 라벨링한 뒤 한 장씩 검수한다.",
      "YOLO 형식이나 마스크 이미지로 내보낸다."]),
    ("segmentation", "Segmentation", "분할 모델 학습",
     ["이미지와 마스크 폴더를 지정한다.",
      "모델과 학습 설정을 고르고 학습을 시작한다.",
      "학습된 모델로 새 이미지를 추론하고 결과를 저장한다."]),
    ("yolo", "Object Detection", "객체 인식 학습/추론",
     ["이미지와 라벨(YOLO 형식) 폴더를 지정한다.",
      "모델 계열(YOLO, RT-DETR)과 크기를 골라 학습을 시작한다.",
      "학습된 가중치로 이미지 폴더를 추론한다."]),
    ("patchcore", "PatchCore", "이상 탐지 학습/추론",
     ["정상 이미지 폴더로 학습한다.",
      "검사할 이미지를 넣어 이상 점수를 계산한다.",
      "히트맵으로 이상 위치를 확인한다."]),
    ("hsi_viewer", "HSI Viewer", "초분광 큐브 보기",
     ["열기 버튼이나 파일 끌어놓기로 초분광 큐브를 불러온다.",
      "이미지를 클릭하면 그 위치의 스펙트럼이 오른쪽 그래프에 나온다.",
      "SAM 3 글자 프롬프트로 객체를 나누고 클러스터링으로 묶는다."]),
]
READY = {"hsi_viewer", "segmentation", "yolo", "patchcore", "labeler"}  # 실제 도구가 연결된 것. 나머지는 자리표시 페이지 (준비 중)


def build_hsi_viewer():
    """HSI Viewer 창을 만든다. 무거운 import(pyqtgraph, numpy 등)는 여기서 처음 한다."""
    tool_dir = os.path.join(ROOT, "tools", "hsi_viewer")
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)  # tools/hsi_viewer/hsi_viewer 패키지를 import 하려고
    theme.apply_pyqtgraph_defaults()
    from hsi_viewer.app import MainWindow as HsiWindow
    return HsiWindow()


def build_segmentation():
    """Segmentation 창을 만든다. torch 등 무거운 import는 탭이 열릴 때 처음 한다."""
    from tools.segmentation.app import MainWindow as SegWindow  # app.py 안에서 onnxruntime을 먼저 불러온다
    return SegWindow()


def build_yolo():
    """Object Detection(YOLO, RT-DETR) 창을 만든다. torch, ultralytics 같은 무거운 import는 탭 안에서 필요할 때 처음 한다."""
    from tools.yolo.app import MainWindow as YoloWindow
    return YoloWindow()


def build_labeler():
    """Labeler 창을 만든다."""
    from tools.labeler.app import MainWindow as LabelerWindow
    return LabelerWindow()


def build_patchcore():
    """PatchCore 창을 만든다. anomalib, torch는 학습/추론을 누를 때 처음 불러온다."""
    from tools.patchcore.app import MainWindow as PatchcoreWindow
    return PatchcoreWindow()


BUILDERS = {"labeler": build_labeler, "hsi_viewer": build_hsi_viewer, "segmentation": build_segmentation, "yolo": build_yolo,
            "patchcore": build_patchcore}


class HelpDialog(QDialog):
    """도구별 사용법(3줄)을 한 화면에 보여주는 다이얼로그."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("도움말")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        for key, name, _, lines in TOOLS:
            head = name.replace("\n", " ") + ("" if key in READY else " (준비 중)")
            title = QLabel(head)
            title.setObjectName("CardTitle")
            layout.addWidget(title)
            body = QLabel("\n".join(f"{i}. {t}" for i, t in enumerate(lines, 1)))
            body.setObjectName("Muted")
            layout.addWidget(body)
        row = QHBoxLayout()
        row.addStretch(1)
        close = QPushButton("닫기")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        layout.addLayout(row)


class HomeImage(QLabel):
    """홈 왼쪽 그림 영역. assets/home.png를 비율 지켜 영역에 맞춘다. 없으면 빈 칸."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(200)
        self._pix = QPixmap(HOME_IMAGE) if os.path.exists(HOME_IMAGE) else None
        if self._pix is not None and self._pix.isNull():
            self._pix = None

    def resizeEvent(self, ev):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(ev)


class RightPanel(QWidget):
    """홈 오른쪽 영역. 뒤에 아주 연한 하늘색 큰 원을 깔고, 오른쪽 끝에서 일부가 잘리게 둔다."""

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.HOME_CIRCLE))
        d = int(self.height() * 1.3)
        p.drawEllipse(self.width() - int(d * 0.55), (self.height() - d) // 2, d, d)  # 원의 오른쪽 45%는 영역 밖
        p.end()


def _section(title):
    """칸 제목(작은 회색 글씨)."""
    lab = QLabel(title)
    lab.setObjectName("HomeSection")
    return lab


def _list():
    """헤더 없는 표 형태 목록(3열). 배경은 투명해서 뒤의 원이 보인다."""
    t = QTreeWidget()
    t.setObjectName("HomeList")
    t.setColumnCount(3)
    t.setHeaderHidden(True)
    t.setRootIsDecorated(False)
    t.setIndentation(0)
    t.setFocusPolicy(Qt.NoFocus)
    t.setTextElideMode(Qt.ElideMiddle)  # 긴 경로는 가운데를 줄임
    t.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return t


class HomePage(QWidget):
    """위: 제목줄(Lumen Vision, 도움말), 가운데: 그림(왼쪽 40%) + 오른쪽 세 칸(최근 작업, 내 모델, 작업 환경),
    아래: 번호 붙은 도구 카드 5개. 카드를 누르면 open_tool(key)를 호출한다."""

    def __init__(self, open_tool, show_help, load_sam3, load_project):
        super().__init__()
        self.open_tool = open_tool
        self.load_project = load_project
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 하늘색 제목줄
        bar = QFrame()
        bar.setObjectName("Title")
        top = QHBoxLayout(bar)
        top.addWidget(QLabel("Lumen Vision"))
        sub = QLabel("라벨링부터 검사까지")
        sub.setObjectName("TitleSub")
        top.addWidget(sub)
        top.addStretch(1)
        sam_btn = QPushButton("SAM 3 로드")
        sam_btn.clicked.connect(load_sam3)
        top.addWidget(sam_btn)
        proj_btn = QPushButton("프로젝트 불러오기")
        proj_btn.clicked.connect(lambda: load_project())  # lambda: clicked 신호의 True/False 인자를 경로로 받지 않게
        top.addWidget(proj_btn)
        help_btn = QPushButton("도움말")
        help_btn.clicked.connect(show_help)
        top.addWidget(help_btn)
        layout.addWidget(bar)

        # 가운데: 그림 40% / 오른쪽 60%
        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)
        img_box = QVBoxLayout()
        img_box.setContentsMargins(16, 0, 16, 56)  # 아래를 더 띄워 그림을 조금 위로
        img_box.addWidget(HomeImage())
        middle.addLayout(img_box, 4)

        right = RightPanel()
        col = QVBoxLayout(right)
        col.setContentsMargins(16, 12, 24, 56)  # 아래 여백은 그림 쪽과 같게 맞춤
        col.setSpacing(32)  # 칸 사이 여백
        col.addStretch(1)
        # 1) 최근 작업 5줄: 도구 이름 / 경로 / 날짜
        box = QVBoxLayout()
        box.setSpacing(4)
        box.addWidget(_section("최근 작업"))
        self.recent = _list()
        self.recent.itemClicked.connect(self._open_recent)
        box.addWidget(self.recent)
        col.addLayout(box)
        # 2) 내 모델: Object Detection / Segmentation / PatchCore 세 칸을 가로로, 칸마다 모델 이름만
        box = QVBoxLayout()
        box.setSpacing(4)
        box.addWidget(_section("내 모델"))
        mrow = QHBoxLayout()
        mrow.setSpacing(24)
        self.model_labels = {}
        for key, title in (("yolo", "Object Detection"),("segmentation", "Segmentation"), ("patchcore", "PatchCore")):
            cell = QVBoxLayout()
            cell.setSpacing(2)
            head = QLabel(title)
            head.setObjectName("HomeEnv")  # 작은 회색 글씨
            names = QLabel("")
            names.setObjectName("HomeModel")
            cell.addWidget(head)
            cell.addWidget(names)
            cell.addStretch(1)
            mrow.addLayout(cell, 1)
            self.model_labels[key] = names
        box.addLayout(mrow)
        col.addLayout(box)
        col.addStretch(1)  # 위아래 stretch로 두 덩어리를 세로 가운데에 둔다
        middle.addWidget(right, 6)
        layout.addLayout(middle, 1)

        # 아래: 도구 카드. 번호는 TOOLS 순서대로 01~05
        row = QHBoxLayout()
        row.setContentsMargins(16, 0, 16, 16)
        for n, (key, name, tip, _) in enumerate(TOOLS, 1):
            card = QPushButton()
            card.setObjectName("ToolCard")
            card.setMinimumHeight(104)
            cl = QVBoxLayout(card)
            cl.setContentsMargins(14, 10, 14, 10)
            for text, obj in ((f"{n:02d}", "CardNo"), (name, "ToolTitle"), (tip, "ToolDesc")):
                lab = QLabel(text)
                lab.setObjectName(obj)
                lab.setAttribute(Qt.WA_TransparentForMouseEvents)  # 클릭이 카드 버튼에 전달되게
                cl.addWidget(lab)
            cl.addStretch(1)
            # lambda 기본 인자(k=key)로 현재 key 값을 고정 (안 하면 마지막 key만 쓰임)
            card.clicked.connect(lambda _=False, k=key: open_tool(k))
            row.addWidget(card, 1)
        layout.addLayout(row)
        self.refresh_recent()
        self.refresh_models()

    @staticmethod
    def _fill(tree, rows, empty_text):
        """rows: (칸1, 칸2, 날짜) 튜플 목록을 채우고 만든 항목을 돌려준다. 비었으면 안내 문구 한 줄.
        높이는 실제 줄 수에 맞춰서 칸 사이 간격이 같게 보이게 한다. 날짜 열은 폭 고정, 오른쪽 정렬."""
        tree.clear()
        head = tree.header()
        head.setStretchLastSection(False)
        tree.setFixedHeight(max(len(rows), 1) * 28 + 4)
        if not rows:
            it = QTreeWidgetItem([empty_text])
            it.setFlags(Qt.NoItemFlags)
            tree.addTopLevelItem(it)
            tree.setFirstColumnSpanned(0, tree.rootIndex(), True)  # 안내 문구가 잘리지 않게 전체 폭 사용
            return []
        items = [QTreeWidgetItem(list(r)) for r in rows]
        tree.addTopLevelItems(items)
        for it in items:
            it.setTextAlignment(2, Qt.AlignRight | Qt.AlignVCenter)
        head.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        head.setSectionResizeMode(1, QHeaderView.Stretch)
        head.setSectionResizeMode(2, QHeaderView.Fixed)
        head.resizeSection(2, 110)
        return items

    def refresh_recent(self):
        """common 설정의 최근 항목 5개. 누르면 해당 도구를 연다."""
        names = {k: n for k, n, _, _ in TOOLS}
        recent = [i for i in settings.recent_items() if i.get("tool") in names][:5]
        rows = [(names[i["tool"]], i.get("path", ""), i.get("time", "")) for i in recent]
        for it, i in zip(self._fill(self.recent, rows, "최근 작업이 없습니다"), recent):
            it.setData(0, Qt.UserRole, i["tool"])
            it.setToolTip(1, i.get("path", ""))

    def refresh_models(self):
        """도구별 모델 계열을 한 줄씩 보여준다. 학습된 가중치가 있는 변형은 검은 글씨, 없으면 회색.
        예: UNet (기본, ++) / SegFormer (B0, B2, B4)"""
        on, off = theme.TEXT, theme.MODEL_OFF

        def span(text, has):
            return f'<span style="color:{on if has else off}">{html.escape(text)}</span>'

        for key, families in sysinfo.model_catalog().items():
            lines = []
            for fam, variants in families:
                if len(variants) == 1 and variants[0][0] is None:
                    lines.append(span(fam, variants[0][1]))  # 변형 없는 계열은 괄호 없이
                    continue
                inner = f'<span style="color:{off}">, </span>'.join(span(v, h) for v, h in variants)
                lines.append(f'{span(fam, any(h for _, h in variants))}'
                             f'<span style="color:{off}"> (</span>{inner}<span style="color:{off}">)</span>')
            self.model_labels[key].setText("<br>".join(lines) if lines else span("목록 없음", False))

    def _open_recent(self, item):
        key = item.data(0, Qt.UserRole)
        path = item.toolTip(1)
        if key and project.is_project(path):  # 최근 항목이 프로젝트 파일이면 그 프로젝트를 연다
            self.load_project(path)
        elif key:
            self.open_tool(key)


class ToolPage(QWidget):
    """도구 페이지: 하늘색 제목줄('← 홈', 도구 이름, 프로젝트 경로, 도움말) 아래에 도구 화면을 넣는다.
    content가 없으면 자리표시 글자를 보여준다 (아직 연결 안 된 도구).
    도구 창에 project_state/apply_project가 있으면 프로젝트 불러오기/저장 버튼을 단다."""

    def __init__(self, name, go_home, show_help, load_sam3, content=None, load_project=None, save_project=None):
        super().__init__()
        self.content = content
        self.project_path = ""  # 지금 연 프로젝트 파일 (없으면 "")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # 하늘색 제목줄 (objectName "Title"은 theme.py가 칠함)
        bar = QFrame()
        bar.setObjectName("Title")
        top = QHBoxLayout(bar)
        back = QPushButton("← 홈")
        back.setObjectName("btn_home")
        back.clicked.connect(go_home)
        top.addWidget(back)
        top.addWidget(QLabel(name.replace(chr(10), " ")))
        self.project_label = QLabel("")
        self.project_label.setObjectName("TitleSub")
        top.addWidget(self.project_label)
        top.addStretch(1)
        sam_btn = QPushButton("SAM 3 로드")
        sam_btn.clicked.connect(load_sam3)
        top.addWidget(sam_btn)
        if hasattr(content, "project_state"):
            load_btn = QPushButton("프로젝트 불러오기")
            load_btn.clicked.connect(lambda: load_project())
            top.addWidget(load_btn)
            save_btn = QPushButton("프로젝트 저장")
            save_btn.clicked.connect(lambda: save_project(self))
            top.addWidget(save_btn)
        help_btn = QPushButton("도움말")
        help_btn.clicked.connect(show_help)
        top.addWidget(help_btn)
        layout.addWidget(bar)
        if content is None:
            content = QLabel(f"{name.replace(chr(10), ' ')} (준비 중)")
            content.setAlignment(Qt.AlignCenter)
        layout.addWidget(content, 1)

    def set_project(self, path):
        self.project_path = path
        folder = os.path.basename(os.path.dirname(path))
        self.project_label.setText(f"{folder}/{os.path.basename(path)}")  # 제목줄엔 짧게, 전체 경로는 마우스를 올리면
        self.project_label.setToolTip(path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Toolkit")
        self.resize(1400, 860)
        settings.restore_window("main", self)  # 지난번 위치. 모니터 밖이면 주 모니터 가운데
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.home = HomePage(self.open_tool, self.show_help, self.load_sam3, self.load_project)
        self.stack.addWidget(self.home)
        self._build_env_bar()
        self.pages = {}  # key -> 페이지. 처음 열 때 만든다 (무거운 도구를 홈 화면에서 불러오지 않기 위해)

    def _build_env_bar(self):
        """창 맨 아래 상태 표시줄 한 줄: GPU 이름과 VRAM | SAM 3 있음/없음 | Python 버전.
        nvidia-smi 호출이 있어서 창이 뜬 뒤에 채운다."""
        self.env_labels = [QLabel("") for _ in range(3)]
        bar = self.statusBar()
        for i, lab in enumerate(self.env_labels):
            if i:
                bar.addWidget(QLabel("|"))
            bar.addWidget(lab)
        QTimer.singleShot(0, self._fill_env_bar)

    def _fill_env_bar(self):
        gpu = sysinfo.gpu_info()
        path, ok = sysinfo.sam3_path()
        self.env_labels[0].setText(f"GPU {gpu[0]}  VRAM {gpu[1]:.0f} GB" if gpu else "GPU 없음")
        self.env_labels[1].setText(f"SAM 3 {'있음' if ok else '없음'}")
        self.env_labels[1].setToolTip(path or "제목줄의 SAM 3 로드로 sam3.pt를 고르세요")  # 경로는 마우스를 올리면 보인다
        self.env_labels[2].setText(f"Python {sysinfo.python_version()}")

    def open_tool(self, key):
        if key not in self.pages:
            name = next(n for k, n, _, _ in TOOLS if k == key)
            content = None
            if key in BUILDERS:
                QApplication.setOverrideCursor(Qt.BusyCursor)
                try:
                    content = BUILDERS[key]()
                    self._apply_sam3(content)
                except Exception as e:  # noqa: BLE001  -- 도구가 안 떠도 홈은 유지
                    QMessageBox.critical(self, name.replace("\n", " "), f"도구를 열지 못했습니다.\n{e}")
                    return
                finally:
                    QApplication.restoreOverrideCursor()
            page = ToolPage(name, self.go_home, self.show_help, self.load_sam3, content,
                            self.load_project, self.save_project)
            self.pages[key] = page
            self.stack.addWidget(page)
        self.stack.setCurrentWidget(self.pages[key])
        return True

    def load_project(self, path=None):
        """프로젝트 파일을 골라(또는 path) 그 도구를 열고 저장된 값을 채운다."""
        if not path:
            start = str(settings.get_value("main", "project_dir", ""))
            path, _ = QFileDialog.getOpenFileName(self, "프로젝트 불러오기", start, "프로젝트 (*.json)")
            if not path:
                return
        try:
            key, state = project.load(path)
        except (OSError, ValueError) as e:  # ValueError: json 형식 오류도 여기로 온다
            QMessageBox.critical(self, "프로젝트 불러오기", f"프로젝트를 읽지 못했습니다.\n{e}")
            return
        if key not in BUILDERS:
            QMessageBox.critical(self, "프로젝트 불러오기", f"알 수 없는 도구입니다: {key}")
            return
        if not self.open_tool(key):  # 도구가 안 열리면 open_tool이 이미 오류를 띄웠다
            return
        page = self.pages[key]
        if not hasattr(page.content, "apply_project"):
            QMessageBox.critical(self, "프로젝트 불러오기", "이 도구는 아직 프로젝트를 지원하지 않습니다.")
            return
        note = page.content.apply_project(state)
        page.set_project(path)
        settings.set_value("main", "project_dir", os.path.dirname(path))
        settings.push_recent(key, path)
        self.statusBar().showMessage(f"프로젝트 불러옴: {path}" + (f"  ({note})" if note else ""), 8000)

    def save_project(self, page):
        """도구 페이지의 지금 값을 프로젝트 파일로 저장한다. 처음 저장이면 위치를 묻는다."""
        key = next(k for k, p in self.pages.items() if p is page)
        start = page.project_path or os.path.join(str(settings.get_value("main", "project_dir", "")), "project.json")
        path, _ = QFileDialog.getSaveFileName(self, "프로젝트 저장", start, "프로젝트 (*.json)")
        if not path:
            return
        try:
            project.save(path, key, page.content.project_state())
        except OSError as e:
            QMessageBox.critical(self, "프로젝트 저장", f"저장하지 못했습니다.\n{e}")
            return
        page.set_project(path)
        settings.set_value("main", "project_dir", os.path.dirname(path))
        settings.push_recent(key, path)
        self.statusBar().showMessage(f"프로젝트 저장: {path}", 8000)

    def load_sam3(self):
        """sam3.pt 위치를 직접 고른다. 경로는 common 설정에 저장되고 모든 도구가 공유한다."""
        cur, _ = sysinfo.sam3_path()
        path, _ = QFileDialog.getOpenFileName(self, "SAM 3 로드", cur or "", "SAM 3 가중치 (*.pt);;모든 파일 (*)")
        if not path:
            return
        sysinfo.set_sam3_path(path)
        self._fill_env_bar()
        for page in self.pages.values():
            self._apply_sam3(page.content)

    @staticmethod
    def _apply_sam3(tool):
        """저장된 SAM 3 경로를 도구 창에 알린다 (set_sam_path가 있는 도구만)."""
        if tool is not None and hasattr(tool, "set_sam_path"):
            tool.set_sam_path(sysinfo.sam3_path()[0])

    def go_home(self):
        self.home.refresh_recent()
        self.stack.setCurrentWidget(self.home)

    def show_help(self):
        HelpDialog(self).exec()

    def closeEvent(self, ev):
        settings.save_window("main", self)
        # 안에 넣은 도구 창은 자기 closeEvent에서 설정 저장/종료 처리를 하므로 직접 닫아 준다
        for page in self.pages.values():
            if page.content is not None:
                page.content.close()
        super().closeEvent(ev)


def main():
    app = QApplication(sys.argv)
    theme.apply_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
