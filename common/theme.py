"""모든 도구가 같이 쓰는 테마: 흰 배경 + 하늘색 포인트. (어두운 테마 없음)

화면 구성(카드, 툴바, 버튼 종류 등)은 tools/hsi_viewer/hsi_viewer/theme.py 를 틀로 삼았다.
각 도구는 이 모듈의 색 상수와 apply_theme()만 불러다 쓴다.
objectName으로 모양을 고른다: Primary(하늘색 채움 버튼), Danger, Ghost, Card, CardTitle,
Title(상단 제목줄), Muted, Hint.
"""
from PySide6 import QtGui

# ---- 색 ------------------------------------------------------------------
BG0 = "#ffffff"      # 창 배경
BG1 = "#f5f9fd"      # 패널, 툴바, 상태줄
BG2 = "#ffffff"      # 입력창, 카드
BG3 = "#e6f3fc"      # 마우스 올림
BORDER = "#d3deea"
TEXT = "#1f2933"
MUTED = "#6b7785"
ACCENT = "#2aa3e0"       # 하늘색 포인트
ACCENT_DARK = "#1b86bd"  # 눌림, 글자용 진한 하늘색
ACCENT_LIGHT = "#cfeafa"  # 선택 표시 배경
MODEL_OFF = "#9aa5b1"   # 홈 내 모델: 학습된 가중치가 없는 항목의 회색
HOME_CIRCLE = "#eef7fd"   # 홈 그림 뒤 원 (아주 연한 하늘색)
DANGER = "#d64545"
WARN = "#e0962a"

# 그래프 점/커서/클러스터 색 (HSI Viewer가 사용). 흰 배경에서 보이는 색
CURSOR_R = "#ff6161"
CURSOR_G = "#2fb855"
CURSOR_B = "#6aa3ff"
POINT_COLORS = [
    "#1f2933", "#4fc3f7", "#ffb74d", "#81c784", "#ef5350", "#ba68c8",
    "#ffd54f", "#4dd0e1", "#f06292", "#aed581", "#ff8a65", "#9fa8da",
]
CLUSTER_COLORS = [
    "#4fd1c5", "#ff7a59", "#8e7dff", "#ffd166", "#06d6a0", "#ef476f",
    "#59c3ff", "#c3f73a", "#ff9ff3", "#f9a03f", "#80ffdb", "#d4a5ff",
]

FONT_FAMILY = "Malgun Gothic"
MONO_FAMILY = "Consolas"


def hex_to_rgb(h):
    """'#rrggbb' 문자열을 (r, g, b) 정수 튜플로 바꾼다."""
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

# f-string 안에서 CSS 중괄호는 {{ }} 로 쓴다
QSS = f"""
* {{
    font-family: "{FONT_FAMILY}";
    font-size: 10pt;
    color: {TEXT};
}}
QMainWindow, QWidget {{ background: {BG0}; }}
QLabel, QCheckBox, QRadioButton {{ background: transparent; }}  /* 흐린 패널 위 글자 뒤에 흰 상자가 생기지 않게 */
QToolTip {{
    background: {BG0}; color: {TEXT}; border: 1px solid {ACCENT}; padding: 4px 6px;
}}

/* ---------- 제목줄 (페이지 맨 위 띠) ---------- */
QFrame#Title {{ background: {ACCENT}; border: none; }}
QFrame#Title QLabel {{ background: transparent; color: #ffffff; font-size: 12pt; font-weight: 600; }}
QFrame#Title QPushButton {{
    background: rgba(255, 255, 255, 0.18); color: #ffffff; border: 1px solid rgba(255, 255, 255, 0.55);
}}
QFrame#Title QPushButton:hover {{ background: rgba(255, 255, 255, 0.32); }}

/* ---------- toolbar ---------- */
QToolBar {{
    background: {BG1}; border: none; border-bottom: 1px solid {BORDER};
    padding: 4px 8px; spacing: 6px;
}}
QToolBar QToolButton {{
    background: transparent; border: 1px solid transparent; border-radius: 4px;
    padding: 5px 10px; color: {TEXT};
}}
QToolBar QToolButton:hover {{ background: {BG3}; border-color: {BORDER}; }}
QToolBar QToolButton:pressed, QToolBar QToolButton:checked {{
    background: {ACCENT_LIGHT}; border-color: {ACCENT}; color: {ACCENT_DARK};
}}
QToolBar::separator {{ background: {BORDER}; width: 1px; margin: 4px 6px; }}
QToolButton::menu-indicator {{ image: none; width: 0; height: 0; }}

/* ---------- status bar ---------- */
QStatusBar {{ background: {BG1}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QStatusBar QLabel {{ color: {MUTED}; padding: 0 8px; }}
QStatusBar::item {{ border: none; }}

/* ---------- panels / cards ---------- */
QFrame#Card {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 4px;
}}
QLabel#CardTitle {{ color: {TEXT}; font-size: 10pt; font-weight: 600; }}
QLabel#CardSubtitle {{ color: {MUTED}; font-size: 9pt; }}
QLabel#Muted {{ color: {MUTED}; }}
QLabel#Mono {{ font-family: "{MONO_FAMILY}"; }}
QLabel#Hint {{ color: {MUTED}; font-size: 9pt; }}
QLabel#SectionLabel {{ color: {MUTED}; font-size: 9pt; font-weight: 600; letter-spacing: 1px; }}
QLabel#Accent {{ color: {ACCENT_DARK}; }}

QToolButton#CardToggle {{
    background: transparent; border: none; color: {TEXT}; font-weight: 600; text-align: left; padding: 6px 4px;
}}
QToolButton#CardToggle:hover {{ color: {ACCENT}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

/* ---------- inputs ---------- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 4px 8px; selection-background-color: {ACCENT_LIGHT}; selection-color: {TEXT};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {MUTED}; background: {BG1};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 16px; border: none; background: transparent; subcontrol-origin: border;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {BG3};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {BG2}; border: 1px solid {BORDER}; selection-background-color: {ACCENT_LIGHT};
    selection-color: {TEXT}; outline: none;
}}

/* ---------- buttons ---------- */
QPushButton {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 6px 12px; color: {TEXT};
}}
QPushButton:hover {{ background: {BG3}; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT_LIGHT}; }}
QPushButton:disabled {{ color: {MUTED}; background: {BG1}; border-color: {BORDER}; }}
QPushButton:checked {{ background: {ACCENT_LIGHT}; border-color: {ACCENT}; color: {ACCENT_DARK}; }}
QPushButton#Primary {{
    background: {ACCENT}; color: #ffffff; border: none; font-weight: 600;
}}
QPushButton#Primary:hover {{ background: #44b4ea; }}
QPushButton#Primary:pressed {{ background: {ACCENT_DARK}; }}
QPushButton#Primary:disabled {{ background: #a9d9f1; color: #ffffff; }}
QPushButton#Danger {{ color: {DANGER}; }}
QPushButton#Danger:hover {{ border-color: {DANGER}; }}
QPushButton#Ghost {{ background: transparent; border: 1px solid transparent; }}
QPushButton#Ghost:hover {{ background: {BG3}; border-color: {BORDER}; }}
QPushButton::menu-indicator {{ width: 0; }}

/* 붙어 있는 선택 버튼 (왼쪽 끝 SegL, 가운데 Seg, 오른쪽 끝 SegR) */
QPushButton#Seg, QPushButton#SegL, QPushButton#SegR {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 0; padding: 5px 12px;
}}
QPushButton#Seg:checked, QPushButton#SegL:checked, QPushButton#SegR:checked {{
    background: {ACCENT_LIGHT}; border-color: {ACCENT}; color: {ACCENT_DARK};
}}
QPushButton#SegL {{ border-top-left-radius: 4px; border-bottom-left-radius: 4px; }}
QPushButton#SegR {{ border-top-right-radius: 4px; border-bottom-right-radius: 4px; }}

/* ---------- 홈 화면 ---------- */
QTreeWidget#HomeList {{ background: transparent; border: none; font-size: 11pt; }}
QLabel#HomeSection {{ color: {MUTED}; font-size: 10pt; font-weight: 600; letter-spacing: 1px; }}
QLabel#HomeModel {{ color: {TEXT}; font-size: 11pt; }}
QLabel#HomeEnv {{ color: {MUTED}; font-size: 11pt; }}
QTreeWidget#HomeList::item {{ padding: 3px 6px; border: none; }}
QTreeWidget#HomeList::item:hover {{ background: {BG3}; }}
QPushButton#ToolCard {{
    background: {BG2}; border: 1px solid {BORDER}; border-top: 2px solid {ACCENT}; border-radius: 4px; padding: 0;
}}
QPushButton#ToolCard:hover {{ background: {BG2}; border: 1px solid {ACCENT}; border-top: 2px solid {ACCENT}; }}
QPushButton#ToolCard QLabel {{ background: transparent; }}
QLabel#CardNo {{ color: {ACCENT_DARK}; font-size: 10pt; font-weight: 600; }}
QLabel#ToolTitle {{ color: {TEXT}; font-size: 12pt; font-weight: 600; }}
QLabel#ToolDesc {{ color: {MUTED}; font-size: 11pt; }}
QFrame#Title QLabel#TitleSub {{ font-size: 9pt; font-weight: 400; padding-left: 10px; }}

/* ---------- check / radio ---------- */
QCheckBox, QRadioButton {{ spacing: 7px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px; height: 15px; border: 1px solid {BORDER}; border-radius: 4px; background: {BG2};
}}
QRadioButton::indicator {{ border-radius: 4px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT};
}}
QCheckBox::indicator:disabled {{ background: {BG1}; }}

/* ---------- sliders ---------- */
QSlider {{ background: transparent; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
    background: {BG0}; border: 2px solid {ACCENT};
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT_LIGHT}; }}
QSlider:disabled::sub-page:horizontal {{ background: {BORDER}; }}

/* ---------- lists / tables ---------- */
QListWidget, QListView, QTreeView {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 4px; outline: none;
}}
QListWidget::item, QListView::item {{ padding: 4px 6px; }}
QListWidget::item:hover, QListView::item:hover {{ background: {BG3}; }}
QListWidget::item:selected, QListView::item:selected, QTreeView::item:selected {{
    background: {ACCENT_LIGHT}; color: {TEXT};
}}
QTableView, QTableWidget {{
    background: {BG2}; alternate-background-color: {BG1}; border: 1px solid {BORDER};
    border-radius: 4px; gridline-color: {BORDER}; selection-background-color: {ACCENT_LIGHT};
    selection-color: {TEXT}; outline: none;
}}
QHeaderView::section {{
    background: {BG1}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER};
    padding: 4px 6px; font-weight: 600; font-size: 9pt;
}}
QTableView::item {{ padding: 2px 6px; }}
QTableCornerButton::section {{ background: {BG1}; border: none; }}

/* ---------- splitter / scrollbar ---------- */
QSplitter::handle {{ background: {BG1}; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}
QSplitter::handle:hover {{ background: {ACCENT_LIGHT}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; min-height: 24px; border-radius: 5px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; min-width: 24px; border-radius: 5px; }}
QScrollBar::handle:horizontal:hover {{ background: {ACCENT}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ---------- progress ---------- */
QProgressBar {{
    background: {BG1}; border: 1px solid {BORDER}; border-radius: 5px; max-height: 8px; text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}

/* ---------- menus ---------- */
QMenu {{ background: {BG0}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT_LIGHT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}

/* ---------- tabs ---------- */
QTabWidget::pane {{ border: none; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: transparent; color: {MUTED}; padding: 6px 12px; border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {ACCENT_DARK}; border-bottom-color: {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QGroupBox {{ border: 1px solid {BORDER}; border-radius: 4px; margin-top: 10px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {ACCENT_DARK}; }}

/* ---------- 단계 표시줄 (Segmentation 등): 끝난 단계는 하늘색, 지금 단계는 진한 테두리 ---------- */
QFrame#StepBar {{ background: {BG1}; border-bottom: 1px solid {BORDER}; }}
QPushButton#Step {{ border-radius: 4px; padding: 5px 16px; background: {BG2}; border: 1px solid {BORDER}; color: {MUTED}; }}
QPushButton#Step[done="true"] {{ background: {ACCENT_LIGHT}; border-color: {ACCENT}; color: {ACCENT_DARK}; }}
QPushButton#Step[current="true"] {{ border: 2px solid {ACCENT}; color: {TEXT}; font-weight: 600; }}
QLabel#StepArrow {{ color: {MUTED}; }}
/* 기존 도구 코드가 쓰는 이름 */
QLabel#hint {{ color: {MUTED}; font-size: 9pt; }}
QLabel#section_title {{ color: {TEXT}; font-size: 10pt; font-weight: 600; }}
QPushButton#secondary {{ background: {BG2}; }}

QMessageBox {{ background: {BG0}; }}
QDialog {{ background: {BG0}; }}
"""


def apply_theme(app):
    """QApplication에 밝은 팔레트와 스타일시트를 적용한다."""
    app.setStyle("Fusion")  # 어느 OS에서든 같은 모양
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(BG0))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor(BG2))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(BG1))
    pal.setColor(QtGui.QPalette.Text, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor(BG2))
    pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(ACCENT))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    pal.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(BG0))
    pal.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(TEXT))
    pal.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(MUTED))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
    app.setFont(QtGui.QFont(FONT_FAMILY, 10))


def apply_pyqtgraph_defaults():
    """그래프를 쓰는 도구가 페이지를 열 때 부른다 (pyqtgraph는 무거워서 여기서 늦게 불러옴)."""
    import pyqtgraph as pg
    pg.setConfigOptions(imageAxisOrder="row-major", background=BG1, foreground=TEXT, antialias=True, useOpenGL=False)
