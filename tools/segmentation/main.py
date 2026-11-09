"""Segmentation 도구 단독 실행용 (평소에는 Vision Toolkit 홈에서 연다).
실행: 이 폴더에서 python main.py  -- 공통 테마를 쓰려면 Vision Toolkit 루트가 경로에 있어야 한다."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # tools.segmentation.* 와 common 을 import 하려고

from tools.segmentation.app import MainWindow  # noqa: E402  (onnxruntime을 PySide6보다 먼저 불러오려고 app을 먼저 import)
from PySide6.QtWidgets import QApplication  # noqa: E402
from common import theme  # noqa: E402


def main():
    app = QApplication(sys.argv)
    theme.apply_theme(app)
    window = MainWindow()
    window.setWindowTitle("Segmentation")
    window.resize(1400, 860)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
