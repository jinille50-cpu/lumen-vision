"""데이터 탭: 정상 이미지 폴더(학습용)와 테스트 폴더(정상/불량 자동 인식)."""
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from common import settings
from tools.patchcore import core
from tools.patchcore.ui_common import TOOL, ImageView, pick_dir, primary_button, two_pane


class DataTab(QWidget):
    step_done = Signal()   # 단계 표시줄에 '끝남'을 알리는 신호

    def __init__(self, work):
        super().__init__()
        self.work = work
        self.scan = {"good": [], "defects": {}, "masks": {}}
        self._build()

    def _build(self):
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 8, 0)

        g1 = QGroupBox("학습 이미지 (정상만)")
        v1 = QVBoxLayout(g1)
        row = QHBoxLayout()
        self.normal_edit = QLineEdit()
        self.normal_edit.setPlaceholderText("정상 이미지 폴더")
        self.normal_edit.editingFinished.connect(self._refresh_normal)
        b = QPushButton("찾기")
        b.clicked.connect(lambda: self._browse(self.normal_edit, self._refresh_normal))
        row.addWidget(self.normal_edit, 1)
        row.addWidget(b)
        v1.addLayout(row)
        self.normal_info = QLabel("")
        self.normal_info.setObjectName("Hint")
        v1.addWidget(self.normal_info)
        lv.addWidget(g1)

        g2 = QGroupBox("테스트 폴더 (평가용)")
        v2 = QVBoxLayout(g2)
        row = QHBoxLayout()
        self.test_edit = QLineEdit()
        self.test_edit.setPlaceholderText("good/ 와 불량 하위 폴더가 있는 폴더")
        self.test_edit.editingFinished.connect(self._refresh_test)
        b = QPushButton("찾기")
        b.clicked.connect(lambda: self._browse(self.test_edit, self._refresh_test))
        row.addWidget(self.test_edit, 1)
        row.addWidget(b)
        v2.addLayout(row)
        self.test_info = QLabel("")
        self.test_info.setObjectName("Hint")
        self.test_info.setWordWrap(True)
        v2.addWidget(self.test_info)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["폴더", "구분", "이미지 수", "마스크"])
        self.tree.setRootIsDecorated(False)
        self.tree.itemClicked.connect(self._show_item)
        v2.addWidget(self.tree, 1)
        lv.addWidget(g2, 1)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 0, 0, 0)
        self.preview = ImageView("폴더를 고르면 대표 이미지가 보입니다")
        rv.addWidget(self.preview, 1)
        self.preview_name = QLabel("")
        self.preview_name.setObjectName("Hint")
        rv.addWidget(self.preview_name)

        self.save_btn = primary_button("저장하고 다음")
        self.save_btn.clicked.connect(self._save)
        two_pane(self, left, right, self.save_btn)

    # ── 동작 ─────────────────────────────────────────────────
    def _browse(self, edit, after):
        path = pick_dir(self, edit.text())
        if path:
            edit.setText(path)
            after()

    def _refresh_normal(self):
        imgs = core.list_images(self.normal_edit.text().strip())
        self.normal_info.setText(f"이미지 {len(imgs)}장" if imgs else "이미지가 없습니다")
        if imgs:
            self._show(imgs[0])

    def _refresh_test(self):
        self.tree.clear()
        self.scan = core.scan_test_folder(self.test_edit.text().strip())
        sc = self.scan
        if sc["good"]:
            QTreeWidgetItem(self.tree, ["good", "정상", str(len(sc["good"])), "-"]).setData(0, Qt.UserRole, sc["good"][0])
        for name, imgs in sc["defects"].items():
            has = "있음" if name in sc["masks"] else "없음"
            QTreeWidgetItem(self.tree, [name, "불량", str(len(imgs)), has]).setData(0, Qt.UserRole, imgs[0])
        for i in range(4):
            self.tree.resizeColumnToContents(i)
        n_def = sum(len(v) for v in sc["defects"].values())
        if not sc["good"] and not sc["defects"]:
            self.test_info.setText("이미지를 찾지 못했습니다")
        elif not sc["defects"]:
            self.test_info.setText(f"정상 {len(sc['good'])}장. 불량 폴더가 없어 AUROC는 계산할 수 없습니다")
        else:
            mk = "마스크 있음(픽셀 AUROC 계산)" if core.has_all_masks(sc) else "마스크 없음(이미지 AUROC만)"
            self.test_info.setText(f"정상 {len(sc['good'])}장, 불량 {n_def}장, {mk}")

    def _show_item(self, item):
        self._show(item.data(0, Qt.UserRole))

    def _show(self, path):
        pm = QPixmap(path)
        if not pm.isNull():
            self.preview.set_pixmap(pm)
            self.preview_name.setText(path)

    def _save(self):
        normal = self.normal_edit.text().strip()
        if not core.list_images(normal):
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "데이터", "정상 이미지 폴더에 이미지가 없습니다.")
            return
        test = self.test_edit.text().strip()
        self.work.set("normal_dir", normal)
        self.work.set("test_dir", test)
        settings.set_value(TOOL, "normal_dir", normal)
        settings.set_value(TOOL, "test_dir", test)
        settings.push_recent(TOOL, normal)
        self.step_done.emit()

    def load_last(self):
        for key, edit, fn in (("normal_dir", self.normal_edit, self._refresh_normal),
                              ("test_dir", self.test_edit, self._refresh_test)):
            val = str(settings.get_value(TOOL, key, ""))
            if val and os.path.isdir(val):
                edit.setText(val)
                fn()
