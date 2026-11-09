"""추론 탭: 폴더의 이미지에 히트맵을 겹쳐 보고, 임계값 슬라이더로 불량 판정 목록을 만든다."""
import os

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSlider, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from tools.patchcore import core
from tools.patchcore.ui_common import ImageView, Worker, pick_dir, pick_open, primary_button, to_pixmap, two_pane


class InferTab(QWidget):
    step_done = Signal()

    def __init__(self, work):
        super().__init__()
        self.work = work
        self.results = {}   # 이미지 경로 -> (점수, anomaly_map)
        self.worker = None
        self._build()
        self.work.changed.connect(self._on_work)
        self._on_work("threshold")

    def _build(self):
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 8, 0)

        g = QGroupBox("모델과 이미지")
        gv = QVBoxLayout(g)
        self.ckpt_edit = QLineEdit()
        self.ckpt_edit.setPlaceholderText("모델 파일 (.ckpt)")
        self.ckpt_edit.editingFinished.connect(lambda: self.work.set("ckpt", self.ckpt_edit.text().strip()))
        r = QHBoxLayout()
        r.addWidget(self.ckpt_edit, 1)
        b = QPushButton("찾기")
        b.clicked.connect(self._browse_ckpt)
        r.addWidget(b)
        gv.addLayout(r)
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("검사할 이미지 폴더")
        r = QHBoxLayout()
        r.addWidget(self.dir_edit, 1)
        b = QPushButton("찾기")
        b.clicked.connect(self._browse_dir)
        r.addWidget(b)
        gv.addLayout(r)
        self.run_btn = QPushButton("추론 실행")
        self.run_btn.clicked.connect(self._run)
        gv.addWidget(self.run_btn)
        lv.addWidget(g)

        g2 = QGroupBox("임계값")
        v2 = QVBoxLayout(g2)
        row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self._slider_changed)
        self.thr_lbl = QLabel("0.50")
        row.addWidget(self.slider, 1)
        row.addWidget(self.thr_lbl)
        v2.addLayout(row)
        self.count_lbl = QLabel("")
        self.count_lbl.setObjectName("Hint")
        v2.addWidget(self.count_lbl)
        lv.addWidget(g2)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["파일", "점수", "판정"])
        self.tree.setRootIsDecorated(False)
        self.tree.currentItemChanged.connect(lambda cur, _: cur and self._show(cur.data(0, Qt.UserRole)))
        lv.addWidget(self.tree, 1)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 0, 0, 0)
        self.view = ImageView("추론을 실행하면 히트맵이 겹쳐 보입니다")
        rv.addWidget(self.view, 1)
        self.view_info = QLabel("")
        self.view_info.setObjectName("Hint")
        rv.addWidget(self.view_info)

        self.next_btn = primary_button("저장하고 다음")
        self.next_btn.clicked.connect(self.step_done.emit)
        two_pane(self, left, right, self.next_btn)

    # ── 작업 상태 연동 ───────────────────────────────────────
    def _on_work(self, key):
        if key == "ckpt" and self.ckpt_edit.text() != self.work.get("ckpt"):
            self.ckpt_edit.setText(self.work.get("ckpt"))
            self.results = {}
        elif key == "threshold":
            v = int(round(self.work.get("threshold") * 100))
            if self.slider.value() != v:
                self.slider.setValue(v)
            self.thr_lbl.setText(f"{self.work.get('threshold'):.2f}")

    def thr(self):
        return self.slider.value() / 100.0

    def _slider_changed(self, v):
        self.work.set("threshold", v / 100.0)
        self.thr_lbl.setText(f"{v / 100.0:.2f}")
        self._refresh_list()

    def _browse_ckpt(self):
        p = pick_open(self, "모델 선택", "체크포인트 (*.ckpt)", self.ckpt_edit.text())
        if p:
            self.work.set("ckpt", p)
            cfg = core.load_config(p)
            if cfg.get("size"):
                self.work.set("size", tuple(cfg["size"]))

    def _browse_dir(self):
        p = pick_dir(self, self.dir_edit.text())
        if p:
            self.dir_edit.setText(p)

    # ── 추론 ─────────────────────────────────────────────────
    def _run(self):
        ckpt, folder = self.work.get("ckpt"), self.dir_edit.text().strip()
        if not os.path.isfile(ckpt):
            QMessageBox.warning(self, "추론", "모델 파일을 먼저 고르세요.")
            return
        if not core.list_images(folder):
            QMessageBox.warning(self, "추론", "이미지 폴더에 이미지가 없습니다.")
            return
        size = self.work.get("size")
        self.run_btn.setEnabled(False)
        self.view_info.setText("추론 중")
        self.worker = Worker(lambda log: core.predict_folder(ckpt, folder, size))
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _failed(self, msg):
        self.run_btn.setEnabled(True)
        self.view_info.setText("")
        QMessageBox.critical(self, "추론 오류", msg)

    def _done(self, res):
        self.run_btn.setEnabled(True)
        self.results = res
        self.view_info.setText(f"{len(res)}장 추론 완료")
        self._refresh_list()
        if self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))

    def _refresh_list(self):
        """임계값이 바뀔 때마다 판정만 다시 계산한다 (모델은 다시 돌리지 않는다)."""
        cur = self.tree.currentItem().data(0, Qt.UserRole) if self.tree.currentItem() else None
        self.tree.blockSignals(True)
        self.tree.clear()
        n_bad = 0
        for p, (score, _) in sorted(self.results.items(), key=lambda kv: -kv[1][0]):   # 점수 높은 것부터
            bad = score >= self.thr()
            n_bad += bad
            it = QTreeWidgetItem(self.tree, [os.path.basename(p), f"{score:.3f}", "불량" if bad else "정상"])
            it.setData(0, Qt.UserRole, p)
            if p == cur:
                self.tree.setCurrentItem(it)
        self.tree.blockSignals(False)
        for i in range(3):
            self.tree.resizeColumnToContents(i)
        self.count_lbl.setText(f"불량 {n_bad}장 / 전체 {len(self.results)}장" if self.results else "")
        if cur:
            self._show(cur)

    def _show(self, path):
        if path not in self.results:
            return
        from PIL import Image
        import matplotlib
        score, amap = self.results[path]
        img = Image.open(path).convert("RGB")
        base = np.asarray(img.resize((amap.shape[1], amap.shape[0])))
        a = np.clip(amap, 0, 1)   # 0~1로 맞춰진 anomaly map
        heat = (matplotlib.colormaps["jet"](a)[..., :3] * 255).astype(np.uint8)
        alpha = (a * 0.6)[..., None]   # 이상 점수가 높은 곳만 진하게 겹친다
        over = (base * (1 - alpha) + heat * alpha).astype(np.uint8)
        self.view.set_pixmap(to_pixmap(over))
        bad = score >= self.thr()
        self.view_info.setText(f"{path}  점수 {score:.3f}  {'불량' if bad else '정상'}")
