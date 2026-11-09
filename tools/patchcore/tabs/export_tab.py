"""내보내기 탭: 최종 모델을 ONNX + config로 내보낸다 (<체크포인트 폴더>/export/<모델>_<날짜-시각>/)."""
import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGroupBox, QLabel, QMessageBox, QTextEdit, QVBoxLayout, QWidget

from tools.patchcore import core
from tools.patchcore.ui_common import Worker, primary_button, two_pane


class ExportTab(QWidget):
    step_done = Signal()

    def __init__(self, work):
        super().__init__()
        self.work = work
        self.worker = None
        self._build()

    def _build(self):
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 8, 0)
        g = QGroupBox("내보낼 모델")
        gv = QVBoxLayout(g)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        gv.addWidget(self.info)
        hint = QLabel("model.onnx, model_config.json, README.md, Results 요약이 한 폴더에 만들어진다. "
                      "ONNX 파일 안에도 config가 들어간다.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        gv.addWidget(hint)
        lv.addWidget(g)
        lv.addStretch(1)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 0, 0, 0)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        rv.addWidget(self.log)

        self.btn = primary_button("최종 모델 내보내기")
        self.btn.clicked.connect(self._export)
        two_pane(self, left, right, self.btn)

    def showEvent(self, e):
        super().showEvent(e)
        w = self.work
        self.info.setText(f"모델: {w.get('ckpt') or '(없음)'}\n입력 크기: {w.get('size')[0]}x{w.get('size')[1]}\n"
                          f"임계값: {w.get('threshold'):.2f}")

    def _export(self):
        ckpt = self.work.get("ckpt")
        if not os.path.isfile(ckpt):
            QMessageBox.warning(self, "내보내기", "모델 파일이 없습니다. 학습하거나 평가 탭에서 모델을 고르세요.")
            return
        size, thr = self.work.get("size"), self.work.get("threshold")
        self.btn.setEnabled(False)
        self.worker = Worker(lambda log: core.export_onnx(ckpt, size, thr, log))
        self.worker.log.connect(self.log.append)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _done(self, out_dir):
        self.btn.setEnabled(True)
        self.log.append(f"내보내기 완료: {out_dir}")
        self.step_done.emit()

    def _failed(self, msg):
        self.btn.setEnabled(True)
        QMessageBox.critical(self, "내보내기 오류", msg)
