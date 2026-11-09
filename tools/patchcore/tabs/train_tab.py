"""학습 탭: 설정 요약, 학습 시작, 진행 표시와 로그. 끝나면 평가 탭으로 넘어가고 모델이 자동으로 로드된다."""
import shutil
import threading

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox, QLabel, QMessageBox, QProgressBar, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from tools.patchcore import core
from tools.patchcore.ui_common import Worker, two_pane


class TrainTab(QWidget):
    step_done = Signal()

    def __init__(self, work, model_tab):
        super().__init__()
        self.work = work
        self.model_tab = model_tab
        self.worker = None
        self._build()

    def _build(self):
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 8, 0)
        g = QGroupBox("학습 설정")
        gv = QVBoxLayout(g)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        gv.addWidget(self.summary)
        note = QLabel("PatchCore는 정상 이미지의 특징을 모으는 방식이라 에포크 없이 1회 통과로 끝난다.")
        note.setObjectName("Hint")
        note.setWordWrap(True)
        gv.addWidget(note)
        lv.addWidget(g)
        self.start_btn = QPushButton("학습 시작")
        self.start_btn.setObjectName("Primary")
        self.start_btn.clicked.connect(self._start)
        lv.addWidget(self.start_btn)
        self.stop_btn = QPushButton("학습 중단")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        lv.addWidget(self.stop_btn)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1)
        self.bar.setTextVisible(False)
        lv.addWidget(self.bar)
        lv.addStretch(1)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 0, 0, 0)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        rv.addWidget(self.log)
        two_pane(self, left, right)

    def showEvent(self, e):
        super().showEvent(e)
        p = self.model_tab.params()
        self.summary.setText(
            f"정상 폴더: {self.work.get('normal_dir') or '(없음)'}\n"
            f"테스트 폴더: {self.work.get('test_dir') or '(없음)'}\n"
            f"백본 {p['backbone']}, 층 {', '.join(p['layers'])}\n"
            f"coreset {p['coreset']}, 입력 {p['size'][0]}x{p['size'][1]}")

    def _start(self):
        normal = self.work.get("normal_dir")
        if not normal:
            QMessageBox.warning(self, "학습", "데이터 탭에서 정상 이미지 폴더를 먼저 저장하세요.")
            return
        p = self.model_tab.params()
        cfg = dict(p, normal_dir=normal, test_dir=self.work.get("test_dir"),
                   out_dir=core.new_run_dir(normal, p["backbone"]))
        self.log.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stop_event = threading.Event()
        self.bar.setRange(0, 0)   # 끝을 알 수 없는 진행 표시
        self.cfg = cfg

        def job(log):
            ckpt = core.train(cfg, log, self.stop_event)
            core.save_config(ckpt, {k: cfg[k] for k in ("backbone", "layers", "coreset", "size", "normal_dir", "test_dir")})
            return ckpt

        self.worker = Worker(job)
        self.worker.log.connect(self.log.append)
        self.worker.done.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _stop(self):
        """학습 중단: 다음 배치가 시작될 때 멈춘다 (모델은 저장하지 않음)"""
        if self.worker is not None and self.worker.isRunning():
            self.stop_event.set()
            self.stop_btn.setEnabled(False)
            self.log.append("중단 요청… 현재 배치가 끝나면 멈춥니다")

    def _finished(self, ckpt):
        self.stop_btn.setEnabled(False)
        self.bar.setRange(0, 1)
        self.bar.setValue(1)
        self.start_btn.setEnabled(True)
        self.work.set("ckpt", ckpt)
        self.log.append("학습 완료")
        self.step_done.emit()

    def _failed(self, msg):
        self.stop_btn.setEnabled(False)
        if self.stop_event.is_set():   # 사용자가 멈춘 것: 오류창 없이 안내만
            self.bar.setRange(0, 1)
            self.bar.setValue(0)
            self.start_btn.setEnabled(True)
            shutil.rmtree(self.cfg["out_dir"], ignore_errors=True)   # 반쪽짜리 실행 폴더 정리
            self.log.append("학습을 중단했습니다 (모델은 저장되지 않음)")
            return
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.start_btn.setEnabled(True)
        QMessageBox.critical(self, "학습 오류", msg)
