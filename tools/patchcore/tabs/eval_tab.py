"""평가 탭: 이미지/픽셀 AUROC, 정상·불량 점수 분포, 임계값 자동 제안. 결과는 체크포인트 폴더 Results/ 에 저장."""
import json
import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from common import theme
from tools.patchcore import core
from tools.patchcore.ui_common import Worker, pick_dir, pick_open, primary_button, two_pane


class EvalTab(QWidget):
    step_done = Signal()

    def __init__(self, work):
        super().__init__()
        self.work = work
        self.result = None
        self.worker = None
        self._build()
        self.work.changed.connect(self._on_work)

    def _build(self):
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 8, 0)

        g = QGroupBox("모델과 테스트 폴더")
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
        self.test_edit = QLineEdit()
        self.test_edit.setPlaceholderText("테스트 폴더")
        self.test_edit.editingFinished.connect(lambda: self.work.set("test_dir", self.test_edit.text().strip()))
        r = QHBoxLayout()
        r.addWidget(self.test_edit, 1)
        b = QPushButton("찾기")
        b.clicked.connect(self._browse_test)
        r.addWidget(b)
        gv.addLayout(r)
        lv.addWidget(g)

        self.run_btn = QPushButton("평가 실행")
        self.run_btn.clicked.connect(self._run)
        lv.addWidget(self.run_btn)

        g2 = QGroupBox("결과")
        v2 = QVBoxLayout(g2)
        self.lbl_img = QLabel("이미지 AUROC: -")
        self.lbl_pix = QLabel("픽셀 AUROC: -")
        self.lbl_cnt = QLabel("정상 - / 불량 -")
        self.lbl_thr = QLabel("제안 임계값: -")
        for w in (self.lbl_img, self.lbl_pix, self.lbl_cnt, self.lbl_thr):
            v2.addWidget(w)
        self.apply_btn = QPushButton("임계값 적용")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_thr)
        v2.addWidget(self.apply_btn)
        self.status = QLabel("")
        self.status.setObjectName("Hint")
        self.status.setWordWrap(True)
        v2.addWidget(self.status)
        lv.addWidget(g2)
        lv.addStretch(1)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 0, 0, 0)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(5, 4))
        self.canvas = Canvas(self.fig)
        rv.addWidget(self.canvas, 1)

        self.save_btn = primary_button("저장하고 다음")
        self.save_btn.clicked.connect(self.step_done.emit)
        two_pane(self, left, right, self.save_btn)

    # ── 작업 상태 연동 ───────────────────────────────────────
    def _on_work(self, key):
        if key == "ckpt" and self.ckpt_edit.text() != self.work.get("ckpt"):
            self.ckpt_edit.setText(self.work.get("ckpt"))
            self.result = None
        elif key == "test_dir" and self.test_edit.text() != self.work.get("test_dir"):
            self.test_edit.setText(self.work.get("test_dir"))

    def load_trained(self):
        """학습이 끝난 뒤: 모델과 테스트 폴더를 채우고 바로 평가한다."""
        self._on_work("ckpt")
        self._on_work("test_dir")
        cfg = core.load_config(self.work.get("ckpt"))
        if cfg.get("size"):
            self.work.set("size", tuple(cfg["size"]))
        if self.test_edit.text().strip():
            self._run()

    def _browse_ckpt(self):
        p = pick_open(self, "모델 선택", "체크포인트 (*.ckpt)", self.ckpt_edit.text())
        if p:
            self.work.set("ckpt", p)
            cfg = core.load_config(p)
            if cfg.get("size"):
                self.work.set("size", tuple(cfg["size"]))

    def _browse_test(self):
        p = pick_dir(self, self.test_edit.text())
        if p:
            self.work.set("test_dir", p)

    # ── 평가 ─────────────────────────────────────────────────
    def _run(self):
        ckpt, test = self.work.get("ckpt"), self.work.get("test_dir")
        if not os.path.isfile(ckpt):
            QMessageBox.warning(self, "평가", "모델 파일을 먼저 고르세요.")
            return
        scan = core.scan_test_folder(test)
        if not scan["good"] and not scan["defects"]:
            QMessageBox.warning(self, "평가", "테스트 폴더에서 이미지를 찾지 못했습니다.")
            return
        size = self.work.get("size")
        self.run_btn.setEnabled(False)
        self.status.setText("평가 중")
        self.worker = Worker(lambda log: core.evaluate(ckpt, scan, size, log))
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _failed(self, msg):
        self.run_btn.setEnabled(True)
        self.status.setText("")
        QMessageBox.critical(self, "평가 오류", msg)

    def _done(self, res):
        self.run_btn.setEnabled(True)
        self.result = res
        f = lambda v: "-" if v is None else f"{v:.4f}"
        self.lbl_img.setText(f"이미지 AUROC: {f(res['image_auroc'])}")
        self.lbl_pix.setText("픽셀 AUROC: " + (f(res["pixel_auroc"]) if res["pixel_auroc"] is not None else "- (마스크 없음)"))
        self.lbl_cnt.setText(f"정상 {res['n_good']}장 / 불량 {res['n_defect']}장")
        thr = res["threshold"]
        self.lbl_thr.setText("제안 임계값: " + (f"{thr:.3f}" if thr is not None else "- (정상/불량이 모두 필요)"))
        self.apply_btn.setEnabled(thr is not None)
        self._plot(res)
        self.status.setText(self._save_results(res))

    def _plot(self, res):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        good = [r[2] for r in res["rows"] if r[1] == 0]
        bad = [r[2] for r in res["rows"] if r[1] == 1]
        bins = 20
        if good:
            ax.hist(good, bins=bins, range=(0, 1), alpha=0.6, label="정상", color=theme.ACCENT)
        if bad:
            ax.hist(bad, bins=bins, range=(0, 1), alpha=0.6, label="불량", color=theme.DANGER)
        if res["threshold"] is not None:
            ax.axvline(res["threshold"], color=theme.TEXT, linestyle="--", linewidth=1, label="제안 임계값")
        ax.set_xlabel("이상 점수")
        ax.set_ylabel("이미지 수")
        ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def _save_results(self, res):
        """체크포인트 폴더/Results 에 요약 json과 점수 분포 그림을 저장한다."""
        out = os.path.join(os.path.dirname(self.work.get("ckpt")), "Results")
        try:
            os.makedirs(out, exist_ok=True)
            summary = {k: res[k] for k in ("n_good", "n_defect", "image_auroc", "pixel_auroc", "threshold")}
            summary["scores"] = [{"image": r[0], "label": "불량" if r[1] else "정상", "score": r[2]} for r in res["rows"]]
            with open(os.path.join(out, "eval_summary.json"), "w", encoding="utf-8") as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=2)
            self.fig.savefig(os.path.join(out, "score_distribution.png"), dpi=120)
            return f"결과 저장: {out}"
        except OSError as e:
            return f"결과 저장 실패: {e}"

    def _apply_thr(self):
        thr = self.result["threshold"]
        self.work.set("threshold", float(thr))
        self.status.setText(f"추론 탭 임계값을 {thr:.3f}로 지정했습니다")
