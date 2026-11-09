import glob
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QGridLayout, QLineEdit,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QTextEdit, QSpinBox, QListWidget, QListWidgetItem, QCheckBox,
    QTabWidget, QDoubleSpinBox, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal
from tools.segmentation.utils.file_dialog import pick_dir, pick_open, pick_save


# ═══════════════════════════════════════════════════════════════
#  QuantWorker — 양자화 → FP32 평가 → INT8 평가
# ═══════════════════════════════════════════════════════════════
class QuantWorker(QThread):
    rows_ready = Signal(list)
    int8_ready = Signal(str)      # 만들어진 INT8 파일 경로
    log_msg   = Signal(str)
    finished   = Signal()

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            self._run_quant()
        except Exception as e:
            import traceback
            self.log_msg.emit(f"[오류] {e}\n{traceback.format_exc()}")
        finally:
            self.finished.emit()

    def _run_quant(self):
        import tempfile
        from tools.segmentation.utils.quantize_onnx import (
            benchmark_onnx, compare_rows, cuda_available, evaluate_onnx, failed_row,
            pick_calibration_files, quantize_int8, tensorrt_status, trt_fail_reason, trt_providers, trt_rows,
        )
        c = self.config
        log = self.log_msg.emit
        threads = c["threads"]
        warmup, runs = c.get("bench_warmup", 3), c.get("bench_runs", 20)     # 기본: 워밍업 3회 + 20회
        files = pick_calibration_files(c["calib_dir"], c["img_ext"], c["calib_count"])
        if not files:
            log(f"[오류] 캘리브레이션 이미지 없음: {c['calib_dir']}")
            return
        if len(files) < c["calib_count"]:
            log(f"  [주의] 이미지가 {len(files)}장뿐이라 전부 사용합니다")

        trt_ok, trt_reason = tensorrt_status()
        use_trt = c["use_trt"] and trt_ok
        gpu = cuda_available()
        steps = 3 + (2 if use_trt else 0)
        log(f"[1/{steps}] 정적 양자화 (INT8)")
        int8_path = quantize_int8(c["onnx_path"], files, c["excluded"], log=log)
        self.int8_ready.emit(int8_path)

        with tempfile.TemporaryDirectory() as tmp:
            val_img, val_lbl = c["val_img_dir"], c["val_lbl_dir"]
            if c.get("eval_limit"):      # 확인용: 검증 이미지를 몇 장만 쓴다
                from tools.segmentation.utils.sensitivity import make_val_subset
                val_img, val_lbl, n = make_val_subset(val_img, val_lbl, c["img_ext"], c["lbl_ext"],
                                                      c["eval_limit"], tmp)
                log(f"  평가 이미지 {n}장만 사용")
            # 속도는 검증 첫 장 한 장으로 잰다 (정확도 평가 루프와 분리)
            sample = sorted(glob.glob(os.path.join(val_img, f"*.{c['img_ext']}")))[0]
            log(f"  속도 측정: 워밍업 {warmup}회 + {runs}회, CPU {threads}스레드" + (", GPU(CUDA) 포함" if gpu else ""))

            evals, speeds, gpu_speeds = {}, {}, {}
            for step, (name, path) in enumerate([("FP32", c["onnx_path"]), ("INT8", int8_path)], 2):
                log(f"[{step}/{steps}] {name} 평가 (CPU)")
                evals[name], _ = evaluate_onnx(path, c["onnx_path"], val_img, val_lbl,
                                               c["img_ext"], c["lbl_ext"], log=log, threads=threads)
                m, sd, dev = benchmark_onnx(path, c["onnx_path"], sample, threads=threads,
                                            warmup=warmup, runs=runs)
                speeds[name] = (m, sd)
                log(f"  CPU 속도 {m:.2f} ± {sd:.2f} ms ({dev})")
                if gpu:
                    try:
                        gpu_speeds[name] = benchmark_onnx(path, c["onnx_path"], sample,
                                                          providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                                                          threads=threads, warmup=warmup, runs=runs)
                        log(f"  GPU 속도 {gpu_speeds[name][0]:.2f} ± {gpu_speeds[name][1]:.2f} ms ({gpu_speeds[name][2]})")
                    except Exception as e:
                        log(f"  [경고] GPU 속도 측정 실패: {str(e).splitlines()[0][:200] if str(e) else type(e).__name__}")
            rows = compare_rows(c["onnx_path"], int8_path, evals["FP32"], evals["INT8"], speeds, threads,
                                gpu_speeds if len(gpu_speeds) == 2 else None)

            if not trt_ok:
                rows.append(failed_row("TensorRT", trt_reason))     # 못 불러온 원인을 표에 남긴다
            elif use_trt:
                trt = {}
                # FP32 기준은 TensorRT FP16, 같은 INT8 .onnx 는 TensorRT INT8 로 평가
                for step, (name, path, is_int8) in enumerate(
                        [("TensorRT FP16", c["onnx_path"], False), ("TensorRT INT8", int8_path, True)], 4):
                    log(f"[{step}/{steps}] {name} 평가")
                    try:
                        res, _ = evaluate_onnx(path, c["onnx_path"], val_img, val_lbl,
                                               c["img_ext"], c["lbl_ext"], log=log, trt_int8=is_int8)
                        providers, _cache = trt_providers(path, is_int8)
                        m, sd, _dev = benchmark_onnx(path, c["onnx_path"], sample, providers=providers,
                                                     threads=0, warmup=warmup, runs=runs)
                        trt[name] = (res, m, sd)
                    except Exception as e:
                        # TensorRT 가 실패해도 CPU 비교표는 보여준다
                        reason = trt_fail_reason(e)
                        log(f"  [경고] {name} 평가 실패, CPU 결과만 표시합니다: {reason}")
                        rows.append(failed_row(name, reason))
                        break
                else:
                    rows += trt_rows(c["onnx_path"], int8_path, trt["TensorRT FP16"], trt["TensorRT INT8"])
        self.rows_ready.emit(rows)
        if c["wandb_enabled"]:
            from tools.segmentation.utils.quantize_onnx import COLUMNS
            from tools.segmentation.utils.wandb_logger import log_quant_table
            log_quant_table(c["onnx_path"], rows, COLUMNS, log=log)


# ═══════════════════════════════════════════════════════════════
#  ExportWorker — 최종 모델 내보내기 (모델 + config + 범례 + README + Results)
# ═══════════════════════════════════════════════════════════════
class ExportWorker(QThread):
    log_msg = Signal(str)
    done    = Signal(str)     # 만들어진 폴더 경로 (실패하면 빈 글자)

    def __init__(self, model_path, fmt, calib_dir, img_ext, calib_count):
        super().__init__()
        self.args = (model_path, fmt, calib_dir, img_ext, calib_count)

    def run(self):
        out = ""
        try:
            from tools.segmentation.utils.export_bundle import export_bundle
            model_path, fmt, calib_dir, img_ext, calib_count = self.args
            out = export_bundle(model_path, fmt, calib_dir, img_ext, calib_count, log=self.log_msg.emit)
        except Exception as e:
            self.log_msg.emit(f"[오류] 내보내기 실패: {type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}")
        self.done.emit(out)


# ═══════════════════════════════════════════════════════════════
#  SensitivityWorker — 민감도 분석 (레이어 SQNR / 블록 민감도 / 자동 제외)
# ═══════════════════════════════════════════════════════════════
class SensitivityWorker(QThread):
    progress     = Signal(int, int)
    result_ready = Signal(str, object)      # (단계 이름, 결과)
    log_msg      = Signal(str)
    finished     = Signal()

    def __init__(self, mode, config, ranking=None, target=0.01, blocks=None):
        super().__init__()
        self.mode, self.config = mode, config
        self.ranking, self.target, self.blocks = ranking or [], target, blocks
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        from tools.segmentation.utils.sensitivity import SensitivityRunner, Stopped
        try:
            with SensitivityRunner(self.config, log=self.log_msg.emit, progress=self.progress.emit,
                                   stop=lambda: self._stop) as runner:
                if self.mode == "layers":
                    result = runner.layer_sqnr()
                elif self.mode == "blocks":
                    result = runner.run_blocks()
                else:
                    result = runner.run_auto(self.ranking, self.target, blocks=self.blocks)
            self.result_ready.emit(self.mode, result)
        except Stopped:
            self.log_msg.emit("[중지] 분석을 멈췄습니다")
        except Exception as e:
            import traceback
            self.log_msg.emit(f"[오류] {e}\n{traceback.format_exc()}")
        finally:
            self.finished.emit()


# ═══════════════════════════════════════════════════════════════
#  QuantTab UI
# ═══════════════════════════════════════════════════════════════
class QuantTab(QWidget):
    step_done = Signal()   # 단계 표시줄에 '끝남'을 알리는 신호
    def __init__(self, shared_state):
        super().__init__()
        self.state     = shared_state
        self.worker    = None
        self._data_tab = None
        self._rows     = []
        self._int8_path = ""
        self._sens     = {}       # 민감도 분석 결과 {layers, blocks, auto}
        self._ranking  = []       # ①의 오차 큰 순 레이어 이름
        self._ranking_for = ""    # 그 순위를 만든 .onnx 경로
        self._blocks_for = ""     # 블록 민감도 결과를 만든 .onnx 경로
        self.sens_worker = None
        self._build()
        # 현재 작업 상태: 모델(.pth)이 정해지면 옆의 .onnx로, 검증 폴더는 평가 폴더로 미리 채운다
        self.state = shared_state
        self.work = shared_state["work"]
        self.work.changed.connect(self._on_work)
        for k in ("model_path", "onnx_path", "img_dir", "lbl_dir", "img_ext"):
            self._on_work(k)

    def set_data_tab(self, data_tab):
        self._data_tab = data_tab

    def _on_work(self, key):
        val = self.work.get(key)
        if not val: return
        if key == "model_path":
            # 모델(.pt) 옆에 .onnx가 있으면 그것, 없으면 작업 상태에 저장된 .onnx
            onnx = os.path.splitext(val)[0] + ".onnx"
            if not os.path.isfile(onnx):
                onnx = self.work.get("onnx_path")
            if onnx and os.path.isfile(onnx) and self.onnx_path.text().strip() != onnx:
                self.onnx_path.setText(onnx)
                self._load_layers()
        elif key == "onnx_path":
            if os.path.isfile(val) and self.onnx_path.text().strip() != val:
                self.onnx_path.setText(val)
                self._load_layers()
        elif key == "img_dir":
            self.val_img.setText(val)
        elif key == "lbl_dir":
            self.val_lbl.setText(val)
        elif key == "img_ext":
            if self.img_ext.findText(val) >= 0: self.img_ext.setCurrentText(val)

    def showEvent(self, event):
        """탭을 열 때 비어 있는 경로를 데이터 탭 값으로 채운다."""
        super().showEvent(event)
        dt = self._data_tab
        if dt is None:
            return
        if not self.calib_dir.text().strip():
            self.calib_dir.setText(dt.train_img_edit.text().strip())
        if not self.val_img.text().strip():
            self.val_img.setText(dt.val_img_edit.text().strip())
        if not self.val_lbl.text().strip():
            self.val_lbl.setText(dt.val_lbl_edit.text().strip())   # 작업 상태 값이 있으면 위에서 이미 채워져 있다
        self.img_ext.setCurrentText(self.work.get("img_ext") or dt.img_ext.currentText())
        self.lbl_ext.setCurrentText(dt.lbl_ext.currentText())

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)

        # ── 왼쪽 설정 패널 ───────────────────────────────────────
        left_w = QWidget()
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner  = QWidget(); scroll.setWidget(inner)
        lv = QVBoxLayout(inner)
        lv.setSpacing(10); lv.setContentsMargins(14, 14, 14, 14)

        mg_g = QGroupBox("ONNX 모델")
        mg = QGridLayout(mg_g)
        mg.addWidget(QLabel("모델 파일 (.onnx):"), 0, 0)
        self.onnx_path = QLineEdit(); self.onnx_path.setPlaceholderText("학습 때 .pth 옆에 저장된 .onnx")
        self.onnx_path.editingFinished.connect(self._load_layers)
        # 경량화에서 쓰는 .onnx는 작업 상태에 저장 (추론 탭이 'ONNX'로 불러 씀)
        self.onnx_path.textChanged.connect(lambda t: self.work.set("onnx_path", t) if t.lower().endswith(".onnx") else None)
        mg.addWidget(self.onnx_path, 0, 1)
        btn_m = QPushButton("찾기"); btn_m.setObjectName("secondary")
        btn_m.clicked.connect(self._browse_onnx); mg.addWidget(btn_m, 0, 2)
        mg.setColumnStretch(1, 1); lv.addWidget(mg_g)

        dg_g = QGroupBox("데이터")
        dg = QGridLayout(dg_g)
        self.calib_dir = QLineEdit(); self.calib_dir.setPlaceholderText("학습 이미지 폴더")
        self.val_img   = QLineEdit(); self.val_img.setPlaceholderText("검증 이미지 폴더")
        self.val_lbl   = QLineEdit(); self.val_lbl.setPlaceholderText("검증 라벨 폴더")
        for r, (label, edit) in enumerate([("캘리브레이션 이미지:", self.calib_dir),
                                           ("검증 이미지:", self.val_img),
                                           ("검증 라벨:", self.val_lbl)]):
            dg.addWidget(QLabel(label), r, 0)
            dg.addWidget(edit, r, 1)
            btn = QPushButton("찾기"); btn.setObjectName("secondary")
            # e=edit: 반복 변수를 지금 값으로 묶어 둔다 (안 그러면 전부 마지막 칸을 가리킴)
            btn.clicked.connect(lambda _, e=edit: self._browse_dir(e))
            dg.addWidget(btn, r, 2)
        dg.addWidget(QLabel("캘리브레이션 장수:"), 3, 0)
        self.calib_count = QSpinBox(); self.calib_count.setRange(100, 200); self.calib_count.setValue(150)
        dg.addWidget(self.calib_count, 3, 1)
        dg.addWidget(QLabel("이미지 확장자:"), 4, 0)
        self.img_ext = QComboBox(); self.img_ext.addItems(["png", "jpg", "tiff", "tif", "mat", "npy"])
        dg.addWidget(self.img_ext, 4, 1)
        dg.addWidget(QLabel("라벨 확장자:"), 5, 0)
        self.lbl_ext = QComboBox(); self.lbl_ext.addItems(["png", "jpg", "tiff", "tif", "mat", "npy"])
        dg.addWidget(self.lbl_ext, 5, 1)
        hint = QLabel("비율로 분리, K-fold는 검증 분할이 저장되지 않으므로 검증 폴더를 직접 지정하세요")
        hint.setObjectName("hint"); hint.setWordWrap(True)
        dg.addWidget(hint, 6, 0, 1, 3)
        dg.setColumnStretch(1, 1); lv.addWidget(dg_g)

        from tools.segmentation.utils.quantize_onnx import tensorrt_status
        trt_g = QGroupBox("설정")
        tv = QVBoxLayout(trt_g)
        self.trt_check = QCheckBox("TensorRT로도 평가")
        self.trt_check.setToolTip("같은 INT8 .onnx를 TensorRT로 평가하고, FP32 기준은 TensorRT FP16으로 평가합니다. "
                                  "엔진 캐시는 .onnx 옆 trt_cache 폴더에 저장됩니다")
        trt_ok, trt_reason = tensorrt_status()
        self.trt_hint = QLabel(""); self.trt_hint.setObjectName("hint"); self.trt_hint.setWordWrap(True)
        if trt_ok:
            self.trt_hint.setText("첫 실행은 엔진 빌드로 오래 걸립니다 (빌드 시간은 로그에 따로 표시, 다음부터 캐시 사용)")
        else:
            self.trt_check.setChecked(False); self.trt_check.setEnabled(False)
            self.trt_hint.setText(f"사용할 수 없음: {trt_reason}")   # 이유에 'TensorRT 라이브러리 설치 필요' 포함
        tv.addWidget(self.trt_check); tv.addWidget(self.trt_hint)
        th = QHBoxLayout()
        th.addWidget(QLabel("CPU 스레드 수:"))
        self.threads = QSpinBox(); self.threads.setRange(1, os.cpu_count() or 1)
        self.threads.setValue(os.cpu_count() or 1)       # 기본: 전체 코어
        self.threads.setToolTip("속도 측정과 평가에 쓰는 CPU 스레드 수 (기본 전체 코어)")
        th.addWidget(self.threads); th.addStretch()
        tv.addLayout(th)
        lv.addWidget(trt_g)

        ly_g = QGroupBox("양자화할 레이어  (체크 해제 = FP32 유지)")
        lyv = QVBoxLayout(ly_g)
        self.preset_btn = QPushButton("첫 Conv와 출력 헤드 제외")
        self.preset_btn.setObjectName("secondary")
        self.preset_btn.clicked.connect(self._apply_preset)
        lyv.addWidget(self.preset_btn)
        self.layer_list = QListWidget(); self.layer_list.setMinimumHeight(260)
        self.layer_list.itemChanged.connect(self._update_count)
        lyv.addWidget(self.layer_list)
        self.count_label = QLabel("ONNX 파일을 불러오세요"); self.count_label.setObjectName("hint")
        lyv.addWidget(self.count_label)
        lv.addWidget(ly_g)

        self.run_btn = QPushButton("평가 시작")
        self.run_btn.clicked.connect(self._run)
        lv.addWidget(self.run_btn)

        sn_g = QGroupBox("민감도 분석  (캘리브레이션 20장, 검증 50장 사용)")
        sg = QVBoxLayout(sn_g)
        self.sens_layer_btn = QPushButton("① 레이어별 오차 (SQNR)")
        self.sens_block_btn = QPushButton("② 블록별 민감도 (mIoU 하락)")
        for b, mode in ((self.sens_layer_btn, "layers"), (self.sens_block_btn, "blocks")):
            b.setObjectName("secondary")
            b.clicked.connect(lambda _, m=mode: self._sens_start(m))
            sg.addWidget(b)
        tg = QHBoxLayout()
        tg.addWidget(QLabel("목표 mIoU 차이:"))
        self.sens_target = QDoubleSpinBox(); self.sens_target.setRange(0.1, 50.0)
        self.sens_target.setSingleStep(0.1); self.sens_target.setValue(1.0); self.sens_target.setSuffix(" %p")
        tg.addWidget(self.sens_target); tg.addStretch()
        sg.addLayout(tg)
        self.sens_auto_btn = QPushButton("③ 자동 제외  (①을 먼저 실행)")
        self.sens_auto_btn.setObjectName("secondary"); self.sens_auto_btn.setEnabled(False)
        self.sens_auto_btn.clicked.connect(lambda: self._sens_start("auto"))
        sg.addWidget(self.sens_auto_btn)
        pr = QHBoxLayout()
        self.sens_progress = QProgressBar(); self.sens_progress.setTextVisible(False)
        self.sens_stop_btn = QPushButton("중지"); self.sens_stop_btn.setObjectName("secondary")
        self.sens_stop_btn.setEnabled(False); self.sens_stop_btn.clicked.connect(self._sens_stop)
        pr.addWidget(self.sens_progress, 1); pr.addWidget(self.sens_stop_btn)
        sg.addLayout(pr)
        self.sens_csv_btn = QPushButton("분석 결과 CSV 저장"); self.sens_csv_btn.setObjectName("secondary")
        self.sens_csv_btn.setEnabled(False); self.sens_csv_btn.clicked.connect(self._save_sens_csv)
        sg.addWidget(self.sens_csv_btn)
        sh = QLabel("③은 FP32 대비 mIoU 차이가 목표 안에 들어오는 최소 개수를 찾아 위 체크박스에 반영합니다")
        sh.setObjectName("hint"); sh.setWordWrap(True); sg.addWidget(sh)
        lv.addWidget(sn_g)
        lv.addStretch()

        lw = QVBoxLayout(left_w); lw.setContentsMargins(0, 0, 0, 0); lw.addWidget(scroll)
        splitter.addWidget(left_w)

        # ── 오른쪽 결과 ──────────────────────────────────────────
        from tools.segmentation.utils.quantize_onnx import COLUMNS
        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setContentsMargins(14, 14, 14, 14); rv.setSpacing(8)

        self.right_tabs = QTabWidget()
        cmp_page = QWidget()
        cv = QVBoxLayout(cmp_page); cv.setContentsMargins(0, 8, 0, 0); cv.setSpacing(8)
        res_l = QLabel("FP32 / INT8 비교  (CUDA가 있으면 GPU 줄, TensorRT 평가를 켜면 TensorRT 줄 추가)"); res_l.setObjectName("section_title")
        cv.addWidget(res_l)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)  # 긴 머리글 잘림 방지
        self.table.horizontalHeader().setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.ResizeToContents)  # 실행 장치
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(290)   # CPU 3줄 + GPU 2줄 + TensorRT 2줄
        cv.addWidget(self.table)
        self.int8_label = QLabel(""); self.int8_label.setObjectName("hint")
        self.int8_label.setWordWrap(True)
        self.int8_label.setTextInteractionFlags(Qt.TextSelectableByMouse)   # 경로를 복사할 수 있게
        cv.addWidget(self.int8_label)
        self.csv_btn = QPushButton("CSV 저장"); self.csv_btn.setObjectName("secondary")
        self.csv_btn.setEnabled(False)
        self.csv_btn.clicked.connect(self._save_csv)
        cv.addWidget(self.csv_btn)
        cv.addStretch()
        self.right_tabs.addTab(cmp_page, "비교표")
        self.right_tabs.addTab(self._build_sens_page(), "민감도 분석")
        rv.addWidget(self.right_tabs, 2)

        log_l = QLabel("로그"); log_l.setObjectName("section_title"); rv.addWidget(log_l)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        rv.addWidget(self.log_box, 1)

        # 최종 모델 내보내기: 형식을 고르고 export/<모델이름>_<날짜-시각>/ 폴더로 모은다
        ex = QHBoxLayout()
        ex.addStretch()
        ex.addWidget(QLabel("형식:"))
        self.export_fmt = QComboBox(); self.export_fmt.addItems(["PyTorch", "ONNX", "ONNX INT8"])
        self.export_fmt.setToolTip("ONNX INT8은 왼쪽 캘리브레이션 이미지 폴더로 경량화합니다")
        ex.addWidget(self.export_fmt)
        self.export_btn = QPushButton("최종 모델 내보내기")
        self.export_btn.clicked.connect(self._export_final)
        ex.addWidget(self.export_btn)
        rv.addLayout(ex)

        splitter.addWidget(right_w)
        splitter.setSizes([420, 680])
        outer.addWidget(splitter)

    # ── 민감도 분석 화면 ─────────────────────────────────────────
    def _make_canvas(self):
        """막대그래프용 (Figure, 캔버스). matplotlib 이 없으면 (None, None)."""
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
            from matplotlib.figure import Figure
        except ImportError:
            return None, None
        fig = Figure(figsize=(6, 4), facecolor="white")
        return fig, Canvas(fig)

    def _build_sens_page(self):
        page = QWidget()
        pv = QVBoxLayout(page); pv.setContentsMargins(0, 8, 0, 0)
        self.sens_tabs = QTabWidget()

        def make_table(headers, stretch_col, max_h):
            t = QTableWidget(0, len(headers))
            t.setHorizontalHeaderLabels(headers)
            t.horizontalHeader().setSectionResizeMode(stretch_col, QHeaderView.Stretch)
            t.verticalHeader().setVisible(False)
            t.setMaximumHeight(max_h)
            return t

        # ① 레이어 SQNR: 표 + 막대그래프
        w1 = QWidget(); v1 = QVBoxLayout(w1)
        self.sqnr_table = make_table(["순위", "레이어", "연산", "SQNR (dB)"], 1, 190)
        v1.addWidget(self.sqnr_table)
        self.sqnr_fig, self.sqnr_canvas = self._make_canvas()
        if self.sqnr_canvas:
            self.sqnr_canvas.setMinimumHeight(480)      # 막대 30개의 이름이 겹치지 않을 높이
        v1.addWidget(self.sqnr_canvas if self.sqnr_canvas else QLabel("matplotlib 미설치"), 1)
        self.sens_tabs.addTab(self._scrolled(w1), "① 레이어 SQNR")

        # ② 블록 민감도: 막대그래프 + 표
        w2 = QWidget(); v2 = QVBoxLayout(w2)
        self.block_fig, self.block_canvas = self._make_canvas()
        if self.block_canvas:
            self.block_canvas.setMinimumHeight(420)     # 블록이 많아도 이름이 겹치지 않게 (넘치면 스크롤)
        v2.addWidget(self.block_canvas if self.block_canvas else QLabel("matplotlib 미설치"), 1)
        self.block_table = make_table(["블록", "레이어 수", "mIoU", "하락 (%p)"], 0, 190)
        v2.addWidget(self.block_table)
        self.sens_tabs.addTab(self._scrolled(w2), "② 블록 민감도")

        # ③ 자동 제외: 요약 + 시도 기록
        w3 = QWidget(); v3 = QVBoxLayout(w3)
        self.auto_label = QLabel("①을 실행한 뒤 ③ 자동 제외를 누르세요"); self.auto_label.setWordWrap(True)
        v3.addWidget(self.auto_label)
        self.auto_table = make_table(["제외 개수", "mIoU", "FP32 대비 (%p)"], 0, 400)
        v3.addWidget(self.auto_table); v3.addStretch()
        self.sens_tabs.addTab(w3, "③ 자동 제외")

        pv.addWidget(self.sens_tabs)
        return page

    @staticmethod
    def _scrolled(widget):
        """차트가 창보다 커도 잘리지 않도록 스크롤 영역으로 감싼다."""
        area = QScrollArea(); area.setWidgetResizable(True); area.setWidget(widget)
        return area

    @staticmethod
    def _fill(table, rows):
        """rows: 줄마다 셀 글자 리스트. 숫자 칸은 오른쪽 정렬."""
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                item = QTableWidgetItem(str(text))
                if c > 0 and str(text).replace(".", "").replace("-", "").replace("+", "").isdigit():
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(r, c, item)

    @staticmethod
    def _short(name, width=38):
        s = "/".join(name.strip("/").split("/")[-3:])      # 이름 끝 3조각만
        return s if len(s) <= width else "…" + s[-(width - 1):]

    def _bars(self, fig, canvas, labels, values, xlabel, title):
        if canvas is None:
            return
        import matplotlib
        from matplotlib import font_manager
        have = {f.name for f in font_manager.fontManager.ttflist}
        for font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
            if font in have:
                matplotlib.rcParams["font.family"] = font
                break
        matplotlib.rcParams["axes.unicode_minus"] = False
        fig.clear()
        ax = fig.add_subplot(111)
        pos = list(range(len(labels)))
        ax.barh(pos, values, color="#2563eb")
        ax.set_yticks(pos); ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()           # 첫 항목이 맨 위로
        ax.set_xlabel(xlabel, fontsize=8); ax.set_title(title, fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        canvas.draw()

    # ── 민감도 분석 실행 ─────────────────────────────────────────
    def _sens_inputs(self):
        """분석에 필요한 입력을 확인해 설정 dict 로 돌려준다. 문제가 있으면 로그에 적고 None."""
        from tools.segmentation.utils.quantize_onnx import read_config
        onnx_path = self.onnx_path.text().strip()
        calib = self.calib_dir.text().strip()
        val_img = self.val_img.text().strip()
        val_lbl = self.val_lbl.text().strip()
        if not os.path.isfile(onnx_path):
            self.log_box.append("[오류] ONNX 파일을 선택해주세요"); return None
        if not os.path.isdir(calib):
            self.log_box.append("[오류] 캘리브레이션 이미지 폴더를 선택해주세요"); return None
        if not os.path.isdir(val_img) or not os.path.isdir(val_lbl):
            self.log_box.append("[오류] 검증 이미지와 라벨 폴더를 선택해주세요"); return None
        try:
            read_config(onnx_path)
        except OSError:
            self.log_box.append("[오류] .onnx 옆에 <모델이름>_config.json 이 필요합니다"); return None
        return {"onnx_path": onnx_path, "calib_dir": calib, "val_img_dir": val_img, "val_lbl_dir": val_lbl,
                "img_ext": self.img_ext.currentText(), "lbl_ext": self.lbl_ext.currentText()}

    def _sens_busy(self, busy):
        for b in (self.sens_layer_btn, self.sens_block_btn, self.run_btn):
            b.setEnabled(not busy)
        self.sens_auto_btn.setEnabled(not busy and bool(self._ranking))
        self.sens_stop_btn.setEnabled(busy)
        if not busy:
            self.sens_progress.setValue(0)

    def _sens_start(self, mode):
        running = [w for w in (self.worker, self.sens_worker) if w is not None and w.isRunning()]
        if running:
            self.log_box.append("[오류] 다른 작업이 실행 중입니다"); return
        c = self._sens_inputs()
        if c is None:
            return
        if mode == "auto" and (not self._ranking or self._ranking_for != c["onnx_path"]):
            self.log_box.append("[오류] 이 모델의 ① 레이어별 오차를 먼저 실행해주세요"); return
        self.log_box.clear()
        self._sens_busy(True)
        self.sens_worker = SensitivityWorker(mode, c, ranking=self._ranking,
                                             target=self.sens_target.value() / 100.0,
                                             blocks=self._sens.get("blocks") if self._blocks_for == c["onnx_path"] else None)
        self.sens_worker.progress.connect(
            lambda cur, total: (self.sens_progress.setRange(0, total), self.sens_progress.setValue(cur)))
        self.sens_worker.log_msg.connect(self.log_box.append)
        self.sens_worker.result_ready.connect(self._on_sens_result)
        self.sens_worker.finished.connect(lambda: self._sens_busy(False))
        self.sens_worker.start()

    def _sens_stop(self):
        if self.sens_worker is not None:
            self.sens_worker.stop()
            self.log_box.append("중지 요청: 지금 하던 평가가 끝나면 멈춥니다")

    def _on_sens_result(self, mode, res):
        from tools.segmentation.utils.sensitivity import TOP_N
        self.right_tabs.setCurrentIndex(1)
        if mode == "layers":
            self._sens["layers"] = res
            self._ranking = [r["레이어"] for r in res]
            self._ranking_for = self.sens_worker.config["onnx_path"]    # 순위를 만든 모델
            top = res[:TOP_N]
            self._fill(self.sqnr_table, [[i, self._short(r["레이어"], 60), r["연산"], f"{r['SQNR (dB)']:.2f}"]
                                         for i, r in enumerate(top, 1)])
            self._bars(self.sqnr_fig, self.sqnr_canvas,
                       [f"{i}. {self._short(r['레이어'])}" for i, r in enumerate(top, 1)],
                       [r["SQNR (dB)"] for r in top], "SQNR (dB)  낮을수록 오차가 큼",
                       f"레이어별 출력 오차 — 오차 큰 상위 {len(top)}개 / 전체 {len(res)}개")
            self.sens_tabs.setCurrentIndex(0)
            self.log_box.append(f"레이어별 오차 완료: 측정 {len(res)}개, 가장 큰 오차 {res[0]['SQNR (dB)']:.1f} dB")
        elif mode == "blocks":
            self._sens["blocks"] = res
            self._blocks_for = self.sens_worker.config["onnx_path"]     # 블록 결과를 만든 모델
            self._fill(self.block_table, [[r["블록"], r["레이어 수"], f"{r['mIoU']:.4f}", f"{100 * r['하락']:+.2f}"]
                                          for r in res])
            self._bars(self.block_fig, self.block_canvas, [self._short(r["블록"]) for r in res],
                       [100 * r["하락"] for r in res], "mIoU 하락 (%p)  FP32 - 이 블록만 INT8",
                       "블록별 민감도 (그래프 순서)")
            self.sens_tabs.setCurrentIndex(1)
            self.log_box.append(f"블록별 민감도 완료: 블록 {len(res)}개")
        else:
            self._sens["auto"] = res
            state = "달성" if res["reached"] else "미달 (후보를 전부 되돌려도 목표에 못 미침)"
            self.auto_label.setText(
                f"제외 {res['k_found']}개 → 줄이기 후 {res['k']}개  →  mIoU {res['miou']:.4f}  (FP32 {res['base']:.4f}, "
                f"차이 {100 * (res['base'] - res['miou']):+.2f}%p, 목표 {100 * res['target']:.2f}%p {state})")
            self._fill(self.auto_table, [[t["제외 개수"], f"{t['mIoU']:.4f}", f"{100 * t['차이']:+.2f}"]
                                         for t in res["trials"]])
            self._set_excluded(res["excluded"])
            self.sens_tabs.setCurrentIndex(2)
            self.log_box.append(f"자동 제외 완료: 찾은 {res['k_found']}개 중 {res['k']}개만 FP32로 유지 → 체크박스에 반영했습니다")
        self.sens_csv_btn.setEnabled(True)

    def _set_excluded(self, names):
        """names 에 있는 레이어만 체크 해제(FP32 유지), 나머지는 체크(INT8)."""
        names = set(names)
        self.layer_list.blockSignals(True)
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            item.setCheckState(Qt.Unchecked if item.data(Qt.UserRole) in names else Qt.Checked)
        self.layer_list.blockSignals(False)
        self._update_count()

    def _save_sens_csv(self):
        from tools.segmentation.utils.sensitivity import save_csv
        default = os.path.splitext(self.onnx_path.text().strip())[0] + "_sensitivity.csv"
        path = pick_save(self, "민감도 분석 저장", "CSV (*.csv)", default, self.state)
        if path:
            save_csv(path, self._sens.get("layers"), self._sens.get("blocks"), self._sens.get("auto"))
            self.log_box.append(f"저장: {path}")

    # ── helpers ──────────────────────────────────────────────────
    def _browse_onnx(self):
        path = pick_open(self, "ONNX 모델", "ONNX (*.onnx);;All (*)", self.onnx_path.text(), self.state)
        if path:
            self.onnx_path.setText(path)
            self._load_layers()

    def _browse_dir(self, edit):
        path = pick_dir(self, edit.text(), self.state)
        if path:
            edit.setText(path)

    def _load_layers(self):
        from tools.segmentation.utils.quantize_onnx import list_layers
        path = self.onnx_path.text().strip()
        self.layer_list.clear()
        if not os.path.isfile(path):
            self.count_label.setText("ONNX 파일을 불러오세요")
            return
        try:
            layers = list_layers(path)
        except Exception as e:
            self.log_box.append(f"[오류] ONNX 읽기 실패: {e}")
            return
        self.layer_list.blockSignals(True)      # 항목마다 개수를 다시 세지 않도록
        for name, op in layers:
            item = QListWidgetItem(f"{op}   {name}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, name)     # 표시 글자와 별도로 레이어 이름 보관
            self.layer_list.addItem(item)
        self.layer_list.blockSignals(False)
        self._update_count()

    def _apply_preset(self):
        from tools.segmentation.utils.quantize_onnx import default_excluded
        path = self.onnx_path.text().strip()
        if not os.path.isfile(path):
            self.log_box.append("[오류] ONNX 파일을 먼저 불러오세요")
            return
        excluded = default_excluded(path)
        self.layer_list.blockSignals(True)
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            item.setCheckState(Qt.Unchecked if item.data(Qt.UserRole) in excluded else Qt.Checked)
        self.layer_list.blockSignals(False)
        self._update_count()

    def _excluded_layers(self):
        items = (self.layer_list.item(i) for i in range(self.layer_list.count()))
        return [it.data(Qt.UserRole) for it in items if it.checkState() != Qt.Checked]

    def _update_count(self, *args):
        total = self.layer_list.count()
        off = len(self._excluded_layers())
        self.count_label.setText(f"전체 {total}개 중 양자화 {total - off}개, 제외 {off}개")

    def _run(self):
        from tools.segmentation.utils.quantize_onnx import read_config
        onnx_path = self.onnx_path.text().strip()
        calib = self.calib_dir.text().strip()
        val_img = self.val_img.text().strip()
        val_lbl = self.val_lbl.text().strip()
        if not os.path.isfile(onnx_path):
            self.log_box.append("[오류] ONNX 파일을 선택해주세요"); return
        if not os.path.isdir(calib):
            self.log_box.append("[오류] 캘리브레이션 이미지 폴더를 선택해주세요"); return
        if not os.path.isdir(val_img) or not os.path.isdir(val_lbl):
            self.log_box.append("[오류] 검증 이미지와 라벨 폴더를 선택해주세요"); return
        try:
            cfg = read_config(onnx_path)
        except OSError:
            self.log_box.append("[오류] .onnx 옆에 <모델이름>_config.json 이 필요합니다"); return

        self.log_box.clear()
        if "input_scale" not in cfg:
            self.log_box.append("[주의] json에 input_scale이 없어 입력을 정규화하지 않습니다. "
                                "학습 때 정규화했다면 결과가 틀립니다")
        if os.path.normcase(os.path.abspath(calib)) == os.path.normcase(os.path.abspath(val_img)):
            self.log_box.append("[주의] 캘리브레이션과 검증 폴더가 같습니다")

        self.run_btn.setEnabled(False); self.csv_btn.setEnabled(False)
        self.table.setRowCount(0); self._rows = []
        self.int8_label.setText("")
        self.worker =QuantWorker({
            "onnx_path": onnx_path, "calib_dir": calib, "calib_count": self.calib_count.value(),
            "val_img_dir": val_img, "val_lbl_dir": val_lbl,
            "img_ext": self.img_ext.currentText(), "lbl_ext": self.lbl_ext.currentText(),
            "excluded": self._excluded_layers(),
            "wandb_enabled": self.state.get("wandb_enabled", False),
            "use_trt": self.trt_check.isChecked(), "threads": self.threads.value(),
        })
        self.worker.int8_ready.connect(self._on_int8)
        self.worker.rows_ready.connect(self._on_rows)
        self.worker.log_msg.connect(self.log_box.append)
        self.worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.worker.start()

    def _on_int8(self, path):
        self._int8_path = path
        self.int8_label.setText(f"INT8 파일: {path}")

    def _on_rows(self, rows):
        from tools.segmentation.utils.quantize_onnx import COLUMNS
        self._rows = rows
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, key in enumerate(COLUMNS):
                value = row[key]
                text = value if isinstance(value, str) else f"{value:.4f}" if c <= 3 else f"{value:.2f}"
                self.table.setItem(r, c, QTableWidgetItem(text))
        self.csv_btn.setEnabled(True)
        self.log_box.append("양자화 평가 완료")
        self.step_done.emit()
        self.log_box.append(f"INT8 파일 경로: {self._int8_path}")   # 평가 로그에 묻히지 않게 끝에 한 번 더

    def _export_final(self):
        model = self.work.get("model_path")
        if not model or not os.path.isfile(model):
            self.log_box.append("[오류] 내보낼 모델(.pt)이 없습니다. 평가 또는 추론 탭에서 모델을 먼저 고르세요"); return
        self.export_btn.setEnabled(False)
        self.log_box.append(f"[최종 모델 내보내기] {self.export_fmt.currentText()}  {model}")
        self.export_worker = ExportWorker(model, self.export_fmt.currentText(), self.calib_dir.text().strip(),
                                          self.img_ext.currentText(), self.calib_count.value())
        self.export_worker.log_msg.connect(self.log_box.append)
        self.export_worker.done.connect(lambda out: self.export_btn.setEnabled(True))
        self.export_worker.start()

    def _save_csv(self):
        from tools.segmentation.utils.quantize_onnx import save_csv
        default = os.path.splitext(self.onnx_path.text().strip())[0] + "_quant_compare.csv"
        path = pick_save(self, "비교표 저장", "CSV (*.csv)", default, self.state)
        if path:
            save_csv(path, self._rows)
            self.log_box.append(f"저장: {path}")
