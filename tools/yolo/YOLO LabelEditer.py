print("=" * 50)
print(" YOLO Label Editor 로딩 중... (10~20초 소요)")
print(" 창이 뜰 때까지 이 콘솔을 닫지 마세요.")
print("=" * 50)

import sys
import json
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('TkAgg')  # PyInstaller exe에서 Agg로 빠지지 않도록 명시
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, Slider
from tkinter import filedialog
import tkinter as tk
import os
import tifffile as tiff 
import cv2
from ultralytics import YOLO

# exe(PyInstaller)로 실행될 때는 exe가 있는 폴더를 기준 경로로 사용
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COLOR_PALETTE = ['red', 'blue', 'green', 'magenta', 'purple', 'black', 'orange',
                 'cyan', 'brown', 'olive', 'teal', 'navy', 'crimson', 'darkgreen']

class MSILabelerV2:
    def __init__(self):
        # 기본 설정
        self.default_model_path = os.path.join(BASE_DIR, 'best.pt')
        self.class_file = os.path.join(BASE_DIR, 'classes.json')
        self.class_names = {0: 'PET', 1: 'PS', 2: 'PE', 3: 'PP', 4: 'PMMA', 5: 'Paper', 6: 'Aluminum'}
        self.load_classes()
        
        self.data = None
        self.model = None
        self.boxes = []  # [x1, y1, x2, y2, cls]
        self.selected_idx = None
        self.image_list = []   # 현재 폴더의 이미지 경로 리스트
        self.image_idx = -1
        self.modified = False  # 사용자가 박스를 편집했는지 (자동저장 여부 판단용)
        
        # [수정 2] 초기 채널값을 텍스트박스 표시값(1,3,4)에 맞춰 인덱스(0,2,3)로 설정
        self.rgb_bands = [0, 2, 3] 
        self.brightness = 1.0
        self.loaded_filename = ""
        
        # 드래그 관련 상태
        self.is_drawing = False
        self.is_resizing = False
        self.start_pos = None
        
        # UI 구성
        self.fig = plt.figure(figsize=(15, 9))
        self.ax = self.fig.add_axes([0.05, 0.15, 0.7, 0.75])
        self.ax.axis('off')
        
        # 클래스 라벨 창 (Legend)
        self.ax_legend = self.fig.add_axes([0.8, 0.65, 0.15, 0.25])
        self.ax_legend.axis('off')
        self.update_legend()

        # 컨트롤 패널
        self.fig.text(0.8, 0.58, "False Color Channels:", fontsize=10, fontweight='bold')
        ax_r = self.fig.add_axes([0.8, 0.53, 0.03, 0.03]); self.tb_r = TextBox(ax_r, 'R ', initial='1')
        ax_g = self.fig.add_axes([0.85, 0.53, 0.03, 0.03]); self.tb_g = TextBox(ax_g, 'G ', initial='3')
        ax_b = self.fig.add_axes([0.9, 0.53, 0.03, 0.03]); self.tb_b = TextBox(ax_b, 'B ', initial='4')
        
        self.tb_r.on_submit(lambda v: self.update_channels())
        self.tb_g.on_submit(lambda v: self.update_channels())
        self.tb_b.on_submit(lambda v: self.update_channels())

        # 밝기 슬라이더
        ax_bright = self.fig.add_axes([0.8, 0.495, 0.13, 0.02])
        self.slider_bright = Slider(ax_bright, 'Brightness', 0.1, 5.0, valinit=1.0)
        self.slider_bright.on_changed(self.update_brightness)

        # Percentile 스트레칭 슬라이더 (표시 명암 범위 조절)
        self.pct_low = 1.0
        self.pct_high = 99.0
        ax_plow = self.fig.add_axes([0.8, 0.465, 0.13, 0.02])
        self.slider_plow = Slider(ax_plow, 'Min %', 0.0, 50.0, valinit=self.pct_low)
        self.slider_plow.on_changed(self.update_percentile)
        ax_phigh = self.fig.add_axes([0.8, 0.435, 0.13, 0.02])
        self.slider_phigh = Slider(ax_phigh, 'Max %', 50.0, 100.0, valinit=self.pct_high)
        self.slider_phigh.on_changed(self.update_percentile)

        # 예측 confidence 임계값 (Enter로 적용 시 현재 이미지 재예측)
        self.conf = 0.30
        ax_conf = self.fig.add_axes([0.85, 0.402, 0.05, 0.025])
        self.tb_conf = TextBox(ax_conf, 'Conf ', initial=f'{self.conf:.2f}')
        self.tb_conf.on_submit(self.update_conf)

        ax_load_model = self.fig.add_axes([0.8, 0.28, 0.13, 0.05])
        self.btn_model = Button(ax_load_model, 'Load New Model')
        self.btn_model.on_clicked(self.manual_load_model)

        ax_edit_cls = self.fig.add_axes([0.8, 0.22, 0.13, 0.05])
        self.btn_edit_cls = Button(ax_edit_cls, 'Edit Classes')
        self.btn_edit_cls.on_clicked(self.edit_classes)

        ax_load_img = self.fig.add_axes([0.8, 0.35, 0.13, 0.05])
        self.btn_img = Button(ax_load_img, 'Open Image')
        self.btn_img.on_clicked(self.load_image)

        ax_load_lbl = self.fig.add_axes([0.8, 0.09, 0.13, 0.05])
        self.btn_load_lbl = Button(ax_load_lbl, 'Load Labels')
        self.btn_load_lbl.on_clicked(self.load_labels_button)

        ax_save = self.fig.add_axes([0.8, 0.15, 0.13, 0.06], facecolor='honeydew')
        self.btn_save = Button(ax_save, 'SAVE YOLO TXT')
        self.btn_save.on_clicked(self.save_labels)

        # 이전/다음 이미지 네비게이션
        ax_prev = self.fig.add_axes([0.05, 0.04, 0.08, 0.06])
        self.btn_prev = Button(ax_prev, '< Prev')
        self.btn_prev.on_clicked(self.prev_image)
        ax_next = self.fig.add_axes([0.14, 0.04, 0.08, 0.06])
        self.btn_next = Button(ax_next, 'Next >')
        self.btn_next.on_clicked(self.next_image)
        self.info_text = self.fig.text(0.24, 0.07, "No image loaded", fontsize=10, va='center')

        # 좌우 화살표를 이미지 전환에 쓰기 위해 matplotlib 기본 단축키에서 제거
        for keymap, key in (('keymap.back', 'left'), ('keymap.forward', 'right')):
            if key in plt.rcParams[keymap]:
                plt.rcParams[keymap].remove(key)

        # 초기 모델 로드
        self.auto_load_model()

        # 이벤트 바인딩
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def update_legend(self):
        self.ax_legend.clear()
        self.ax_legend.set_title("[ Class Labels ]", fontsize=10)
        keys = sorted(self.class_names)
        step = min(0.12, 0.9 / max(1, len(keys)))
        for pos, i in enumerate(keys):
            self.ax_legend.text(0.1, 0.85 - (pos * step), f"{i}: {self.class_names[i]}",
                               color=self.class_colors[i], fontweight='bold', fontsize=11)
        self.ax_legend.axis('off')

    def refresh_colors(self):
        self.class_colors = {i: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in self.class_names}

    def load_classes(self):
        if os.path.exists(self.class_file):
            try:
                with open(self.class_file, 'r', encoding='utf-8') as f:
                    self.class_names = {int(k): v for k, v in json.load(f).items()}
                print(f"Classes loaded: {self.class_file}")
            except Exception as e:
                print(f"classes.json 로드 실패, 기본값 사용: {e}")
        self.refresh_colors()

    def save_classes(self):
        try:
            with open(self.class_file, 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in sorted(self.class_names.items())}, f,
                          ensure_ascii=False, indent=2)
            print(f"Classes saved: {self.class_file}")
        except Exception as e:
            print(f"classes.json 저장 실패: {e}")

    def edit_classes(self, event):
        root = tk.Tk()
        root.title("Edit Classes")
        root.attributes('-topmost', True)
        entries = {}
        frame = tk.Frame(root)
        frame.pack(padx=12, pady=10)

        def add_row(idx, name):
            row = tk.Frame(frame)
            row.pack(fill='x', pady=2)
            tk.Label(row, text=f"{idx} :", width=4, anchor='e').pack(side='left')
            e = tk.Entry(row, width=22)
            e.insert(0, name)
            e.pack(side='left', padx=4)
            entries[idx] = e

        for i in sorted(self.class_names):
            add_row(i, self.class_names[i])

        def on_add():
            idx = max(entries) + 1 if entries else 0
            add_row(idx, f"Class{idx}")

        def on_apply():
            new_names = {}
            for idx, e in entries.items():
                name = e.get().strip()
                if name:
                    new_names[idx] = name
            if new_names:
                self.class_names = new_names
                self.refresh_colors()
                self.save_classes()
                self.update_legend()
                self.update_display()
            root.destroy()

        btns = tk.Frame(root)
        btns.pack(pady=(0, 10))
        tk.Button(btns, text="+ Add Class", width=12, command=on_add).pack(side='left', padx=5)
        tk.Button(btns, text="Apply & Save", width=12, command=on_apply).pack(side='left', padx=5)
        tk.Label(root, text="* 이름을 비우면 해당 클래스는 목록에서 제외됩니다",
                 fg='gray').pack(pady=(0, 8))
        root.mainloop()

    def auto_load_model(self):
        if os.path.exists(self.default_model_path):
            self.model = YOLO(self.default_model_path)
            print(f"Default model loaded: {self.default_model_path}")
        else:
            print("Default model not found. Please load manually.")

    def manual_load_model(self, event):
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(filetypes=[('YOLO model', '*.pt')])
        if path:
            self.model = YOLO(path)
            print(f"New model loaded: {path}")
        root.destroy()

    def update_channels(self, text=None):
        try:
            # 1-based 입력을 0-based 인덱스로 변환
            r = int(self.tb_r.text) - 1
            g = int(self.tb_g.text) - 1
            b = int(self.tb_b.text) - 1
            self.rgb_bands = [r, g, b]
            self.update_display()
        except Exception as e:
            print(f"채널 입력 오류: {e}")

    def update_brightness(self, val):
        self.brightness = val
        self.update_display()

    def update_percentile(self, val=None):
        self.pct_low = self.slider_plow.val
        self.pct_high = self.slider_phigh.val
        self.update_display()

    def update_conf(self, text):
        try:
            val = float(text)
            if not (0.0 < val < 1.0):
                raise ValueError("0과 1 사이 값이어야 합니다")
        except ValueError as e:
            print(f"Conf 입력 오류: {e}")
            self.tb_conf.set_val(f'{self.conf:.2f}')
            return
        self.conf = val
        # 새 임계값으로 현재 이미지 재예측 (수동 편집한 박스는 사라짐)
        if self.data is not None and self.model:
            self.run_prediction()
            self.selected_idx = None
            self.modified = False
            self.update_display()
            print(f"conf={val:.2f} 재예측: {len(self.boxes)} boxes")

    def load_image(self, event):
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(filetypes=[('Image', '*.mat *.tif *.tiff'), ('All files', '*.*')])
        root.destroy()
        if not path: return
        self.open_image_path(path)

    def open_image_path(self, path):
        ext = path.lower().split('.')[-1]
        if ext == 'mat':
            mat = sio.loadmat(path)
            data = mat.get('image', mat.get(next(k for k in mat if not k.startswith('_'))))
        elif ext in ['tiff', 'tif']:
            data = np.transpose(tiff.imread(path), (1, 2, 0))
        else:
            print(f"지원하지 않는 형식: {ext}")
            return
        self.data = data.astype(np.float32)
        self.loaded_filename = path
        self.selected_idx = None
        self.modified = False
        self.build_image_list(path)

        # 기본 동작: 모델 자동 예측 (저장된 라벨 확인은 Load Labels 버튼으로)
        if self.model:
            self.run_prediction()
        else:
            self.boxes = []

        name = os.path.basename(path)
        self.info_text.set_text(f"[{self.image_idx + 1}/{len(self.image_list)}] {name}")
        self.update_display()

    def load_labels_button(self, event):
        if self.data is None:
            print("이미지를 먼저 열어주세요.")
            return
        # 같은 이름의 txt가 있으면 바로 로드, 없으면 탐색창
        txt_path = os.path.splitext(self.loaded_filename)[0] + '.txt'
        if not os.path.exists(txt_path):
            root = tk.Tk(); root.withdraw()
            txt_path = filedialog.askopenfilename(
                title='라벨 txt 선택',
                filetypes=[('YOLO label', '*.txt')])
            root.destroy()
        if txt_path:
            self.load_labels_txt(txt_path)
            self.selected_idx = None
            self.update_display()

    def build_image_list(self, path):
        # tkinter가 반환하는 경로는 / 구분자라서 정규화 후 비교해야 함
        folder = os.path.dirname(path)
        exts = ('.mat', '.tif', '.tiff')
        try:
            files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
            self.image_list = [os.path.normpath(os.path.join(folder, f)) for f in files]
            norm = os.path.normpath(path).lower()
            self.image_idx = [p.lower() for p in self.image_list].index(norm)
            print(f"폴더 이미지 {len(self.image_list)}개 중 {self.image_idx + 1}번째")
        except (OSError, ValueError) as e:
            print(f"이미지 리스트 구성 실패 ({e}) - 단일 이미지 모드")
            self.image_list = [path]
            self.image_idx = 0

    def load_labels_txt(self, txt_path):
        h, w = self.data.shape[:2]
        self.boxes = []
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5: continue
                    cls = int(float(parts[0]))
                    xc, yc, nw, nh = map(float, parts[1:5])
                    x1, x2 = (xc - nw / 2) * w, (xc + nw / 2) * w
                    y1, y2 = (yc - nh / 2) * h, (yc + nh / 2) * h
                    self.boxes.append([x1, y1, x2, y2, cls])
            self.modified = False  # 파일 내용과 동일한 상태
            print(f"Labels loaded: {txt_path} ({len(self.boxes)} boxes)")
        except Exception as e:
            print(f"라벨 로드 실패: {e}")

    def prev_image(self, event=None):
        self.step_image(-1)

    def next_image(self, event=None):
        self.step_image(1)

    def step_image(self, delta):
        if not self.image_list or self.image_idx < 0: return
        new_idx = self.image_idx + delta
        if not (0 <= new_idx < len(self.image_list)):
            print("이미지 리스트의 끝입니다.")
            return
        self.autosave_labels()
        self.open_image_path(self.image_list[new_idx])

    def autosave_labels(self):
        # 사용자가 편집한 경우에만 자동저장 (예측 결과만 있는 이미지가 txt를 덮어쓰지 않도록)
        if not self.modified or not self.loaded_filename or self.data is None: return
        txt_path = os.path.splitext(self.loaded_filename)[0] + '.txt'
        if self.boxes or os.path.exists(txt_path):
            self.write_labels(txt_path)

    def run_prediction(self):
        import tempfile
        temp_tiff = os.path.join(tempfile.gettempdir(), 'temp_inf.tiff')
        ar = self.data.transpose((2, 0, 1))
        cv2.imwritemulti(temp_tiff, [ar[i] for i in range(min(6, ar.shape[0]))])
        # agnostic_nms: 클래스가 달라도 같은 위치에 겹친 박스는 conf 높은 것 하나만 남김
        results = self.model.predict(temp_tiff, conf=self.conf, agnostic_nms=True, verbose=False)
        self.boxes = []
        if results[0].boxes:
            for b in results[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                cls = int(b.cls[0].cpu().numpy())
                self.boxes.append([x1, y1, x2, y2, cls])
        if os.path.exists(temp_tiff): os.remove(temp_tiff)

    def update_display(self):
        self.ax.clear()
        if self.data is not None:
            # 6채널 데이터를 초과하는 인덱스 방지 예외처리 포함
            try:
                img = self.data[:, :, self.rgb_bands]
                norm_img = np.zeros_like(img)
                for i in range(3):
                    ch = img[:,:,i]
                    low, high = np.percentile(ch, [self.pct_low, self.pct_high])
                    norm_img[:,:,i] = np.clip(((ch - low) / (high - low + 1e-8)) * self.brightness, 0, 1)
                self.ax.imshow(norm_img)
            except IndexError:
                print("Error: Channel index out of range.")
            
            for i, (x1, y1, x2, y2, cls) in enumerate(self.boxes):
                is_sel = (i == self.selected_idx)
                color = self.class_colors.get(cls, 'white')
                edge_color = 'yellow' if is_sel else color
                lw = 3 if is_sel else 1.5
                
                # 드래그 방향에 상관없이 사각형 그리기
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor=edge_color, linewidth=lw)
                self.ax.add_patch(rect)
                self.ax.text(xmin, ymin-5, f"{cls}:{self.class_names.get(cls,'?')}", color=edge_color, fontsize=9, fontweight='bold')
        
        self.ax.axis('off')
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax: return
        
        if event.button == 3: # 우클릭: 선택
            self.selected_idx = None
            # [수정 1] 드래그 방향에 관계없이 선택되도록 min/max 좌표 적용
            for i, (x1, y1, x2, y2, cls) in enumerate(self.boxes):
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                if xmin < event.xdata < xmax and ymin < event.ydata < ymax:
                    self.selected_idx = i
                    break
        
        elif event.button == 1: # 좌클릭: 생성 혹은 리사이징
            if self.selected_idx is not None:
                x1, y1, x2, y2, cls = self.boxes[self.selected_idx]
                # 우하단 코너 근처 클릭 시 리사이징
                if abs(event.xdata - x2) < 20 and abs(event.ydata - y2) < 20:
                    self.is_resizing = True
                    self.modified = True
                    return
            
            self.is_drawing = True
            self.modified = True
            self.start_pos = (event.xdata, event.ydata)
            self.boxes.append([event.xdata, event.ydata, event.xdata, event.ydata, 0])
            self.selected_idx = len(self.boxes) - 1
            
        self.update_display()

    def on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        
        if self.is_drawing:
            self.boxes[-1][2] = event.xdata
            self.boxes[-1][3] = event.ydata
            self.update_display()
        elif self.is_resizing and self.selected_idx is not None:
            self.boxes[self.selected_idx][2] = event.xdata
            self.boxes[self.selected_idx][3] = event.ydata
            self.update_display()

    def on_release(self, event):
        self.is_drawing = False
        self.is_resizing = False

    def on_key(self, event):
        if event.key == 'left':
            self.prev_image(); return
        if event.key == 'right':
            self.next_image(); return
        if self.selected_idx is None: return
        if event.key is not None and event.key.isdigit() and int(event.key) in self.class_names:
            self.boxes[self.selected_idx][4] = int(event.key)
            self.modified = True
        elif event.key in ['delete', 'backspace']:
            self.boxes.pop(self.selected_idx)
            self.selected_idx = None
            self.modified = True
        self.update_display()

    def save_labels(self, event):
        if not self.loaded_filename or self.data is None: return
        txt_path = os.path.splitext(self.loaded_filename)[0] + ".txt"
        self.write_labels(txt_path)
        self.modified = False

    def write_labels(self, txt_path):
        h, w = self.data.shape[:2]
        with open(txt_path, 'w') as f:
            for x1, y1, x2, y2, cls in self.boxes:
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                xc, yc = (xmin + xmax)/2/w, (ymin + ymax)/2/h
                nw, nh = (xmax - xmin)/w, (ymax - ymin)/h
                f.write(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
        print(f"Saved: {txt_path}")

if __name__ == "__main__":
    try:
        app = MSILabelerV2()
        plt.show()
    except Exception:
        import traceback
        err_path = os.path.join(BASE_DIR, 'error_log.txt')
        with open(err_path, 'w', encoding='utf-8') as f:
            f.write(traceback.format_exc())
        print(traceback.format_exc())
        print(f"\n오류가 {err_path} 에 저장되었습니다. 아무 키나 누르면 종료합니다.")
        input()