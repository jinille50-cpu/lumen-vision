# Lumen app

![Lumen app](docs/home.png)

라벨링, 학습, 평가, 추론까지 한 창에서 이어서 하는 비전 도구 모음입니다. Python 3.10 기준입니다.

## 앱 소개

### Labeler

적은 손작업으로 분할과 검출 학습 데이터를 만들기 위한 라벨링 앱.

- SAM 3로 점이나 박스를 몇 번 찍어 영역 마스크 만들기
- 한국어로 대상을 적거나 칸을 비워 두면 VLM(Ollama)이 이미지를 보고 영어 프롬프트를 만들어 SAM 3로 바로 찾기
- 폴더 전체를 글자 프롬프트로 한 번에 자동 라벨링
- 학습한 Segmentation, Object Detection 모델로 새 이미지에 1차 라벨링한 뒤, 점수 낮은 것부터 승인, 수정, 버리기. 고친 라벨로 다시 학습하는 반복에 맞춘 흐름
- 초분광 큐브(mat, npy, hdr, h5, 다채널 tif)를 열어 밴드를 골라 false color로 보며 라벨링, 클래스별 평균 스펙트럼을 CSV로 저장

![Labeler](docs/labeler.png)

### HSI Viewer

초분광 큐브를 눈으로 확인하고 물체별 스펙트럼을 뽑아 비교하기 위한 뷰어.

- 큐브를 열어 false color, 단일 밴드, PCA 합성으로 보기
- 이미지 위에 포인트나 ROI를 찍어 스펙트럼을 겹쳐 비교하고 CSV로 저장
- SAM 3에 글자 프롬프트를 주어 물체를 찾고 객체별 스펙트럼 확인
- 찾은 객체를 스펙트럼 특징으로 K개 그룹으로 분류. 방법(KMeans 등)과 객체, 픽셀, ROI 단위를 고를 수 있음
- 결과를 CSV, 라벨 맵(npy), 오버레이 PNG로 저장

![HSI Viewer](docs/hsi_viewer.png)

### Segmentation

분할 모델을 데이터 준비부터 학습, 평가, 추론, 경량화까지 한 흐름으로 돌리는 앱.

- 이미지와 라벨 폴더를 파일 이름으로 짝지어 불러오고, 폴더 분리, 비율 분리, K-fold 중에서 선택
- UNet 계열, DeepLabV3, SegFormer, TransUNet, DINOv2와 DINOv3 기반 모델, 다채널 초분광 전용 모델 중에서 골라 학습. 증강 미리보기와 조기 종료, 실시간 학습 곡선 제공
- 평가에서 mIoU, F1, Confusion Matrix, PR과 ROC 곡선 등을 골라 저장
- 추론에서 폴더 이미지의 분할 결과와 오버레이 저장. 정답 라벨이 없는 이미지도 가능
- 경량화에서 ONNX로 내보내고 INT8 양자화 후 FP32와 정확도 비교, 민감도 분석으로 양자화에서 뺄 레이어 고르기

![Segmentation 학습](docs/segmentation_train.png)

### Object Detection

라벨링한 데이터로 객체 인식 모델을 학습하고 평가해 모델 파일로 내보내는 앱.

- YOLOv8, YOLOv10, YOLO11, YOLO12, YOLO26, RT-DETR 중 모델과 크기를 골라 학습
- 폴더 분리, 비율 분리, K-fold, Full Data 학습을 지원하고, K-fold 평균 best epoch로 전체 데이터 학습 가능
- 다채널 TIFF에서 쓸 채널을 고르거나 밴드 3개를 false color로 바꿔 학습과 추론에 사용
- 평가 탭에서 지표와 결과 그림 확인, 폴더 이미지에 추론
- PyTorch, ONNX, INT8 ONNX 형식으로 내보내기

### PatchCore

불량 이미지 없이 정상 이미지만으로 불량을 찾는 이상 탐지 앱.

- 정상 이미지 폴더로 학습. 백본(WideResNet50, ResNet18, ResNet34, ResNet50, ResNeXt50), 사용할 층, coreset 비율, 입력 크기 선택. 학습 중간에 멈출 수 있음
- 정상과 불량이 섞인 테스트 폴더로 이미지 AUROC, 점수 분포, 픽셀 AUROC(마스크가 있을 때) 확인
- F1이 가장 높은 임계값을 자동으로 제안
- 폴더 이미지에 히트맵을 겹쳐 보고 임계값 슬라이더로 불량 판정 목록 만들기
- ONNX로 내보내기

![PatchCore 추론](docs/patchcore.png)

Sample images: MVTec AD 2 (Vial), Heckler-Kram et al., IJCV 2026.
Licensed under CC BY-NC-SA 4.0. Heatmap overlays added.

## 설치 순서

1. GPU에 맞는 torch, torchvision을 먼저 설치한다 (https://pytorch.org/get-started/locally/).
   - RTX 50 시리즈: cu128 — `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
   - 그 외 NVIDIA GPU: 드라이버에 맞는 빌드(예: cu126)
   - GPU 없이 CPU만: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
2. CPU판 onnxruntime이 깔려 있으면 먼저 지우기: `pip uninstall onnxruntime`
3. `pip install -r requirements.txt`

연구실 PC에서 돌아가는 정확한 버전은 `requirements-lock.txt`에 기록돼 있다 (참고용).
