# Real-time Pothole Detection & Measurement

Pipeline Python cho bài toán:
- Detect pothole (YOLOv8)
- Estimate depth (Depth Anything V2 ONNX hoặc fallback relative depth)
- Estimate surface area qua BEV/IPM xấp xỉ
- Classify severity (minor/moderate/severe)
- Xuất file đánh giá theo yêu cầu bài toán

## 1) Cài đặt

```bash
pip install -r requirements.txt
```

## 2) Cấu trúc

- `data/dataset.yaml`: config dataset YOLO
- `scripts/train_yolo.py`: train YOLOv8
- `scripts/export_onnx.py`: export ONNX detector
- `scripts/run_realtime.py`: chạy pipeline real-time
- `scripts/evaluate.py`: tạo depth/area error report
- `src/`: mã nguồn chính
- `configs/`: camera + severity rule
- `android-onnx-sample/`: Kotlin mẫu (LetterBox + ONNX YOLOv8 + Overlay) copy vào Android Studio

## 3) Train detector

```bash
python scripts/train_yolo.py --model yolov8n.pt --data data/dataset.yaml --epochs 200
```

## 4) Export ONNX

```bash
python scripts/export_onnx.py --weights runs/detect/runs/train/pothole_yolov8/weights/best.pt
```

**Quant ONNX dynamic INT8** (weights) — đường dẫn dùng `/` (ưu tiên `cd pothole`):

```bash
cd pothole
python -c "from onnxruntime.quantization import quantize_dynamic, QuantType; quantize_dynamic('runs/detect/runs/pothole/train_runs/weights/best.onnx', 'runs/detect/runs/pothole/train_runs/weights/best_int8.onnx', weight_type=QuantType.QInt8)"
```

Hoặc đường dẫn đầy đủ không dùng `cd`:

```bash
python -c "from onnxruntime.quantization import quantize_dynamic, QuantType; quantize_dynamic('d:/NguyenDuyAn_test/pothole/runs/detect/runs/pothole/train_runs/weights/best.onnx', 'd:/NguyenDuyAn_test/pothole/runs/detect/runs/pothole/train_runs/weights/best_int8.onnx', weight_type=QuantType.QInt8)"
```

## 5) Chạy real-time inference

```bash
python scripts/run_realtime.py ^
  --video path/to/video.mp4 ^
  --yolo-onnx runs/train/pothole_yolov8/weights/best.onnx ^
  --depth-onnx path/to/depth_anything_v2.onnx ^
  --output-dir outputs
```

Nếu chưa có depth ONNX, có thể bỏ `--depth-onnx` để chạy với fallback relative depth.

## 6) Evaluate depth/area

Ground truth CSV cần các cột:
- `frame_idx,x1,y1,x2,y2,gt_depth_m,gt_area_m2`

Chạy:

```bash
python scripts/evaluate.py --results-csv outputs/results.csv --gt-csv path/to/ground_truth.csv --output-dir outputs
```

## 7) Output files

Pipeline sinh:
- `outputs/results.csv`
- `outputs/metrics.json`
- `outputs/fps_log.txt`
- `outputs/depth_error_report.txt` (khi chạy evaluate)
- `outputs/area_error_report.txt` (khi chạy evaluate)

## 8) Stereo pipeline (tuỳ chọn)

Nếu bạn có cặp camera stereo đã calibrate, dùng:

```bash
python scripts/run_stereo.py --video-left left.mp4 --video-right right.mp4 --yolo-onnx runs/detect/runs/train/pothole_yolov8/weights/best.onnx --stereo-calib configs/stereo_calib_template.yaml --output-dir outputs_stereo
```

File calibration mẫu: `configs/stereo_calib_template.yaml`

## Ghi chú kỹ thuật

- Depth từ monocular là relative, đang scale về metric bằng heuristic theo `camera_height_m`.
- BEV/IPM dùng homography xấp xỉ; để giảm sai số nên calibrate camera nội/ngoại tại hiện trường.
