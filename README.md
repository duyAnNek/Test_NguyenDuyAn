# EV Computer Vision & Visual Localization

Monorepo gồm hai nhánh chính: **(A) perception ổ gà (pothole)** chạy Python + ONNX trên CPU/GPU tùy cấu hình, và **(B) ROS 2 `gps_visual`** — giám sát toàn vẹn GPS, visual odometry, làn đường, cơ sở landmark và fusion EKF. Báo cáo kỹ thuật chi tiết: [`AI_ENGINEER_TECHNICAL_REPORT.tex`](AI_ENGINEER_TECHNICAL_REPORT.tex) (biên dịch bằng XeLaTeX / LuaLaTeX).

---

## Cấu trúc thư mục

| Thư mục / tệp | Mô tả |
|----------------|--------|
| [`pothole/`](pothole/) | Pipeline phát hiện ổ gà, depth (ONNX hoặc fallback), BEV/diện tích, severity; script train/eval/export/realtime. |
| [`gps_system/`](gps_system/) | Workspace ROS 2; package [`src/gps_visual`](gps_system/src/gps_visual) (nodes, launch, config). |
| [`AI_ENGINEER_TECHNICAL_REPORT.tex`](AI_ENGINEER_TECHNICAL_REPORT.tex) | Báo cáo kỹ thuật (pipeline, failure analysis, hướng tối ưu). |

---

## Yêu cầu môi trường

### Phần A — `pothole`

- **Python** 3.10+ (khuyến nghị; kiểm tra tương thích với PyTorch bạn cài).
- **Phụ thuộc:** xem [`pothole/requirements.txt`](pothole/requirements.txt) (`ultralytics`, `torch`, `onnxruntime`, `opencv-python`, `numpy`, `pyyaml`, …).
- **GPU:** tùy chọn (CUDA) cho train/torch; pipeline realtime có thể chạy CPU với ONNX.

### Phần B — ROS 2 `gps_visual`

- **ROS 2** (ví dụ Humble trên Linux; trên Windows có thể gặp hạn chế symlink khi `colcon build` — xem báo cáo, cân nhắc WSL2).
- Python packages: `rclpy`, `numpy`, `opencv`, `onnxruntime`, `cv_bridge`, … (khai báo trong [`package.xml`](gps_system/src/gps_visual/package.xml)).

---

## Phần A: Pothole perception

### Cài đặt

```bash
cd pothole
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Cấu hình

- [`pothole/configs/camera_config.yaml`](pothole/configs/camera_config.yaml) — `camera_height_m`, `meters_per_pixel_bev`, homography (template).
- [`pothole/configs/severity_rules.yaml`](pothole/configs/severity_rules.yaml) — ngưỡng **severity** (rule-based) theo `depth_m` và `area_m2`.
- [`pothole/data/dataset.yaml`](pothole/data/dataset.yaml) — dataset YOLO (train/val/test), 3 lớp ví dụ: `minor`, `moderate`, `severe`.

### Export ONNX (detector)

Từ trọng số `.pt` (Ultralytics):

```bash
cd pothole
python scripts/export_onnx.py --weights path/to/best.pt --imgsz 640
```

### Depth ONNX (tùy chọn)

Có script export depth trong [`pothole/scripts/export_depth_onnx.py`](pothole/scripts/export_depth_onnx.py) — dùng khi bạn đã có quy trình export phù hợp model monocular depth của repo.

### Chạy realtime trên video

```bash
cd pothole
python scripts/run_realtime.py \
  --video path/to/video.mp4 \
  --yolo-onnx outputs/weights/best.onnx \
  --output-dir outputs \
  --conf-thres 0.25 \
  --depth-interval 5 \
  --depth-input-size 518 \
  --metrics-json test_mAP/detection_metrics.json \
  --save-video
```

Trên **PowerShell**, có thể ghi **một dòng** hoặc dùng dấu `` ` `` cuối dòng thay cho `\`.

Tham số chính:

| Tham số | Ý nghĩa |
|---------|---------|
| `--yolo-onnx` | Đường dẫn file ONNX detector YOLOv8. |
| `--depth-onnx` | (Tùy chọn) ONNX depth; nếu thiếu hoặc lỗi file → fallback depth nhẹ. |
| `--depth-interval` | Chỉ tính depth mỗi N frame, tái sử dụng map ở giữa. |
| `--depth-input-size` | Kích thước input depth (nhỏ hơn thường nhanh hơn). |
| `--metrics-json` | JSON chứa `mAP@0.5` để hiển thị tham chiếu trên overlay (nếu có). |
| `--save-video` | Lưu video overlay; không đối số → `<output-dir>/<tên>_detected.mp4`. |
| `--no-visualize` | Không hiện cửa sổ (encode nhanh hơn khi chỉ `--save-video`). |

### Script khác (tham khảo)

| Script | Mục đích |
|--------|-----------|
| [`scripts/eval_detection.py`](pothole/scripts/eval_detection.py) | Đánh giá detection / metrics JSON. |
| [`scripts/evaluate_test_estimation.py`](pothole/scripts/evaluate_test_estimation.py) | So GT CSV với ước lượng depth/area. |
| [`scripts/evaluate.py`](pothole/scripts/evaluate.py) | Đánh giá tổng quát (theo repo). |
| [`scripts/run_onnx_detect.py`](pothole/scripts/run_onnx_detect.py) | Chạy ONNX detect đơn giản. |
| [`scripts/run_ultralytics_onnx.py`](pothole/scripts/run_ultralytics_onnx.py) | Inference qua Ultralytics + ONNX. |
| [`scripts/analyze_depth_area.py`](pothole/scripts/analyze_depth_area.py) | Phân tích depth/diện tích. |
| [`scripts/analyze_test_set.py`](pothole/scripts/analyze_test_set.py) | Phân tích tập test. |
| [`scripts/smoke_onnx_runtime.py`](pothole/scripts/smoke_onnx_runtime.py) | Kiểm tra nhanh ONNX Runtime. |

Module lõi: [`pothole/src/pipeline.py`](pothole/src/pipeline.py), [`geometry.py`](pothole/src/geometry.py), [`severity.py`](pothole/src/severity.py), [`config.py`](pothole/src/config.py).

---

## Phần B: ROS 2 `gps_visual`

### Build

```bash
cd gps_system
colcon build --packages-select gps_visual
# Linux:
source install/setup.bash
# Windows (cmd):
call install\setup.bat
```

### Launch hệ thống

```bash
ros2 launch gps_visual system_b.launch.py
```

Launch khởi chạy các node: `landmark_db_node`, `visual_odometry_node`, `lane_detection_node`, `gps_monitor_node`, `sensor_fusion_node`. Tham số: [`gps_system/src/gps_visual/config/params.yaml`](gps_system/src/gps_visual/config/params.yaml).

### Kiểm thử

- [`gps_system/src/gps_visual/test/test_part_b_scenarios.py`](gps_system/src/gps_visual/test/test_part_b_scenarios.py) — kịch bản pytest (chạy trong môi trường ROS/colcon test nếu đã cấu hình).

---

## Gợi ý roadmap ngắn

- Quantize INT8 depth/detector; tách thread depth; hiệu chuẩn camera ngoại tại chặt chẽ hơn template.
- CI build ROS trên Linux; phát triển perception trên Windows/WSL theo nhu cầu.
- Geo-tag cảnh báo pothole với pose sau EKF (tích hợp hai nhánh).

---

*Nếu đường dẫn hoặc tên script thay đổi trong repo, cập nhật bảng và lệnh mẫu cho khớp.*
