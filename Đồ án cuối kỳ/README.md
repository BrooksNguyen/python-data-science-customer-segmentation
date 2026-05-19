  # Đồ án cuối kỳ Python cho khoa học dữ liệu 
  # Phân cụm khách hàng 
  
## 1. Giới thiệu
- Bối cảnh: xây dựng pipeline tự động cho tiền xử lý và phân cụm dữ liệu khách hàng/marketing.
- Mục đích: chuẩn hóa dữ liệu, so sánh nhiều thuật toán phân cụm và xuất báo cáo/biểu đồ phục vụ phân tích hành vi khách hàng.
- Dữ liệu: file `data/marketing_campaign.csv` (đường dẫn, cột datetime, cột bỏ, tham số xử lý được cấu hình trong `config.yaml`).

## 2. Các tính năng chính
| Module | Tính năng | Mô tả |
|--------|-----------|-------|
| **Preprocessing** |  Phân tích EDA tự động | Tạo report thống kê, correlation, missing values |
| |  Xử lý missing values | Nhiều chiến lược tự động (mean/median/mode/auto) |
| |  Phát hiện outliers | IQR, Z-Score, Isolation Forest |
| |  Chuẩn hóa dữ liệu | Standard, MinMax, Robust scaling |
| |  Feature Engineering | Tạo features từ datetime, mã hóa categorical |
| **Clustering** |  Đa thuật toán | KMeans, Gaussian Mixture Model, Agglomerative |
| |  Tối ưu hóa thông minh | Bayesian Optimization với Optuna |
| |  So sánh metrics | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| |  Trực quan hóa | Heatmaps, radar charts, scatter matrices |
| |  Quản lý mô hình | Lưu/load pipeline, export predictions |

## 3. Cách cài đặt & sử dụng
1. Chuẩn bị môi trường
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   pip install -r requirements.txt
   ```
2. Cấu hình trong `config.yaml` (đường dẫn dữ liệu, cột thời gian, cột loại bỏ, chiến lược xử lý thiếu/ngoại lệ, thuật toán phân cụm và tham số).
3. Chạy pipeline chính
   ```bash
   python main.py
   ```
   - Kết quả tiền xử lý: `preprocessing_output/`
   - Báo cáo/biểu đồ & dự đoán: `clustering_results/`
   - Mô hình & pipeline đã lưu: `saved_models/`
4. Chạy thử nhanh với ví dụ
   ```bash
   python examples/example_preprocessing.py
   python examples/example_clustering.py
   ```

## 4. Cấu trúc thư mục dự án
```
.
├── config.yaml
├── main.py
├── requirements.txt
├── data/
│   └── marketing_campaign.csv
├── src/
│   ├── preprocessing/   # Xử lý dữ liệu đầu vào
│   │   ├── data_pipeline.py      # Pipeline tự động: EDA → missing → outlier → encode → scale → feature time
│   │   ├── data_loader.py        # Đọc dữ liệu
│   │   ├── data_analyzer.py      # EDA, thống kê, báo cáo
│   │   ├── missing_handler.py    # Chiến lược xử lý missing (auto/mean/median/mode)
│   │   ├── outlier_handler.py    # Phát hiện/điều trị ngoại lệ (IQR/Z-Score/Isolation Forest)
│   │   ├── categorical_encoder.py# Mã hóa categorical (auto/one-hot/label)
│   │   ├── feature_scaler.py     # Chuẩn hóa (standard/minmax/robust)
│   │   ├── feature_engineer.py   # Tạo đặc trưng thời gian
│   │   └── base_processor.py     # Lớp cơ sở cho processors
│   ├── clustering/      # Thuật toán, so sánh, tối ưu, trực quan, lưu mô hình
│   │   ├── trainer.py            # Orchestrator: chạy toàn bộ clustering pipeline
│   │   ├── model_comparator.py   # Grid search so sánh KMeans/GMM, tính metrics
│   │   ├── fine_tuner.py         # Bayesian Optimization (Optuna) cho thuật toán chọn
│   │   ├── visualizer.py         # Vẽ scatter/pca/heatmap/radar, lưu hình
│   │   ├── model_manager.py      # Lưu/đọc pipeline, export predictions & model card
│   │   ├── data_preprocessor.py  # Chuẩn bị dữ liệu numeric, PCA, lưu báo cáo PCA
│   │   ├── base.py               # Lớp cơ sở chung
│   │   └── __init__.py
│   └── utils/           # Tiện ích chung
│       ├── config_loader.py      # Đọc config YAML, truy cập tham số an toàn
│       ├── logger.py             # Cấu hình logger màu, ghi log
│       └── __init__.py
├── examples/
│   ├── example_preprocessing.py
│   └── example_clustering.py
├── preprocessing_output/  # Dữ liệu & log sau tiền xử lý
├── clustering_results/    # Báo cáo, biểu đồ, dự đoán
├── saved_models/          # Mô hình/pipeline đã huấn luyện
├── logs/                  # Log chi tiết từng bước
└── README.md
```

## 5. Tác giả 
- [HOÀNG NGỌC DUY] — MSSV: [23280050]
- [NGUYỄN THÁI HOÀNG] — MSSV: [23280059]
- [NGUYỄN PHÚC BÁCH] — MSSV: [23280039]   