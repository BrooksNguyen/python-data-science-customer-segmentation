# Customer Segmentation Pipeline 🎯
*Automated Machine Learning Pipeline for Customer Behavior Analysis*

Đồ án môn học **Python cho Khoa học Dữ liệu** - Khoa Khoa học Dữ liệu, Trường Đại học Khoa học Tự nhiên (HCMUS).
**Sinh viên thực hiện:** Nguyễn Phúc Bách (MSSV: 23280039)

## 1. Giới thiệu (Overview)
Dự án xây dựng một hệ thống (pipeline) tự động hóa hoàn toàn quy trình xử lý và phân cụm dữ liệu khách hàng. Mục tiêu là phân tích hành vi mua sắm, từ đó giúp các chiến dịch marketing mục tiêu (targeted marketing) đạt hiệu quả cao hơn.

**Dataset:** `marketing_campaign.csv` (2,240 bản ghi chứa thông tin nhân khẩu học và lịch sử mua sắm).

## 2. Kiến trúc Hệ thống (Architecture)
Hệ thống được thiết kế theo hướng hướng đối tượng (OOP) và module hóa cao, chia thành 3 phần chính:

| Module | Tính năng nổi bật (Features) |
|--------|------------------------------|
| **Preprocessing** | Tự động EDA, xử lý Missing Values (auto/mean/median/mode), loại bỏ Outliers (IQR/Z-Score/Isolation Forest), Encoding và Feature Engineering (Datetime features). |
| **Clustering** | Hỗ trợ KMeans, GMM, Agglomerative. Tích hợp **Bayesian Optimization (Optuna)** để tìm kiếm siêu tham số tối ưu thay vì Grid Search thông thường. |
| **Evaluation & Viz** | Giảm chiều dữ liệu bằng PCA (giữ 95% phương sai), đánh giá qua Silhouette, Davies-Bouldin. Tự động xuất biểu đồ Scatter, Heatmap, Radar chart. |

## 3. Cài đặt & Sử dụng (Installation & Usage)
Cấu hình toàn bộ pipeline thông qua file `config.yaml` mà không cần can thiệp vào source code.

```bash
# 1. Clone repository
git clone [https://github.com/your-username/customer-segmentation-python.git](https://github.com/your-username/customer-segmentation-python.git)
cd customer-segmentation-python

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Chạy Pipeline toàn trình
python main.py
