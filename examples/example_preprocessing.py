# examples/example_preprocessing.py
"""
Ví dụ sử dụng Data Preprocessing Pipeline
"""
import sys
import os
import pandas as pd
import numpy as np

# Thêm đường dẫn gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.preprocessing import DataLoader, DataPreprocessorPipeline
    print(" Import thành công!")
except ImportError as e:
    print(f" Lỗi import: {e}")
    sys.exit(1)

def create_sample_data():
    """Tạo dữ liệu mẫu"""
    np.random.seed(42)
    n_samples = 500
    
    data = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'gender': np.random.choice(['Male', 'Female', None], n_samples, p=[0.48, 0.48, 0.04]),
        'city': np.random.choice(['Hanoi', 'HCM', 'Danang', 'Hue', 'Cantho', None], 
                                n_samples, p=[0.3, 0.3, 0.2, 0.1, 0.09, 0.01]),
        'purchase_amount': np.random.randn(n_samples) * 1000 + 5000,
        'last_purchase_date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'satisfaction_score': np.random.randint(1, 11, n_samples)
    })
    
    # Thêm missing values
    data.loc[10:20, 'age'] = np.nan
    data.loc[30:40, 'income'] = np.nan
    data.loc[50:60, 'purchase_amount'] = np.nan
    
    # Thêm outliers
    data.loc[100:105, 'income'] = 5000000  # Outliers
    data.loc[110:115, 'purchase_amount'] = -20000  # Outliers âm
    
    return data

def main():
    print("="*60)
    print("VÍ DỤ TIỀN XỬ LÝ DỮ LIỆU")
    print("="*60)
    
    # 1. Tạo dữ liệu mẫu
    print("\n1. Tạo dữ liệu mẫu...")
    data = create_sample_data()
    print(f"   Kích thước: {data.shape}")
    print(f"   Columns: {list(data.columns)}")
    
    # Lưu dữ liệu thô
    data.to_csv('sample_data.csv', index=False)
    
    # 2. Sử dụng DataLoader
    print("\n2. Sử dụng DataLoader...")
    loader = DataLoader.from_file('sample_data.csv')
    print(f"   Đã đọc: {len(loader.data)} dòng")
    
    # 3. Tạo và chạy pipeline
    print("\n3. Tạo DataPreprocessorPipeline...")
    pipeline = DataPreprocessorPipeline(loader.data)
    
    print("\n4. Chạy xử lý tự động...")
    pipeline.auto_process(config={
        'missing_strategy': 'auto',
        'outlier_method': 'iqr',
        'outlier_treatment': 'cap',
        'encode_method': 'auto',
        'scale_method': 'standard',
        'datetime_features': True
    })
    
    # 4. Lấy kết quả
    print("\n5. Lấy dữ liệu đã xử lý...")
    processed_data = pipeline.get_data()
    print(f"   Kích thước sau xử lý: {processed_data.shape}")
    
    # 5. Hiển thị tổng kết
    print("\n6. Tổng kết quá trình xử lý:")
    pipeline.show_summary()
    
    # 6. Xuất kết quả
    print("\n7. Xuất dữ liệu đã xử lý...")
    pipeline.to_csv('processed_data.csv')
    
    # Dọn dẹp file tạm
    if os.path.exists('sample_data.csv'):
        os.remove('sample_data.csv')
    
    print("\n" + "="*60)
    print("HOÀN THÀNH VÍ DỤ TIỀN XỬ LÝ")
    print("="*60)

if __name__ == "__main__":
    main()