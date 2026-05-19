import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from .base_processor import BaseProcessor


class FeatureScaler(BaseProcessor):
    """Class chuyên chuẩn hóa dữ liệu bằng scikit-learn"""

    def __init__(self, data: Optional[pd.DataFrame] = None):
        super().__init__(data, name="FeatureScaler")
        self.scaler_objects: Dict[str, Any] = {}  # Lưu đối tượng scaler
        self.scaling_params: Dict[str, Any] = {}

    def process(self, method: str = 'standard', columns: Optional[List] = None, **kwargs) -> 'FeatureScaler':
        """
        Chuẩn hóa dữ liệu

        Args:
            method: Phương pháp chuẩn hóa
            columns: Danh sách cột cần chuẩn hóa
            **kwargs: Tham số bổ sung cho scaler

        Returns:
            FeatureScaler
        """
        return self.scale_data(method, columns, **kwargs)

    def scale_data(self, method: str = 'standard', columns: Optional[List] = None, **kwargs) -> 'FeatureScaler':
        """
        Chuẩn hóa dữ liệu bằng scikit-learn scalers

        Args:
            method: Phương pháp chuẩn hóa
                'standard' - StandardScaler (z-score)
                'minmax' - MinMaxScaler (về [0, 1])
                'robust' - RobustScaler (chống outliers)
            columns: Danh sách cột cần chuẩn hóa (None = tất cả cột kiểu số)
            **kwargs: Tham số bổ sung cho scaler

        Returns:
            FeatureScaler
        """
        if self.data is None:
            raise ValueError("Không có dữ liệu để chuẩn hóa")

        print("\n" + "=" * 60)
        print(" CHUẨN HÓA DỮ LIỆU")
        print("=" * 60)

        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Kiểm tra columns có tồn tại không
            missing_cols = [col for col in columns if col not in self.data.columns]
            if missing_cols:
                print(f"  Các cột không tồn tại: {missing_cols}")
                columns = [col for col in columns if col in self.data.columns]

        if len(columns) == 0:
            print("  Không có cột số để chuẩn hóa")
            return self

        print(f"Chuẩn hóa {len(columns)} cột số bằng {method.upper()}...")

        data_copy = self.data.copy()
        scaling_info = {}

        # Chọn scaler dựa trên method
        if method == 'standard':
            scaler = StandardScaler(**kwargs)
        elif method == 'minmax':
            scaler = MinMaxScaler(**kwargs)
        elif method == 'robust':
            scaler = RobustScaler(**kwargs)
        else:
            raise ValueError(f"Phương pháp không được hỗ trợ: {method}")

        # Lấy dữ liệu số từ các cột được chọn
        numeric_data = data_copy[columns].values

        # Fit và transform dữ liệu
        scaled_data = scaler.fit_transform(numeric_data)

        # Cập nhật dữ liệu đã chuẩn hóa
        data_copy[columns] = scaled_data

        # Lưu thông tin scaling
        scaling_info = {
            'scaler_type': type(scaler).__name__,
            'method': method,
            'columns': columns,
            'scaler_params': scaler.get_params(),
            'scaler_object': scaler
        }

        # Lưu thống kê nếu có
        if hasattr(scaler, 'mean_'):
            scaling_info['means'] = scaler.mean_.tolist()
        if hasattr(scaler, 'scale_'):
            scaling_info['scales'] = scaler.scale_.tolist()
        if hasattr(scaler, 'data_min_'):
            scaling_info['data_min'] = scaler.data_min_.tolist()
            scaling_info['data_max'] = scaler.data_max_.tolist()

        self.data = data_copy
        self.scaler_objects[method] = scaler
        self.scaling_params = {'method': method, 'columns': columns, **kwargs}

        # In thông tin
        print(f" Đã chuẩn hóa bằng {method.upper()}")
        print(f"  Số cột: {len(columns)}")
        print(f"  Scaler: {type(scaler).__name__}")

        # In ví dụ về vài cột
        if len(columns) <= 10:
            print(f"  Cột đã xử lý: {', '.join(columns)}")
        else:
            print(f"  Cột đã xử lý: {', '.join(columns[:5])} ... và {len(columns) - 5} cột khác")

        self._log_action('scale_data', scaling_info)

        return self
