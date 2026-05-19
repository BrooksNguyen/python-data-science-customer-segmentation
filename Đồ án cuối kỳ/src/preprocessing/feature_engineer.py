import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from .base_processor import BaseProcessor


class FeatureEngineer(BaseProcessor):
    """Class chuyên tạo đặc trưng mới"""

    def __init__(self, data: Optional[pd.DataFrame] = None):
        super().__init__(data, name="FeatureEngineer")

    def process(self, datetime_column) -> 'FeatureEngineer':
        """
        Process data - có thể dùng để tạo features tự động

        Returns:
            FeatureEngineer
        """
        return self.create_datetime_features(datetime_column)

    def create_datetime_features(self, datetime_column: str) -> 'FeatureEngineer':
        """
        Tạo đặc trưng từ cột datetime

        Tham số:
        -----------
        datetime_column : str
            Cột chứa dữ liệu thời gian
        """
        if self.data is None:
            print("Chưa có dữ liệu")
            return self

        print("\n" + "=" * 60)
        print("TẠO ĐẶC TRƯNG THỜI GIAN")
        print("=" * 60)

        if datetime_column not in self.data.columns:
            print(f"Cột {datetime_column} không tồn tại")
            return self

        data_copy = self.data.copy()

        try:
            # Chuyển đổi sang datetime
            data_copy[datetime_column] = pd.to_datetime(data_copy[datetime_column])

            # Tạo các đặc trưng cơ bản
            data_copy[f'{datetime_column}_year'] = data_copy[datetime_column].dt.year
            data_copy[f'{datetime_column}_month'] = data_copy[datetime_column].dt.month
            data_copy[f'{datetime_column}_day'] = data_copy[datetime_column].dt.day
            data_copy[f'{datetime_column}_dayofweek'] = data_copy[datetime_column].dt.dayofweek
            # Tạo đặc trưng boolean
            data_copy[f'{datetime_column}_is_weekend'] = data_copy[datetime_column].dt.dayofweek.isin([5, 6]).astype(
                int)
            data_copy[f'{datetime_column}_is_month_start'] = data_copy[datetime_column].dt.is_month_start.astype(int)
            data_copy[f'{datetime_column}_is_month_end'] = data_copy[datetime_column].dt.is_month_end.astype(int)

            print(f"\n Đã tạo đặc trưng từ cột {datetime_column}:")
            new_features = [col for col in data_copy.columns if col.startswith(f'{datetime_column}_')]
            for feat in new_features:
                print(f"   • {feat}")

            # Ghi log
            self._log_action('create_datetime_features', {
                'datetime_column': datetime_column,
                'new_features': new_features
            })

        except Exception as e:
            print(f" Lỗi khi xử lý datetime: {str(e)}")

        self.data = data_copy
        return self
