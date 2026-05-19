import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from .base_processor import BaseProcessor


class MissingValueHandler(BaseProcessor):
    """Class chuyên xử lý missing values"""

    def __init__(self, data: Optional[pd.DataFrame] = None):
        super().__init__(data, name="MissingValueHandler")
        self.missing_strategies: Dict[str, Dict] = {}
        self.fill_values: Dict[str, Any] = {}

    def process(self, strategy: str = 'auto', **kwargs) -> 'MissingValueHandler':
        """
        Xử lý missing values

        Args:
            strategy: Phương pháp xử lý
                'auto' - Tự động chọn phương pháp tốt nhất
                'mean', 'median', 'mode', 'ffill', 'bfill'
            **kwargs: Tham số bổ sung

        Returns:
            MissingValueHandler
        """
        return self.handle_missing(strategy, **kwargs)

    def handle_missing(self, strategy: str = 'auto', **kwargs) -> 'MissingValueHandler':
        """
        Xử lý missing values với nhiều phương pháp

        Args:
            strategy: Phương pháp xử lý
            **kwargs: Tham số bổ sung
                - threshold: Ngưỡng xóa cột (default: 0.8)

        Returns:
            MissingValueHandler
        """
        if self.data is None:
            raise ValueError("Không có dữ liệu để xử lý")

        print("\n" + "=" * 60)
        print(" XỬ LÝ MISSING VALUES")
        print("=" * 60)

        data_copy = self.data.copy()
        missing_before = data_copy.isnull().sum().sum()

        if strategy == 'auto':
            print("   Chế độ AUTO: Tự động chọn phương pháp...")
            data_copy = self._handle_missing_auto(data_copy, **kwargs)
        else:
            print(f"  Dùng phương pháp: {strategy}")
            data_copy = self._handle_missing_manual(data_copy, strategy, **kwargs)

        missing_after = data_copy.isnull().sum().sum()
        self.data = data_copy

        print(f"\n   KẾT QUẢ:")
        print(f"   Missing trước: {missing_before:,}")
        print(f"   Missing sau: {missing_after:,}")
        print(f"   Đã xử lý: {missing_before - missing_after:,} giá trị")

        self._log_action('handle_missing', {
            'strategy': strategy,
            'missing_before': missing_before,
            'missing_after': missing_after,
            'handled_count': missing_before - missing_after,
            'strategies_used': self.missing_strategies
        })

        return self

    def _handle_missing_auto(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Đưa ra phương pháp tự động xử lí missing_value"""
        threshold = kwargs.get('threshold', 0.8)  # Ngưỡng xóa cột (80%)

        for col in data.columns:
            missing_count = data[col].isnull().sum()
            if missing_count == 0:
                continue

            missing_percent = (missing_count / len(data)) * 100
            col_type = str(data[col].dtype)

            # Lưu strategy đã dùng
            self.missing_strategies[col] = {
                'missing_percent': missing_percent,
                'dtype': col_type,
                'missing_count': missing_count
            }

            # QUY TẮC AUTO:
            # 1. Nếu missing > threshold% -> XÓA CỘT
            if missing_percent > (threshold * 100):
                print(f"       {col}: {missing_percent:.1f}% missing -> XÓA CỘT")
                data = data.drop(columns=[col])
                self.missing_strategies[col]['strategy'] = 'drop_column'
                continue

            # 2. Nếu missing < 1% -> XÓA DÒNG
            elif missing_percent < 1:
                print(f"       {col}: {missing_percent:.1f}% missing -> XÓA DÒNG")
                data = data.dropna(subset=[col])
                self.missing_strategies[col]['strategy'] = 'drop_rows'
                continue

            # 3. Dựa vào kiểu dữ liệu
            if 'int' in col_type or 'float' in col_type:
                # Dữ liệu số -> Dùng MEDIAN (robust với outliers)
                fill_val = data[col].median()
                data[col] = data[col].fillna(fill_val)
                self.missing_strategies[col]['strategy'] = 'median'
                self.missing_strategies[col]['fill_value'] = fill_val
                self.fill_values[col] = fill_val
                print(f"      {col}: Dùng median = {fill_val:.2f}")

            elif 'datetime' in col_type:
                # Dữ liệu thời gian -> Dùng forward fill
                data[col] = data[col].ffill()  # forward fill
                data[col] = data[col].bfill()  # backward fill
                self.missing_strategies[col]['strategy'] = 'ffill'
                print(f"      {col}: Dùng forward/backward fill")

            elif 'bool' in col_type:
                # Boolean -> Dùng mode
                mode_val = data[col].mode()[0] if not data[col].mode().empty else False
                data[col] = data[col].fillna(mode_val)
                self.missing_strategies[col]['strategy'] = 'mode'
                self.missing_strategies[col]['fill_value'] = mode_val
                self.fill_values[col] = mode_val
                print(f"    {col}: Dùng mode = {mode_val}")

            else:
                # Dữ liệu phân loại -> Dùng mode
                mode_val = data[col].mode()[0] if not data[col].mode().empty else "MISSING"
                data[col].fillna(mode_val, inplace=True)
                self.missing_strategies[col]['strategy'] = 'mode'
                self.missing_strategies[col]['fill_value'] = mode_val
                self.fill_values[col] = mode_val
                print(f"      {col}: Dùng mode = '{mode_val}'")

        return data

    def _handle_missing_manual(self, data: pd.DataFrame, strategy: str, **kwargs) -> pd.DataFrame:
        """Xử lý missing thủ công"""
        for col in data.columns:
            if data[col].isnull().any():
                fill_val = None

                if strategy == 'mean' and pd.api.types.is_numeric_dtype(data[col]):
                    fill_val = data[col].mean()
                    data[col] = data[col].fillna(fill_val)
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(data[col]):
                    fill_val = data[col].median()
                    data[col] = data[col].fillna(fill_val)
                elif strategy == 'mode':
                    mode_val = data[col].mode()[0] if not data[col].mode().empty else "MISSING"
                    fill_val = mode_val
                    data[col] = data[col].fillna(fill_val)
                elif strategy == 'ffill':
                    data[col] = data[col].fillna(method='ffill')
                elif strategy == 'bfill':
                    data[col] = data[col].fillna(method='bfill')
                elif strategy == 'zero':
                    fill_val = 0
                    data[col] = data[col].fillna(0)
                elif strategy == 'constant':
                    constant_val = kwargs.get('constant_value', 0)
                    fill_val = constant_val
                    data[col].fillna(constant_val, inplace=True)

                self.missing_strategies[col] = {
                    'strategy': strategy,
                    'fill_value': fill_val
                }
                if fill_val is not None:
                    self.fill_values[col] = fill_val

        return data


    def get_missing_report(self) -> Dict:
        """
        Tạo báo cáo về missing values

        Returns:
            Dict chứa báo cáo
        """
        if self.data is None:
            return {}

        missing_summary = {
            'total_missing_before': sum(info.get('missing_count', 0)
                                        for info in self.missing_strategies.values()),
            'strategies_used': self.missing_strategies,
            'fill_values': self.fill_values,
            'current_missing': self.data.isnull().sum().sum(),
            'missing_by_column': self.data.isnull().sum().to_dict()
        }

        return missing_summary
