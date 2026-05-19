import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
from .base_processor import BaseProcessor


class DataLoader(BaseProcessor):
    """Class chuyên đọc và xuất dữ liệu"""

    def __init__(self, data: Optional[pd.DataFrame] = None):
        super().__init__(data, name="DataLoader")

    @classmethod
    def from_file(cls, file_path: str, **kwargs) -> 'DataLoader':
        """
        Tạo DataLoader từ file

        Args:
            file_path: Đường dẫn file
            **kwargs: Tham số cho pandas read functions

        Returns:
            DataLoader với dữ liệu đã đọc
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File không tồn tại: {file_path}")

        try:
            # Xác định định dạng file
            suffix = file_path.suffix.lower()

            if suffix == '.csv':
                data = pd.read_csv(file_path, **kwargs)
            elif suffix in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path, **kwargs)
            elif suffix == '.json':
                data = pd.read_json(file_path, **kwargs)
            else:
                raise ValueError(f"Định dạng file không được hỗ trợ: {suffix}")

            loader = cls(data)
            loader._log_action('load_file', {
                'file_path': str(file_path),
                'format': suffix,
                'rows': len(data),
                'columns': len(data.columns),
                'kwargs': kwargs
            })

            print(f"   Đọc file thành công: {file_path.name}")
            print(f"   Kích thước: {len(data):,} dòng × {len(data.columns):,} cột")

            return loader

        except Exception as e:
            print(f"   Lỗi khi đọc file: {str(e)}")
            raise

    def to_file(self, file_path: str, format: str = None, **kwargs) -> 'DataLoader':
        """
        Xuất dữ liệu ra file

        Args:
            file_path: Đường dẫn file
            format: Định dạng file
            **kwargs: Tham số cho pandas write functions

        Returns:
            DataLoader
        """
        if self.data is None:
            raise ValueError("Không có dữ liệu để xuất")

        file_path = Path(file_path)

        # Xác định định dạng nếu không được cung cấp
        if format is None:
            format = file_path.suffix.lower().replace('.', '')

        try:
            if format == 'csv':
                self.data.to_csv(file_path, index=False, **kwargs)
            elif format in ['xlsx', 'excel']:
                self.data.to_excel(file_path, index=False, **kwargs)
            elif format == 'json':
                self.data.to_json(file_path, orient='records', **kwargs)
            else:
                raise ValueError(f"Định dạng không được hỗ trợ: {format}")

            self._log_action('export_file', {
                'file_path': str(file_path),
                'format': format,
                'shape': self.data.shape,
                'kwargs': kwargs
            })

            print(f"   Xuất dữ liệu thành công: {file_path}")
            print(f"   Kích thước: {self.data.shape[0]:,} dòng × {self.data.shape[1]:,} cột")

        except Exception as e:
            print(f"   Lỗi khi xuất file: {str(e)}")
            raise

        return self

    def process(self) -> pd.DataFrame:
        pass