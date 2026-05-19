import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from .base_processor import BaseProcessor


class CategoricalEncoder(BaseProcessor):
    """Class chuyên mã hóa biến phân loại"""

    def __init__(self, data: Optional[pd.DataFrame] = None):
        super().__init__(data, name="CategoricalEncoder")

    def process(self, columns: Optional[List] = None,
                method: str = 'auto',
                threshold: int = 10) -> 'CategoricalEncoder':
        return self.encode_categorical(columns, method, threshold)

    def encode_categorical(self, columns: Optional[List] = None,
                           method: str = 'auto',
                           threshold: int = 10) -> 'CategoricalEncoder':
        """
        Mã hóa biến phân loại

        Args:
            columns: Danh sách cột cần mã hóa (None = tất cả cột categorical)
            method: Phương pháp mã hóa
                'auto' - Tự động chọn OneHot hoặc Label
                'onehot' - Chỉ dùng OneHot Encoding
                'label' - Chỉ dùng Label Encoding
                'frequency' - Mã hóa bằng tần suất (tự định nghĩa)
            threshold: Ngưỡng chọn phương pháp (chỉ dùng khi method='auto')

        Returns:
            CategoricalEncoder
        """
        if self.data is None:
            print("Chưa có dữ liệu")
            return self

        print("\n" + "=" * 60)
        print("MÃ HÓA BIẾN PHÂN LOẠI")
        print("=" * 60)

        # Tìm cột phân loại
        if columns is None:
            cat_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            cat_cols = [col for col in columns if col in self.data.columns]

        if len(cat_cols) == 0:
            print("Không có cột phân loại")
            return self

        print(f"Phát hiện {len(cat_cols)} cột phân loại:")

        data_copy = self.data.copy()
        cols_before = len(data_copy.columns)

        if method == 'auto':
            print("Chế độ AUTO: Tự động chọn phương pháp...")
            data_copy = self._encode_auto(data_copy, cat_cols, threshold)
        elif method == 'onehot':
            print("Dùng OneHot Encoding...")
            data_copy = self._encode_onehot(data_copy, cat_cols)
        elif method == 'label':
            print("Dùng Label Encoding...")
            data_copy = self._encode_label(data_copy, cat_cols)
        elif method == 'frequency':
            print("Dùng Frequency Encoding...")
            data_copy = self._encode_frequency(data_copy, cat_cols)
        else:
            print(f"Phương pháp không được hỗ trợ: {method}")
            return self

        cols_after = len(data_copy.columns)
        self.data = data_copy

        print(f"\nHOÀN THÀNH")
        print(f"   Số cột trước: {cols_before}")
        print(f"   Số cột sau: {cols_after}")
        print(f"   Tăng: {cols_after - cols_before} cột")

        self._log_action('encode_categorical', {
            'method': method,
            'columns_processed': len(cat_cols),
            'cols_before': cols_before,
            'cols_after': cols_after
        })

        return self

    def _encode_auto(self, data: pd.DataFrame, columns: List, threshold: int) -> pd.DataFrame:
        """Mã hóa tự động theo ngưỡng"""
        for col in columns:
            unique_count = data[col].nunique()

            # QUY TẮC AUTO: ≤ threshold -> OneHot, > threshold -> Frequency
            if unique_count <= threshold:
                # OneHot Encoding
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=False)
                data = data.drop(columns=[col])
                data = pd.concat([data, dummies], axis=1)
                print(f"   {col:15}: {unique_count:3} values -> ONEHOT ({len(dummies.columns)} cột mới)")
            else:
                # Label Encoding
                unique_vals = data[col].dropna().unique()
                mapping = {val: i for i, val in enumerate(unique_vals)}
                data[col] = data[col].map(mapping)
                print(f"   {col:15}: {unique_count:3} values -> LABEL")

        return data

    def _encode_onehot(self, data: pd.DataFrame, columns: List) -> pd.DataFrame:
        """OneHot Encoding cho tất cả cột"""
        for col in columns:
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=False)
            data = data.drop(columns=[col])
            data = pd.concat([data, dummies], axis=1)
            print(f"   {col}: {len(dummies.columns)} cột mới")

        return data

    def _encode_label(self, data: pd.DataFrame, columns: List) -> pd.DataFrame:
        """Label Encoding cho tất cả cột"""
        for col in columns:
            unique_vals = data[col].dropna().unique()
            mapping = {val: i for i, val in enumerate(unique_vals)}
            data[col] = data[col].map(mapping)
            print(f"   {col}: -> mã số 0-{len(unique_vals) - 1}")

        return data

    def _encode_frequency(self, data: pd.DataFrame, columns: List) -> pd.DataFrame:
        """Frequency Encoding cho tất cả cột"""
        for col in columns:
            freq = data[col].value_counts(normalize=True)
            data[col] = data[col].map(freq)
            print(f"   {col}: mã hóa bằng tần suất")

        return data