import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from .base_processor import BaseProcessor


class OutlierHandler(BaseProcessor):
    """Class chuyên phát hiện và xử lý outliers"""

    def __init__(self, data: Optional[pd.DataFrame] = None):
        super().__init__(data, name="OutlierHandler")
        self.outliers_info: Dict = {}
        self.detection_params: Dict = {}

    def process(self, columns: Optional[List] = None, method: str = 'auto', treatment: str = 'auto', **kwargs) -> 'OutlierHandler':
        """
        Phát hiện và xử lý outliers

        Args:
            method: Phương pháp phát hiện
            treatment: Phương pháp xử lý
            **kwargs: Tham số bổ sung

        Returns:
            OutlierHandler
        """
        return self.handle_outliers(columns, method, treatment, **kwargs)

    def _detect_outliers(self, columns: Optional[List] = None, method: str = 'iqr', **kwargs) -> Dict:
        """
        Phát hiện outliers với nhiều phương pháp

        Args:
            columns: Danh sách cột cần kiểm tra
            method: Phương pháp phát hiện
                'iqr' - Interquartile Range
                'zscore' - Z-Score
                'isolation_forest' - Isolation Forest
            **kwargs: Tham số bổ sung
                - threshold: Ngưỡng cho IQR/Z-Score
                - contamination: Tỷ lệ outliers cho Isolation Forest
                - n_estimators: Số cây cho Isolation Forest

        Returns:
            Dict chứa thông tin outliers
        """
        if self.data is None:
            raise ValueError("Không có dữ liệu để phân tích")

        print("\n" + "=" * 60)
        print("   PHÁT HIỆN OUTLIERS")
        print("=" * 60)

        # Xác định cột số
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        if len(columns) == 0:
            print("    Không có cột số để kiểm tra")
            return {}

        outliers_info = {}

        print(f"\n   Kiểm tra {len(columns)} cột số bằng {method.upper()}...")

        for col in columns:
            if col not in self.data.columns:
                continue

            series = self.data[col].dropna()

            # Bỏ qua nếu ít dữ liệu
            if len(series) < 10:
                print(f"       Bỏ qua {col}: Quá ít dữ liệu ({len(series)} samples)")
                continue

            try:
                if method == 'iqr':
                    info = self._detect_outliers_iqr(series, col, **kwargs)
                elif method == 'zscore':
                    info = self._detect_outliers_zscore(series, col, **kwargs)
                elif method == 'isolation_forest':
                    info = self._detect_outliers_isolation_forest(series, col, **kwargs)
                else:
                    print(f"      Phương pháp không được hỗ trợ: {method}")
                    continue

                if info['outlier_count'] > 0:
                    outliers_info[col] = info
                    print(f"      {col:20}: {info['outlier_count']} outliers ({info['outlier_percent']:.1f}%)")

            except Exception as e:
                print(f"      Lỗi khi xử lý cột {col}: {str(e)}")
                continue

        # Hiển thị tổng kết
        self._display_outliers_summary(outliers_info, method)
        self.outliers_info = outliers_info
        self.detection_params = {'method': method, **kwargs}

        return outliers_info

    def _detect_outliers_iqr(self, series: pd.Series, col_name: str, **kwargs) -> Dict:
        """Phát hiện outliers bằng phương pháp IQR"""
        threshold = kwargs.get('threshold', 1.5)

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = series[(series < lower_bound) | (series > upper_bound)]

        return {
            'method': 'iqr',
            'column': col_name,
            'Q1': float(Q1),
            'Q3': float(Q3),
            'IQR': float(IQR),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'threshold': threshold,
            'outlier_count': int(len(outliers)),
            'outlier_percent': float((len(outliers) / len(series)) * 100),
            'outlier_indices': outliers.index.tolist(),
            'outlier_values': outliers.tolist()
        }

    def _detect_outliers_zscore(self, series: pd.Series, col_name: str, **kwargs) -> Dict:
        """Phát hiện outliers bằng phương pháp Z-Score"""
        threshold = kwargs.get('threshold', 3.0)

        mean = series.mean()
        std = series.std()

        if std == 0:
            return {
                'method': 'zscore',
                'column': col_name,
                'mean': float(mean),
                'std': float(std),
                'outlier_count': 0,
                'outlier_percent': 0.0,
                'note': 'Standard deviation is zero'
            }

        z_scores = np.abs((series - mean) / std)
        outliers = series[z_scores > threshold]

        return {
            'method': 'zscore',
            'column': col_name,
            'mean': float(mean),
            'std': float(std),
            'threshold': threshold,
            'outlier_count': int(len(outliers)),
            'outlier_percent': float((len(outliers) / len(series)) * 100),
            'outlier_indices': outliers.index.tolist(),
            'outlier_values': outliers.tolist(),
            'max_z_score': float(z_scores.max())
        }

    def _detect_outliers_isolation_forest(self, series: pd.Series, col_name: str, **kwargs) -> Dict:
        """Phát hiện outliers bằng Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest

            # Tham số mặc định
            contamination = kwargs.get('contamination', 'auto')
            n_estimators = kwargs.get('n_estimators', 100)
            random_state = kwargs.get('random_state', 42)
            max_samples = kwargs.get('max_samples', 'auto')

            # Chuẩn bị dữ liệu cho Isolation Forest
            X = series.values.reshape(-1, 1)

            # Khởi tạo và fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                max_samples=max_samples,
                random_state=random_state,
                n_jobs=-1
            )

            # Fit và predict
            outlier_labels = iso_forest.fit_predict(X)

            # Lấy các điểm outliers
            outlier_mask = outlier_labels == -1
            outliers = series[outlier_mask]

            # Tính anomaly scores
            anomaly_scores = iso_forest.score_samples(X)

            return {
                'method': 'isolation_forest',
                'column': col_name,
                'contamination': contamination,
                'n_estimators': n_estimators,
                'random_state': random_state,
                'max_samples': max_samples,
                'outlier_count': int(len(outliers)),
                'outlier_percent': float((len(outliers) / len(series)) * 100),
                'outlier_indices': outliers.index.tolist(),
                'outlier_values': outliers.tolist(),
                'anomaly_scores': {
                    'min': float(anomaly_scores.min()),
                    'max': float(anomaly_scores.max()),
                    'mean': float(anomaly_scores.mean()),
                    'std': float(anomaly_scores.std())
                }
            }

        except ImportError:
            print(f"     Không thể import Isolation Forest")
            print(f"   Chuyển sang dùng IQR method")
            return self._detect_outliers_iqr(series, col_name, **kwargs)

    def _display_outliers_summary(self, outliers_info: Dict, method: str):
        """Hiển thị tổng kết outliers"""
        if not outliers_info:
            print("\n   Không phát hiện outliers")
            return

        print("\n" + "-" * 60)
        print("   TỔNG KẾT OUTLIERS")
        print("-" * 60)

        total_outliers = sum(info['outlier_count'] for info in outliers_info.values())
        total_rows = len(self.data)

        print(f"Phương pháp: {method.upper()}")
        print(f"Tổng outliers phát hiện: {total_outliers:,}")
        print(f"Số cột có outliers: {len(outliers_info)}")
        print(f"Tỷ lệ outliers: {total_outliers / (total_rows * len(outliers_info)) * 100:.2f}%")

        # Hiển thị chi tiết từng cột
        print(f"\n   CHI TIẾT CÁC CỘT CÓ OUTLIERS:")
        print("-" * 60)
        print(f"{'Cột':<20} {'Outliers':<10} {'Tỷ lệ':<10} {'Phương pháp':<15}")
        print("-" * 60)

        for col, info in sorted(outliers_info.items(),
                                key=lambda x: x[1]['outlier_percent'],
                                reverse=True):
            print(f"{col:<20} {info['outlier_count']:<10} {info['outlier_percent']:<10.1f}% {info['method']:<15}")

        print("-" * 60)

    def handle_outliers(self, columns: Optional[List] = None, method: str = 'auto', treatment: str = 'auto', **kwargs) -> 'OutlierHandler':
        """
        Phát hiện và xử lý outliers

        Args:
            columns: Danh sách cột cần được xử lí
            method: Phương pháp phát hiện
            treatment: Phương pháp xử lý
                'remove' - Xóa outliers
                'cap' - Giới hạn outliers
                'impute' - Thay thế bằng median/mean
                'transform' - Biến đổi dữ liệu
            **kwargs: Tham số bổ sung

        Returns:
            OutlierHandler
        """
        if self.data is None:
            raise ValueError("Không có dữ liệu để phân tích")

        print("\n" + "=" * 60)
        print("    XỬ LÝ OUTLIERS")
        print("=" * 60)


        # Xác định cột số
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        if len(columns) == 0:
            print("    Không có cột số để kiểm tra")
            return {}

        # Phát hiện outliers
        if method == 'auto':
            # Tự động chọn phương pháp dựa trên số lượng cột
            if len(columns) < 5:
                method = 'iqr'
            else:
                method = 'isolation_forest'
            print(f"   Tính năng Auto chọn phương pháp: {method.upper()}")

        outliers_info = self._detect_outliers(columns = columns, method=method, **kwargs)

        if not outliers_info:
            print("   Không có outliers để xử lý")
            return self

        # Xử lý outliers
        data_copy = self._treat_outliers(outliers_info, treatment, **kwargs)

        rows_before = len(self.data)
        rows_after = len(data_copy)
        rows_removed = rows_before - rows_after

        self.data = data_copy

        print(f"\n   HOÀN THÀNH XỬ LÝ OUTLIERS")
        print(f"   Dòng trước: {rows_before:,}")
        print(f"   Dòng sau: {rows_after:,}")
        if rows_removed > 0:
            print(f"   Đã xóa: {rows_removed:,} dòng")

        self._log_action('handle_outliers', {
            'method': method,
            'treatment': treatment,
            'total_outliers': sum(info['outlier_count'] for info in outliers_info.values()),
            'columns_with_outliers': list(outliers_info.keys()),
            'rows_before': rows_before,
            'rows_after': rows_after,
            'rows_removed': rows_removed
        })

        return self

    def _treat_outliers(self, outliers_info: Dict, treatment: str, **kwargs) -> pd.DataFrame:
        """
        Xử lý outliers: Tự động chọn phương pháp phù hợp cho TỪNG CỘT
        """
        data_copy = self.data.copy()

        # Tập hợp các dòng cần xóa (nếu có cột nào quyết định dùng phương pháp remove)
        rows_to_drop = set()

        print(f"   Chi tiết xử lý từng cột (Chế độ: {treatment.upper()}):")

        for col, info in outliers_info.items():
            # 1. Xác định phương pháp cho cột này
            current_treatment = treatment

            if treatment == 'auto':
                # Tính tỷ lệ outliers riêng cho cột này
                outlier_ratio = info['outlier_percent'] / 100.0

                # QUY TẮC AUTO CHO TỪNG CỘT:
                if outlier_ratio < 0.05:  # < 5%: Ít ngoại lai -> Capping (Kẹp giá trị)
                    current_treatment = 'cap'
                elif outlier_ratio < 0.15:  # 5-15%: Trung bình -> Impute (Thay thế bằng median)
                    current_treatment = 'impute'
                else:  # > 15%: Quá nhiều ngoại lai -> Transform (Biến đổi Log/Sqrt)
                    current_treatment = 'transform'

            # 2. Thực hiện xử lý theo phương pháp đã chọn
            try:
                if current_treatment == 'remove':
                    # Chỉ đánh dấu dòng cần xóa, sẽ xóa một thể sau cùng
                    rows_to_drop.update(info['outlier_indices'])
                    print(f"      • {col:<20}: REMOVE ({info['outlier_count']} dòng)")

                elif current_treatment == 'cap':
                    if info['method'] == 'iqr':
                        data_copy[col] = data_copy[col].clip(
                            lower=info['lower_bound'],
                            upper=info['upper_bound']
                        )
                    elif info['method'] == 'zscore':
                        lower = info['mean'] - 3 * info['std']
                        upper = info['mean'] + 3 * info['std']
                        data_copy[col] = data_copy[col].clip(lower=lower, upper=upper)
                    else:
                        # Fallback về IQR
                        Q1 = data_copy[col].quantile(0.25)
                        Q3 = data_copy[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        data_copy[col] = data_copy[col].clip(lower=lower_bound, upper=upper_bound)

                    print(f"      • {col:<20}: CAP (Giới hạn biên)")

                elif current_treatment == 'impute':
                    impute_method = kwargs.get('impute_method', 'median')

                    if impute_method == 'median':
                        impute_val = self.data[col].median()
                    elif impute_method == 'mean':
                        impute_val = self.data[col].mean()
                    elif impute_method == 'mode':
                        impute_val = self.data[col].mode()[0] if not self.data[col].mode().empty else 0
                    else:
                        impute_val = self.data[col].median()

                    data_copy.loc[info['outlier_indices'], col] = impute_val
                    print(f"      • {col:<20}: IMPUTE ({impute_method}={impute_val:.2f})")

                elif current_treatment == 'transform':
                    transform_method = kwargs.get('transform_method', 'log')

                    if transform_method == 'log':
                        min_val = data_copy[col].min()
                        shift = abs(min_val) + 1 if min_val <= 0 else 0
                        data_copy[col] = np.log(data_copy[col] + shift)
                    elif transform_method == 'sqrt':
                        min_val = data_copy[col].min()
                        shift = abs(min_val) + 1 if min_val < 0 else 0
                        data_copy[col] = np.sqrt(data_copy[col] + shift)

                    print(f"      • {col:<20}: TRANSFORM ({transform_method})")

            except Exception as e:
                print(f"      Lỗi xử lý cột {col}: {e}")

        # 3. Thực hiện xóa dòng (nếu có cột nào chọn 'remove')
        if rows_to_drop:
            rows_before = len(data_copy)
            data_copy = data_copy[~data_copy.index.isin(rows_to_drop)]
            print(f"      -> Đã xóa tổng cộng {rows_before - len(data_copy):,} dòng do chứa outliers")

        return data_copy
