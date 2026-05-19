import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from typing import List, Optional, Dict, Any
from .base import BaseClusteringComponent


class DataPreprocessor(BaseClusteringComponent):
    """Xử lý toàn bộ bước tiền xử lý dữ liệu (không bao gồm chuẩn hóa)"""

    def __init__(
            self,
            data: Optional[pd.DataFrame] = None,
            random_state: int = 42,
            name: str = None,
            log_to_console: bool = False
    ):
        super().__init__(
            data=data,
            name=name or "DataPreprocessor",
            log_to_console=log_to_console
        )
        self.pca = None
        self.X_pca = None
        self.X_numeric = None  # Thay thế X_scaled
        self.feature_names = None
        self.random_state = random_state

    def load_and_prepare(
            self,
            data: pd.DataFrame,
            target_column: Optional[str] = None,
            drop_columns: List[str] = None
    ) -> np.ndarray:
        """
        Tải dữ liệu và chọn các cột dạng số

        Tham s?:
            data: DataFrame đầu vào
            target_column: Cột nhãn (tùy chọn) cho các tác vụ có giám sát
            drop_columns: Danh sách cột cần loại bỏ

        Tr? v?:
            Ma trận đặc trưng dạng số (đã được chuẩn hóa kiểu dữ liệu)
        """
        if drop_columns:
            cols = [c for c in drop_columns if c in data.columns]
            data = data.drop(columns=cols)

        if target_column and target_column in data.columns:
            self.y = data[target_column].values
            data = data.drop(columns=[target_column])

        # Chỉ lấy các cột số và chuyển boolean thành số
        for col in data.select_dtypes(include=['bool']).columns:
            data[col] = data[col].astype(int)
        numeric_data = data.select_dtypes(include=[np.number])
        self.feature_names = numeric_data.columns.tolist()
        self.X_numeric = numeric_data.values

        print(f"   Đã chọn {len(self.feature_names)} đặc trưng dạng số")
        print(f"   Kích thước dữ liệu: {self.X_numeric.shape}")

        return self.X_numeric

    def apply_pca(self, n_components: Optional[int] = None, variance_threshold: float = 0.95):
        """
        Áp dụng giảm chiều bằng PCA

        Tham s?:
            n_components: Số thành phần (nếu None thì dùng variance_threshold)
            variance_threshold: Ngưỡng phương sai tối thiểu cần giữ

        Tr? v?:
            Dữ liệu sau khi biến đổi PCA
        """
        if self.X_numeric is None:
            raise ValueError("Must call load_and_prepare() first")

        if n_components is None:
            self.pca = PCA(n_components=variance_threshold, random_state=self.random_state)
        else:
            self.pca = PCA(n_components=n_components, random_state=self.random_state)

        self.X_pca = self.pca.fit_transform(self.X_numeric)

        print(f"   Đã áp dụng PCA: {self.X_numeric.shape[1]} đặc trưng -> {self.X_pca.shape[1]} thành phần")
        print(f"   Tỷ lệ phương sai giữ lại: {self.pca.explained_variance_ratio_.sum():.3f}")

        return self.X_pca

    def _get_pca_loadings(self) -> pd.DataFrame:
        """
        Lấy hệ số đóng góp (loading) của PCA cho từng thành phần

        Tr? v?:
            DataFrame chứa hệ số của mỗi thành phần
        """
        if self.pca is None:
            raise ValueError("Must call apply_pca() first")

        if self.feature_names is None:
            raise ValueError("Feature names not available")

        n_components = self.pca.components_.shape[0]
        loadings_df = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i + 1}' for i in range(n_components)],
            index=self.feature_names
        )

        return loadings_df

    def _get_explained_variance(self) -> Dict[str, Any]:
        """
        Lấy thông tin phương sai được giải thích

        Tr? v?:
            Từ điển mô tả thông tin phương sai
        """
        if self.pca is None:
            raise ValueError("Must call apply_pca() first")

        explained_var = self.pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        return {
            'individual': explained_var.tolist(),
            'cumulative': cumulative_var.tolist(),
            'total': explained_var.sum(),
            'n_components': len(explained_var)
        }
    @staticmethod
    def split_data(data, test_size: float = 0.2, random_state = 42):
        """
        Chia dữ liệu thành tập train/test

        Tr? v?:
            X_train, X_test, y_train, y_test (nếu y tồn tại)
        """
        return train_test_split(
            data,
            test_size=test_size,
            random_state=random_state
        )

    def get_feature_importance_pca(self, n_top: int = 10) -> Dict[str, List[Any]]:
        """
        Lấy các đặc trưng nổi bật nhất cho từng thành phần PCA

        Tham s?:
            n_top: Số đặc trưng trả về cho mỗi thành phần

        Tr? v?:
            Từ điển đặc trưng nổi bật của từng thành phần
        """
        if self.pca is None:
            raise ValueError("Must call apply_pca() first")

        if self.feature_names is None:
            raise ValueError("Feature names not available")

        result = {}
        n_components = self.pca.components_.shape[0]

        for i in range(n_components):
            component_loadings = self.pca.components_[i]

            # Lấy đặc trưng có giá trị loading lớn nhất (giá trị tuyệt đối)
            top_indices = np.argsort(np.abs(component_loadings))[-n_top:][::-1]

            top_features = []
            for idx in top_indices:
                feature_name = self.feature_names[idx]
                loading_value = component_loadings[idx]
                top_features.append({
                    'feature': feature_name,
                    'loading': float(loading_value),
                    'abs_loading': float(abs(loading_value))
                })

            result[f'PC{i + 1}'] = top_features

        return result

    def get_config(self) -> Dict[str, Any]:
        return {
            'pca_components': self.pca.n_components_ if self.pca else None,
            'feature_names': self.feature_names,
            'random_state': self.random_state,
            'data_shape': self.X_numeric.shape if self.X_numeric is not None else None,
            'pca_applied': self.pca is not None
        }

    def save_pca_report(self, output_dir: str = "."):
        """
        Lưu báo cáo PCA đầy đủ

        Tham s?:
            output_dir: Thư mục lưu các tệp báo cáo
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 1. Lưu hệ số loading
        if self.pca is not None and self.feature_names is not None:
            loadings_df = self._get_pca_loadings()
            loadings_df.to_csv(os.path.join(output_dir, 'pca_loadings.csv'))
            print(f"Đã lưu hệ số PCA tại: {os.path.join(output_dir, 'pca_loadings.csv')}")

        # 2. Lưu thông tin phương sai
        if self.pca is not None:
            var_info = self._get_explained_variance()
            var_df = pd.DataFrame({
                'Component': [f'PC{i + 1}' for i in range(len(var_info['individual']))],
                'Individual_Variance': var_info['individual'],
                'Cumulative_Variance': var_info['cumulative']
            })
            var_df.to_csv(os.path.join(output_dir, 'pca_variance.csv'), index=False)
            print(f"Đã lưu thông tin phương sai tại: {os.path.join(output_dir, 'pca_variance.csv')}")

        # 3. Lưu mức độ quan trọng của đặc trưng
        if self.pca is not None and self.feature_names is not None:
            feature_importance = self.get_feature_importance_pca()

            # Chuyển sang DataFrame
            importance_data = []
            for pc, features in feature_importance.items():
                for feat in features:
                    importance_data.append({
                        'Component': pc,
                        'Feature': feat['feature'],
                        'Loading': feat['loading'],
                        'Abs_Loading': feat['abs_loading']
                    })

            importance_df = pd.DataFrame(importance_data)
            importance_df.to_csv(os.path.join(output_dir, 'pca_feature_importance.csv'), index=False)
            print(f"Đã lưu mức độ quan trọng của đặc trưng tại: {os.path.join(output_dir, 'pca_feature_importance.csv')}")
