import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from src.preprocessing import FeatureScaler
from .model_comparator import ModelComparator
from .visualizer import ClusteringVisualizer
from .model_manager import ModelManager
from .fine_tuner import ClusteringFineTuner
from .data_preprocessor import DataPreprocessor


class ClusteringPipeline(ModelComparator, ClusteringVisualizer, ModelManager, ClusteringFineTuner):
    """
    Pipeline phân cụm đầy đủ, tích hợp tất cả các thành phần
    """

    def __init__(self, experiment_name: str = "Clustering_Experiment", random_state: int = 42):
        # Khởi tạo tất cả lớp cha
        ModelComparator.__init__(self, random_state)
        ClusteringVisualizer.__init__(self, random_state)
        ModelManager.__init__(self, random_state)
        ClusteringFineTuner.__init__(self, random_state)

        # Khởi tạo DataPreprocessor riêng
        self.random_state = random_state
        self.data_preprocessor = DataPreprocessor(random_state)
        self.experiment_name = experiment_name
        self.fitted = False
        self.feature_scaler = None
        self.pca = None
        self.feature_names = None
        self.X_processed = None
        self.best_labels = None
        self.best_model = None
        self.best_algorithm = None
        self.report_dir = None

    def _save_figure(self, filename: str):
        """Lưu hình ảnh vào thư mục báo cáo"""
        if self.report_dir:
            filepath = os.path.join(self.report_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath
        return None

    def run_full_pipeline(
            self,
            data: pd.DataFrame,
            algorithms_config: Dict[str, Dict[str, Any]],
            drop_columns: List[str] = None,
            scale_method: str = 'standard',
            use_pca: bool = True,
            pca_variance: float = 0.95,
            fine_tune_algorithm: Optional[str] = None,
            fine_tune_trials: int = 30,
            evaluate_metrics: List[str] = None,
            export_results: bool = True,
            report_output_dir: str = "clustering_report"
    ) -> Dict[str, Any]:
        """
        Chạy toàn bộ pipeline phân cụm

        Tham s?:
            data: DataFrame đầu vào
            algorithms_config: Từ điển cấu hình các thuật toán
            drop_columns: Danh sách cột cần bỏ
            scale_method: Phương pháp scale ('standard', 'minmax', 'robust')
            use_pca: Có áp dụng PCA hay không
            pca_variance: Mức phương sai cần giữ lại khi dùng PCA
            fine_tune_algorithm: Thuật toán ('kmeans', 'gmm', 'agglo') cần tinh chỉnh (None để bỏ qua)
            fine_tune_trials: Số lần thử tinh chỉnh
            evaluate_metrics: Danh sách chỉ số đánh giá
            export_results: Có xuất kết quả hay không
            report_output_dir: Thư mục lưu báo cáo trực quan

        Tr? v?:
            Từ điển kết quả của pipeline
        """
        # Tạo thư mục lưu báo cáo
        self.report_dir = report_output_dir
        os.makedirs(self.report_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f" BẮT ĐẦU PIPELINE PHÂN CỤM: {self.experiment_name}")
        print(f" Thư mục báo cáo: {self.report_dir}")
        print('=' * 60)

        # ===== BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU VỚI DataPreprocessor =====
        print("\n BƯỚC 1: Tiền xử lý dữ liệu với DataPreprocessor")
        print('-' * 40)

        # Khởi tạo DataPreprocessor
        print("Đang khởi tạo DataPreprocessor...")
        self.data_preprocessor = DataPreprocessor(
            data=data,
            random_state=self.random_state,
            name=f"{self.experiment_name}_preprocessor",
            log_to_console=True
        )

        self.report_dir = report_output_dir
        os.makedirs(self.report_dir, exist_ok=True)

        # 1.1. Tải và chuẩn bị dữ liệu (chọn cột số)
        print("\n1.1 Đang tải và chuẩn bị dữ liệu...")
        X_numeric = self.data_preprocessor.load_and_prepare(
            data=data,
            drop_columns=drop_columns
        )

        # Kiểm tra dữ liệu
        if X_numeric is None or X_numeric.size == 0:
            raise ValueError("No numeric data available after preprocessing")

        # 1.2. Áp dụng PCA nếu được yêu cầu
        if use_pca:
            print(f"\n1.2 Đang áp dụng PCA (ngưỡng phương sai: {pca_variance})...")
            X_processed = self.data_preprocessor.apply_pca(variance_threshold=pca_variance)

            # Lấy thông tin PCA
            self.pca = self.data_preprocessor.pca
            pca_config = self.data_preprocessor.get_config()


            # Cập nhật tên đặc trưng cho các thành phần PCA
            self.feature_names = [f'PC{i + 1}' for i in range(X_processed.shape[1])]

            # Lưu báo cáo PCA
            pca_report_dir = os.path.join(self.report_dir, "pca_analysis")
            self.data_preprocessor.save_pca_report(output_dir=pca_report_dir)
            print(f" Đã lưu phân tích PCA tại: {pca_report_dir}")

        else:
            # Không dùng PCA
            X_processed = X_numeric
            self.feature_names = self.data_preprocessor.feature_names
            print(" Không áp dụng PCA, sử dụng đặc trưng gốc")

        # Lưu dữ liệu đã xử lý
        self.X_processed = X_processed
        print(f"\n  Đã hoàn tất tiền xử lý!")
        print(f"   Kích thước dữ liệu cuối cùng: {self.X_processed.shape}")
        print(f"   Số lượng đặc trưng: {len(self.feature_names)}")

        print("\n BƯỚC 2: Huấn luyện và so sánh mô hình")
        print('-' * 40)

        # Định nghĩa bộ chỉ số đánh giá mặc định nếu chưa được cung cấp
        if evaluate_metrics is None:
            evaluate_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']

        # Huấn luyện và so sánh mô hình
        comparison_df = self.compare_algorithms(
            X=self.X_processed,
            algorithms_config=algorithms_config,
            score_metric=evaluate_metrics[0]  # Use first metric for selection
        )

        # Hiển thị kết quả so sánh
        print("\n Kết quả so sánh mô hình:")
        print("-" * 50)
        print(comparison_df.sort_values(evaluate_metrics[0], ascending=False).head())

        # Bước 3: Tinh chỉnh (tùy chọn)
        if fine_tune_algorithm:
            print("\n BƯỚC 3: Tinh chỉnh mô hình")
            print('-' * 40)

            # Kiểm tra thuật toán có trong cấu hình hay không
            if fine_tune_algorithm in algorithms_config:
                best_fine_tuned_params = self.bayesian_optimization(
                    X=self.X_processed,
                    algorithm=fine_tune_algorithm,
                    n_trials=fine_tune_trials
                )

                print(f" Tham số tốt nhất sau tinh chỉnh: {best_fine_tuned_params}")
                print(f" Điểm tốt nhất: {self.best_score:.4f}")

                # Update best model with tuned parameters
                if fine_tune_algorithm == 'kmeans':
                    from sklearn.cluster import KMeans
                    self.best_model = KMeans(
                        **best_fine_tuned_params,
                        random_state=self.random_state
                    )
                elif fine_tune_algorithm == 'gmm':
                    from sklearn.mixture import GaussianMixture
                    self.best_model = GaussianMixture(
                        **best_fine_tuned_params,
                        random_state=self.random_state
                    )

                # Refit with tuned parameters
                self.best_model.fit(self.X_processed)
                self.best_labels = self.best_model.predict(self.X_processed)
            else:
                print(f" Thuật toán '{fine_tune_algorithm}' không có trong cấu hình, bỏ qua bước tinh chỉnh")

        # Bước 4: Trực quan hóa và phân tích
        print("\n BƯỚC 4: Trực quan hóa và phân tích")
        print('-' * 40)

        # Set pipeline state for visualizer
        self.set_pipeline_state(
            report_dir=self.report_dir,
            best_labels=self.best_labels,
            X_processed=self.X_processed,
            feature_names=self.feature_names,
            pca=self.pca
        )

        # Tạo toàn bộ biểu đồ
        visualization_files = self.create_visualization_pipeline()

        #  Lưu kết quả thực nghiệm đơn giản
        print("\n Lưu kết quả thực nghiệm...")

        # Tạo DataFrame kết quả
        if hasattr(self, 'results_df') and self.results_df is not None:
            # Lưu ra CSV
            results_csv = os.path.join(self.report_dir, 'experiment_results.csv')
            self.results_df.to_csv(results_csv, index=False)
            print(f"   Lưu kết quả: {results_csv}")

        # Bước 5: Xuất kết quả và quản lý mô hình
        if export_results:
            print("\n BƯỚC 5: Xuất kết quả và quản lý mô hình")
            print('-' * 40)

            # Prepare metadata
            metadata = {
                'experiment_name': self.experiment_name,
                'best_algorithm': self.best_algorithm,
                'best_params': self.best_params,
                'best_score': self.best_score,
                'pipeline_config': {
                    'scale_method': scale_method,
                    'use_pca': use_pca,
                    'pca_variance': pca_variance if use_pca else None,
                    'feature_names': self.feature_names,
                    'processed_shape': self.X_processed.shape
                },
                'metrics': {
                    'silhouette': silhouette_score(self.X_processed,
                                                   self.best_labels) if self.best_labels is not None else None,
                    'n_clusters': len(np.unique(self.best_labels)) if self.best_labels is not None else None
                },
                'visualization_files': visualization_files
            }

            # Lưu toàn bộ pipeline
            if self.best_model is not None:
                model_dir = self.save_model(
                    model=self.best_model,
                    preprocessor=self.feature_scaler,
                    metadata=metadata,
                    model_name=f"{self.best_algorithm}_{self.experiment_name}"
                )

                # Xuất file dự đoán
                predictions_file = os.path.join(self.report_dir, f"predictions_{self.experiment_name}.csv")
                self.save_predictions(
                    predictions=self.best_labels,
                    original_data=data,
                    output_path=predictions_file
                )

                # Xuất model card
                model_card_file = os.path.join(self.report_dir, f"model_card_{self.experiment_name}.json")
                metrics_dict = {}
                if self.best_labels is not None:
                    try:
                        metrics_dict = {
                            'silhouette': silhouette_score(self.X_processed, self.best_labels),
                            'davies_bouldin': davies_bouldin_score(self.X_processed, self.best_labels),
                            'calinski_harabasz': calinski_harabasz_score(self.X_processed, self.best_labels),
                            'n_clusters': len(np.unique(self.best_labels))
                        }
                    except:
                        pass

                self.export_model_card(
                    model=self.best_model,
                    metrics=metrics_dict,
                    features=self.feature_names if not use_pca else [f'PC{i + 1}' for i in
                                                                     range(self.X_processed.shape[1])],
                    output_path=model_card_file
                )

        # Mark pipeline as fitted
        self.fitted = True
        # Prepare final results
        results = {
            'best_model': self.best_model,
            'best_labels': self.best_labels,
            'best_algorithm': self.best_algorithm,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'processed_data': self.X_processed,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance if hasattr(self, 'feature_importance') else None,
            'comparison_results': comparison_df,
            'pipeline_fitted': self.fitted,
            'report_directory': self.report_dir,
            'visualization_files': visualization_files
        }

        print(f"\n{'=' * 60}")
        print(f" ĐÃ HOÀN THÀNH PIPELINE: {self.experiment_name}")
        print(f" Tất cả tệp đã lưu tại: {self.report_dir}")
        print('=' * 60)

        return results

    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        """
        Dự đoán cụm cho dữ liệu mới bằng pipeline đã huấn luyện

        Tham s?:
            new_data: DataFrame mới cần dự đoán

        Tr? v?:
            Nhãn cụm
        """
        if not self.fitted:
            raise ValueError("Pipeline not fitted. Call run_full_pipeline() first.")

        if self.best_model is None:
            raise ValueError("No trained model available.")

        # Tiền xử lý dữ liệu mới (theo đúng quy trình trước đó)
        if self.feature_scaler is not None:
            # Chuẩn hóa dữ liệu
            new_data_scaled = self.feature_scaler.transform(new_data[self.feature_names].values)

            # Áp dụng PCA nếu đã dùng khi huấn luyện
            if self.pca is not None:
                new_data_processed = self.pca.transform(new_data_scaled)
            else:
                new_data_processed = new_data_scaled

            # Dự đoán cụm
            if hasattr(self.best_model, 'predict'):
                predictions = self.best_model.predict(new_data_processed)
            else:
                predictions = self.best_model.fit_predict(new_data_processed)

            return predictions
        else:
            raise ValueError("Feature scaler not available.")

    def get_cluster_profiles(self, original_data: pd.DataFrame) -> pd.DataFrame:
        """
        Lấy thống kê đặc trưng cho từng cụm

        Tham s?:
            original_data: DataFrame gốc chứa các đặc trưng

        Tr? v?:
            DataFrame mô tả hồ sơ từng cụm
        """
        if self.best_labels is None:
            raise ValueError("No cluster labels available.")

        # Gắn nhãn cụm vào dữ liệu
        data_with_clusters = original_data.copy()
        data_with_clusters['Cluster'] = self.best_labels

        # Tính thống kê cho từng cụm
        cluster_profiles = []

        for cluster in np.unique(self.best_labels):
            cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]

            profile = {
                'Cluster': cluster,
                'Size': len(cluster_data),
                'Size_Percentage': len(cluster_data) / len(data_with_clusters) * 100
            }

            # Bổ sung giá trị trung bình cho các cột số
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'Cluster':
                    profile[f'{col}_mean'] = cluster_data[col].mean()
                    profile[f'{col}_std'] = cluster_data[col].std()

            cluster_profiles.append(profile)

        return pd.DataFrame(cluster_profiles)

    def get_config(self) -> Dict[str, Any]:
        """Lấy cấu hình pipeline"""
        config = {
            'experiment_name': self.experiment_name,
            'fitted': self.fitted,
            'best_algorithm': self.best_algorithm,
            'best_score': self.best_score,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'has_pca': self.pca is not None,
            'report_directory': self.report_dir
        }

        # Add parent class configs
        config.update(ModelComparator.get_config(self))
        config.update(ModelManager.get_config(self))

        return config
