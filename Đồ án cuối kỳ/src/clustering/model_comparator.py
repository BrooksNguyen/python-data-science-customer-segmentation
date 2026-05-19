import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
from .base import BaseClusteringComponent

class ModelComparator(BaseClusteringComponent):
    """So sánh mô hình và tối ưu siêu tham số"""

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.comparison_results = []
        self.best_model = None
        self.best_params = {}
        self.best_score = -1
        self.results_df = None
        self.best_algorithm = None

    def compare_algorithms(
        self,
        X: np.ndarray,
        algorithms_config: Dict[str, Dict[str, Any]],
        score_metric: str = 'silhouette'
    ) -> pd.DataFrame:
        """
        So sánh nhiều thuật toán phân cụm

        Tham s?:
            X: Ma trận đặc trưng
            algorithms_config: Cấu hình cho từng thuật toán
            score_metric: Chỉ số đánh giá ('silhouette', 'davies_bouldin', 'calinski_harabasz')

        Tr? v?:
            DataFrame chứa kết quả so sánh
        """
        self.comparison_results = []

        for algo_name, config in algorithms_config.items():
            print(f"\n Đang thử {algo_name.upper()}...")

            # Lấy lưới tham số
            param_grid = config.get('param_grid', {})

            # Sinh tất cả tổ hợp tham số
            for params in ParameterGrid(param_grid):
                try:
                    # Khởi tạo mô hình
                    model = self._create_model(algo_name, params)

                    # Huấn luyện và dự đoán
                    model.fit(X)
                    labels = model.predict(X)

                    # Tính toán chỉ số đánh giá
                    metrics = self._calculate_metrics(X, labels)

                    # Lưu kết quả
                    result = {
                        'algorithm': algo_name,
                        'params': params.copy(),
                        'labels': labels,
                        'model': model,
                        **metrics
                    }
                    self.comparison_results.append(result)

                    print(f"  Tham số: {params} | Silhouette: {metrics['silhouette']:.4f}")

                except Exception as e:
                    print(f"  Lỗi với bộ tham số {params}: {str(e)}")

        # Tạo DataFrame kết quả
        self.results_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != 'labels' and k != 'model'}
            for r in self.comparison_results
        ])

        # Chọn mô hình tốt nhất
        self._select_best_model(score_metric)

        return self.results_df

    def _create_model(self, algo_name: str, params: Dict[str, Any]):
        """Khởi tạo mô hình dựa trên tên thuật toán"""
        model_map = {
            'kmeans': KMeans,
            'gmm': GaussianMixture,
        }

        if algo_name not in model_map:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        # Lọc tham số hợp lệ cho thuật toán tương ứng
        model_class = model_map[algo_name]
        valid_params = {k: v for k, v in params.items()
                       if k in model_class.__init__.__code__.co_varnames}

        return model_class(**valid_params, random_state=self.random_state)

    def _calculate_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Tính toán các chỉ số đánh giá phân cụm"""
        n_clusters = len(np.unique(labels))

        if n_clusters == 1:
            return {
                'silhouette': -1,
                'davies_bouldin': float('inf'),
                'calinski_harabasz': 0
            }

        return {
            'silhouette': silhouette_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
            'n_clusters': n_clusters
        }

    def _select_best_model(self, score_metric: str = 'silhouette'):
        """Chọn mô hình tốt nhất dựa trên chỉ số đánh giá (mặc định là silhouette)"""
        if self.results_df is None or self.results_df.empty:
            return

        # Điểm cao tốt hơn cho silhouette và calinski_harabasz
        # Điểm thấp hơn tốt hơn cho davies_bouldin
        if score_metric in ['silhouette', 'calinski_harabasz']:
            best_idx = self.results_df[score_metric].idxmax()
        else:  # davies_bouldin
            best_idx = self.results_df[score_metric].idxmin()

        best_result = self.comparison_results[best_idx]

        self.best_model = best_result['model']
        self.best_params = best_result['params']
        self.best_score = best_result[score_metric]
        self.best_labels = best_result['labels']
        self.best_algorithm = best_result['algorithm']

        print(f"\n Mô hình tốt nhất: {best_result['algorithm']}")
        print(f"   Tham số: {best_result['params']}")
        print(f"   {score_metric}: {self.best_score:.4f}")

    def plot_metrics_comparison(self, algorithms_to_plot: List[str] = None, save_path: str = None):
        """Vẽ biểu đồ so sánh các chỉ số giữa các thuật toán"""
        if self.results_df is None or self.results_df.empty:
            print("Không có kết quả để vẽ biểu đồ")
            return None

        df = self.results_df.copy()

        # Đơn giản hóa bước kiểm tra dữ liệu
        required_cols = ['algorithm', 'n_clusters', 'silhouette']
        if not all(col in df.columns for col in required_cols):
            print(f"Thiếu cột cần thiết. Các cột hiện có: {df.columns.tolist()}")
            return None

        # Chuyển n_clusters về kiểu số
        df['n_clusters'] = pd.to_numeric(df['n_clusters'], errors='coerce')
        df = df.dropna(subset=['n_clusters'])
        df['n_clusters'] = df['n_clusters'].astype(int)

        if algorithms_to_plot:
            df = df[df['algorithm'].isin(algorithms_to_plot)]

        if df.empty:
            return None

        # Đơn giản hóa phần vẽ biểu đồ
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics_config = [
            ('silhouette', 'Silhouette Score\n(Cao hơn là tốt hơn)', 'higher'),
            ('davies_bouldin', 'Chỉ số Davies-Bouldin\n(Thấp hơn là tốt hơn)', 'lower'),
            ('calinski_harabasz', 'Chỉ số Calinski-Harabasz\n(Cao hơn là tốt hơn)', 'higher')
        ]

        algorithms = df['algorithm'].unique()
        colors = plt.cm.tab10(np.arange(len(algorithms)) / max(len(algorithms), 1))

        for idx, (metric, title, direction) in enumerate(metrics_config):
            ax = axes[idx]

            if metric not in df.columns:
                ax.text(0.5, 0.5, f'Không có dữ liệu {metric}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                continue

            for i, algo in enumerate(algorithms):
                algo_data = df[df['algorithm'] == algo]
                if algo_data.empty:
                    continue

                # Vẽ dữ liệu
                ax.scatter(algo_data['n_clusters'], algo_data[metric],
                           color=colors[i], label=algo if idx == 0 else "",
                           s=80, alpha=0.7, edgecolors='w', linewidth=0.5)

                # Vẽ đường xu hướng
                sorted_data = algo_data.sort_values('n_clusters')
                if len(sorted_data) > 1:
                    ax.plot(sorted_data['n_clusters'], sorted_data[metric],
                            color=colors[i], linewidth=1.5, alpha=0.5)

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Number of Clusters')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            if idx == 0:
                ax.legend(title='Algorithm', fontsize=9, title_fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            return save_path

        return fig

    def get_config(self) -> Dict[str, Any]:
        return {
            'best_algorithm': type(self.best_model).__name__ if self.best_model else None,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'random_state': self.random_state
        }
