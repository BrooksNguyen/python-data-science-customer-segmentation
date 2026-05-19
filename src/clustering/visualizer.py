import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from math import pi
from typing import List, Optional, Dict, Any, Union
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy import stats
from .base import BaseClusteringComponent


class ClusteringVisualizer(BaseClusteringComponent):
    """Thực hiện trực quan hóa kết quả phân cụm với hỗ trợ tích hợp pipeline"""

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.set_style()
        self.report_dir = None
        self.best_labels = None
        self.X_processed = None
        self.feature_names = None
        self.pca = None

    def set_style(self):
        """Thiết lập phong cách vẽ biểu đồ thống nhất"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['font.size'] = 12

    def set_pipeline_state(self, report_dir: str, best_labels: np.ndarray,
                           X_processed: np.ndarray, feature_names: List[str],
                           pca: Optional[Any] = None):
        """Thiết lập trạng thái pipeline để phục vụ trực quan hóa"""
        self.report_dir = report_dir
        self.best_labels = best_labels
        self.X_processed = X_processed
        self.feature_names = feature_names
        self.pca = pca

    def _save_figure(self, filename: str):
        """Lưu hình ảnh vào thư mục báo cáo"""
        if self.report_dir:
            filepath = os.path.join(self.report_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath
        return None

    def create_visualization_pipeline(self, vis_data: pd.DataFrame = None) -> Dict[str, str]:
        """
        Tạo đầy đủ các hình ảnh trực quan cho kết quả phân cụm

        Tham s?:
            vis_data: DataFrame để trực quan hóa (nếu None sẽ tự tạo từ X_processed)

        Tr? v?:
            Từ điển đường dẫn tới các tệp hình ảnh đã lưu
        """
        if self.best_labels is None:
            raise ValueError("No cluster labels available. Call set_pipeline_state() first.")

        visualization_files = {}

        # Chuẩn bị dữ liệu trực quan nếu chưa được cung cấp
        if vis_data is None:
            vis_data = self._prepare_visualization_data()

        print("Đang tạo các hình minh họa...")

        # 1. Phân bố kích thước cụm
        try:
            self.plot_cluster_sizes(self.best_labels)
            visualization_files['cluster_sizes'] = self._save_figure('01_cluster_sizes.png')
            print(f"   Đã lưu biểu đồ kích thước cụm: {visualization_files['cluster_sizes']}")
        except Exception as e:
            print(f" Không thể tạo biểu đồ kích thước cụm: {e}")

        # 2. Ma trận scatter (dùng các đặc trưng nổi bật)
        if vis_data.shape[1] > 1:
            try:
                self.plot_scatter_matrix(
                    data=vis_data,
                    labels=self.best_labels,
                    n_features=min(4, vis_data.shape[1])
                )
                visualization_files['scatter_matrix'] = self._save_figure('02_scatter_matrix.png')
                print(f"   Đã lưu scatter matrix: {visualization_files['scatter_matrix']}")
            except Exception as e:
                print(f" Không thể tạo scatter matrix: {e}")

        # 3. Heatmap cho mức độ quan trọng của đặc trưng
        try:
            if len(self.best_labels) > 0:
                self.plot_heatmap(
                    data=vis_data,
                    labels=self.best_labels,
                    n_features=min(10, vis_data.shape[1])
                )
                visualization_files['heatmap'] = self._save_figure('03_feature_heatmap.png')
                print(f"  Đã lưu heatmap đặc trưng: {visualization_files['heatmap']}")

                # 4. Biểu đồ radar
                try:
                    self.plot_radar_chart(
                        data=vis_data,
                        labels=self.best_labels,
                        n_features=min(6, vis_data.shape[1])
                    )
                    visualization_files['radar_chart'] = self._save_figure('04_radar_chart.png')
                    print(f"   Đã lưu biểu đồ radar: {visualization_files['radar_chart']}")
                except Exception as e:
                    print(f"  Không thể tạo biểu đồ radar: {e}")
        except Exception as e:
            print(f" Không thể tạo heatmap: {e}")

        # 5. Biểu đồ scatter 2D (khi có PCA hoặc dữ liệu >= 2 chiều)
        if self.X_processed is not None and self.X_processed.shape[1] >= 2:
            try:
                self.plot_scatter_2d(
                    X=self.X_processed,
                    labels=self.best_labels,
                    feature_names=self.feature_names,
                    title=f'Clustering Results - {self.report_dir}'
                )
                visualization_files['scatter_2d'] = self._save_figure('05_scatter_2d.png')
                print(f"   Đã lưu biểu đồ scatter 2D: {visualization_files['scatter_2d']}")
            except Exception as e:
                print(f" Không thể tạo biểu đồ scatter 2D: {e}")

        # 6. Phân tích mức độ quan trọng của đặc trưng
        if self.X_processed is not None and len(np.unique(self.best_labels)) > 1:
            try:
                self.plot_feature_importance(
                    X=self.X_processed,
                    labels=self.best_labels,
                    feature_names=self.feature_names
                )
                visualization_files['feature_importance'] = self._save_figure('06_feature_importance.png')
                print(f"  Đã lưu biểu đồ mức độ quan trọng: {visualization_files['feature_importance']}")

                # Tạo báo cáo chi tiết
                try:
                    importance_df = self.create_comprehensive_report(
                        X=self.X_processed,
                        labels=self.best_labels,
                        feature_names=self.feature_names,
                        output_dir=self.report_dir
                    )
                    if not importance_df.empty:
                        print(f"  Đã tạo báo cáo mức độ quan trọng của đặc trưng")
                except Exception as e:
                    print(f"  Không thể tạo báo cáo mức độ quan trọng của đặc trưng: {e}")
            except Exception as e:
                print(f" Không thể tạo biểu đồ mức độ quan trọng của đặc trưng: {e}")

        # 7. Biểu đồ scatter 3D (khi dữ liệu có từ 3 chiều)
        if self.X_processed is not None and self.X_processed.shape[1] >= 3:
            try:
                self.plot_scatter_3d(
                    X=self.X_processed,
                    labels=self.best_labels,
                    feature_names=self.feature_names
                )
                visualization_files['scatter_3d'] = self._save_figure('07_scatter_3d.png')
                print(f"   Đã lưu biểu đồ scatter 3D: {visualization_files['scatter_3d']}")
            except Exception as e:
                print(f" Không thể tạo biểu đồ scatter 3D: {e}")

        return visualization_files

    def _prepare_visualization_data(self) -> pd.DataFrame:
        """Chuẩn bị dữ liệu phục vụ trực quan hóa"""
        if self.X_processed is None:
            raise ValueError("No processed data available")

        # Chuyển dữ liệu đã xử lý về DataFrame để vẽ biểu đồ
        if self.pca is not None:
            # Với dữ liệu đã giảm chiều PCA, tạo tên cột dễ hiểu
            n_components = self.X_processed.shape[1]
            if n_components == 2:
                vis_columns = ['PC1', 'PC2']
            elif n_components == 3:
                vis_columns = ['PC1', 'PC2', 'PC3']
            else:
                vis_columns = [f'PC{i + 1}' for i in range(n_components)]

            vis_data = pd.DataFrame(self.X_processed, columns=vis_columns)
        else:
            # Sử dụng tên đặc trưng gốc
            vis_data = pd.DataFrame(self.X_processed, columns=self.feature_names)

        return vis_data

    def plot_scatter_matrix(
            self,
            data: pd.DataFrame,
            labels: np.ndarray,
            features: List[str] = None,
            n_features: int = 4
    ):
        """
        Tạo ma trận scatter để trực quan hóa các cụm
        """
        if features is None:
            features = self._select_top_features(data, labels, n_features)

        plot_data = data[features].copy()
        plot_data['Cluster'] = labels

        # Tạo biểu đồ pairplot
        g = sns.pairplot(
            plot_data,
            hue='Cluster',
            palette='viridis',
            plot_kws={'alpha': 0.7, 's': 50},
            diag_kind='kde'
        )

        g.fig.suptitle('Pairwise Feature Relationships by Cluster', y=1.02)
        return g.fig

    def plot_radar_chart(
            self,
            data: pd.DataFrame,
            labels: np.ndarray,
            features: List[str] = None,
            n_features: int = 6
    ):
        """
        Tạo biểu đồ radar cho hồ sơ từng cụm
        """
        if features is None:
            features = self._select_top_features(data, labels, n_features)

        # Tính giá trị trung bình theo từng cụm
        cluster_means = data.groupby(labels)[features].mean()

        # Chuẩn hóa dữ liệu
        scaler = MinMaxScaler()
        normalized_means = pd.DataFrame(
            scaler.fit_transform(cluster_means),
            columns=features,
            index=cluster_means.index
        )

        # Chuẩn bị dữ liệu cho biểu đồ radar
        categories = list(features)
        N = len(categories)

        # Góc của từng trục
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # Tạo biểu đồ
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Màu sắc cho từng cụm
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_means)))

        for idx, (cluster_idx, row) in enumerate(normalized_means.iterrows()):
            values = row.values.flatten().tolist()
            values += values[:1]

            ax.plot(angles, values, linewidth=2, linestyle='solid',
                    label=f'Cluster {cluster_idx}', color=colors[idx])
            ax.fill(angles, values, color=colors[idx], alpha=0.25)

        # Thêm nhãn trục
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)

        # Thêm chú thích
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Cluster Profiles - Radar Chart', size=20, y=1.1)
        return fig

    def plot_heatmap(
            self,
            data: pd.DataFrame,
            labels: np.ndarray,
            features: List[str] = None,
            n_features: int = 10
    ):
        """
        Tạo heatmap thể hiện mức độ quan trọng của đặc trưng theo cụm
        """
        if features is None:
            features = self._select_top_features(data, labels, n_features)

        # Tính z-score cho từng cụm
        cluster_means = data.groupby(labels)[features].mean()
        global_means = data[features].mean()
        global_stds = data[features].std()

        z_scores = (cluster_means - global_means) / global_stds

        # Tạo heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            z_scores.T,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Z-Score'},
            ax=ax
        )

        ax.set_title('Feature Z-Scores by Cluster', fontsize=16)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        return fig

    def plot_cluster_sizes(self, labels: np.ndarray):
        """Vẽ biểu đồ cột thể hiện kích thước các cụm"""
        unique, counts = np.unique(labels, return_counts=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar([f'Cluster {c}' for c in unique], counts,
                      color=plt.cm.Set3(np.arange(len(unique))))

        # Thêm giá trị trên mỗi cột
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{count}\n({count / len(labels):.1%})',
                    ha='center', va='bottom')

        ax.set_title('Cluster Sizes Distribution', fontsize=16)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Points')
        ax.set_ylim(0, max(counts) * 1.1)
        ax.grid(axis='y', alpha=0.3)
        return fig

    # ====== 2D SCATTER PLOT FUNCTIONS ======

    def plot_scatter_2d(
            self,
            X: np.ndarray,
            labels: np.ndarray,
            feature_names: List[str] = None,
            components: tuple = (0, 1),
            figsize: tuple = (10, 8),
            show_centroids: bool = True,
            alpha: float = 0.7,
            s: int = 50,
            title: str = None
    ) -> plt.Figure:
        """
        Tạo biểu đồ scatter 2D cho kết quả phân cụm

        Tham s?:
            X: Ma trận đặc trưng (từ 2 chiều trở lên)
            labels: Nhãn cụm
            feature_names: Tên các đặc trưng/thành phần
            components: Cặp thành phần/dimension cần vẽ (mặc định: 2 thành phần đầu)
            show_centroids: Có hiển thị tâm cụm hay không
            alpha: Độ trong suốt của điểm
            s: Kích thước điểm
            title: Tiêu đề biểu đồ

        Tr? v?:
            Đối tượng Figure của matplotlib
        """
        if X.shape[1] < 2:
            raise ValueError("X must have at least 2 dimensions for 2D scatter plot")

        # Lấy hai thành phần cần vẽ
        comp1, comp2 = components
        if comp1 >= X.shape[1] or comp2 >= X.shape[1]:
            raise ValueError(f"Components {components} exceed available dimensions {X.shape[1]}")

        x_data = X[:, comp1]
        y_data = X[:, comp2]

        # Tạo Figure
        fig, ax = plt.subplots(figsize=figsize)

        # Lấy danh sách cụm
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Bảng màu cho từng cụm
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

        # Vẽ từng cụm
        for i, cluster_label in enumerate(unique_labels):
            # Điểm thuộc cụm hiện tại
            mask = labels == cluster_label
            ax.scatter(
                x_data[mask],
                y_data[mask],
                color=colors[i],
                label=f'Cluster {cluster_label}',
                alpha=alpha,
                s=s,
                edgecolors='w',
                linewidth=0.5
            )

            # Hiển thị tâm cụm nếu cần
            if show_centroids:
                centroid_x = np.mean(x_data[mask])
                centroid_y = np.mean(y_data[mask])
                ax.scatter(
                    centroid_x, centroid_y,
                    color=colors[i],
                    marker='X',
                    s=200,
                    edgecolors='black',
                    linewidth=2,
                    label=f'Centroid {cluster_label}' if i == 0 else ""
                )

        # Thêm nhãn trục
        if feature_names is not None and len(feature_names) > max(comp1, comp2):
            x_label = feature_names[comp1]
            y_label = feature_names[comp2]
        else:
            x_label = f'Component {comp1 + 1}'
            y_label = f'Component {comp2 + 1}'

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)

        # Thêm tiêu đề
        if title:
            plot_title = title
        else:
            plot_title = f'2D Scatter Plot of Clusters (n={n_clusters})'

        ax.set_title(plot_title, fontsize=14, fontweight='bold')

        # Thêm lưới
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Thêm chú thích
        ax.legend(loc='best', fontsize=10)

        # Bổ sung thông tin phân bố cụm
        unique, counts = np.unique(labels, return_counts=True)
        cluster_info = "\n".join([f'Cluster {c}: {cnt} ({cnt / len(labels):.1%})'
                                  for c, cnt in zip(unique, counts)])

        ax.text(0.02, 0.98, cluster_info,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig

    def plot_scatter_3d(
            self,
            X: np.ndarray,
            labels: np.ndarray,
            feature_names: List[str] = None,
            components: tuple = (0, 1, 2),
            figsize: tuple = (12, 10),
            alpha: float = 0.7,
            s: int = 50,
            title: str = None
    ) -> plt.Figure:
        """
        Tạo biểu đồ scatter 3D cho kết quả phân cụm
        """
        if X.shape[1] < 3:
            raise ValueError("X must have at least 3 dimensions for 3D scatter plot")

        # Lấy ba thành phần cần vẽ
        comp1, comp2, comp3 = components

        # Tạo biểu đồ 3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Lấy danh sách cụm
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        # Vẽ từng cụm
        for i, cluster_label in enumerate(unique_labels):
            mask = labels == cluster_label
            ax.scatter(
                X[mask, comp1],
                X[mask, comp2],
                X[mask, comp3],
                color=colors[i],
                label=f'Cluster {cluster_label}',
                alpha=alpha,
                s=s,
                edgecolors='w',
                linewidth=0.5
            )

        # Labels
        if feature_names is not None:
            ax.set_xlabel(feature_names[comp1] if comp1 < len(feature_names) else f'Comp {comp1 + 1}')
            ax.set_ylabel(feature_names[comp2] if comp2 < len(feature_names) else f'Comp {comp2 + 1}')
            ax.set_zlabel(feature_names[comp3] if comp3 < len(feature_names) else f'Comp {comp3 + 1}')
        else:
            ax.set_xlabel(f'Component {comp1 + 1}')
            ax.set_ylabel(f'Component {comp2 + 1}')
            ax.set_zlabel(f'Component {comp3 + 1}')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'3D Scatter Plot of Clusters', fontsize=14, fontweight='bold')

        ax.legend(loc='best')

        return fig

    # ====== FEATURE IMPORTANCE FUNCTIONS ======

    def calculate_feature_importance(self, X: np.ndarray, labels: np.ndarray,
                                     method: str = 'anova') -> Dict:
        """
        Tính mức độ quan trọng của đặc trưng cho kết quả phân cụm

        Tham s?:
            X: Ma trận đặc trưng
            labels: Nhãn cụm
            method: 'anova' hoặc 'silhouette'

        Tr? v?:
            Từ điển điểm quan trọng
        """
        if method == 'anova':
            return self._anova_feature_importance(X, labels)
        elif method == 'silhouette':
            return self._silhouette_feature_importance(X, labels)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _anova_feature_importance(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Kiểm định ANOVA F giữa các cụm cho từng đặc trưng"""
        n_features = X.shape[1]
        n_clusters = len(np.unique(labels))

        importance_scores = {}
        p_values = {}

        for feature_idx in range(n_features):
            # Gom nhóm dữ liệu theo cụm cho đặc trưng hiện tại
            groups = []
            for cluster_id in range(n_clusters):
                cluster_data = X[labels == cluster_id, feature_idx]
                if len(cluster_data) > 0:
                    groups.append(cluster_data)

            # Chỉ thực hiện ANOVA khi có ít nhất 2 nhóm
            if len(groups) >= 2:
                try:
                    f_stat, p_val = stats.f_oneway(*groups)
                    importance_scores[feature_idx] = f_stat
                    p_values[feature_idx] = p_val
                except:
                    importance_scores[feature_idx] = 0
                    p_values[feature_idx] = 1
            else:
                importance_scores[feature_idx] = 0
                p_values[feature_idx] = 1

        return {
            'importance_scores': importance_scores,
            'p_values': p_values,
            'method': 'anova_f_test'
        }

    def _silhouette_feature_importance(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Tính mức độ quan trọng dựa trên mức giảm điểm silhouette"""
        if len(np.unique(labels)) <= 1:
            return {'importance_scores': {i: 0 for i in range(X.shape[1])},
                    'method': 'silhouette_reduction'}

        try:
            base_silhouette = silhouette_score(X, labels)
        except:
            base_silhouette = 0

        n_features = X.shape[1]
        importance_scores = {}

        for feature_idx in range(n_features):
            try:
                # Loại bỏ một đặc trưng
                mask = [i for i in range(n_features) if i != feature_idx]
                X_reduced = X[:, mask]

                # Tính silhouette khi bỏ đặc trưng này
                reduced_silhouette = silhouette_score(X_reduced, labels)

                # Mức quan trọng = độ giảm silhouette khi bỏ đặc trưng
                importance = base_silhouette - reduced_silhouette
                importance_scores[feature_idx] = importance
            except:
                importance_scores[feature_idx] = 0

        return {
            'importance_scores': importance_scores,
            'method': 'silhouette_reduction',
            'base_silhouette': base_silhouette
        }

    def plot_feature_importance(
            self,
            X: np.ndarray,
            labels: np.ndarray,
            feature_names: List[str] = None,
            methods: List[str] = None,
            top_n: int = 15,
            figsize: tuple = (15, 5)
    ) -> plt.Figure:
        """
        Vẽ mức độ quan trọng của đặc trưng với nhiều phương pháp
        """
        if methods is None:
            methods = ['anova', 'silhouette']

        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

        n_methods = len(methods)
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)

        if n_methods == 1:
            axes = [axes]

        for idx, method in enumerate(methods):
            ax = axes[idx]

            try:
                # Tính mức độ quan trọng
                results = self.calculate_feature_importance(X, labels, method)
                importance_dict = results.get('importance_scores', {})

                if not importance_dict:
                    ax.text(0.5, 0.5, f'Không có dữ liệu cho {method}',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{method.upper()} Feature Importance')
                    continue

                # Chuyển về danh sách đã sắp xếp
                sorted_features = sorted(importance_dict.items(),
                                         key=lambda x: x[1],
                                         reverse=True)

                # Lấy top_n
                top_n_actual = min(top_n, len(sorted_features))
                top_features = sorted_features[:top_n_actual]

                # Lấy tên và điểm
                top_names = [feature_names[i] for i, _ in top_features]
                top_scores = [score for _, score in top_features]

                # Vẽ biểu đồ
                colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_scores)))
                bars = ax.barh(range(len(top_scores)), top_scores, color=colors)

                ax.set_yticks(range(len(top_scores)))
                ax.set_yticklabels(top_names)
                ax.set_xlabel('Importance Score')
                ax.set_title(f'{method.upper()} Feature Importance')
                ax.invert_yaxis()

                # Thêm nhãn giá trị
                if len(top_scores) > 0:
                    max_score = max(top_scores)
                    for i, (bar, score) in enumerate(zip(bars, top_scores)):
                        ax.text(score + 0.01 * max_score, i, f'{score:.3f}',
                                va='center', fontsize=9)

            except Exception as e:
                ax.text(0.5, 0.5, f'Lỗi: {str(e)[:50]}...',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{method.upper()} (Error)')

        plt.suptitle(f'Feature Importance Analysis (n_clusters={len(np.unique(labels))})',
                     fontsize=14, y=1.02)
        plt.tight_layout()

        return fig

    def create_comprehensive_report(
            self,
            X: np.ndarray,
            labels: np.ndarray,
            feature_names: List[str] = None,
            output_dir: str = '.'
    ) -> pd.DataFrame:
        """
        Tạo báo cáo đầy đủ về mức độ quan trọng của đặc trưng
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

        # Tính mức độ quan trọng bằng tất cả phương pháp
        methods = ['anova', 'silhouette']
        results = {}

        for method in methods:
            try:
                method_results = self.calculate_feature_importance(X, labels, method)
                importance_scores = method_results.get('importance_scores', {})

                # Lưu điểm số
                for feature_idx, score in importance_scores.items():
                    if feature_idx < len(feature_names):
                        if feature_idx not in results:
                            results[feature_idx] = {'feature': feature_names[feature_idx]}
                        results[feature_idx][f'{method}_score'] = float(score)

                        # Với ANOVA, lưu thêm p-value
                        if method == 'anova' and 'p_values' in method_results:
                            p_val = method_results['p_values'].get(feature_idx, 1.0)
                            results[feature_idx]['anova_p_value'] = float(p_val)

            except Exception as e:
                print(f"  Cảnh báo: Không tính được {method}: {e}")

        # Tạo DataFrame kết quả
        if results:
            df = pd.DataFrame.from_dict(results, orient='index')

            # Tính điểm tổng hợp
            score_columns = [col for col in df.columns if 'score' in col and col != 'anova_p_value']
            if score_columns:
                # Chuẩn hóa từng cột điểm
                for col in score_columns:
                    col_data = df[col]
                    if col_data.max() > col_data.min():
                        df[f'{col}_norm'] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                    else:
                        df[f'{col}_norm'] = 0.5

                # Tính điểm trung bình sau chuẩn hóa
                norm_cols = [f'{col}_norm' for col in score_columns]
                if norm_cols:
                    df['combined_importance'] = df[norm_cols].mean(axis=1)
                    df = df.sort_values('combined_importance', ascending=False)

            # Lưu ra CSV
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'unsupervised_feature_importance.csv')
            df.to_csv(output_path, index=False)

            print(f"   Đã lưu mức độ quan trọng đặc trưng (không giám sát) tại: {output_path}")

            return df
        else:
            print("  Không tạo được kết quả mức độ quan trọng của đặc trưng")
            return pd.DataFrame()

    # ====== HELPER FUNCTIONS ======
    def _select_top_features(self, data: pd.DataFrame, labels: np.ndarray, n_features: int) -> List[str]:
        """Chọn n đặc trưng tốt nhất dựa trên giá trị F của ANOVA"""
        try:
            # Kiểm tra điều kiện đầu vào
            if len(np.unique(labels)) <= 1:
                # Chỉ có 1 cụm, không thể tính ANOVA
                print(" Chỉ có 1 cụm, sử dụng phương sai thay thế")
                variances = data.var().sort_values(ascending=False)
                return variances.head(n_features).index.tolist()

            # Kiểm tra mỗi cụm có ít nhất 2 điểm
            unique_labels, counts = np.unique(labels, return_counts=True)
            if any(count < 2 for count in counts):
                print(" Có cụm có dưới 2 điểm, sử dụng phương sai thay thế")
                variances = data.var().sort_values(ascending=False)
                return variances.head(n_features).index.tolist()

            # Tính ANOVA an toàn
            try:
                selector = SelectKBest(f_classif, k=min(n_features, data.shape[1]))
                selector.fit(data, labels)

                # Lấy chỉ số đặc trưng được chọn
                selected_indices = selector.get_support(indices=True)
                return [data.columns[i] for i in selected_indices]

            except Exception as e:
                print(f" ANOVA gặp lỗi: {e}. Sử dụng phương sai thay thế.")
                variances = data.var().sort_values(ascending=False)
                return variances.head(n_features).index.tolist()

        except Exception as e:
            print(f" Chọn đặc trưng thất bại: {e}. Sử dụng toàn bộ đặc trưng.")
            return data.columns[:min(n_features, data.shape[1])].tolist()

    def get_config(self) -> Dict[str, Any]:
        return {
            'visualization_style': 'seaborn-darkgrid',
            'random_state': self.random_state,
            'report_dir': self.report_dir
        }
