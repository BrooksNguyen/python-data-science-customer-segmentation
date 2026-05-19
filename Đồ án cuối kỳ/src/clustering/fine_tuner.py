import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

try:
    import optuna

    OPTUNA_AVAILABLE = True
    print(" Đã nhập Optuna thành công")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("  Không tìm thấy Optuna. Sẽ dùng phương án dự phòng cho tối ưu Bayesian.")
    print(" Cài đặt bằng lệnh: pip install optuna")

from .base import BaseClusteringComponent


class ClusteringFineTuner(BaseClusteringComponent):
    """Tinh chỉnh nâng cao cho các mô hình phân cụm"""

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.best_params = {}
        self.best_score = -1
        self.optimization_history = []
        self.feature_importance = None
        self.has_optuna = OPTUNA_AVAILABLE

    def bayesian_optimization(
            self,
            X: np.ndarray,
            algorithm: str = 'kmeans',
            n_trials: int = 50,
            param_space: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Tối ưu Bayesian cho siêu tham số phân cụm

        Tham s?:
            X: Ma trận đặc trưng
            algorithm: 'kmeans', 'gmm' hoặc 'agglo'
            n_trials: Số lần thử tối ưu
            param_space: Không gian tham số tự định nghĩa

        Tr? v?:
            Bộ tham số tốt nhất tìm được
        """
        print("\n" + "=" * 60)
        print(" TỐI ƯU BAYESIAN")
        print("=" * 60)

        # Kiểm tra optuna có sẵn không
        if not self.has_optuna:
            print("  Chưa cài đặt Optuna!")
            print("Đang dùng GridSearch thay thế...")
            return self._grid_search_fallback(X, algorithm, n_trials)

        from sklearn.cluster import KMeans
        from sklearn.mixture import GaussianMixture

        if param_space is None:
            param_space = self._get_default_param_space(algorithm)

        def objective(trial):
            # Đề xuất tham số
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )

            # Tạo và huấn luyện mô hình
            try:
                if algorithm == 'kmeans':
                    model = KMeans(
                        n_clusters=params.get('n_clusters', 3),
                        init=params.get('init', 'k-means++'),
                        n_init=params.get('n_init', 10),
                        max_iter=params.get('max_iter', 300),
                        random_state=self.random_state
                    )
                elif algorithm == 'gmm':
                    model = GaussianMixture(
                        n_components=params.get('n_components', 3),
                        covariance_type=params.get('covariance_type', 'full'),
                        max_iter=params.get('max_iter', 100),
                        random_state=self.random_state
                    )
                else:
                    return 0.0

                # Huấn luyện và dự đoán
                labels = model.fit_predict(X)

                # Tính điểm đánh giá (cao hơn là tốt hơn)
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X, labels)
                else:
                    score = 0.0

                # Lưu thông tin từng lần thử
                trial_info = {
                    'params': params.copy(),
                    'score': score,
                    'algorithm': algorithm
                }
                self.optimization_history.append(trial_info)

                return score

            except Exception as e:
                return 0.0

        # Chạy tối ưu với Optuna
        print(f"Đang chạy tối ưu Bayesian bằng Optuna...")
        print(f"Thuật toán: {algorithm}")
        print(f"Số lần thử: {n_trials}")

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Lưu kết quả tốt nhất
        self.best_params = study.best_params
        self.best_score = study.best_value

        print(f"\n Đã hoàn tất tối ưu Bayesian")
        print(f"   Điểm tốt nhất: {self.best_score:.4f}")
        print(f"   Tham số tốt nhất: {self.best_params}")
        print("=" * 60)

        return self.best_params

    def _grid_search_fallback(self, X, algorithm, n_trials):
        """Phương án dự phòng bằng GridSearch khi không có Optuna"""
        print(f"Sử dụng GridSearch thay thế cho {algorithm}")

        from sklearn.cluster import KMeans
        from sklearn.mixture import GaussianMixture
        from sklearn.model_selection import ParameterGrid

        # Dùng không gian tham số mặc định
        param_space = self._get_default_param_space(algorithm)

        # Chuyển không gian tham số sang định dạng ParameterGrid
        param_grid = {}
        for param, config in param_space.items():
            if config['type'] == 'int':
                # Tạo dải giá trị int với số lượng hợp lý
                low, high = config['low'], config['high']
                step = max(1, (high - low) // min(n_trials, 5))
                param_grid[param] = list(range(low, high + 1, step))
            elif config['type'] == 'categorical':
                param_grid[param] = config['choices']
            elif config['type'] == 'float':
                # Tạo 3-5 giá trị cho biến float
                low, high = config['low'], config['high']
                if config.get('log', False):
                    param_grid[param] = np.logspace(np.log10(low), np.log10(high), num=min(5, n_trials)).tolist()
                else:
                    param_grid[param] = np.linspace(low, high, num=min(5, n_trials)).tolist()

        best_score = -1
        best_params = {}

        print(f"Cấu hình GridSearch: {param_grid}")

        # Thực hiện GridSearch đơn giản
        for params in ParameterGrid(param_grid):
            try:
                if algorithm == 'kmeans':
                    model = KMeans(**params, random_state=self.random_state)
                elif algorithm == 'gmm':
                    model = GaussianMixture(**params, random_state=self.random_state)
                elif algorithm == 'agglo':
                    from sklearn.cluster import AgglomerativeClustering
                    model = AgglomerativeClustering(**params)
                else:
                    continue

                if algorithm == 'agglo':
                    labels = model.fit_predict(X)
                else:
                    labels = model.fit_predict(X)

                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X, labels)

                    # Lưu thông tin từng lần thử
                    trial_info = {
                        'params': params.copy(),
                        'score': score,
                        'algorithm': algorithm
                    }
                    self.optimization_history.append(trial_info)

                    if score > best_score:
                        best_score = score
                        best_params = params

                        print(f"  Kết quả tốt hơn: {params} | Điểm: {score:.4f}")

            except Exception as e:
                continue

        self.best_params = best_params
        self.best_score = best_score

        print(f"\n Đã hoàn tất GridSearch")
        print(f"   Điểm tốt nhất: {self.best_score:.4f}")
        print(f"   Tham số tốt nhất: {self.best_params}")
        print("=" * 60)

        return self.best_params

    def _get_default_param_space(self, algorithm: str) -> Dict:
        """Trả về không gian tham số mặc định cho từng thuật toán"""

        if algorithm == 'kmeans':
            return {
                'n_clusters': {'type': 'int', 'low': 2, 'high': 10},
                'init': {'type': 'categorical', 'choices': ['k-means++', 'random']},
                'n_init': {'type': 'int', 'low': 5, 'high': 20},
                'max_iter': {'type': 'int', 'low': 100, 'high': 500}
            }

        elif algorithm == 'gmm':
            return {
                'n_components': {'type': 'int', 'low': 2, 'high': 10},
                'covariance_type': {
                    'type': 'categorical',
                    'choices': ['full', 'tied', 'diag', 'spherical']
                },
                'max_iter': {'type': 'int', 'low': 50, 'high': 200},
                'reg_covar': {'type': 'float', 'low': 1e-6, 'high': 1e-1, 'log': True}
            }

        else:  # agglo
            return {
                'n_clusters': {'type': 'int', 'low': 2, 'high': 10},
                'linkage': {
                    'type': 'categorical',
                    'choices': ['ward', 'complete', 'average', 'single']
                },
                'affinity': {
                    'type': 'categorical',
                    'choices': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
                }
            }

    def get_config(self) -> Dict[str, Any]:
        return {
            'has_optuna': self.has_optuna,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_trials': len(self.optimization_history),
            'random_state': self.random_state
        }
