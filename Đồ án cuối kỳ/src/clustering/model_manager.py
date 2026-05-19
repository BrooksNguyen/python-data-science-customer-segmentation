import joblib
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from .base import BaseClusteringComponent

class ModelManager(BaseClusteringComponent):
    """Quản lý lưu, tải và triển khai mô hình phân cụm"""
    
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.model_metadata = {}
    
    def save_model(
        self,
        model: Any,
        preprocessor: Any,
        metadata: Dict[str, Any],
        directory: str = "saved_models",
        model_name: str = "clustering_model"
    ) -> str:
        """
        Lưu toàn bộ pipeline phân cụm

        Tham s?:
            model: Mô hình phân cụm đã huấn luyện
            preprocessor: Đối tượng tiền xử lý dữ liệu
            metadata: Thông tin mô tả bổ sung
            directory: Thư mục lưu mô hình
            model_name: Tên mô hình

        Tr? v?:
            Đường dẫn tới mô hình đã lưu
        """
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(directory, exist_ok=True)
        
        # Sinh timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(directory, f"{model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Lưu mô hình
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(model, model_path)
        
        # Lưu bộ tiền xử lý
        preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        joblib.dump(preprocessor, preprocessor_path)
        
        # Lưu metadata
        metadata.update({
            'saved_at': timestamp,
            'model_type': type(model).__name__,
            'model_path': model_path,
            'preprocessor_path': preprocessor_path
        })
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f" Đã lưu pipeline tại: {model_dir}")
        return model_dir
    
    def load_model(self, model_dir: str) -> tuple:
        """
        Tải mô hình đã lưu

        Tham s?:
            model_dir: Thư mục chứa mô hình đã lưu

        Tr? v?:
            Bộ giá trị (model, preprocessor, metadata)
        """
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Tải mô hình
        model_path = os.path.join(model_dir, "model.pkl")
        model = joblib.load(model_path)
        
        # Tải bộ tiền xử lý
        preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        preprocessor = joblib.load(preprocessor_path)
        
        # Tải metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f" Đã tải pipeline từ: {model_dir}")
        return model, preprocessor, metadata
    
    def save_predictions(
        self,
        predictions: np.ndarray,
        original_data: pd.DataFrame,
        output_path: str = "predictions.csv",
        include_features: bool = True
    ):
        """
        Lưu kết quả dự đoán cụm

        Tham s?:
            predictions: Nhãn cụm
            original_data: DataFrame gốc
            output_path: Đường dẫn lưu kết quả
            include_features: Có kèm đặc trưng gốc hay không
        """
        if include_features:
            result_df = original_data.copy()
            result_df['cluster'] = predictions
        else:
            result_df = pd.DataFrame({
                'cluster': predictions
            })
        
        result_df.to_csv(output_path, index=False)
        print(f" Đã lưu dự đoán tại: {output_path}")
        
        # In phân bố cụm
        cluster_counts = pd.Series(predictions).value_counts().sort_index()
        print("\n Phân bố cụm:")
        for cluster, count in cluster_counts.items():
            percentage = count / len(predictions) * 100
            print(f"   Cụm {cluster}: {count} mẫu ({percentage:.1f}%)")
    
    def export_model_card(
        self,
        model: Any,
        metrics: Dict[str, float],
        features: List[str],
        output_path: str = "model_card.json"
    ):
        """
        Xuất model card với chỉ số hiệu năng và metadata

        Tham s?:
            model: Mô hình đã huấn luyện
            metrics: Bộ chỉ số đánh giá
            features: Danh sách đặc trưng sử dụng
            output_path: Đường dẫn lưu model card
        """
        model_card = {
            'model_info': {
                'type': type(model).__name__,
                'parameters': self._extract_model_params(model),
                'features_used': features,
                'created_at': datetime.now().isoformat()
            },
            'performance_metrics': metrics,
            'interpretation': self._generate_interpretation(metrics)
        }
        
        with open(output_path, 'w') as f:
            json.dump(model_card, f, indent=4)
        
        print(f" Đã lưu model card tại: {output_path}")
    
    def _extract_model_params(self, model: Any) -> Dict[str, Any]:
        """Trích xuất tham số mô hình"""
        params = model.get_params() if hasattr(model, 'get_params') else {}
        return params
    
    def _generate_interpretation(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Diễn giải ý nghĩa các chỉ số"""
        interpretation = {}
        
        if 'silhouette' in metrics:
            score = metrics['silhouette']
            if score > 0.7:
                interpretation['silhouette'] = "Phân tách cụm rất rõ ràng"
            elif score > 0.5:
                interpretation['silhouette'] = "Phân tách cụm khá tốt"
            elif score > 0.25:
                interpretation['silhouette'] = "Phân tách cụm yếu"
            else:
                interpretation['silhouette'] = "Gần như không có sự phân tách"
        
        return interpretation
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'model_metadata': self.model_metadata,
            'random_state': self.random_state
        }
