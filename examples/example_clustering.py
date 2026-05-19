# examples/run_clustering_complete.py
"""
Ví dụ hoàn chỉnh chạy Clustering Pipeline với tất cả components
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Thêm đường dẫn gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.clustering import (
        DataPreprocessor,
        ModelComparator,
        ClusteringVisualizer,
        ModelManager,
        ClusteringPipeline
    )

    print(" Import clustering components thành công!")
except ImportError as e:
    print(f" Lỗi import clustering components: {e}")
    sys.exit(1)


def create_customer_clustering_data(n_samples=500):
    """Tạo dữ liệu khách hàng mẫu cho clustering"""
    np.random.seed(42)

    # Tạo 3 cụm khách hàng rõ ràng
    # Cụm 1: Khách hàng trẻ, thu nhập thấp
    n1 = n_samples // 3
    cluster1 = pd.DataFrame({
        'age': np.random.normal(25, 3, n1),
        'income': np.random.normal(30000, 5000, n1),
        'spending': np.random.normal(500, 100, n1),
        'frequency': np.random.poisson(2, n1),
        'true_cluster': 0
    })

    # Cụm 2: Khách hàng trung niên, thu nhập trung bình
    n2 = n_samples // 3
    cluster2 = pd.DataFrame({
        'age': np.random.normal(40, 4, n2),
        'income': np.random.normal(60000, 10000, n2),
        'spending': np.random.normal(1500, 300, n2),
        'frequency': np.random.poisson(5, n2),
        'true_cluster': 1
    })

    # Cụm 3: Khách hàng cao cấp
    n3 = n_samples - n1 - n2
    cluster3 = pd.DataFrame({
        'age': np.random.normal(50, 5, n3),
        'income': np.random.normal(100000, 20000, n3),
        'spending': np.random.normal(5000, 1000, n3),
        'frequency': np.random.poisson(8, n3),
        'true_cluster': 2
    })

    # Kết hợp các cụm
    data = pd.concat([cluster1, cluster2, cluster3], ignore_index=True)

    # Thêm ID khách hàng
    data['customer_id'] = range(1, len(data) + 1)

    # Xáo trộn dữ liệu
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    return data

def example_full_pipeline():
    """Ví dụ 5: Sử dụng ClusteringPipeline hoàn chỉnh"""
    print("\n" + "=" * 60)
    print("VÍ DỤ 5: FULL CLUSTERING PIPELINE")
    print("=" * 60)

    # 1. Tạo dữ liệu
    print("\n1. Tạo dữ liệu mẫu...")
    data = create_customer_clustering_data(500)

    # 2. Khởi tạo pipeline
    print("\n2. Khởi tạo ClusteringPipeline...")
    pipeline = ClusteringPipeline(
        experiment_name="Customer_Segmentation_Analysis",
        random_state=42
    )

    # 3. Định nghĩa cấu hình thuật toán
    print("\n3. Thiết lập cấu hình thuật toán...")
    algorithms_config = {
        'kmeans': {
            'param_grid': {
                'n_clusters': [3, 4, 5],
                'init': ['k-means++'],
                'n_init': [10]
            }
        },
        'gmm': {
            'param_grid': {
                'n_components': [3, 4, 5],
                'covariance_type': ['full', 'diag']
            }
        }
    }

    # 4. Chạy full pipeline
    print("\n4. Chạy toàn bộ pipeline...")
    try:
        results = pipeline.run_full_pipeline(
            data=data,
            algorithms_config=algorithms_config,
            target_column='true_cluster',
            drop_columns=['customer_id'],
            scale_method='standard',
            use_pca=True,
            pca_variance=0.95
        )

        print("\n Pipeline chạy thành công!")

        # 5. Visualization
        print("\n5. Tạo visualizations...")
        if hasattr(pipeline, 'best_labels'):
            visualizer = ClusteringVisualizer(random_state=42)
            features_data = data[['age', 'income', 'spending', 'frequency']]

            # Plot cluster sizes
            visualizer.plot_cluster_sizes(pipeline.best_labels)

            # Plot heatmap
            visualizer.plot_heatmap(
                data=features_data,
                labels=pipeline.best_labels,
                n_features=4
            )

        # 6. Lưu kết quả
        print("\n6. Lưu kết quả...")
        manager = ModelManager(random_state=42)

        # Lưu predictions
        if hasattr(pipeline, 'best_labels'):
            manager.save_predictions(
                predictions=pipeline.best_labels,
                original_data=data,
                output_path='final_predictions.csv'
            )

    except Exception as e:
        print(f" Lỗi khi chạy pipeline: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Hàm chính chạy tất cả ví dụ"""
    print("=" * 70)
    print("CLUSTERING PIPELINE - DEMONSTRATION")
    print("=" * 70)
    example_full_pipeline()
    # Dọn dẹp file tạm
    print("\n Dọn dẹp file tạm...")
    temp_files = [
        'clustering_predictions.csv',
        'model_card.json',
        'final_predictions.csv',
        'custom_model_card.json',
        'custom_predictions.csv'
    ]

    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"   Đã xóa: {file}")
            except:
                pass

    # Xóa thư mục saved_models nếu rỗng
    if os.path.exists('saved_models'):
        try:
            if not os.listdir('saved_models'):
                os.rmdir('saved_models')
                print("   Đã xóa thư mục saved_models")
        except:
            pass

    print("\n" + "=" * 70)
    print("HOÀN THÀNH DEMONSTRATION")
    print("=" * 70)


if __name__ == "__main__":
    # Thiết lập matplotlib
    plt.style.use('seaborn-v0_8-darkgrid')

    # Chạy chương trình
    main()