import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from src.utils.config_loader import ConfigLoader  # <--- Import mới
from src.preprocessing.data_pipeline import DataPreprocessorPipeline
from src.clustering.trainer import ClusteringPipeline
from src.preprocessing.data_loader import DataLoader
from src.clustering.visualizer import ClusteringVisualizer


def main():
    # ====== 0. LOAD CONFIG ======
    print(" Đang tải cấu hình...")
    try:
        config_loader = ConfigLoader("config.yaml")
    except Exception as e:
        print(f"Lỗi: {e}")
        print("Đang sử dụng cấu hình mặc định hoặc dừng chương trình.")
        return

    # Lấy các tham số từ config
    DATA_PATH = config_loader.get("data.raw_path", "data/marketing_campaign.csv")
    DATETIME_COL = config_loader.get("data.datetime_column", [])
    DROP_COLS = config_loader.get("data.drop_columns", [])

    # ====== 1. ĐỌC DỮ LIỆU ======
    print(f"\n Đọc dữ liệu từ: {DATA_PATH}")
    try:
        data_loader = DataLoader.from_file(DATA_PATH)
        data = data_loader.get_data()
    except Exception as e:
        print(f" Lỗi đọc file: {e}")
        return

    # Chuyển cột datetime
    if DATETIME_COL in data.columns:
        print(f"   Converting datetime column: {DATETIME_COL}")
        data[DATETIME_COL] = pd.to_datetime(data[DATETIME_COL], errors='coerce')

    # ====== 2. PREPROCESSING ======
    print("\n  Chạy preprocessing pipeline...")

    preprocessor = DataPreprocessorPipeline(
        data=data,
        output_dir=config_loader.get("preprocessing.output_dir", "preprocessing_output")
    )

    # Lấy tham số preprocessing từ config
    process_config = {
        'missing_strategy': config_loader.get("preprocessing.missing_strategy"),
        'outlier_method': config_loader.get("preprocessing.outlier_method"),
        'outlier_treatment': config_loader.get("preprocessing.outlier_treatment"),
        'encode_method': config_loader.get("preprocessing.encode_method"),
        'scale_method': config_loader.get("preprocessing.scale_method"),
        'datetime_features': config_loader.get("preprocessing.create_datetime_features")
    }

    preprocessor.auto_process(config=process_config)
    cleaned_data = preprocessor.get_data()

    # ====== 3. CHỌN FEATURES ======
    print("\n Chuẩn bị dữ liệu clustering...")
    clustering_data = cleaned_data.copy()

    # Drop các cột định nghĩa trong config
    cols_to_drop = [c for c in DROP_COLS if c in clustering_data.columns]
    if cols_to_drop:
        print(f"   Loại bỏ cột: {cols_to_drop}")
        clustering_data.drop(columns=cols_to_drop, inplace=True)

    # Lọc lại lần cuối để chỉ lấy cột số (quan trọng cho PCA)
    clustering_data = clustering_data.select_dtypes(include=[np.number, 'bool'])

    # ====== 4. CLUSTERING PIPELINE ======
    print("\n Chạy clustering pipeline...")

    clustering_pipeline = ClusteringPipeline(
        experiment_name=config_loader.get("project.name"),
        random_state=config_loader.get("project.random_state")
    )

    # Lấy cấu hình thuật toán từ yaml
    algos_config = {}
    yaml_algos = config_loader.get("clustering.algorithms", {})

    # Chỉ thêm thuật toán nào có enabled: true
    for name, cfg in yaml_algos.items():
        if cfg.get('enabled', False):
            algos_config[name] = {'param_grid': cfg['param_grid']}

    # Chạy pipeline
    results = clustering_pipeline.run_full_pipeline(
        data=clustering_data,
        algorithms_config=algos_config,
        drop_columns=[],  # Đã drop ở trên rồi
        scale_method=config_loader.get("preprocessing.scale_method"),
        use_pca=config_loader.get("clustering.use_pca"),
        pca_variance=config_loader.get("clustering.pca_variance"),
        fine_tune_algorithm="kmeans" if config_loader.get("clustering.fine_tune") else None,
        export_results=config_loader.get("clustering.export_results"),
        report_output_dir=config_loader.get("clustering.report_output_dir")
    )

    # ====== 5. VISUALIZATION SAU CLUSTERING ======
    if results and results['best_labels'] is not None:
        print("\n" + "=" * 60)
        print("TỔNG HỢP KẾT QUẢ")
        print("=" * 60)

        best_labels = results['best_labels']
        best_model = results['best_model']

        original_mapped = data.loc[cleaned_data.index].copy()
        original_mapped['Cluster'] = results['best_labels']

        print(f" Thuật toán tốt nhất: {results['best_algorithm']}")
        print(f" Silhouette Score: {results['best_score']:.4f}")

        # Thống kê nhanh
        print("\n Phân phối Cluster:")
        print(original_mapped['Cluster'].value_counts().sort_index())
        unique, counts = np.unique(best_labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            percent = count / len(best_labels) * 100
            print(f"   Cluster {cluster}: {count} KH ({percent:.1f}%)")

        # Tính toán thống kê từng cluster
        print("\n Thống kê từng cluster:")
        for cluster in unique:
            cluster_data = original_mapped[original_mapped['Cluster'] == cluster]

            if 'Income' in cluster_data.columns:
                avg_income = cluster_data['Income'].mean()
                print(f"Cluster {cluster}:")
                print(f"   • Số KH: {len(cluster_data)}")
                print(f"   • Income TB: ${avg_income:,.0f}")

                # Tính tổng chi tiêu nếu có các cột Mnt
                mnt_columns = [col for col in cluster_data.columns if 'Mnt' in col]
                if mnt_columns:
                    total_spending = cluster_data[mnt_columns].sum().sum()
                    avg_spending = total_spending / len(cluster_data)
                    print(f"   • Spending TB: ${avg_spending:,.0f}")

        # Cluster profile
        print("\n Cluster Profile:")
        if 'Education' in original_mapped.columns:
            for cluster in unique:
                cluster_data = original_mapped[original_mapped['Cluster'] == cluster]
                if 'Education' in cluster_data.columns:
                    top_edu = cluster_data['Education'].mode()[0] if not cluster_data[
                        'Education'].mode().empty else 'N/A'
                    print(f"Cluster {cluster}: Education phổ biến - {top_edu}")

        print("\n" + "=" * 60)
        print(" TẤT CẢ KẾT QUẢ ĐƯỢC LƯU TRONG:")
        print(f"• preprocessing_output/    - Kết quả preprocessing")
        print(f"• clustering_results/      - Kết quả clustering pipeline")
        print("=" * 60)

    else:
        print(" Clustering không thành công!")


if __name__ == "__main__":
    main()
