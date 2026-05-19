import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path


class DataPreprocessorPipeline:
    """
    Pipeline tổng hợp tất cả các processor
    Quản lý và chạy tự động toàn bộ quá trình xử lý dữ liệu
    """

    def __init__(self, data: Optional[pd.DataFrame] = None, output_dir: str = "pipeline_output"):
        """
        Khởi tạo pipeline với dữ liệu

        Tham số:
        -----------
        data : pandas.DataFrame, optional
            Dữ liệu đầu vào
        output_dir : str
            Thư mục lưu kết quả
        """
        self.data = data
        self.original_data = data.copy() if data is not None else None
        self.processed_data = None
        self.processing_history = []

        # Thêm thuộc tính output_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Import các processor (lazy import để tránh lỗi)
        self.DataAnalyzer = None
        self.MissingValueHandler = None
        self.OutlierHandler = None
        self.FeatureScaler = None
        self.CategoricalEncoder = None
        self.FeatureEngineer = None

        print(" Pipeline đã được khởi tạo")
        print(f"   Output directory: {self.output_dir}")

    def _import_processors(self):
        """Import các processor khi cần"""
        if self.DataAnalyzer is None:
            try:
                from .data_analyzer import DataAnalyzer
                from .missing_handler import MissingValueHandler
                from .outlier_handler import OutlierHandler
                from .feature_scaler import FeatureScaler
                from .categorical_encoder import CategoricalEncoder
                from .feature_engineer import FeatureEngineer

                self.DataAnalyzer = DataAnalyzer
                self.MissingValueHandler = MissingValueHandler
                self.OutlierHandler = OutlierHandler
                self.FeatureScaler = FeatureScaler
                self.CategoricalEncoder = CategoricalEncoder
                self.FeatureEngineer = FeatureEngineer
                print("   Đã import tất cả processors")

            except ImportError as e:
                # Nếu không import được, tạo các class đơn giản
                print(f"    Không thể import processors: {e}")
                print("   Cần đảm bảo các module sau đã được tạo:")
                print("   - data_analyzer.py")
                print("   - missing_handler.py")
                print("   - outlier_handler.py")
                print("   - feature_scaler.py")
                print("   - categorical_encoder.py")
                print("   - feature_engineer.py")

    def _log_step(self, step_name: str, details: Dict):
        """Ghi log cho từng bước xử lý"""
        log_entry = {
            'step': step_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'details': details
        }
        self.processing_history.append(log_entry)

    def auto_process(self, config: Optional[Dict] = None) -> 'DataPreprocessorPipeline':
        """
        Tự động xử lý toàn bộ dữ liệu
        Chạy pipeline: Analyze → Missing → Outliers → Encode → Scale

        Tham số:
        -----------
        config : dict, optional
            Cấu hình xử lý

        Returns:
        --------
        DataPreprocessorPipeline
        """
        if self.data is None:
            print("   Chưa có dữ liệu")
            return self

        print("\n" + "=" * 60)
        print("   BẮT ĐẦU XỬ LÝ TỰ ĐỘNG")
        print("=" * 60)

        # Import processors nếu cần
        self._import_processors()

        # Cấu hình mặc định
        if config is None:
            config = {
                'missing_strategy': 'auto',
                'outlier_method': 'auto',
                'outlier_treatment': 'auto',
                'encode_method': 'auto',
                'scale_method': 'standard',
                'datetime_features': True
            }

        original_shape = self.data.shape
        data = self.data.copy()

        print(f"  Dữ liệu ban đầu: {original_shape}")

        # =============== BƯỚC 1: PHÂN TÍCH DỮ LIỆU ===============
        print("\n1.  PHÂN TÍCH DỮ LIỆU VÀ TẠO BÁO CÁO")

        # Kiểm tra xem DataAnalyzer đã import chưa
        if self.DataAnalyzer is None:
            print("    Không thể thực hiện phân tích vì DataAnalyzer chưa được import")
            self._log_step('analyze', {'error': 'DataAnalyzer not imported'})
        else:
            analyzer = self.DataAnalyzer(data)

            try:
                # Tạo thư mục cho báo cáo
                report_dir = self.output_dir / "analysis_reports"
                report_dir.mkdir(exist_ok=True)

                print("   - Thực hiện phân tích và tạo báo cáo...")

                # Gọi generate_report() để làm cả hai việc
                analysis = analyzer.process(output_dir=str(report_dir))

                # Log thông tin
                self._log_step('analyze', {
                    'shape': data.shape,
                    'report_created': True,
                    'report_dir': str(report_dir),
                    'analysis_summary': {
                        'rows': analysis['shape'][0],
                        'cols': analysis['shape'][1],
                        'missing': sum(stats['count'] for stats in analysis['missing_values'].values())
                    }
                })

                print(f"    Đã tạo báo cáo tại: {report_dir}")

            except Exception as e:
                print(f"    Không thể tạo báo cáo đầy đủ: {e}")
                print("   -> Thực hiện phân tích cơ bản...")

                # Fallback: chỉ phân tích cơ bản
                analysis = analyzer.analyze()
                self._log_step('analyze_basic', {'shape': data.shape, 'analysis': analysis})

                print("    Đã thực hiện phân tích cơ bản")

        # =============== BƯỚC 2: XỬ LÝ MISSING VALUES ===============
        print("\n2.  XỬ LÝ MISSING VALUES")
        if self.MissingValueHandler is None:
            print("    Bỏ qua vì MissingValueHandler chưa được import")
            self._log_step('handle_missing', {'skipped': True})
        else:
            missing_handler = self.MissingValueHandler(data)
            missing_handler.process(strategy=config['missing_strategy'])
            data = missing_handler.data
            self._log_step('handle_missing', {
                'strategy': config['missing_strategy'],
                'shape_before': original_shape,
                'shape_after': data.shape
            })
            print(f"   Đã xử lý missing values")

        # =============== BƯỚC 3: XỬ LÝ OUTLIERS ===============
        print("\n3.  XỬ LÝ OUTLIERS")
        if self.OutlierHandler is None:
            print("    Bỏ qua vì OutlierHandler chưa được import")
            self._log_step('handle_outliers', {'skipped': True})
        else:
            outlier_handler = self.OutlierHandler(data)
            outlier_handler.process(
                method=config['outlier_method'],
                treatment=config['outlier_treatment']
            )
            data = outlier_handler.data
            self._log_step('handle_outliers', {
                'method': config['outlier_method'],
                'treatment': config['outlier_treatment'],
                'shape': data.shape
            })
            print(f"   Đã xử lý outliers")

        # =============== BƯỚC 4: MÃ HÓA CATEGORICAL ===============
        print("\n4.  MÃ HÓA CATEGORICAL")
        if self.CategoricalEncoder is None:
            print("    Bỏ qua vì CategoricalEncoder chưa được import")
            self._log_step('encode_categorical', {'skipped': True})
        else:
            encoder = self.CategoricalEncoder(data)
            encoder.process(method=config['encode_method'])
            data = encoder.data
            self._log_step('encode_categorical', {
                'method': config['encode_method'],
                'shape': data.shape
            })
            print(f"   Đã mã hóa categorical")

        # =============== BƯỚC 5: TẠO FEATURES MỚI ===============
        if config.get('datetime_features', True):
            print("\n5.  TẠO FEATURES MỚI")
            if self.FeatureEngineer is None:
                print("    Bỏ qua vì FeatureEngineer chưa được import")
                self._log_step('create_features', {'skipped': True})
            else:
                # Tìm cột datetime
                datetime_cols = data.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
                if datetime_cols:
                    feature_engineer = self.FeatureEngineer(data)
                    for col in datetime_cols[:2]:  # Chỉ xử lý 2 cột đầu
                        feature_engineer.process(col)
                    data = feature_engineer.data
                    self._log_step('create_features', {
                        'datetime_columns': datetime_cols,
                        'shape': data.shape
                    })
                    print(f"   Đã tạo features từ {len(datetime_cols)} datetime columns")
                else:
                    print("    Không tìm thấy datetime columns")
                    self._log_step('create_features', {'skipped': 'No datetime columns'})

        # =============== BƯỚC 6: CHUẨN HÓA DỮ LIỆU ===============
        print("\n6.  CHUẨN HÓA DỮ LIỆU")
        if self.FeatureScaler is None:
            print("    Bỏ qua vì FeatureScaler chưa được import")
            self._log_step('scale_data', {'skipped': True})
        else:
            scaler = self.FeatureScaler(data)
            scaler.process(method=config['scale_method'])
            data = scaler.data
            self._log_step('scale_data', {
                'method': config['scale_method'],
                'shape': data.shape
            })
            print(f"   Đã chuẩn hóa dữ liệu")

        # Lưu kết quả
        self.processed_data = data

        # Lưu dữ liệu đã xử lý
        processed_file = self.output_dir / "processed_data.csv"
        data.to_csv(processed_file, index=False)
        print(f"   Đã lưu dữ liệu đã xử lý: {processed_file}")

        print("\n" + "=" * 60)
        print("   XỬ LÝ HOÀN THÀNH!")
        print("=" * 60)
        print(f"   Trước: {original_shape}")
        print(f"   Sau: {data.shape}")
        print(f"   Đã thực hiện: {len(self.processing_history)} bước")
        print(f"   Kết quả lưu tại: {self.output_dir}")

        # Lưu log xử lý
        self._save_processing_log()

        return self

    def _save_processing_log(self):
        """Lưu log xử lý ra file"""
        import json
        from datetime import datetime

        log_file = self.output_dir / f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        log_data = {
            'pipeline_info': {
                'created_at': datetime.now().isoformat(),
                'original_shape': self.original_data.shape if self.original_data is not None else None,
                'processed_shape': self.processed_data.shape if self.processed_data is not None else None,
                'output_dir': str(self.output_dir)
            },
            'processing_history': self.processing_history
        }

        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"    Đã lưu processing log: {log_file}")
        except Exception as e:
            print(f"    Không thể lưu processing log: {e}")

    def get_data(self) -> pd.DataFrame:
        """
        Lấy dữ liệu đã xử lý

        Returns:
        --------
        pandas.DataFrame
        """
        if self.processed_data is not None:
            return self.processed_data
        return self.data

    def to_csv(self, filepath: str = None, index: bool = False):
        """
        Xuất dữ liệu đã xử lý ra file CSV

        Tham số:
        -----------
        filepath : str, optional
            Đường dẫn file (mặc định: output_dir/processed_data.csv)
        index : bool
            Có lưu index không
        """
        data_to_save = self.get_data()

        if data_to_save is None:
            print("   Không có dữ liệu để xuất")
            return

        if filepath is None:
            # Tạo tên file mặc định
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"processed_data_{timestamp}.csv"
        else:
            filepath = Path(filepath)

        try:
            data_to_save.to_csv(filepath, index=index)
            print(f"    Đã xuất dữ liệu ra: {filepath}")
            print(f"    Kích thước: {data_to_save.shape}")
            print(f"    Thư mục: {filepath.parent}")
        except Exception as e:
            print(f"    Lỗi khi xuất file: {str(e)}")
