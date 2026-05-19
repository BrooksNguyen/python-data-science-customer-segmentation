import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from .base_processor import BaseProcessor
from datetime import datetime
from pathlib import Path

class DataAnalyzer(BaseProcessor):
    """Class chuyên phân tích dữ liệu"""

    def __init__(self, data: Optional[pd.DataFrame] = None):
        super().__init__(data, name="DataAnalyzer")

    def process(self, output_dir: str = "reports") -> Dict:
        """
        Tạo báo cáo phân tích chi tiết

        Args:
            output_dir: Thư mục lưu báo cáo

        Returns:
            Dict chứa báo cáo
        """
        return self.generate_report(output_dir)

    def analyze(self) -> Dict[str, Any]:
        """
        Phân tích toàn diện dữ liệu

        Returns:
            Dict chứa kết quả phân tích
        """
        if self.data is None:
            raise ValueError("Không có dữ liệu để phân tích")

        print("\n" + "=" * 70)
        print("PHÂN TÍCH DỮ LIỆU TOÀN DIỆN")
        print("=" * 70)

        analysis = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'data_types': self.data.dtypes.astype(str).to_dict(),
            'missing_values': self._get_missing_stats(),
            'numeric_stats': self._get_numeric_stats(),
            'categorical_stats': self._get_categorical_stats(),
            'memory_usage': self._get_memory_usage(),
            'duplicates': self._get_duplicate_info(),
            'correlation': self._get_correlation_info()
        }

        # Hiển thị tổng quan
        self._display_summary(analysis)

        self._log_action('analyze', {
            'summary': {
                'rows': analysis['shape'][0],
                'columns': analysis['shape'][1],
                'missing_total': sum(stats['count'] for stats in analysis['missing_values'].values()),
                'numeric_columns': len(analysis['numeric_stats']),
                'categorical_columns': len(analysis['categorical_stats'])
            }
        })

        return analysis

    def _get_missing_stats(self) -> Dict:
        """Thống kê missing values"""
        missing_stats = {}
        for col in self.data.columns:
            count = self.data[col].isnull().sum()
            percent = (count / len(self.data)) * 100
            missing_stats[col] = {
                'count': int(count),
                'percent': round(percent, 2),
                'dtype': str(self.data[col].dtype)
            }
        return missing_stats

    def _get_numeric_stats(self) -> Dict:
        """Thống kê cột numeric"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        stats = {}
        for col in numeric_cols:
            try:
                series = self.data[col].dropna()
                stats[col] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'median': float(series.median()),
                    'q1': float(series.quantile(0.25)),
                    'q3': float(series.quantile(0.75)),
                    'skew': float(series.skew()),
                    'kurtosis': float(series.kurtosis()),
                    'zeros': int((series == 0).sum()),
                    'zeros_percent': float((series == 0).sum() / len(series) * 100)
                }
            except:
                stats[col] = {}
        return stats

    def _get_categorical_stats(self) -> Dict:
        """Thống kê cột categorical"""
        cat_cols = self.data.select_dtypes(include=['object', 'category']).columns
        stats = {}
        for col in cat_cols:
            try:
                series = self.data[col].dropna()
                value_counts = series.value_counts()
                stats[col] = {
                    'unique_count': int(series.nunique()),
                    'missing_count': int(self.data[col].isnull().sum()),
                    'top_value': value_counts.index[0] if len(value_counts) > 0 else None,
                    'top_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'top_percent': float((value_counts.iloc[0] / len(series)) * 100) if len(value_counts) > 0 else 0,
                    'value_distribution': value_counts.head(10).to_dict()
                }
            except:
                stats[col] = {}
        return stats

    def _get_memory_usage(self) -> Dict:
        """Thống kê memory usage"""
        memory_usage = self.data.memory_usage(deep=True)
        return {
            'total_bytes': int(memory_usage.sum()),
            'total_mb': memory_usage.sum() / 1024 ** 2,
            'total_gb': memory_usage.sum() / 1024 ** 3,
            'by_column': memory_usage.to_dict(),
            'optimization_potential': self._estimate_memory_optimization()
        }

    def _get_duplicate_info(self) -> Dict:
        """Thông tin về duplicate rows"""
        duplicates = self.data.duplicated()
        return {
            'duplicate_rows': int(duplicates.sum()),
            'duplicate_percent': float(duplicates.sum() / len(self.data) * 100),
            'duplicate_indices': self.data[duplicates].index.tolist()
        }

    def _get_correlation_info(self) -> Dict:
        """Thông tin correlation"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return {'matrix': None, 'high_correlations': []}

        corr_matrix = self.data[numeric_cols].corr()

        # Tìm các cặp có correlation cao
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]

                if pd.notna(corr_value) and abs(corr_value) > 0.8:
                    high_corr.append({
                        'col1': col1,
                        'col2': col2,
                        'correlation': float(corr_value)
                    })

        return {
            'matrix': corr_matrix,
            'high_correlations': high_corr
        }

    def _estimate_memory_optimization(self) -> Dict:
        """Ước tính tiềm năng tối ưu memory"""
        optimization = {
            'categorical_columns': [],
            'numeric_columns': [],
            'total_savings_mb': 0
        }

        # Kiểm tra categorical columns
        cat_cols = self.data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            unique_ratio = self.data[col].nunique() / len(self.data)
            if unique_ratio < 0.5:  # Nếu unique values ít hơn 50%
                current_memory = self.data[col].memory_usage(deep=True)
                optimization['categorical_columns'].append({
                    'column': col,
                    'unique_ratio': float(unique_ratio),
                    'current_memory_mb': current_memory / 1024 ** 2
                })

        # Kiểm tra numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            min_val = self.data[col].min()
            max_val = self.data[col].max()

            # Kiểm tra xem có thể downcast không
            if min_val >= 0:
                if max_val < 256:
                    optimization['numeric_columns'].append({
                        'column': col,
                        'suggested_type': 'uint8',
                        'range': f"[{min_val}, {max_val}]"
                    })
                elif max_val < 65535:
                    optimization['numeric_columns'].append({
                        'column': col,
                        'suggested_type': 'uint16',
                        'range': f"[{min_val}, {max_val}]"
                    })

        return optimization

    def _display_summary(self, analysis: Dict):
        """Hiển thị tổng quan phân tích"""
        print(f" Kích thước: {analysis['shape'][0]:,} dòng × {analysis['shape'][1]:,} cột")
        print(f" Dung lượng: {analysis['memory_usage']['total_mb']:.2f} MB")

        # Kiểu dữ liệu
        dtype_counts = pd.Series(analysis['data_types']).value_counts()
        print(f"\n Kiểu dữ liệu:")
        for dtype, count in dtype_counts.items():
            print(f"  • {dtype}: {count} cột")

        # Missing values
        missing_cols = [col for col, stats in analysis['missing_values'].items()
                        if stats['count'] > 0]
        print(f"\n  Giá trị thiếu: {len(missing_cols)}/{len(analysis['columns'])} cột")
        if missing_cols:
            total_missing = sum(stats['count'] for stats in analysis['missing_values'].values())
            print(f"  • Tổng missing: {total_missing:,} giá trị")

            # Hiển thị top 5 cột có nhiều missing nhất
            sorted_missing = sorted(analysis['missing_values'].items(),
                                    key=lambda x: x[1]['percent'], reverse=True)[:5]
            for col, stats in sorted_missing:
                if stats['count'] > 0:
                    print(f"  • {col:20}: {stats['count']:>7,}")

        # Duplicates
        dup_info = analysis['duplicates']
        if dup_info['duplicate_rows'] > 0:
            print(f"\n Duplicate rows: {dup_info['duplicate_rows']:,} ({dup_info['duplicate_percent']:.1f}%)")

        # High correlations
        high_corr = analysis['correlation']['high_correlations']
        if high_corr:
            print(f"\n High correlations (>0.8): {len(high_corr)} cặp")
            for corr in high_corr[:3]:
                print(f"  • {corr['col1']} ↔ {corr['col2']}: {corr['correlation']:.3f}")

    def generate_report(self, output_dir: str = "reports") -> Dict:
        """
        Tạo báo cáo phân tích chi tiết

        Args:
            output_dir: Thư mục lưu báo cáo

        Returns:
            Dict chứa báo cáo
        """
        from pathlib import Path
        import json
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Tạo thư mục con cho report analysis
        report_analysis_dir = output_dir / "report_analysis"
        report_analysis_dir.mkdir(exist_ok=True)

        print(f"\n Thư mục báo cáo: {output_dir}")
        print(f" Thư mục visualization: {report_analysis_dir}")

        # Thực hiện phân tích
        analysis = self.analyze()

        # Lưu báo cáo JSON vào report_analysis
        report_file = report_analysis_dir / f"analysis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)

        # Tạo visualization và lưu vào report_analysis
        self._create_visualizations(report_analysis_dir)

        self._log_action('generate_report', {
            'report_file': str(report_file),
            'output_dir': str(output_dir),
            'report_analysis_dir': str(report_analysis_dir)
        })

        print(f"\n Đã tạo báo cáo tại: {report_file}")
        print(f" Visualizations được lưu trong: {report_analysis_dir}")

        # Liệt kê các file đã tạo
        print(f"\n Các file đã tạo:")
        json_files = list(report_analysis_dir.glob("*.json"))
        image_files = list(report_analysis_dir.glob("*.png"))

        if json_files:
            print(f"   • JSON reports: {len(json_files)} file")
            for file in json_files:
                file_size = file.stat().st_size / 1024  # KB
                print(f"     - {file.name} ({file_size:.1f} KB)")

        if image_files:
            print(f"   • Visualizations: {len(image_files)} file")
            for file in image_files:
                file_size = file.stat().st_size / 1024  # KB
                print(f"     - {file.name} ({file_size:.1f} KB)")

        return analysis

    def _create_visualizations(self, output_dir: Path):
        """Tạo các biểu đồ EDA - Mở rộng với boxplot và visualizations đầy đủ"""
        print("   - Tạo EDA visualizations...")

        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from matplotlib.gridspec import GridSpec

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns

        # =============== BIỂU ĐỒ 1: NUMERIC DISTRIBUTIONS (TẤT CẢ BIẾN) ===============
        if len(numeric_cols) > 0:
            # Tạo figure cho tất cả biến số
            n_numeric = len(numeric_cols)
            n_cols = 4  # Số cột trên mỗi hàng
            n_rows = int(np.ceil(n_numeric / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))

            # Flatten axes nếu có nhiều hàng
            if n_rows > 1 or n_cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]

            for idx, col in enumerate(numeric_cols):
                if idx < len(axes):
                    data_clean = self.data[col].dropna()

                    if len(data_clean) > 0:
                        # Vẽ histogram với KDE
                        sns.histplot(data_clean, bins=30, kde=True, ax=axes[idx],
                                     color='skyblue', edgecolor='black', alpha=0.7)

                        # Thêm thống kê
                        mean_val = data_clean.mean()
                        median_val = data_clean.median()

                        axes[idx].axvline(mean_val, color='red', linestyle='--',
                                          linewidth=1.5, label=f'Mean: {mean_val:.2f}')
                        axes[idx].axvline(median_val, color='green', linestyle='--',
                                          linewidth=1.5, label=f'Median: {median_val:.2f}')

                        axes[idx].set_title(f'{col}', fontsize=11, fontweight='bold')
                        axes[idx].set_xlabel('Giá trị')
                        axes[idx].set_ylabel('Tần suất')
                        axes[idx].legend(fontsize=8)
                        axes[idx].grid(True, alpha=0.3)
                    else:
                        axes[idx].text(0.5, 0.5, 'Không có dữ liệu',
                                       ha='center', va='center', transform=axes[idx].transAxes)

            # Ẩn các axes không sử dụng
            for idx in range(len(numeric_cols), len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle('PHÂN PHỐI TẤT CẢ BIẾN SỐ', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / '1_numeric_distributions_all.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"     Biểu đồ 1: Phân phối {len(numeric_cols)} biến số")

        # =============== BIỂU ĐỒ 2: CATEGORICAL DISTRIBUTIONS (TẤT CẢ BIẾN) ===============
        if len(categorical_cols) > 0:
            # Tạo figure cho tất cả biến phân loại
            n_categorical = len(categorical_cols)
            n_cols = 3  # Số cột trên mỗi hàng
            n_rows = int(np.ceil(n_categorical / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

            # Flatten axes nếu có nhiều hàng
            if n_rows > 1 or n_cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]

            for idx, col in enumerate(categorical_cols):
                if idx < len(axes):
                    value_counts = self.data[col].value_counts()

                    # Giới hạn hiển thị 10 giá trị phổ biến nhất
                    display_counts = value_counts.head(10)

                    if len(display_counts) > 0:
                        colors = plt.cm.Set2(np.arange(len(display_counts)))

                        bars = axes[idx].bar(range(len(display_counts)), display_counts.values,
                                             color=colors, edgecolor='black')

                        # Thêm số liệu trên mỗi cột
                        for bar_idx, (x, y) in enumerate(zip(range(len(display_counts)),
                                                             display_counts.values)):
                            axes[idx].text(x, y, f'{y:,}', ha='center', va='bottom',
                                           fontsize=9, fontweight='bold')

                        axes[idx].set_title(f'{col} ({len(value_counts)} categories)',
                                            fontsize=11, fontweight='bold')
                        axes[idx].set_xlabel('Giá trị')
                        axes[idx].set_ylabel('Số lượng')
                        axes[idx].set_xticks(range(len(display_counts)))
                        axes[idx].set_xticklabels([str(label)[:15] + '...'
                                                   if len(str(label)) > 15 else str(label)
                                                   for label in display_counts.index],
                                                  rotation=45, ha='right', fontsize=9)
                        axes[idx].grid(True, alpha=0.3, axis='y')
                    else:
                        axes[idx].text(0.5, 0.5, 'Không có dữ liệu',
                                       ha='center', va='center', transform=axes[idx].transAxes)

            # Ẩn các axes không sử dụng
            for idx in range(len(categorical_cols), len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle('PHÂN PHỐI TẤT CẢ BIẾN PHÂN LOẠI', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / '2_categorical_distributions_all.png',
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"     Biểu đồ 2: Phân phối {len(categorical_cols)} biến phân loại")

        # =============== BIỂU ĐỒ 3: BOXPLOTS (TẤT CẢ BIẾN SỐ) ===============
        if len(numeric_cols) > 0:
            # Tạo figure cho tất cả boxplots
            n_numeric = len(numeric_cols)
            n_cols = 4  # Số cột trên mỗi hàng
            n_rows = int(np.ceil(n_numeric / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))

            # Flatten axes nếu có nhiều hàng
            if n_rows > 1 or n_cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]

            for idx, col in enumerate(numeric_cols):
                if idx < len(axes):
                    data_clean = self.data[col].dropna()

                    if len(data_clean) > 0:
                        # Vẽ boxplot
                        boxplot = axes[idx].boxplot(data_clean, vert=True, patch_artist=True,
                                                    widths=0.7, showmeans=True,
                                                    meanprops={"marker": "o",
                                                               "markerfacecolor": "red",
                                                               "markeredgecolor": "black",
                                                               "markersize": 6})

                        # Tô màu cho box
                        boxplot['boxes'][0].set_facecolor('lightblue')
                        boxplot['boxes'][0].set_edgecolor('black')
                        boxplot['boxes'][0].set_alpha(0.7)

                        # Tính toán thống kê
                        q1 = np.percentile(data_clean, 25)
                        q3 = np.percentile(data_clean, 75)
                        iqr = q3 - q1
                        median_val = np.median(data_clean)
                        mean_val = np.mean(data_clean)

                        # Hiển thị thông tin thống kê
                        stats_text = f"Min: {data_clean.min():.2f}\n"
                        stats_text += f"Q1: {q1:.2f}\n"
                        stats_text += f"Median: {median_val:.2f}\n"
                        stats_text += f"Mean: {mean_val:.2f}\n"
                        stats_text += f"Q3: {q3:.2f}\n"
                        stats_text += f"Max: {data_clean.max():.2f}\n"
                        stats_text += f"IQR: {iqr:.2f}"

                        # Hiển thị outliers nếu có
                        outliers = data_clean[(data_clean < q1 - 1.5 * iqr) |
                                              (data_clean > q3 + 1.5 * iqr)]
                        if len(outliers) > 0:
                            stats_text += f"\nOutliers: {len(outliers)}"

                        axes[idx].text(0.02, 0.98, stats_text, transform=axes[idx].transAxes,
                                       fontsize=8, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                        axes[idx].set_title(f'{col}', fontsize=11, fontweight='bold')
                        axes[idx].set_ylabel('Giá trị')
                        axes[idx].grid(True, alpha=0.3, axis='y')
                    else:
                        axes[idx].text(0.5, 0.5, 'Không có dữ liệu',
                                       ha='center', va='center', transform=axes[idx].transAxes)

            # Ẩn các axes không sử dụng
            for idx in range(len(numeric_cols), len(axes)):
                axes[idx].set_visible(False)

            plt.suptitle('BOXPLOT TẤT CẢ BIẾN SỐ', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / '3_boxplots_all.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"     Biểu đồ 3: Boxplot {len(numeric_cols)} biến số")

        # =============== BIỂU ĐỒ 4: CORRELATION MATRIX ===============
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = self.data[numeric_cols].corr()

                plt.figure(figsize=(max(10, len(numeric_cols) * 0.8),
                                    max(8, len(numeric_cols) * 0.6)))

                sns.heatmap(corr_matrix, annot=True, fmt='.2f',
                            cmap='coolwarm', center=0, square=True,
                            cbar_kws={'shrink': 0.8, 'label': 'Correlation'})

                plt.title('CORRELATION MATRIX', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(output_dir / '4_correlation_matrix.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("     Biểu đồ 4: Correlation matrix")
            except Exception as corr_error:
                print(f"     Lỗi correlation matrix: {corr_error}")

        # =============== BIỂU ĐỒ 5: SCATTER PLOTS (CHO 6 BIẾN ĐẦU TIÊN) ===============
        if len(numeric_cols) >= 2:
            try:
                # Chọn 6 biến đầu tiên cho scatter matrix
                scatter_cols = numeric_cols[:min(6, len(numeric_cols))]

                if len(scatter_cols) >= 2:
                    scatter_data = self.data[scatter_cols].dropna()

                    if len(scatter_data) > 0:
                        g = sns.pairplot(scatter_data, diag_kind='kde',
                                         plot_kws={'alpha': 0.6, 's': 20},
                                         diag_kws={'fill': True})

                        g.fig.suptitle('SCATTER MATRIX', fontsize=16, fontweight='bold', y=1.02)
                        plt.tight_layout()
                        plt.savefig(output_dir / '5_scatter_matrix.png',
                                    dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"     Biểu đồ 5: Scatter matrix {len(scatter_cols)} biến")
            except Exception as scatter_error:
                print(f"      Lỗi scatter matrix: {scatter_error}")
