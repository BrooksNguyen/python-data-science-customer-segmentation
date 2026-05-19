# File: src/preprocessing/__init__.py
"""
Data Preprocessing Toolkit
"""

# Import các class từ các module
from .base_processor import BaseProcessor
from .data_loader import DataLoader
from .data_analyzer import DataAnalyzer
from .missing_handler import MissingValueHandler
from .outlier_handler import OutlierHandler
from .feature_scaler import FeatureScaler
from .categorical_encoder import CategoricalEncoder
from .feature_engineer import FeatureEngineer
from .data_pipeline import DataPreprocessorPipeline

__version__ = "1.0.0"
__author__ = "Data Preprocessing Toolkit Team"

__all__ = [
    'BaseProcessor',
    'DataLoader',
    'DataAnalyzer',
    'MissingValueHandler',
    'OutlierHandler',
    'FeatureScaler',
    'CategoricalEncoder',
    'FeatureEngineer',
    'DataPreprocessorPipeline'
]