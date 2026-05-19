from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from ..utils.logger import setup_file_logger

class BaseProcessor(ABC):
    """Lớp cơ sở cho tất cả processor"""

    def __init__(self, data: Optional[pd.DataFrame] = None,
                 name: str = None):
        """
        Khởi tạo BaseProcessor

        Args:
            data: DataFrame chứa dữ liệu
            name: Tên của processor
        """
        self.data = data
        self.original_data = data.copy() if data is not None else None
        self.processing_history: List[Dict] = []

        # Thiết lập logger
        self.name = name or self.__class__.__name__
        self.logger = setup_file_logger(self.name)

        # Lưu trữ metadata
        self.metadata: Dict[str, Any] = {
            'processor_name': self.name,
            'created_at': datetime.now().isoformat(),
            'data_shape': data.shape if data is not None else None
        }

    def _log_action(self, action: str, details: Dict, level: str = "info"):
        """
        Ghi log hành động xử lý

        Args:
            action: Tên hành động
            details: Chi tiết hành động
            level: Mức độ log
        """
        log_entry = {
            'action': action,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            'details': details
        }
        self.processing_history.append(log_entry)

        # Ghi log theo level
        log_message = f"{action}: {details}"
        if level == "info":
            self.logger.info(log_message)
        elif level == "warning":
            self.logger.warning(log_message)
        elif level == "error":
            self.logger.error(log_message)
        elif level == "debug":
            self.logger.debug(log_message)

    def get_data(self) -> pd.DataFrame:
        """Trả về DataFrame đã xử lý"""
        return self.data

    def get_processing_history(self) -> List[Dict]:
        """Trả về lịch sử xử lý"""
        return self.processing_history

    def get_metadata(self) -> Dict:
        """Trả về metadata của processor"""
        return self.metadata

    @abstractmethod
    def process(self, **kwargs) -> 'BaseProcessor':
        """
        Phương thức trừu tượng để xử lý dữ liệu
        Các class con phải implement phương thức này
        """
        pass
