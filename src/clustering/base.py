from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from ..utils.logger import setup_file_logger


class BaseClusteringComponent(ABC):
    """Lớp cơ sở trừu tượng cho tất cả các thành phần phân cụm"""

    def __init__(self, data: Optional[pd.DataFrame] = None,
                 name: str = None,
                 log_to_console: bool = False):

        # Thiết lập logger chỉ ghi file
        self.name = name or self.__class__.__name__
        self.logger = setup_file_logger(
            name=self.name,
            log_dir="logs",  # Thư mục logs
            log_to_console=log_to_console
        )

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Trả về cấu hình của thành phần"""
        pass
