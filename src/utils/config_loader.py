import yaml
import os
from typing import Dict, Any


class ConfigLoader:
    """
    Class quản lý việc đọc và truy xuất cấu hình từ file YAML
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Đọc file YAML"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Không tìm thấy file cấu hình: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"Đã load cấu hình từ: {self.config_path}")
            return config
        except Exception as e:
            raise ValueError(f"Lỗi định dạng file config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Lấy giá trị config theo key (hỗ trợ nested key bằng dấu chấm)
        Ví dụ: get('data.raw_path')
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_full_config(self) -> Dict[str, Any]:
        return self.config
