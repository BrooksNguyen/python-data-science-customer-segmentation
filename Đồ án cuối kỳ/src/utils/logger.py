import logging
import os
from datetime import datetime
from pathlib import Path


def setup_file_logger(name: str, log_dir: str = "logs",
                      log_to_console: bool = False) -> logging.Logger:
    """
    Setup logger chỉ ghi vào file

    Args:
        name: Tên logger
        log_dir: Thư mục lưu log
        log_to_console: Có in ra console không (mặc định: False)

    Returns:
        logging.Logger
    """
    # Tạo thư mục logs nếu chưa có
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Tên file log với timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = log_path / f"{name}_{timestamp}.log"

    # Tạo logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Xóa handlers cũ
    if logger.handlers:
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Console handler (chỉ thêm nếu cần)
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Chỉ hiển thị warning/error
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Formatter cho file
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger