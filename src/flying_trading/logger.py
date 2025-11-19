import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
    log_dir: str = "logs",
) -> logging.Logger:
    """
    Настройка логирования для приложения.
    
    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Имя файла для логов (если None, логи только в консоль)
        log_dir: Директория для логов
    
    Returns:
        Настроенный logger
    """
    # Создаем директорию для логов если нужно
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        log_file_path = log_path / log_file
    else:
        log_file_path = None

    # Настройка формата
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Настройка root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (если указан файл)
    if log_file_path:
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Получить logger с указанным именем.
    
    Args:
        name: Имя logger (обычно __name__ модуля)
    
    Returns:
        Logger
    """
    return logging.getLogger(name)

