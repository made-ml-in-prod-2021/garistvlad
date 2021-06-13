import sys
import logging


def setup_logger(name: str) -> logging.Logger:
    """Configure logger"""
    base_logger = logging.getLogger(name)
    base_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    base_logger.addHandler(console_handler)
    return base_logger
