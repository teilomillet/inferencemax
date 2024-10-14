# inferencemax/utils/logger.py

import sys
from pathlib import Path
from loguru import logger

def setup_logger(log_file: str = "inferencemax.log", retention: str = "10 days", rotation: str = "100 MB", level: str = "INFO"):
    """
    Set up Loguru logger with console and file outputs.
    
    Args:
        log_file (str): Path to the log file.
        retention (str): How long to keep log files.
        rotation (str): When to rotate the log file.
        level (str): Minimum log level to record.
    """
    # Remove any existing handlers
    logger.remove()
    
    # Add a handler to write to console
    logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level=level)
    
    # Add a handler to write to file
    log_path = Path(log_file)
    logger.add(log_path, rotation=rotation, retention=retention, level=level)

    return logger

# Create a logger instance
inferencemax_logger = setup_logger()

# Convenience functions for different log levels
def debug(message: str, *args, **kwargs):
    inferencemax_logger.debug(message, *args, **kwargs)

def info(message: str, *args, **kwargs):
    inferencemax_logger.info(message, *args, **kwargs)

def warn(message: str, *args, **kwargs):
    inferencemax_logger.warning(message, *args, **kwargs)

def error(message: str, *args, **kwargs):
    inferencemax_logger.error(message, *args, **kwargs)

def critical(message: str, *args, **kwargs):
    inferencemax_logger.critical(message, *args, **kwargs)

# Function to change log level dynamically
def set_log_level(level: str):
    inferencemax_logger.remove()
    setup_logger(level=level)
    info(f"Log level changed to {level}")