import logging
import sys
from pathlib import Path
from pythonjsonlogger import jsonlogger
from config import config

def setup_logger(name: str) -> logging.Logger:
    """Setup logger with JSON formatting"""
    
    logger = logging.getLogger(name)
    
    # Get level from config, ensure it's uppercase for getattr
    level_name = str(config.LOG_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)
    
    # Ensure level is an integer (standard logging levels are ints)
    if not isinstance(level, int):
        level = logging.INFO
        
    logger.setLevel(level)
    
    # Create logs directory if it doesn't exist
    log_dir = Path(config.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON formatter
    json_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(json_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create default logger
logger = setup_logger('voice_detection')