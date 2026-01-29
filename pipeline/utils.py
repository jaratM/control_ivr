import sys
from loguru import logger
import time
import os

# Global variable to store the log file path for the current run
_current_log_file = None


def setup_logging(config, log_file_path=None):
    """
    Configure logging based on config settings.
    
    Args:
        config: Configuration dictionary with logging settings
        log_file_path: Specific log file path to use (for workers to use the same file as main process)
    """
    global _current_log_file
    
    logger.remove()
    logger.add(sys.stderr, level=config['logging']['level'])
    
    # Add file logging if configured
    if 'file' in config.get('logging', {}):
        if log_file_path:
            # Workers use the provided path
            _current_log_file = log_file_path
        elif _current_log_file is None:
            # Main process generates the timestamped filename
            log_file = config['logging']['file']
            os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
            _current_log_file = f'{log_file}_{time.strftime("%Y%m%d_%H%M%S")}.log'
        
        # Use enqueue=True for process-safe logging from multiple workers
        logger.add(_current_log_file, rotation="500 MB", level=config['logging']['level'], enqueue=True)
    
    return _current_log_file


def get_current_log_file():
    """Return the current log file path for workers to use."""
    return _current_log_file

