import sys
from loguru import logger
import time


def setup_logging(config):
    logger.remove()
    logger.add(sys.stderr, level=config['logging']['level'])
    # Add file logging if file path is specified in config
    if 'file' in config.get('logging', {}):
        log_file = config['logging']['file']
        timestamped_file = f'{log_file}_{time.strftime("%Y%m%d_%H%M%S")}.log'
        logger.add(timestamped_file, rotation="500 MB", level=config['logging']['level'])

