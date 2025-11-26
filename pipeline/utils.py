import sys
from loguru import logger

def setup_logging(config):
    logger.remove()
    logger.add(sys.stderr, level=config['logging']['level'])
    logger.add(config['logging']['file'], rotation="500 MB", level=config['logging']['level'])

