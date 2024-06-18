
import os
import sys 
# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logging_config import configure_get_logger
import config



def main():
    if not os.path.exists(config.OUTPUT_PATH):
        os.makedirs(config.OUTPUT_PATH)
    logger = configure_get_logger(config.OUTPUT_PATH)

    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
    
    # Your application logic here

if __name__ == '__main__':
    main()
