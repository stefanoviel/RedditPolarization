import os
import time
import logging
from datetime import datetime
import logging.config

def setup_logging(log_file: str = 'app.log', log_level: str = 'DEBUG') -> None:
    """
    Set up logging configuration

    Parameters:

    log_file (str): The name of the log file to write logs to.
    log_level (str): The logging level. Can be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.

    """

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            },
        },
        'handlers': {
            'console': {
                'level': log_level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': log_level,
                'class': 'logging.FileHandler',
                'filename': log_file,
                'formatter': 'standard',
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file'],
                'level': log_level,
                'propagate': True,
            },
        }
    }

    logging.config.dictConfig(logging_config)



def configure_get_logger(output_dir: str, experiment_name:str, log_level: str = 'DEBUG', executed_file_name: str = __file__) -> logging.Logger:
    """Get the logger named as the current file"""

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{current_time}.log"

    log_file = os.path.join(output_dir, "logs", log_file_name)
    print("LOG file", log_file, output_dir, log_file_name)

    if not os.path.exists(os.path.join(output_dir, "logs")):
        os.makedirs(os.path.join(output_dir, "logs"))

    print(output_dir)
    print("LOG file", log_file)

    setup_logging(log_file, log_level)
    return logging.getLogger(executed_file_name)
