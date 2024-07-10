import os
import logging
import logging.config
from datetime import datetime

def setup_logging(log_file: str, log_level: str, logger_name: str) -> None:
    """
    Set up logging configuration specific to the given logger name.

    Parameters:

    log_file (str): The name of the log file to write logs to.
    log_level (str): The logging level.
    logger_name (str): The name of the logger to configure.

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
            logger_name: {  # configure logger for your application namespace
                'handlers': ['console', 'file'],
                'level': log_level,
                'propagate': False,
            },
        }
    }

    logging.config.dictConfig(logging_config)

def configure_get_logger(output_dir: str, experiment_name:str, log_level: str = 'DEBUG', executed_file_name: str = __file__) -> logging.Logger:
    """Get the logger named after the executed file to isolate logging to this application."""

    logger_name = os.path.splitext(os.path.basename(executed_file_name))[0]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{logger_name}_{current_time}.log"

    log_dir = os.path.join(output_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, log_file_name)

    setup_logging(log_file, log_level, logger_name)
    return logging.getLogger(logger_name)
