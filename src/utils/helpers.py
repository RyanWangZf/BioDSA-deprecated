import atexit
import logging
import os
import json
from datetime import datetime

def get_datetime_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger_name': record.name,
            'module': record.module,
            'function': record.funcName,
            'message': record.getMessage()
        }
        return json.dumps(log_data)


class Logger():
    def __init__(self, logger_name, log_filename):
        self.file_handler = logging.FileHandler(log_filename)
        self.file_handler.setLevel(logging.INFO)
        self.json_formatter = JSONFormatter()
        self.file_handler.setFormatter(self.json_formatter)
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.file_handler)

    # To log string
    def log(self, message):
        self.logger.info(message)

    # To log dictionary (agent reponse)
    def log_json(self, message):
        self.logger.info(json.dumps(message))
    
    def close(self):
        self.file_handler.close()
        self.logger.removeHandler(self.file_handler)

def setup_logger(run_id, log_dir='./logs'):
    os.makedirs(log_dir, exist_ok=True)
    log_fname = f"{log_dir}/{run_id}.log"
    logger = logging.getLogger()  # get root logger
    file_handler = logging.FileHandler(log_fname, mode="a", delay=False)
    file_handler.setFormatter(
        logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)  # all other loggers propagate to root; write to one log file from root
    print(f"Log path: {log_fname}")
    atexit.register(lambda: print(f"Log path: {log_fname}"))


def printj(obj, indent=2, logger=None):
    fn = print if logger is None else logger
    fn(json.dumps(obj, indent=indent))

