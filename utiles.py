import atexit
import os
import logging
import torch
import datetime
from pathlib import Path

project_path = Path(__file__).parent
log_path = project_path / "log"
if not log_path.exists():
    os.mkdir(log_path)

def get_device(only_cpu=True):
    if torch.cuda.is_available() and not only_cpu:
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_logger(name=None):
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path / "dqn_log.log", mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def rename_train_log():
    logging.shutdown()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    src_path = log_path / "dqn_log.log"
    dst_path = log_path / f"dqn_log_{timestamp}.log"
    os.rename(src_path, dst_path)
