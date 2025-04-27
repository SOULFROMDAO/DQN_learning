import logging

import torch

def get_device(only_cpu=True):
    if torch.cuda.is_available() and not only_cpu:
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_logger(name=None):
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    handler1 = logging.StreamHandler()
    handler1.setFormatter(formatter)
    handler1.setLevel(logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler1)
    return logger