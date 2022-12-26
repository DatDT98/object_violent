import json
import logging
import os
from logging.config import dictConfig
import datetime

DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"


def setup_logging_json(default_path='logging.json', default_level=logging.INFO, env_key='LOG_CFG'):
    """
    Setup logging configuration
    :param default_path:
    :param default_level:
    :param env_key:
    :return:
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
            dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
