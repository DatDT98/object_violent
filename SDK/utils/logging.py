import logging

log_fmt = '%(asctime)s|%(levelname)s|%(message)s'
logging.basicConfig(format=log_fmt, level=logging.INFO)
logger = logging.getLogger("SDK")