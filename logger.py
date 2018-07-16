import logging


def init_console_logger(logger, verbose=False):
    stream_handler = logging.StreamHandler()
    if verbose:
        stream_handler.setLevel(logging.DEBUG)
    else:
        stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler("model.log")
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)