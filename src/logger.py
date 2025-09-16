from types import MethodType
import logging

formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s')

def log_caught_exception(logger, exception):
    """
    Log a caught exception at the debug level.
    """
    logger.debug('{}: {}'.format(exception.__class__, str(exception)))


def get_logger(module_name, level='DEBUG'):
    '''
    Returns a logger object with standard log output formatting that writes log
    messages to standard out. Logging outputs include the module_name string
    to enable tracing of the source of the log message.
    '''
    logging.captureWarnings(True)

    if level is None:
        level = logging.INFO
    logger = logging.getLogger(module_name)
    logger.setLevel('DEBUG')
    logger.propagate = True

    # add a console handler, no need to set level here as the level is set for
    # the logger
    handlers = logger.handlers
    console_handler = None
    for h in handlers:
        if isinstance(h, logging.StreamHandler):
            console_handler = h

    if console_handler is None:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
        logging.getLogger('py.warnings').addHandler(console_handler)

    logger.caught_exception = MethodType(log_caught_exception, logger)
    return logger