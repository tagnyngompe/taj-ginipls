import logging
import os

def setup_custom_logger(log_filename, console_level=logging.INFO, logfile_level=logging.DEBUG):
    log_fmt = '%(asctime)s:%(name)s:%(module)s.%(funcName)s():line %(lineno)s:%(levelname)-8s %(message)-8s (%(pathname)s)'
    #logging.basicConfig(level=console_level, format=log_fmt)
    # create logger with name
    logger = logging.getLogger(__name__)
    # create file handler which logs even info messages
    fh = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
    fh.setLevel(logfile_level)
    # create console handler with a higher log level
    # ch = logging.StreamHandler()
    # ch.setLevel(console_level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(log_fmt)
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    # logger.addHandler(ch)
    return logger


def init_logging(log_file=None, append=False, console_loglevel=logging.INFO):
    """Set up logging to file and console."""
    file_format = "%(asctime)s:%(name)s:%(module)s.%(funcName)s():line %(lineno)s:%(levelname)-8s: %(message)-8s: (%(pathname)s)"
    console_format = "(asctime)s:%(module)s.%(funcName)s():line %(lineno)-8s%(message)s"
    if log_file is not None:
        if append:
            filemode_val = 'a'
        else:
            filemode_val = 'w'
        logging.basicConfig(level=logging.DEBUG,
                            #format="%(asctime)s %(levelname)s %(threadName)s %(name)s %(message)s",
                            format=file_format,
                            # datefmt='%m-%d %H:%M',
                            filename=log_file,
                            filemode=filemode_val)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(console_loglevel)
    # set a format which is simpler for console use
    formatter = logging.Formatter(console_format)
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    LOG = logging.getLogger(__name__) 
    return LOG

# log file
log_filename = "taj-ginipls.log"
#logger = setup_custom_logger(log_filename)
logger = init_logging(log_filename, )

if __name__ == '__main__':
  logger.info("ICI")
  logger.debug("ICI debug")