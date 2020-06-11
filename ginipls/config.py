import logging
import os


# log file
LOG_FILENAME = "taj-ginipls.log"

def init_logging(log_file=None, append=False, console_loglevel=logging.INFO):
    """Set up logging to file and console."""
    file_format = "%(asctime)s:%(module)s.%(funcName)s():line %(lineno)s:%(levelname)-8s: %(message)-8s : (%(pathname)s)"
    console_format = "%(asctime)s:%(module)s.%(funcName)s():line %(lineno)s:%(levelname)-8s%(message)s"
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
    return logging.getLogger(__name__) 

GLOBAL_LOGGER = init_logging(log_file=LOG_FILENAME, append=False)


if __name__ == '__main__':
  logger.info("ICI")
  logger.debug("ICI debug")
