import logging
import sys

#
# for colorful print
#


class COLOR:
    grey = "\x1b[37;20m"
    black = "\x1b[30;20m"
    green = "\x1b[32m"
    magenta = "\x1B[35m"
    cyan = "\x1B[36m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"


class Highlight:
    def __init__(self, color):
        self.color = color

    def __enter__(self):
        print(self.color, end="")

    def __exit__(self, type, value, traceback):
        print(COLOR.reset, end="")
        sys.stdout.flush()


class CustomFormatter(logging.Formatter):
    """
    colorize logging
    source: https://stackoverflow.com/a/56944256/6109336
    """

    format = "%(asctime)s-%(name)s-%(levelname)s-(%(filename)s:%(lineno)d) %(message)s "

    FORMATS = {
        logging.DEBUG: COLOR.grey + format + COLOR.reset,
        logging.INFO: COLOR.cyan + format + COLOR.reset,
        logging.WARNING: COLOR.yellow + format + COLOR.reset,
        logging.ERROR: COLOR.red + format + COLOR.reset,
        logging.CRITICAL: COLOR.bold_red + format + COLOR.reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class CustomColorLogger(logging.Logger):
    """
    >>> from ? import CustomColorLogger
    >>> import logging
    >>> logger = CustomColorLogger("plugin", level=logging.DEBUG)
    >>> logger.debug("debug msg")
    >>> logger.error("error msg")
    """

    def __init__(self, name):
        super().__init__(name)
        ch = logging.StreamHandler()
        ch.setFormatter(CustomFormatter())
        self.addHandler(ch)
        
logger = CustomColorLogger(name=__name__)