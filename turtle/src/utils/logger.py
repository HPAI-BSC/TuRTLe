# Adapted from
# https://github.com/skypilot-org/skypilot/blob/86dc0f6283a335e4aa37b3c10716f90999f48ab6/sky/sky_logging.py

import logging
import os
import sys
from typing import Optional

CONFIGURE_LOGGING = int(os.getenv("CONFIGURE_LOGGING", "1"))

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

_COLORS = {
    'DEBUG': '\033[36m',       # Cyan
    'INFO': '\033[32m',        # Green
    'WARNING': '\033[33m',     # Yellow
    'ERROR': '\033[31m',       # Red
    'CRITICAL': '\033[31;47m', # Red on White
    'RESET': '\033[0m',        # Reset
}

class NewLineFormatter(logging.Formatter):
    """Adds logging prefix and colors to newlines."""

    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        levelname = record.levelname
        color = _COLORS.get(levelname, _COLORS['RESET'])
        reset = _COLORS['RESET']
        
        # Formatear el mensaje con colores
        message = super().format(record)
        if record.message != "":
            parts = message.split(record.message)
            message = f"{color}{message}{reset}"
            message = message.replace("\n", f"\r\n{parts[0]}")
        return message


_root_logger = logging.getLogger("vllm")
_default_handler: Optional[logging.Handler] = None


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)
    global _default_handler
    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(logging.INFO)
        _root_logger.addHandler(_default_handler)
    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(fmt)
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
if CONFIGURE_LOGGING:
    _setup_logger()


def init_logger(name: str):
    # Use the same settings as above for root logger
    logger = logging.getLogger(name)
    logger.setLevel(os.getenv("LOG_LEVEL", "DEBUG"))

    if CONFIGURE_LOGGING:
        if _default_handler is None:
            raise ValueError(
                "_default_handler is not set up. This should never happen!"
                " Please open an issue on Github.")
        logger.addHandler(_default_handler)
        logger.propagate = False
    return logger
