import logging
import sys
import threading
from pathlib import Path
from typing import Optional

MODEL_REPOID = 'unsloth/phi-4-unsloth-bnb-4bit'

# Define the path to the model directory
MODEL_PATH = (Path(__file__).parent / 'models' / 'phi-4-unsloth-bnb-4bit').resolve()

_lock = threading.Lock()
_logger: Optional[logging.Logger] = None

def getLogger() -> logging.Logger:
    """
    Returns the logger instance for the Local Translator.
    """
    global _lock
    global _logger
    with _lock:
        if _logger is None:
            _logger = logging.getLogger('localtranslator')
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('[Local Translator] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            handler.setLevel(logging.DEBUG)
            _logger.addHandler(handler)
            _logger.setLevel(logging.DEBUG)
            _logger.propagate = False

    return _logger
