import logging
import sys
from typing import Optional


class Logger:
    _instance = None

    def __new__(cls, name: str = "kronos", level: int = logging.INFO):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, name: str = "kronos", level: int = logging.INFO):
        if not hasattr(self, "_initialized") or not self._initialized:
            self.logger = logging.getLogger(name)
            self.logger.setLevel(level)
            if not self.logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self._initialized = True

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)


# Global instance
logger = Logger().logger
