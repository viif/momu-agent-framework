"""日志工具"""

import logging
import sys

import colorama
from colorama import Fore, Style

colorama.init()

_has_setup = False


class ColoredFormatter(logging.Formatter):
    """
    自定义日志格式化器，匹配配色方案。
    """

    COLORS = {
        "DEBUG": Fore.MAGENTA,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        return super().format(record)


def setup_logger(
    name: str = "momu_agent",
    level: str = "WARNING",
    format_string: str | None = None,
) -> logging.Logger:
    """
    显式设置日志记录器（供用户手动配置使用）
    """
    global _has_setup

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    default_format = (
        "%(levelname)-8s %(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
    )

    formatter = ColoredFormatter(
        format_string or default_format, datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    _has_setup = True

    return logger


def get_logger(name: str = "momu_agent") -> logging.Logger:
    """
    获取日志记录器
    """
    global _has_setup

    logger = logging.getLogger(name)

    if not _has_setup and not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        formatter = ColoredFormatter(
            "%(levelname)-8s %(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

    return logger
