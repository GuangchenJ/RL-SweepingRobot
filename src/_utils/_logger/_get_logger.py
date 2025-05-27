# -*- coding: utf-8 -*-
# @Time    : 2023/5/12 22:41
# @Author  : Guangchen Jiang
# @Email   : guangchen98.jiang@gmail.com
# @File    : src/_logger/_get_logger.py
# @Software: PyCharm

import datetime as _datetime
import logging as _logging

# import tzlocal as _tzlocal
from typing import Optional

import pytz as _pytz
from tzlocal import get_localzone_name

from . import FmtRegex
from ._logging import _get_logger


def converter(timestamp):
    dt = _datetime.datetime.fromtimestamp(timestamp)
    tzinfo = _pytz.timezone(get_localzone_name())
    return tzinfo.localize(dt)


class Formatter(_logging.Formatter):
    """override logging.Formatter to use an aware datetime object"""

    def formatTime(self, record, datefmt=None):
        dt = converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec="milliseconds")
            except TypeError:
                s = dt.isoformat()
        return s


def get_logger(
    name: Optional[str] = None,
    level: Optional[int] = None,
    filename: Optional[str] = None,
) -> _logging.Logger:
    logger = _get_logger(name, 3)

    fmtr = Formatter(FmtRegex)
    for hdl in logger.handlers:
        hdl.setFormatter(fmtr)

    if level:
        logger.setLevel(level)
    if filename:
        file_hdlr = _logging.FileHandler(filename=filename)
        file_hdlr.setLevel(_logging.DEBUG)
        logger.addHandler(file_hdlr)

    return logger
