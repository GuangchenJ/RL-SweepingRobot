# -*- coding: utf-8 -*-
# @Time    : 2023/5/12 22:40
# @Author  : Guangchen Jiang
# @Email   : guangchen98.jiang@gmail.com
# @File    : src/_logger/__init__.py.py
# @Software: PyCharm


FmtRegex = "[%(levelname)s] [%(asctime)s] %(message)s - %(filename)s:%(lineno)s"

from ._get_logger import get_logger

__all__ = ["get_logger"]

__author__ = "Guangchen Jiang <guangchen98.jiang@gmail.com>"
__status__ = "test"
__version__ = "0.1"
__date__ = "12nd May 2023"
