#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: __init__.py

"""
Structura: A Python library for analyzing and visualizing the large-scale structure of the universe.
"""

__version__ = "0.1.0"

from .config import LBOX_CMPCH, LBOX_CKPCH, DEFAULT_GRID_SIZE, USE_GPU
from .data_loader import DataLoader
from .density import DensityCalculator
from .visualization import Visualizer
from .utils import set_x_range
from .convert import SimulationDataConverter  # ✅ 추가

__all__ = [
    "LBOX_CMPCH",
    "LBOX_CKPCH",
    "DEFAULT_GRID_SIZE",
    "USE_GPU",
    "DataLoader",
    "DensityCalculator",
    "Visualizer",
    "set_x_range",
    "SimulationDataConverter",  # ✅ 여기도 추가
]
