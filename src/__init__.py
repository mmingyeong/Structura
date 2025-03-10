#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: __init__.py

"""
Structura: A Python Library for Analyzing and Visualizing the Large-Scale Structure of the Universe.

This package comprises a suite of modules designed for cosmological research, including configuration settings, data loading, density calculation, visualization, and simulation data conversion. It provides robust tools for the quantitative analysis and graphical representation of cosmic structures.
"""

__version__ = "0.1.0"

from .config import LBOX_CMPCH, LBOX_CKPCH, DEFAULT_GRID_SIZE, USE_GPU
from .data_loader import DataLoader
from .density import DensityCalculator
from .visualization import Visualizer
from .utils import set_x_range
from .convert import SimulationDataConverter
from .kernel import KernelFunctions
from .save_density_map import save_density_map, save_parameters_info

__all__ = [
    "LBOX_CMPCH",
    "LBOX_CKPCH",
    "DEFAULT_GRID_SIZE",
    "USE_GPU",
    "DataLoader",
    "DensityCalculator",
    "Visualizer",
    "set_x_range",
    "SimulationDataConverter",  # Included in the public API.
    "KernelFunctions",
    "save_density_map",
    "save_parameters_info"
]
