"""
Structura: A Python library for analyzing and visualizing the large-scale structure of the universe.
"""

__version__ = "0.1.0"

from .config import LBOX_CMPCH, LBOX_CKPCH, DEFAULT_GRID_SIZE, USE_GPU
from .data_loader import DataLoader
from .density import DensityCalculator
from .visualization import Visualizer
from .utils import set_x_range

__all__ = [
    "LBOX_CMPCH",  # ✅ LBOX_MPC → LBOX_CMPCH로 변경
    "LBOX_CKPCH",
    "DEFAULT_GRID_SIZE",
    "USE_GPU",
    "DataLoader",
    "DensityCalculator",
    "Visualizer",
    "set_x_range",
]
