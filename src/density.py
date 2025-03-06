#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: density.py
# structura/density.py

import cupy as cp
from config import DEFAULT_GRID_SIZE


class DensityCalculator:
    """Computes density fields from 3D particle distributions."""

    def __init__(self, method="grid", grid_size=DEFAULT_GRID_SIZE):
        self.method = method
        self.grid_size = grid_size

    def compute_density(self, positions):
        """Computes the density field based on the chosen method."""
        if self.method == "grid":
            return self.grid_density(positions)
        else:
            raise ValueError(f"Unknown density calculation method: {self.method}")

    def grid_density(self, positions):
        """Computes a grid-based density field using CuPy for acceleration."""
        grid = cp.zeros(
            (self.grid_size, self.grid_size, self.grid_size), dtype=cp.float32
        )
        norm_positions = (positions / positions.max()) * (self.grid_size - 1)
        indices = norm_positions.astype(cp.int32)

        for x, y, z in indices:
            grid[x, y, z] += 1

        return grid
