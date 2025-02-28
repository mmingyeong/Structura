#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: visualization.py

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from structura.logger import logger
import time

class Visualizer:
    """Provides visualization tools for cosmological datasets."""

    def __init__(self, bins=500, cmap="cividis", dpi=200):
        """
        Initialize the visualization settings.
        
        Args:
            bins (int): Number of bins for histogram.
            cmap (str): Colormap for visualization.
            dpi (int): DPI setting for saving images.
        """
        self.bins = bins
        self.cmap = cmap
        self.dpi = dpi

    def _generate_histogram(self, data):
        """Compute the 2D histogram (Y-Z plane)."""
        hist, yedges, zedges = np.histogram2d(data[:, 1], data[:, 2], bins=self.bins)
        return np.log10(hist + 1), yedges, zedges

    def _add_simulation_info(self, ax, lbox_cMpc, lbox_ckpch, x_range):
        """Add simulation metadata to the plot."""
        resolution_cMpc = lbox_cMpc / self.bins
        resolution_ckpch = lbox_ckpch / self.bins

        stats_text = f"Lbox: {lbox_cMpc} cMpc / {lbox_ckpch} ckpc/h\n" \
                     f"Resolution: {resolution_cMpc:.3f} cMpc/bin ({resolution_ckpch:.1f} ckpc/h/bin)\n" \
                     f"X range: {x_range:.2f} cMpc/h"

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.5, color="white"))

    def plot_2d_histogram(self, positions, results_folder, x_range, lbox_cMpc, lbox_ckpch, save_pdf=False):
        """
        Generate and save a 2D histogram (Y-Z plane).
        """
        # ✅ Ensure results directory exists
        if not os.path.exists(results_folder):
            os.makedirs(results_folder, exist_ok=True)
            logger.info(f"📁 Created directory: {results_folder}")

        # ✅ Compute histogram
        hist, yedges, zedges = self._generate_histogram(positions)

        # ✅ Plot setup
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        im = ax.imshow(hist.T, origin='lower', extent=[yedges[0], yedges[-1], zedges[0], zedges[-1]], cmap=self.cmap, aspect='auto')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Log(Density)")

        # ✅ Add simulation info
        self._add_simulation_info(ax, lbox_cMpc, lbox_ckpch, x_range)

        # ✅ Save plot
        file_name = f"ICS_{time.strftime('%Y%m%d')}_DM_density_YZ_X_{x_range}cMpc_bins{self.bins}.png"
        file_path = os.path.join(results_folder, file_name)

        plt.savefig(file_path, bbox_inches="tight")
        logger.info(f"✅ Plot saved: {file_path}")

        # ✅ Save as PDF if required
        if save_pdf:
            pdf_path = file_path.replace(".png", ".pdf")
            plt.savefig(pdf_path, bbox_inches="tight")
            logger.info(f"✅ PDF version saved: {pdf_path}")

        plt.close()
