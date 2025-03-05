#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: visualization.py

import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from logger import logger

# Optional: These imports are required for FITS and TIFF saving.
try:
    from astropy.io import fits
except ImportError:
    logger.warning("Astropy is not installed. FITS saving will not be available.")

try:
    from PIL import Image
except ImportError:
    logger.warning("Pillow is not installed. TIFF saving using PIL may not be available.")

# Import Numba for JIT compilation.
try:
    from numba import njit
except ImportError:
    logger.warning("Numba is not installed. JIT acceleration will not be available.")
    njit = lambda func: func  # fallback: identity decorator


@njit
def compute_histogram2d_numba(block_chunk, xedges, yedges, axis1, axis2):
    """
    Compute a 2D histogram for a block using Numba JIT for speed.
    이 함수는 각 데이터 포인트에 대해 bin index를 직접 찾아 카운트를 증가시킵니다.
    """
    nx = len(xedges) - 1
    ny = len(yedges) - 1
    hist = np.zeros((nx, ny), dtype=np.float64)
    n_points = block_chunk.shape[0]
    for i in range(n_points):
        val_x = block_chunk[i, axis1]
        val_y = block_chunk[i, axis2]
        # Find bin index for x.
        j = 0
        while j < nx:
            if xedges[j] <= val_x < xedges[j+1]:
                break
            j += 1
        else:
            continue  # value out of range

        # Find bin index for y.
        k = 0
        while k < ny:
            if yedges[k] <= val_y < yedges[k+1]:
                break
            k += 1
        else:
            continue  # value out of range

        hist[j, k] += 1
    return hist


class Visualizer:
    """Provides visualization tools for cosmological datasets."""

    def __init__(self, use_gpu=True):
        """
        Initialize the visualization settings.
        
        Args:
            use_gpu (bool, optional): [Optional Input with Default Value]
                If True, use GPU acceleration via CuPy; otherwise, use NumPy.
                Default is True.
        """
        self.use_gpu = use_gpu

    def compute_2d_histogram(self, data, bins=500, projection_axis=0, scale="log10"):
        """
        Compute a 2D histogram from the input data with the specified scaling.
        
        If the input data is a dask array (e.g., loaded from a large HDF5 file), the histogram
        is computed blockwise in parallel using dask.delayed and Numba-optimized functions.
        Otherwise, the computation uses NumPy or CuPy (if use_gpu is True).
        
        Categorization of arguments:
          - Required Input:
                data (np.ndarray, cp.ndarray, or dask.array.Array): Input data array with shape (n_points, n_dimensions).
          - Optional Inputs with Default Values:
                bins (int): Number of bins for the histogram (default=500).
                projection_axis (int): Axis along which to project the data (default=0).
                scale (str): Scaling transformation. Options are "log10", "log2", "ln", "sqrt", "linear"
                           (default="log10").
          - Auto-generated Inputs:
                None in this function.
        
        Returns:
            tuple: (hist, edges1, edges2) – Transformed histogram and bin edges.
        """
        # Determine histogram axes.
        all_axes = list(range(data.shape[1]))
        try:
            all_axes.remove(projection_axis)
        except ValueError:
            raise ValueError(f"Projection axis {projection_axis} is invalid for data shape {data.shape}")
        axis1, axis2 = all_axes

        # Try to import dask.array and dask.delayed.
        try:
            import dask.array as da
            from dask import delayed
        except ImportError:
            da = None

        # If data is a dask array, process it in parallel.
        if da is not None and isinstance(data, da.Array):
            # Determine bin edges from the global min and max.
            x = data[:, axis1]
            y = data[:, axis2]
            x_min = x.min().compute()
            x_max = x.max().compute()
            y_min = y.min().compute()
            y_max = y.max().compute()
            xedges = np.linspace(x_min, x_max, bins + 1)
            yedges = np.linspace(y_min, y_max, bins + 1)

            # Convert each dask block to a delayed object.
            delayed_chunks = data.to_delayed().ravel()  # flatten all blocks
            delayed_hist_list = []
            for block in delayed_chunks:
                # Use Numba-accelerated function to compute histogram for each block.
                def hist_block(block_chunk):
                    block_chunk = np.asarray(block_chunk)
                    return compute_histogram2d_numba(block_chunk, xedges, yedges, axis1, axis2)
                delayed_hist = delayed(hist_block)(block)
                delayed_hist_list.append(delayed_hist)
            # Sum the delayed histograms.
            total_hist = delayed(sum)(delayed_hist_list)
            hist = total_hist.compute()

            # Apply scaling transformation.
            if scale in ("log10", "log"):
                hist = np.log10(hist + 1)
            elif scale == "log2":
                hist = np.log2(hist + 1)
            elif scale == "ln":
                hist = np.log(hist + 1)
            elif scale == "sqrt":
                hist = np.sqrt(hist)
            return hist, xedges, yedges
        else:
            # If data is not a dask array, proceed using NumPy or CuPy.
            xp = cp if self.use_gpu else np

            # Convert data to appropriate array type.
            if self.use_gpu and not isinstance(data, cp.ndarray):
                data = cp.asarray(data)
            elif not self.use_gpu and isinstance(data, cp.ndarray):
                data = cp.asnumpy(data)

            # Compute the 2D histogram.
            hist, edges1, edges2 = xp.histogram2d(data[:, axis1], data[:, axis2], bins=bins)
            
            # Apply scaling transformation.
            if scale in ("log10", "log"):
                hist = xp.log10(hist + 1)
            elif scale == "log2":
                hist = xp.log2(hist + 1)
            elif scale == "ln":
                hist = xp.log(hist + 1)
            elif scale == "sqrt":
                hist = xp.sqrt(hist)
            # For "linear", no transformation is applied.
            
            # Convert results to NumPy arrays if necessary.
            if self.use_gpu:
                if isinstance(hist, cp.ndarray):
                    hist = hist.get()
                if isinstance(edges1, cp.ndarray):
                    edges1 = edges1.get()
                if isinstance(edges2, cp.ndarray):
                    edges2 = edges2.get()
            return hist, edges1, edges2


    def _save_figure_by_format(self, fig, file_path, fmt, hist):
        """
        Save the figure or data according to the specified format.
        
        For 'PNG', 'PDF', and 'SVG', the figure is saved using plt.savefig().
        For 'FITS', the histogram data is saved using astropy.io.fits.
        For 'TIFF', the figure is saved using plt.savefig() with TIFF format.
        
        Args:
            fig (Figure): Matplotlib figure instance.
            file_path (str): The complete path for saving the file.
            fmt (str): The desired file format.
            hist (np.ndarray): The histogram data, used for FITS saving.
        """
        fmt_upper = fmt.upper()
        if fmt_upper in ["PNG", "PDF", "SVG"]:
            fig.savefig(file_path, bbox_inches="tight")
        elif fmt_upper == "FITS":
            try:
                hdu = fits.PrimaryHDU(data=hist)
                hdu.writeto(file_path, overwrite=True)
            except Exception as e:
                logger.error(f"Failed to save FITS file: {e}")
        elif fmt_upper == "TIFF":
            # Option 1: Using matplotlib's built-in TIFF saving (requires Pillow).
            try:
                fig.savefig(file_path, bbox_inches="tight", format="tiff")
            except Exception as e:
                logger.error(f"Failed to save TIFF file using plt.savefig: {e}")
                # Option 2: Convert figure to image array and save using PIL.
                try:
                    fig.canvas.draw()
                    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    im = Image.fromarray(image_array)
                    im.save(file_path)
                except Exception as e:
                    logger.error(f"Failed to save TIFF file using PIL: {e}")
        else:
            logger.warning(f"Format {fmt} not recognized. Skipping saving for this format.")

    def create_image_plot(self, hist, edges1, edges2, results_folder,
                          title, xlabel, ylabel, projection_axis, x_range, x_center,
                          sampling_rate, lbox_cMpc, lbox_ckpch, x_min, x_max,
                          input_folder, results_dir, bins=500, scale="log10",
                          cmap="cividis", dpi=200, output_formats=None,
                          show_grid=False, file_name_prefix=None, additional_info=None,
                          save_options_record=True):
        """
        Generate and save an image plot from the provided histogram data.
        
        If the 'title' parameter is None, an automatic plot title is generated in the following format:
            executionDateTime-xRange-projAxis-sampling_rate-bins-scale-color
        The saved image file name is set to be identical to the plot title.
        
        Additionally, the output_formats parameter allows selection among:
          - PNG, PDF, SVG: For publication, presentations, and web/LaTeX.
          - FITS, TIFF: For astronomical research and data preservation.
        
        Categorization of arguments:
          - Required Inputs:
                hist (np.ndarray or cp.ndarray): 2D histogram array.
                edges1 (np.ndarray or cp.ndarray): Bin edges for the first dimension.
                edges2 (np.ndarray or cp.ndarray): Bin edges for the second dimension.
                results_folder (str): Directory where the plot images will be saved.
                title (str or None): Title of the plot. If None, auto-generated title is used.
                xlabel (str): Label for the x-axis.
                ylabel (str): Label for the y-axis.
                projection_axis (int): Axis used for the projection.
                x_range (float): Range for the x dimension.
                x_center (float): Center position for the x dimension.
                sampling_rate (float): Sampling rate used.
                lbox_cMpc (float): Box size in cMpc.
                lbox_ckpch (float): Box size in ckpc/h.
                x_min (float): Minimum x value for filtering.
                x_max (float): Maximum x value for filtering.
                input_folder (str): Path to the input folder.
                results_dir (str): Path to the results directory.
                
          - Optional Inputs with Default Values:
                bins (int): Number of bins for the histogram (default=500).
                scale (str): Scaling transformation (default="log10").
                cmap (str): Colormap for the plot (default="cividis").
                dpi (int): Dots per inch for the figure (default=200).
                show_grid (bool): Whether to display a grid on the plot (default=False).
                save_options_record (bool): Whether to record the plot options in a JSON file (default=True).
                
          - Auto-generated Inputs (when None is provided):
                output_formats (list): If None, automatically set to ["PNG", "PDF"].
                file_name_prefix (str): If None, automatically set to the plot title.
                additional_info (dict): [Optional] Additional information to be appended to the file name; if provided, appended as _key:value.
        
        Returns:
            dict: A dictionary containing the paths of the saved files.
        """
        # Auto-generate output_formats if None provided.
        if output_formats is None:
            output_formats = ["PNG", "PDF"]

        # If title is None, auto-generate the plot title.
        if title is None:
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            title = f"{current_datetime}_xRange{x_range}_projAxis{projection_axis}_srate{sampling_rate}_bins{bins}_scale{scale}_color{cmap}"

        # Set file_name_prefix to the plot title if not provided.
        file_name_prefix = file_name_prefix or title
        if additional_info:
            for key, value in additional_info.items():
                file_name_prefix += f"_{key}:{value}"

        # Ensure the results folder exists.
        os.makedirs(results_folder, exist_ok=True)

        # 1) Convert any remaining CuPy arrays to NumPy arrays.
        if isinstance(hist, cp.ndarray):
            hist = hist.get()
        if isinstance(edges1, cp.ndarray):
            edges1 = edges1.get()
        if isinstance(edges2, cp.ndarray):
            edges2 = edges2.get()

        # 2) Build a multi-line annotation (displayed at the top-right of the plot).
        annotation_lines = [
            f"Sampling: {sampling_rate*100:.3f}%",
            f"Projection Axis: {projection_axis}",
            f"Filter Range: {x_min:.2f}-{x_max:.2f} ckpc/h",
            f"Scale: {scale}",
            "",  # blank line for spacing
            f"X Center: {x_center:.2f} cMpc/h | Thickness: {x_range:.2f} cMpc/h",
            f"Lbox: {lbox_cMpc} cMpc / {lbox_ckpch} ckpc/h",
            f"Input Folder: {str(input_folder)}",
            f"Results Folder: {str(results_dir)}"
        ]
        annotation = "\n".join(annotation_lines)

        # 3) Set up the figure and adjust right margin for annotation and colorbar.
        fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
        fig.subplots_adjust(right=0.88)

        # 4) Display the histogram.
        im = ax.imshow(hist.T, origin='lower',
                       extent=[edges1[0], edges1[-1], edges2[0], edges2[-1]],
                       cmap=cmap, aspect='auto')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if show_grid:
            ax.grid(True)

        # 5) Add colorbar on the right side.
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Log10(Density)")

        # 6) Save the figure (or data) in each requested format.
        saved_files = {}
        for fmt in output_formats:
            file_path = os.path.join(results_folder, f"{file_name_prefix}.{fmt.lower()}")
            self._save_figure_by_format(fig, file_path, fmt, hist)
            saved_files[fmt.upper()] = file_path
        plt.close()

        # 7) Organize metadata into a structured JSON file if required.
        if save_options_record:
            metadata = {
                "Plot Info": {
                    "title": title,
                    "xlabel": xlabel,
                    "ylabel": ylabel,
                    "annotation": annotation
                },
                "Filtering Info": {
                    "projection_axis": projection_axis,
                    "x_range": x_range,
                    "x_center": x_center,
                    "sampling_rate": sampling_rate,
                    "x_min": x_min,
                    "x_max": x_max
                },
                "Simulation Info": {
                    "lbox_cMpc": lbox_cMpc,
                    "lbox_ckpch": lbox_ckpch
                },
                "Paths": {
                    "input_folder": str(input_folder),
                    "results_dir": str(results_dir)
                },
                "Rendering Info": {
                    "bins": bins,
                    "scale": scale,
                    "cmap": cmap,
                    "dpi": dpi,
                    "output_formats": output_formats,
                    "show_grid": show_grid
                }
            }

            json_path = os.path.join(results_folder, f"{file_name_prefix}_metadata.json")
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=4)
            saved_files["JSON"] = json_path

        return saved_files

    def record_plot_options(self, results_folder, file_name_prefix, options_dict):
        """
        Record all image plotting options in a JSON file for reproducibility.
        
        Categorization of arguments:
          - Required Inputs:
                results_folder (str): Directory where the JSON file will be saved.
                file_name_prefix (str): Prefix used for naming the JSON file.
                options_dict (dict): Dictionary of plotting options to record.
        
        Returns:
            None
        """
        record_file = os.path.join(results_folder, f"{file_name_prefix}_options.json")
        with open(record_file, "w") as f:
            json.dump(options_dict, f, indent=4)
        logger.info(f"Plot options recorded in {record_file}")
