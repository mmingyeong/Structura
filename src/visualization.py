#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: visualization.py

import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from logger import logger

# Optional: These imports are required for saving in FITS and TIFF formats.
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

    def njit(func):
        return func  # Fallback: identity decorator

@njit
def compute_histogram2d_numba(block_chunk, xedges, yedges, axis1, axis2):
    """
    Compute a 2D histogram for a data block using Numba JIT acceleration.
    
    This function iterates over each data point in the block_chunk, determines the 
    corresponding bin indices for the two specified axes based on provided bin edges, 
    and increments the count of the appropriate bin.
    
    Parameters:
        block_chunk (ndarray): A 2D array containing the data points.
        xedges (array_like): Array of bin edges for the first dimension.
        yedges (array_like): Array of bin edges for the second dimension.
        axis1 (int): Index of the first projection axis.
        axis2 (int): Index of the second projection axis.
    
    Returns:
        ndarray: A 2D histogram array with counts for each bin.
    """
    nx = len(xedges) - 1
    ny = len(yedges) - 1
    hist = np.zeros((nx, ny), dtype=np.float64)
    n_points = block_chunk.shape[0]
    # Note: Due to Numba limitations, internal logging is not feasible.
    for i in range(n_points):
        val_x = block_chunk[i, axis1]
        val_y = block_chunk[i, axis2]
        # Determine bin index for x-axis.
        j = 0
        while j < nx:
            if xedges[j] <= val_x < xedges[j + 1]:
                break
            j += 1
        else:
            continue  # Ignore values outside the specified range.

        # Determine bin index for y-axis.
        k = 0
        while k < ny:
            if yedges[k] <= val_y < yedges[k + 1]:
                break
            k += 1
        else:
            continue  # Ignore values outside the specified range.
        hist[j, k] += 1
    return hist

class Visualizer:
    """
    Provides visualization tools for cosmological datasets.
    """

    def __init__(self, use_gpu=True):
        """
        Initialize visualization settings.
        
        Parameters:
            use_gpu (bool): Flag indicating whether to utilize GPU acceleration.
        """
        logger.debug("Initializing Visualizer. GPU usage: %s", use_gpu)
        self.use_gpu = use_gpu
        logger.debug("Visualizer initialization complete.")

    def compute_2d_histogram(self, data, bins=None, projection_axis=0, scale="log10", chunk_size=1000000):
        """
        Compute a 2D histogram from the input data using the specified scale.
        
        For large datasets, the computation is performed in chunks to prevent memory overflow.
        
        Parameters:
            data (ndarray or dask.array): The input dataset.
            bins (int, optional): Number of bins for the histogram. If None, optimal bins are computed.
            projection_axis (int): The axis to be projected out from the data.
            scale (str): Scaling transformation to apply ('log10', 'log2', 'ln', 'sqrt').
            chunk_size (int): Size of data chunks for processing large datasets.
        
        Returns:
            tuple: A tuple containing the computed histogram and the bin edges for the two dimensions.
        """
        logger.debug("Starting compute_2d_histogram: data shape=%s, projection axis=%d, scale=%s", data.shape, projection_axis, scale)
        # Select the two axes excluding the projection axis.
        all_axes = list(range(data.shape[1]))
        try:
            all_axes.remove(projection_axis)
        except ValueError:
            raise ValueError(f"Projection axis {projection_axis} is invalid for data shape {data.shape}")
        axis1, axis2 = all_axes
        logger.debug("Selected axes: axis1=%d, axis2=%d", axis1, axis2)

        # Attempt to use Dask for parallel CPU processing.
        try:
            import dask.array as da
            from dask import delayed
        except ImportError:
            da = None
            logger.debug("Failed to load Dask library; Dask functionality will be disabled.")

        # Process data as a Dask array for parallel computation.
        if da is not None and isinstance(data, da.Array):
            logger.debug("Input data is of type Dask array.")
            x = data[:, axis1]
            y = data[:, axis2]
            x_min = x.min().compute()
            x_max = x.max().compute()
            y_min = y.min().compute()
            y_max = y.max().compute()
            logger.debug("Dask data min/max values: x_min=%f, x_max=%f, y_min=%f, y_max=%f", x_min, x_max, y_min, y_max)

            if bins is None:
                bins_x, bins_y = self.optimal_bins_2d(x.compute(), y.compute())
            else:
                bins_x = bins_y = bins

            xedges = np.linspace(x_min, x_max, bins_x + 1)
            yedges = np.linspace(y_min, y_max, bins_y + 1)
            logger.debug("Created Dask histogram bin edges: bins_x=%d, bins_y=%d", bins_x, bins_y)

            delayed_chunks = data.to_delayed().ravel()
            delayed_hist_list = []
            for idx, block in enumerate(delayed_chunks):
                logger.debug("Processing Dask chunk %d.", idx)
                def hist_block(block_chunk):
                    block_chunk = np.asarray(block_chunk)
                    return compute_histogram2d_numba(block_chunk, xedges, yedges, axis1, axis2)
                delayed_hist = delayed(hist_block)(block)
                delayed_hist_list.append(delayed_hist)
            total_hist = delayed(sum)(delayed_hist_list)
            hist = total_hist.compute()
            logger.debug("Accumulated histogram over all chunks complete.")

            # Apply scale transformation.
            if scale in ("log10", "log"):
                hist = np.log10(hist + 1)
            elif scale == "log2":
                hist = np.log2(hist + 1)
            elif scale == "ln":
                hist = np.log(hist + 1)
            elif scale == "sqrt":
                hist = np.sqrt(hist)
            logger.debug("Scale transformation (%s) applied successfully.", scale)
            return hist, xedges, yedges

        else:
            # Use GPU acceleration (CuPy) or CPU (NumPy) based on the flag.
            xp = cp if self.use_gpu else np
            logger.debug("Converting arrays based on GPU usage: use_gpu=%s", self.use_gpu)
            if self.use_gpu and not isinstance(data, cp.ndarray):
                data = cp.asarray(data)
                logger.debug("Converted data to CuPy array.")
            elif not self.use_gpu and isinstance(data, cp.ndarray):
                data = cp.asnumpy(data)
                logger.debug("Converted data to NumPy array.")

            x_min = xp.min(data[:, axis1]).item()
            x_max = xp.max(data[:, axis1]).item()
            y_min = xp.min(data[:, axis2]).item()
            y_max = xp.max(data[:, axis2]).item()
            logger.debug("Computed data min/max values: x_min=%f, x_max=%f, y_min=%f, y_max=%f", x_min, x_max, y_min, y_max)

            if bins is None:
                if self.use_gpu:
                    data_x = cp.asnumpy(data[:, axis1])
                    data_y = cp.asnumpy(data[:, axis2])
                else:
                    data_x = data[:, axis1]
                    data_y = data[:, axis2]
                bins_x, bins_y = self.optimal_bins_2d(data_x, data_y)
            else:
                bins_x = bins_y = bins
            logger.debug("Histogram bin counts: bins_x=%d, bins_y=%d", bins_x, bins_y)

            xedges = np.linspace(x_min, x_max, bins_x + 1)
            yedges = np.linspace(y_min, y_max, bins_y + 1)
            logger.debug("Bin edges successfully generated.")

            hist_accum = cp.zeros((bins_x, bins_y), dtype=cp.float64)
            n_points = data.shape[0]
            logger.debug("Total number of data points: %d", n_points)

            for start in range(0, n_points, chunk_size):
                end = min(start + chunk_size, n_points)
                logger.debug("Processing data chunk: start=%d, end=%d", start, end)
                data_chunk = data[start:end, :]
                hist_chunk, _, _ = cp.histogram2d(
                    data_chunk[:, axis1],
                    data_chunk[:, axis2],
                    bins=[bins_x, bins_y],
                    range=[[x_min, x_max], [y_min, y_max]],
                )
                hist_accum += hist_chunk
            logger.debug("Accumulated histogram over all chunks complete.")

            if scale in ("log10", "log"):
                hist_accum = cp.log10(hist_accum + 1)
            elif scale == "log2":
                hist_accum = cp.log2(hist_accum + 1)
            elif scale == "ln":
                hist_accum = cp.log(hist_accum + 1)
            elif scale == "sqrt":
                hist_accum = cp.sqrt(hist_accum)
            logger.debug("Scale transformation (%s) applied successfully.", scale)

            hist = hist_accum.get()
            return hist, xedges, yedges

    def optimal_bins_1d(self, data):
        """
        Calculate the optimal number of bins for one-dimensional data using the Freedman-Diaconis rule.
        
        Parameters:
            data (array_like): The input data array.
        
        Returns:
            int: The optimal number of bins.
        """
        logger.debug("Starting optimal_bins_1d: data size=%d", data.size)
        data = np.asarray(data)
        n = data.size
        if n < 2:
            logger.debug("Insufficient data points; returning 1 bin.")
            return 1

        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        logger.debug("25th and 75th percentiles: %f, %f / IQR: %f", q25, q75, iqr)

        if iqr == 0:
            bins = int(np.ceil(np.log2(n) + 1))
            logger.debug("IQR is zero; applying Sturges' rule: bins=%d", bins)
            return bins

        bin_width = 2 * iqr / (n ** (1 / 3))
        data_range = data.max() - data.min()
        bins = int(np.ceil(data_range / bin_width))
        logger.debug("Freedman-Diaconis rule applied: bin_width=%f, data_range=%f, bins=%d", bin_width, data_range, bins)
        return bins

    def optimal_bins_2d(self, data_x, data_y):
        """
        Calculate the optimal number of bins for each axis of two-dimensional data.
        
        Parameters:
            data_x (array_like): Data for the first axis.
            data_y (array_like): Data for the second axis.
        
        Returns:
            tuple: Optimal number of bins (bins_x, bins_y).
        """
        logger.debug("Starting optimal_bins_2d.")
        bins_x = self.optimal_bins_1d(data_x)
        bins_y = self.optimal_bins_1d(data_y)
        logger.debug("Calculated bin counts: bins_x=%d, bins_y=%d", bins_x, bins_y)
        return bins_x, bins_y

    def _save_figure_by_format(self, fig, file_path, fmt, hist):
        """
        Save the figure or data in the specified format.
        
        Parameters:
            fig (Figure): Matplotlib figure to be saved.
            file_path (str): Path where the file will be saved.
            fmt (str): Format in which to save the file (e.g., 'PNG', 'PDF', 'SVG', 'FITS', 'TIFF').
            hist (ndarray): Histogram data used for saving in certain formats.
        """
        logger.debug("Initiating file save: format=%s, path=%s", fmt, file_path)
        fmt_upper = fmt.upper()
        if fmt_upper in ["PNG", "PDF", "SVG"]:
            fig.savefig(file_path, bbox_inches="tight")
            logger.debug("File saved in %s format successfully.", fmt_upper)
        elif fmt_upper == "FITS":
            try:
                hdu = fits.PrimaryHDU(data=hist)
                hdu.writeto(file_path, overwrite=True)
                logger.debug("FITS file saved successfully: %s", file_path)
            except Exception as e:
                logger.error("FITS file save failed: %s", e)
        elif fmt_upper == "TIFF":
            try:
                fig.savefig(file_path, bbox_inches="tight", format="tiff")
                logger.debug("TIFF file save (plt.savefig) completed: %s", file_path)
            except Exception as e:
                logger.error("TIFF file save using plt.savefig failed: %s", e)
                try:
                    fig.canvas.draw()
                    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    im = Image.fromarray(image_array)
                    im.save(file_path)
                    logger.debug("TIFF file save using PIL completed: %s", file_path)
                except Exception as e:
                    logger.error("TIFF file save using PIL failed: %s", e)
        else:
            logger.warning("Unrecognized format %s. Skipping save.", fmt)
    
    def create_image_plot(
        self,
        hist,
        edges1,
        edges2,
        results_folder,
        title,
        xlabel,
        ylabel,
        projection_axis,
        x_range,
        x_center,
        sampling_rate,
        x_min,
        x_max,
        input_folder,
        results_dir,
        bins=500,
        scale="log10",
        cmap="cividis",
        dpi=200,
        output_formats=None,
        show_grid=False,
        file_name_prefix=None,
        additional_info=None,
        save_options_record=True,
        data_unit="ckpc/h",
        box_size=None,
    ):
        """
        Generate and save an image plot based on the provided histogram data.
        
        Parameters:
            hist (ndarray): The histogram data.
            edges1 (array_like): Bin edges for the first dimension.
            edges2 (array_like): Bin edges for the second dimension.
            results_folder (str): Directory where results will be saved.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            projection_axis (int): The axis used for projection.
            x_range (float): The range of x values for filtering.
            x_center (float): The center value of the x range.
            sampling_rate (float): The sampling rate applied.
            x_min (float): Minimum x value.
            x_max (float): Maximum x value.
            input_folder (str): Directory of input data.
            results_dir (str): Directory for additional results.
            bins (int, optional): Number of bins used. Defaults to 500.
            scale (str, optional): Scale transformation used. Defaults to 'log10'.
            cmap (str, optional): Colormap used for the plot. Defaults to 'cividis'.
            dpi (int, optional): Dots per inch for the plot. Defaults to 200.
            output_formats (list, optional): List of file formats to save. Defaults to ['PNG', 'PDF'].
            show_grid (bool, optional): Whether to show grid lines. Defaults to False.
            file_name_prefix (str, optional): Prefix for the saved file names.
            additional_info (dict, optional): Additional information to include in file names.
            save_options_record (bool, optional): Whether to save plot options as a JSON file.
            data_unit (str, optional): Unit of the data values. Defaults to 'ckpc/h'.
            box_size (float, optional): Box size information for simulation data.
        
        Returns:
            dict: A dictionary mapping output formats to their corresponding file paths.
        """
        logger.debug("Starting create_image_plot: title=%s, results folder=%s", title, results_folder)
        if output_formats is None:
            output_formats = ["PNG", "PDF"]

        os.makedirs(results_folder, exist_ok=True)

        if isinstance(hist, cp.ndarray):
            hist = hist.get()
        if isinstance(edges1, cp.ndarray):
            edges1 = edges1.get()
        if isinstance(edges2, cp.ndarray):
            edges2 = edges2.get()

        bin_resolution_x = np.diff(edges1).mean()
        bin_resolution_y = np.diff(edges2).mean()
        logger.debug("Bin resolution: x=%f, y=%f", bin_resolution_x, bin_resolution_y)

        if title is None:
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            title = (
                f"{current_datetime}_xRange{x_range}_projAxis{projection_axis}_"
                f"srate{sampling_rate}_bins{bins}_scale{scale}_res{bin_resolution_x}"
            )
            logger.debug("Auto-generated title: %s", title)

        file_name_prefix = file_name_prefix or title
        if additional_info:
            for key, value in additional_info.items():
                file_name_prefix += f"_{key}:{value}"
        logger.debug("File name prefix: %s", file_name_prefix)

        annotation_lines = [
            f"Sampling: {sampling_rate * 100:.3f}%",
            f"Projection Axis: {projection_axis}",
            f"Filter Range: {x_min:.2f}-{x_max:.2f} {data_unit}",
            f"Scale: {scale}",
            "",
            f"X Center: {x_center:.2f} cMpc/h | Thickness: {x_range:.2f} cMpc/h",
            f"Input Folder: {str(input_folder)}",
            f"Results Folder: {str(results_dir)}",
            "",
            f"Bin Resolution: {bin_resolution_x:.2f} (x-axis) x {bin_resolution_y:.2f} (y-axis) {data_unit}",
        ]
        annotation = "\n".join(annotation_lines)
        logger.debug("Annotation text created successfully.")

        fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
        fig.subplots_adjust(right=0.88)
        logger.debug("Plot created successfully.")

        im = ax.imshow(
            hist.T,
            origin="lower",
            extent=[edges1[0], edges1[-1], edges2[0], edges2[-1]],
            cmap=cmap,
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if show_grid:
            ax.grid(True)
            logger.debug("Grid display enabled.")

        cbar = plt.colorbar(im, ax=ax)
        if scale in ("log10", "log", "log2", "ln"):
            cbar.set_label("Log-scaled Density")
        else:
            cbar.set_label("Density")

        saved_files = {}
        for fmt in output_formats:
            file_path = os.path.join(results_folder, f"{file_name_prefix}.{fmt.lower()}")
            logger.debug("Attempting to save file: format=%s, path=%s", fmt, file_path)
            self._save_figure_by_format(fig, file_path, fmt, hist)
            saved_files[fmt.upper()] = file_path
            logger.debug("File saved: %s", file_path)

        plt.close()
        logger.debug("Plot closed successfully.")

        if save_options_record:
            metadata = {
                "Plot Info": {
                    "title": title,
                    "xlabel": xlabel,
                    "ylabel": ylabel,
                    "annotation": annotation,
                },
                "Filtering Info": {
                    "projection_axis": projection_axis,
                    "x_range": x_range,
                    "x_center": x_center,
                    "sampling_rate": sampling_rate,
                    "x_min": x_min,
                    "x_max": x_max,
                    "data_unit": data_unit,
                },
                "Simulation Info": {"box_size": box_size, "data_unit": data_unit},
                "Paths": {
                    "input_folder": str(input_folder),
                    "results_dir": str(results_dir),
                },
                "Rendering Info": {
                    "bins": bins,
                    "scale": scale,
                    "cmap": cmap,
                    "dpi": dpi,
                    "output_formats": output_formats,
                    "show_grid": show_grid,
                    "bin_resolution_x": bin_resolution_x,
                    "bin_resolution_y": bin_resolution_y,
                    "data_unit": data_unit,
                },
            }
            json_path = os.path.join(results_folder, f"{file_name_prefix}_metadata.json")
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=4)
            saved_files["JSON"] = json_path
            logger.debug("Metadata JSON file saved successfully: %s", json_path)

        logger.debug("create_image_plot completed. Saved files: %s", saved_files)
        return saved_files

    def record_plot_options(self, results_folder, file_name_prefix, options_dict):
        """
        Record image plot options in a JSON file to ensure reproducibility.
        
        Parameters:
            results_folder (str): Directory where the record will be saved.
            file_name_prefix (str): Prefix for the saved file name.
            options_dict (dict): Dictionary containing plot options.
        """
        record_file = os.path.join(results_folder, f"{file_name_prefix}_options.json")
        with open(record_file, "w") as f:
            json.dump(options_dict, f, indent=4)
        logger.info("Plot options successfully recorded in %s", record_file)
