# 📝 Changelog

## **[2025-02-28] - 🚀 Initial Release: Structura 0.1.0**
### 🎯 Key Features
- **Flexible Data Loading**:
  - Supports `.npz`, `.hdf5`, and `.csv` simulation datasets
  - Filters data based on X-axis range
- **2D Visualization**:
  - Generates **histograms, scatter plots, and density maps**
  - Optimized rendering for large datasets
- **Optimized GPU Acceleration**:
  - Uses `CuPy` for parallelized computations
  - Efficient memory management for large-scale data
- **Execution Logging**:
  - Tracks execution time in **hh:mm:ss** format
  - Logs precise filtering X range in `ckpc/h`

---

## **[2025-03-04] - 🚀 Feature Update: SimulationDataConverter & SystemChecker**
### **🎯 Key Features**
#### **🔹 SimulationDataConverter**
- **Automated HDF5 to NPY/NPZ conversion**:
  - Developed `SimulationDataConverter()` to process cosmological simulation data.
  - Ensures **GPU acceleration** is utilized if available.
- **Runtime Optimization**:
  - **Skips conversion** if output files already exist, reducing redundant computation.
  - Logs **total file count, total size, and average file size** in the output directory.
- **Execution Logging**:
  - Tracks **input file path & output folder**.
  - Displays conversion status with a summary of skipped/completed files.

#### **🔹 SystemChecker**
- **Comprehensive System Diagnostics**:
  - Developed `SystemChecker()` to analyze **GPU, CPU, RAM, and software dependencies**.
  - Detects **available GPUs, CPU cores, and total RAM**.
  - Logs **Python, NumPy, CuPy, and PyTorch versions**.
- **Upgrade Recommendations**:
  - **Python < 3.9** → Upgrade suggested.
  - **PyTorch missing** → Installation recommended for deep learning tasks.
  - **Prevents unnecessary GPU upgrade suggestions** (e.g., **A100 → H100**).
- **Execution Summary**:
  - Displays **last update date** of `SystemChecker` for reference.
  - Uses `verbose=True` option for **extended system diagnostics**.

# 📝 Changelog

## **[2025-03-04] - 🚀 Feature Update: DataLoader & Visualizer Enhancements (Structura 0.2.0)**
### 🎯 Key Features

#### **🔹 DataLoader**
- **Enhanced Image Saving & Format Options**:
  - Improved storage path management and image format selection.
  - Added new function arguments: `projection_axis`, `filtering_range`, `sampling_rate`, `histogram_bin`, and `log_scale`.
- **Conditional GPU Acceleration**:
  - Utilizes `CuPy` when GPU is enabled; defaults to `NumPy` otherwise.
- **Performance Optimization**:
  - Optimized data processing routines to reduce memory usage and execution time.

#### **🔹 Visualizer**
- **Flexible Projection & Log Scale Options**:
  - Now allows the selection of a custom projection axis (instead of fixed 0th axis).
  - Offers multiple log scale transformations: `log10`, `log2`, `ln`, `sqrt`, and `linear`.
- **Robust Image Plot Customization**:
  - Comprehensive argument handling: separates required inputs, defaulted options, and auto-generated parameters (e.g., auto-generated title).
  - Enhanced image plot functionality with configurable title, axis labels, colormap, and metadata overlay.
- **Diverse Image Format Support**:
  - Supports saving images in various formats:
    - **PNG & PDF:** For publication and presentation.
    - **FITS & TIFF:** For astronomical research and data preservation.
    - **SVG:** Ideal for web and LaTeX applications.
- **Performance & Data Processing Enhancements**:
  - Improved HDF5 processing speed via multiprocessing and Dask for parallel data loading.
  - Integrated GPU acceleration using `CuPy` alongside Numba JIT compilation to accelerate custom histogram computations.
