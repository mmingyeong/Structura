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


## **[2025-03-06] - 🚀 Feature Update: Unit Conversions & Visualizer Enhancements (Structura 0.2.1)**
### 🎯 Key Features

#### **🔹 utils.py**
- **New Unit Conversion Functions**:
  - Added `cMpc_to_cMpc_h()` to convert distances from cMpc to cMpc/h.
  - Added `cMpc_h_to_cMpc()` to convert distances from cMpc/h to cMpc.

#### **🔹 Visualizer**
- **Enhanced Logging and Output Details**:
  - Modified to output bins count, resolution, and data unit in generated plots.
- **Automatic Optimal Bins Calculation**:
  - Integrated an algorithm to automatically compute optimal bins based on data count.
- **Recommended Bins for 2D Histograms**:
  - Introduced a function that suggests appropriate bin counts for generating 2D histograms at various resolutions.

#### **🔹 General**
- **Comprehensive Code Improvements**:
  - Enhanced logging, updated docstrings, and refined overall code documentation.
  - Reviewed and updated code to comply with `ruff` linter standards.
  - Upgraded project compatibility to Python 3.10.
  - Added documentation badges to the project README for improved visibility.


## **[2025-03-10] - 🚀 Major Update: Optimized Data Loading with Dask (Structura 0.3.0)**
### 🎯 Key Features

#### **🔹 DataLoader**
- **🚀 Transition to Dask for Parallel Processing**:
  - Replaced `ProcessPoolExecutor` with **Dask** for distributed and efficient parallel processing.
  - Implemented `dask.delayed`, `dask.compute`, and `dask.distributed.Client` to enhance scalability.
  - Added **real-time task monitoring** using `as_completed` and `tqdm` for tracking execution progress.
  - Optimized Dask worker configuration (`heartbeat_interval`, `timeout`) for stability.

- **⚡ Optimized Batch & Task Grouping**:
  - Introduced `BATCH_SIZE = 5` (previously unbatched individual file processing).
  - Implemented `GROUP_SIZE = 2` to **reduce scheduling overhead** and minimize input dependency size.
  - Developed `group_tasks()` function to efficiently merge Dask tasks and optimize execution.

- **🎯 Intelligent GPU Selection**:
  - Introduced `get_least_used_gpu()` function to **automatically select the GPU with the most available memory**.
  - Cached GPU memory status (`_gpu_memory_cache`) with a **5-second refresh interval** to reduce unnecessary queries.
  - Implemented `cupy.cuda.runtime.memGetInfo()` for **real-time GPU memory tracking**.

- **🛠️ Improved Memory Management**:
  - Adopted `np.load(..., mmap_mode="r")` to **minimize memory overhead** and avoid unnecessary copies.
  - Enhanced garbage collection (`gc.collect()`) throughout the pipeline for efficient resource cleanup.

- **📊 Statistical Insights (Optional)**:
  - Introduced `statistics=True` flag to conditionally compute **mean, median, min, and max** across dataset columns.
  - Added **projection axis range logging** for better debugging of filtered datasets.

#### **🔹 Enhanced Error Handling & Logging**
- **🔎 Robust File Type Validation**:
  - Removed `.npz` support to enforce `.npy` usage (prevent unexpected data structure issues).
  - Early validation to prevent processing of unsupported file formats.
  
- **⚠️ Detailed Exception Handling**:
  - Added `MemoryError` handling during `np.concatenate()` to avoid crashes on large datasets.
  - Implemented more descriptive `logger.error()` messages for file loading failures and Dask task errors.

- **🖥️ Improved Execution Logging**:
  - Added **total execution time tracking** for dataset processing.
  - Displayed **real-time batch loading progress** using `tqdm` and `ProgressBar`.

#### **🔹 Optimized Data Concatenation & GPU Processing**
- **🔄 Efficient Data Merging**:
  - Switched from `np.vstack()` to `np.concatenate()` for **better memory efficiency**.
  - Implemented memory-efficient chunk processing to prevent crashes on large datasets.

- **🔥 GPU Acceleration Improvements**:
  - Implemented `cp.cuda.Device(gpu_id).use()` to **explicitly allocate selected GPU** before processing.
  - Ensured seamless transition between **NumPy (CPU) and CuPy (GPU) processing** based on `use_gpu` flag.


## **[2025-03-10] - 🚀 Major Update: Optimized Density Map Comparison & 2D Testing (Structura 0.4.0)**
### 🎯 Key Features

#### **🔹 Density Map Comparison & Analysis**
- **Direct vs. FFT-based Density Calculation Comparison**:
  - Updated **Density_ex.py** to compute and save density maps using the basic (direct kernel) method.
  - Updated **Fft_kde_ex.py** to compute and save density maps using the FFT-based approach.
  - Modified **Plot_density_map.py** to generate figures that visually compare the density maps from both methods.
  - Integrated numerical metrics such as **RMSE, MAE, and Pearson Correlation** to facilitate detailed quantitative analysis.

#### **🔹 2D Testing Framework**
- **2D Dataset Generation**:
  - Developed a new 2D synthetic dataset generation function using inverse transform sampling based on a sine-modulated distribution.
- **2D Density Map Calculation**:
  - Created new classes **DensityCalculator2D** and **FFTKDE2D** to perform kernel density estimation in 2D.
  - This 2D framework significantly reduces memory usage compared to 3D data, enabling faster testing and iterative development.
- **Visualization Enhancements**:
  - Added multiple visualization methods including:
    - Standard imshow-based plots.
    - pcolormesh-based plots with optional log scaling.
    - Scatter plots for direct, cell-by-cell density value comparisons.
  - All figures are saved as high-resolution PNG files to support detailed visual examination.

#### **🔹 Data Loading Improvements**
- **Cube/Subcube Data Loading**:
  - Updated the data loading module in **data_load_ex.py** to load and concatenate data from all `.npy` files within a specified cube.
  - Implemented periodic boundary conditions to seamlessly integrate subcube data for large-scale tests.

