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
