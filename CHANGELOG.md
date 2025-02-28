# 📝 Changelog

## [2025-02-28] - 🚀 Initial Release: Structura 0.1.0
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

### 📖 Documentation
- **Added `README.md`**: Detailed installation & usage instructions
- **Created `CHANGELOG.md`**: Tracks modifications and future plans

### 🔮 Future Plans
- Optimize performance for **multi-GPU environments**
- Support additional simulation formats (e.g., Gadget, Arepo)
