# Structura  

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://beta.ruff.rs/docs/)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue)](https://your-docs-link.com)


**Structura** is a Python library for **analyzing and visualizing the large-scale structure (LSS) of the universe**.  
It is designed to efficiently **load, process, and visualize** cosmological simulation data from **TNG300, TNG100, Illustris**, and more.  

---

## 🚀 Installation  
```bash
git clone https://github.com/mmingyeong/Structura.git
cd Structura
pip install -r requirements.txt
```

---

## 🛠 Usage  

### **1️⃣ Check System Compatibility**
Before running large-scale computations, verify your environment:  
```bash
python src/example/src_ex/check_system_ex.py
```
📌 **Checks:**  
- GPU availability & specs  
- CPU core count & RAM size  
- Python, NumPy, CuPy, and PyTorch versions  

---

### **2️⃣ Convert HDF5 Simulation Data to NPY/NPZ**
Run the conversion script for snapshot data:  
```bash
python src/example/src_ex/converter_ex.py
```
📌 **Features:**  
- Automatically detects and skips files if already converted  
- Supports `.hdf5` input, outputs `.npy` or `.npz`  

---

### **3️⃣ Visualize Simulation Data**
To generate 2D visualizations:  
```bash
python src/example/src_ex/visualization_ex.py
```
📌 **Features:**  
- Supports **scatter plots, density maps, and histograms**  
- Optimized for large cosmological datasets  

---

## 📊 Data Processing Workflow (in progress)

Structura follows a structured data processing pipeline:  

```mermaid
graph LR
    A[check_system_ex.py: System Compatibility Check] --> B[Convert.py: HDF5 → NPY/NPZ Conversion]
    B --> C[visualization_ex.py: Generate 2D Histogram Images]
    C --> D[density_ex.py: Generate Density Map]
    D --> E[analysis_ex.py: Generate Data Analysis Report]
    
    %% Adjust the diagram appearance with more neutral colors
    classDef default fill:#f0f0f0,stroke:#333,stroke-width:2px,font-family:Arial;
    class A,B,C,D,E default;

```
---

## 📂 Project Structure  

```plaintext
Structura/
│── src/                          # Structura source code
│   ├── etc/                       # Configuration files
│   │   ├── config.yml              # User-editable configurations
│   ├── example/                   # Example scripts
│   │   ├── check_system_ex.py      # Example: Run system diagnostics
│   │   ├── converter_ex.py         # Example: Convert HDF5 to NPY
│   │   ├── visualization_ex.py     # Example: Generate visualizations
│   ├── results/                   # Processed data storage
│   ├── structura/                 # Core library modules
│   │   ├── __init__.py             # Package initialization
│   │   ├── config.py               # Configuration settings
│   │   ├── data_loader.py          # Data loading and filtering
│   │   ├── density.py              # Density calculation algorithms
│   │   ├── visualization.py        # 2D/3D visualization tools
│   │   ├── convert.py              # HDF5 to NPY/NPZ conversion
│   │   ├── system_checker.py       # System diagnostics
│   │   ├── utils.py                # Helper functions
│── log/                           # Runtime logs (auto-generated)
│── README.md                      # Documentation
│── CHANGELOG.md                   # Version history
│── LICENSE                        # License information
│── pyproject.toml                 # Project dependencies & setup
│── CONTRIBUTING.md                 # Contribution guidelines
```

---

## 📌 Features  
✔ **Supports Multiple Data Formats** – Load `.npz`, `.hdf5`, `.csv` simulation data  
✔ **Flexible Data Loading** – Filter data by X-axis range and other parameters  
✔ **Density Calculation** – Compute density fields using **SPH (Smoothed Particle Hydrodynamics) and grid-based methods**  
✔ **2D Visualization** – Generate histograms, scatter plots, and density maps  
✔ **Optimized for Large Datasets** – Utilizes `CuPy` for GPU acceleration and parallel processing  
✔ **Execution Logging** – Tracks execution time and filtering ranges  
✔ **Automated Data Conversion** – Convert HDF5 simulation snapshots to NPY/NPZ  
✔ **System Diagnostics** – Check **GPU, CPU, RAM**, and software dependencies before running computations  

---

## ⚖ License  
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.  

---

## 🤝 Contributions  
Contributions are welcome! Feel free to fork the repository and submit pull requests.  
For major changes, please open an issue to discuss your ideas. 🚀  

---

## 📧 Contact  
For questions or collaboration inquiries, contact **Mingyeong Yang** at [mmingyeong@kasi.re.kr](mailto:mmingyeong@kasi.re.kr).  
```

---