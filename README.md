# Structura  
**Structura** is a Python library for **analyzing and visualizing the large-scale structure (LSS) of the universe**.  
It is designed to efficiently **load, process, and visualize** cosmological simulation data from **TNG300, TNG100, Illustris**, and more.  

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
python src/example/check_system_ex.py
```
📌 **Checks:**  
- GPU availability & specs  
- CPU core count & RAM size  
- Python, NumPy, CuPy, and PyTorch versions  

---

### **2️⃣ Convert HDF5 Simulation Data to NPY/NPZ**
Run the conversion script for snapshot data:  
```bash
python src/example/converter_ex.py
```
📌 **Features:**  
- Automatically detects and skips files if already converted  
- Supports `.hdf5` input, outputs `.npy` or `.npz`  

---

### **3️⃣ Visualize Simulation Data**
To generate 2D/3D visualizations:  
```bash
python src/example/visualization_ex.py
```
📌 **Features:**  
- Supports **scatter plots, density maps, and histograms**  
- Optimized for large cosmological datasets  

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

### **🔹 Key Fixes & Enhancements**
✅ **Updated project structure based on the latest folder structure in the image**  
✅ **Added `visualization_ex.py` to the `Usage` section**  
✅ **Clarified what each example script does**  
✅ **Improved directory structure formatting for better readability**  

💡 **Now, the README is 100% aligned with the latest Structura setup!** 🚀🔥  
💡 **You're ready to push this to GitHub!** 🎯