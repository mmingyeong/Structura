
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

---

## 📂 Project Structure  

```plaintext
Structura/
│── src/                          # Structura source code
│   ├── structura/                 # Core library modules
│   │   ├── __init__.py             # Package initialization
│   │   ├── config.py               # Configuration settings
│   │   ├── data_loader.py          # Data loading and filtering
│   │   ├── density.py              # Density calculation algorithms
│   │   ├── visualization.py        # 2D/3D visualization tools
│   │   ├── utils.py                # Helper functions
│   ├── example_load_and_plot.py   # Example usage script
│── examples/                      # Example outputs
│   ├── results/                    # 2D histogram results
│── README.md                      # Documentation
│── CHANGELOG.md                   # Version history
│── LICENSE                        # License information

---

## 🚀 Installation  
```bash
git clone https://github.com/mmingyeong/Structura.git
cd Structura
pip install -r requirements.txt
```

---

## 🔮 Future Plans  
✔ Implement SPH-based density estimation  
✔ Support additional simulation formats (Gadget, Arepo output files)  
✔ Optimize GPU acceleration for large-scale datasets  

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
