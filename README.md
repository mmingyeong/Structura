
# Structura  
**Structura** is a Python library for **analyzing and visualizing the large-scale structure (LSS) of the universe**.  
It is designed to efficiently **load, process, and visualize** cosmological simulation data from **TNG300, TNG100, Illustris**, and more.  

---

## 📌 Features  
✔ **Supports Multiple Data Formats** – Load `.npz`, `.hdf5`, `.csv` simulation data  
✔ **Flexible Data Loading** – Filter data by X-axis range and other parameters  
✔ **Density Calculation** – Compute density fields using **SPH (Smoothed Particle Hydrodynamics) and grid-based methods**  
✔ **2D & 3D Visualization** – Generate histograms, scatter plots, and density maps  
✔ **Optimized for Large Datasets** – Utilizes `CuPy` for GPU acceleration and parallel processing  

---

## 🚀 Installation  
```bash
git clone https://github.com/mmingyeong/Structura.git
cd Structura
pip install -r requirements.txt
```

---

## 📖 Usage Example  
### **1️⃣ Load Simulation Data**  
```python
from structura.data_loader import DataLoader

loader = DataLoader(folder_path="data/", file_type="npz")
positions = loader.load_all()  # Load (X, Y, Z) coordinates
```

### **2️⃣ Compute Density Field**  
```python
from structura.density import DensityCalculator

density_calc = DensityCalculator(method="grid", grid_size=128)
density_map = density_calc.compute_density(positions)
```

### **3️⃣ Visualize the Data**  
```python
from structura.visualization import Visualizer

viz = Visualizer(bins=200)
viz.plot_2d_histogram(positions)  # 2D histogram
viz.plot_3d_scatter(positions, sample_size=50000)  # 3D scatter plot
```

---

## 🔮 Future Plans  
✔ Implement SPH-based density estimation  
✔ Develop 3D volume rendering for density visualization  
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
