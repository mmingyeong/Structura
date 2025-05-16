# README: dataprep

This document explains how the input and output datasets used in the deep learning-based reconstruction of initial conditions were generated. All relevant scripts are organized under the `example/dataprep/` directory.

---

## 🧪 Scientific Purpose

**Objective:** Train a deep learning model to reconstruct the initial condition density field at redshift z=127, using the observed large-scale structure at z=0.  
**Application:** Constrained simulations, structure formation modeling, and backward inference in cosmology.

---

## 🔧 Data Generation Workflow

All preprocessing steps to generate both the input (z=0) and output (z=127) density maps are implemented in this directory.

### 1. **Source Simulation**
- Simulation: **IllustrisTNG300-1**
- Particle type: **Dark Matter only (PartType1)**
- File: `/caefs/data/IllustrisTNG/ics.hdf5` (~500GB)
  - We use only `PartType1/Coordinates` (float32, 187.5GB)

### 2. **Directory Overview**
```
example/dataprep/
├── input_z=0/
│   └── 01_compute_densitymap_all.py      # Density map generation from z=0 snapshot
    └── pbs_submit_array_99.sh            # PBS array job script
├── output_ics/
│   ├── 00_split_hdf5.py                  # Split ics.hdf5 into chunks
│   ├── 01_compute_densitymap_ics.py       # Compute z=127 density map from split files
│   └── pbs_submit_array_127.sh            # PBS array job script
└── README.md                                 # This file
```

### 3. **Initial Condition Density Map (z=127)**

- **Step 1:** Split the full `ics.hdf5` file
  - Script: `00_split_hdf5.py`
  - Output: ~314 chunk files with coordinate datasets only

- **Step 2:** Compute density maps for each chunk
  - Script: `01_compute_densitymap_ics.py`
  - Execution: `qsub pbs_submit_array_127.sh`

- **Parameters:**
  - **Kernels:** Tophat (uniform) / Triangular
  - **Bandwidth:** Silverman's rule
  - **Cutoff:** 3σ (support truncation)
  - **Method:** Particle-centered KDE
  - **Grid bounds:** [0, 205]^3 cMpc/h
  - **Grid resolution:** 0.41 or 0.82 cMpc/h
  - **Grid shape:** typically (500, 500, 500)
  - **Data type:** float64

- **Output format:** HDF5, organized by group keys `kernel_dx{resolution}`

### 4. **z=0 Density Map**
- Script: `input_z=0/01_compute_densitymap_all.py`
- Same KDE parameters as above, applied to TNG300-1 snapshot at z=0

---

## 🌍 Scientific Context

This dataset enables training and evaluation of generative deep learning models that infer initial conditions from the present-day dark matter distribution. The goal is to examine whether large-scale structure traces the cosmic initial state.

- **Input to model:** z=0 DM density map (float64 grid)
- **Output:** z=127 IC density map (float64 grid)
- **Model candidates:** ViT, cGAN, UNet, Diffusion, etc.

### Reference
Sungwook E. Hong et al. (2021), *ApJ* 913, 76  
"Revealing the Local Cosmic Web from Galaxies by Deep Learning"

---

For further details on model training or data access paths, please refer to the main project documentation.

