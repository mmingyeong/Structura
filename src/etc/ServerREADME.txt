
GPU uses in seondeok cluster

CUDA 11.8
ssh anode13   or anode12  or anode11
module unload cuda110
module load cuda118
module load gcc/9.3.0
conda activate new_env # CUDA 11.8
conda activate 110 # CUDA 11.0
export NUMBA_THREADING_LAYER=omp
Python
cd /home/users/mmingyeong/structura/Structura/src/example/

# GPU status check
nvcc --version
nvidia-smi -l 2 # update in every 2 sec`

# Cupy 설치
pip install cupy-cuda11x
or pip install numpy cupy-cuda11x scipy dask distributed

# 참고
conda create -n new_env python=3.10
conda activate new_env
pip install numpy cupy-cuda11x scipy dask distributed
pip install -r requirements.txt


[mmingyeong@anode13 ~]$ module avail

------------------------------------------------- /usr/share/Modules/modulefiles --------------------------------------------------
anaconda3/2023.03/envs/py39 cuda118                     intel18u1                   namaster
cmake/3.24.1                dot                         intel18u1m32                null
cmake/3.24.1.gcc930         gcc/5.5.0                   llvm/14.0.6                 ompi211i
cuda102                     gcc/9.3.0                   module-git                  py37
cuda110                     golang                      module-info                 sgl/3.7.3
cuda111                     gsl26                       modules                     use.own

-------------------------------------------------------- /etc/modulefiles ---------------------------------------------------------
mpi/compat-openmpi16-x86_64  mpi/mpich-x86_64             mpi/mvapich2-2.2-psm2-x86_64 mpi/mvapich2-psm-x86_64
mpi/mpich-3.0-x86_64         mpi/mvapich2-2.0-psm-x86_64  mpi/mvapich2-2.2-psm-x86_64  mpi/mvapich2-x86_64
mpi/mpich-3.2-x86_64         mpi/mvapich2-2.0-x86_64      mpi/mvapich2-2.2-x86_64


2025-03-06 12:31:31,019 - INFO - 🔍 Starting system check (SystemChecker last updated: 2025-03-06)
2025-03-06 12:31:31,084 - INFO - 📅 Environment analysis based on SystemChecker update: 2025-03-06
2025-03-06 12:31:31,084 - WARNING - ⚠️ Python 3.10.16 is outdated. Upgrade to Python 3.9+ recommended.
2025-03-06 12:31:31,084 - WARNING - ⚠️ PyTorch is not installed. Consider installing it for deep learning.
2025-03-06 12:31:31,084 - INFO - 🐍 Python Version: 3.10.16
2025-03-06 12:31:31,084 - INFO - 📦 NumPy Version: 1.23.5
2025-03-06 12:31:31,084 - INFO - 📦 CuPy Version: 13.4.0
2025-03-06 12:31:31,084 - INFO - 📦 PyTorch Version: Not Installed
2025-03-06 12:31:31,084 - INFO - 🔍 Running System Check...
2025-03-06 12:31:31,085 - INFO - 🖥 GPU 0: A100-PCIE-40GB (39.59 GB, 108 CUDA cores, 1.41 GHz)
2025-03-06 12:31:31,085 - INFO - 🖥 GPU 1: A100-PCIE-40GB (39.59 GB, 108 CUDA cores, 1.41 GHz)
2025-03-06 12:31:31,085 - INFO - 🖥 GPU 2: A100-PCIE-40GB (39.59 GB, 108 CUDA cores, 1.41 GHz)
2025-03-06 12:31:31,085 - INFO - 🖥 GPU 3: A100-PCIE-40GB (39.59 GB, 108 CUDA cores, 1.41 GHz)
2025-03-06 12:31:31,085 - INFO - 🖥 GPU 4: A100 80GB PCIe (79.35 GB, 108 CUDA cores, 1.41 GHz)
2025-03-06 12:31:31,085 - INFO - 🖥 GPU 5: A100-PCIE-40GB (39.59 GB, 108 CUDA cores, 1.41 GHz)
2025-03-06 12:31:31,085 - INFO - 🖥 GPU 6: A100-PCIE-40GB (39.59 GB, 108 CUDA cores, 1.41 GHz)
2025-03-06 12:31:31,085 - INFO - 🖥 CPU Cores: 96
2025-03-06 12:31:31,085 - INFO - 💾 Total RAM: 376.56 GB
2025-03-06 12:31:31,085 - INFO - 🚀 Using GPU: True
2025-03-06 12:31:31,085 - INFO - ✅ System check complete.
2025-03-06 12:31:31,085 - INFO - ✅ GPU is available. Computation will be accelerated.
2025-03-06 12:31:31,085 - INFO - ⏳ System check completed in 00:00:00 (0.06 seconds).


=== 전체 키 목록 ===
Header
PartType1
PartType1/Coordinates

=== 이름 및 타입 목록 ===
Header (Group)
PartType1 (Group)
PartType1/Coordinates (Dataset)
PartType1
Coordinates