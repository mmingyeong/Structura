import os
import h5py
import numpy as np

def print_min_max_coords(data_dir, label):
    hdf5_files = [f for f in os.listdir(data_dir) if f.endswith(('.h5', '.hdf5'))]
    if not hdf5_files:
        print(f"No HDF5 files found in {label} directory: {data_dir}")
        return

    file_path = os.path.join(data_dir, hdf5_files[0])
    print(f"\n[{label}] Opening file: {file_path}")

    with h5py.File(file_path, 'r') as f:
        coords_path = 'PartType1/Coordinates'
        if coords_path in f:
            coords = f[coords_path][:]
            print(f"Dataset '{coords_path}' shape: {coords.shape}, dtype: {coords.dtype}")
            print(f"Coordinates min: {np.min(coords, axis=0)}")
            print(f"Coordinates max: {np.max(coords, axis=0)}")
        else:
            print(f"Dataset '{coords_path}' not found in the file.")

def print_min_max_density(data_dir, label):
    hdf5_files = [f for f in os.listdir(data_dir) if f.endswith(('.h5', '.hdf5'))]
    if not hdf5_files:
        print(f"No HDF5 files found in {label} directory: {data_dir}")
        return

    file_path = os.path.join(data_dir, hdf5_files[0])
    print(f"\n[{label}] Opening file: {file_path}")

    with h5py.File(file_path, 'r') as f:
        if 'density' in f:
            data = f['density'][:]
            dataset_name = 'density'
        else:
            # density 데이터셋 없으면 첫 번째 키 사용
            first_key = list(f.keys())[0]
            data = f[first_key][:]
            dataset_name = first_key
        print(f"Dataset '{dataset_name}' shape: {data.shape}, dtype: {data.dtype}")
        print(f"Min value: {np.min(data)}")
        print(f"Max value: {np.max(data)}")

if __name__ == "__main__":
    # 기존 좌표 min/max 출력
    ics_dir = "/caefs/data/IllustrisTNG/snapshot-0-ics"
    z0_dir = "/caefs/data/IllustrisTNG/TNG300_snapshot99"

    print_min_max_coords(ics_dir, "z=127 (Initial Conditions) Coordinates")
    print_min_max_coords(z0_dir, "z=0 (Snapshot 99) Coordinates")

    # 추가 density min/max 출력
    dm99_dir = "/caefs/data/IllustrisTNG/densitymap-99-dm-hdf5"
    ics_density_dir = "/caefs/data/IllustrisTNG/densitymap-ics-hdf5"

    print_min_max_density(dm99_dir, "z=0 Dark Matter Density Map")
    print_min_max_density(ics_density_dir, "z=127 Initial Conditions Density Map")
