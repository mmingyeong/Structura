import os
import glob
import h5py
import numpy as np
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def compute_overdensity_mean(filepath):
    """
    파일 내 밀도 데이터를 읽어 overdensity를 계산하고,
    overdensity 배열의 평균값을 반환.
    """
    try:
        with h5py.File(filepath, 'r') as f:
            if 'density' in f:
                density = f['density'][:]
            else:
                key = list(f.keys())[0]
                density = f[key][:]
        rho_bar = np.mean(density)
        overdensity = (density - rho_bar) / rho_bar
        mean_overdensity = np.mean(overdensity)
        return mean_overdensity
    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        return None

def main():
    data_dirs = [
        "/caefs/data/IllustrisTNG/densitymap-99-dm-hdf5",
        "/caefs/data/IllustrisTNG/densitymap-ics-hdf5"
    ]
    hdf5_files = []
    for d in data_dirs:
        files = sorted(glob.glob(os.path.join(d, "*.hdf5")))
        hdf5_files.extend(files)

    if not hdf5_files:
        logging.error("No HDF5 files found.")
        return

    logging.info(f"Found {len(hdf5_files)} files. Computing mean overdensity...")

    results_99 = {}
    results_ics = {}

    for filepath in hdf5_files:
        mean_od = compute_overdensity_mean(filepath)
        if mean_od is not None:
            filename = os.path.basename(filepath)
            # 경로에 '99'가 포함되면 snapshot 99, 'ics'가 포함되면 ics로 분류
            if "99" in filepath:
                results_99[filename] = mean_od
            elif "ics" in filepath:
                results_ics[filename] = mean_od
            else:
                logging.warning(f"File {filename} does not match known categories.")

    # snapshot 99 결과 출력
    logging.info("=== Snapshot 99 Mean Overdensity ===")
    for fname, val in sorted(results_99.items()):
        logging.info(f"{fname}: Mean overdensity = {val:.6e}")

    # ics 결과 출력
    logging.info("=== ICS Mean Overdensity ===")
    for fname, val in sorted(results_ics.items()):
        logging.info(f"{fname}: Mean overdensity = {val:.6e}")

if __name__ == "__main__":
    main()
