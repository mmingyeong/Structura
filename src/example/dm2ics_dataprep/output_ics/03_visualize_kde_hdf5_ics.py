import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# 설정
IMG_RESULTS_DIR = os.path.join(os.getcwd(), "results")
HDF5_DIR = "/caefs/data/IllustrisTNG/densitymap-ics-hdf5/"
os.makedirs(IMG_RESULTS_DIR, exist_ok=True)

LOG_OFFSET = 1e-6  # ICS 특성상 낮은 밀도 대비 강조

# 조건 분류용 prefix 추출 함수
def parse_condition_name(fname):
    parts = fname.replace(".hdf5", "").split("_")
    kernel = next((p for p in parts if "uniform" in p or "triangular" in p), "kernel?")
    dx = next((p for p in parts if p.startswith("dx")), "dx?")
    return f"{kernel}_{dx}"

def visualize_all_conditions():
    hdf5_files = sorted([
        os.path.join(HDF5_DIR, fname)
        for fname in os.listdir(HDF5_DIR)
        if fname.endswith(".hdf5")
    ])

    if not hdf5_files:
        print(f"[✗] No HDF5 files found in {HDF5_DIR}")
        return

    print(f"[INFO] Found {len(hdf5_files)} files")
    n = len(hdf5_files)
    if n > 4:
        print("[!] More than 4 files detected. Only the first 4 will be shown in 2x2 grid.")
        hdf5_files = hdf5_files[:4]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
    axs = axs.flatten()

    for i, file_path in enumerate(hdf5_files):
        fname = os.path.basename(file_path)
        cond_label = parse_condition_name(fname)

        with h5py.File(file_path, "r") as f:
            density = None
            for key in ["density", "kde_density", "density_map", "density_kde"]:
                if key in f:
                    density = f[key][:]
                    break
            if density is None:
                print(f"[✗] Skipped (no dataset): {fname}")
                continue

            box_size = f.attrs.get("box_size", 205.0)
            if density.shape[0] > 0:
                grid_spacing = f.attrs.get("grid_spacing", box_size / density.shape[0])
            else:
                print(f"[✗] Invalid density shape: {fname}")
                continue

        # x-range projection (partial)
        x_min, x_max = 100, 120  # in cMpc/h
        i_min = int(x_min / grid_spacing)
        i_max = int(x_max / grid_spacing)
        i_max = min(i_max, density.shape[0])

        if i_max - i_min < 5:
            print(f"[!] Skipped (slice too narrow): {fname}")
            continue

        density_cut = density[i_min:i_max, :, :]
        density_proj = np.sum(density_cut, axis=0)
        log_proj = np.log10(density_proj + LOG_OFFSET)

        # ✅ Per-image min/max 계산
        vmin = np.percentile(log_proj, 5)
        vmax = np.percentile(log_proj, 99.9)

        extent = [0, grid_spacing * density.shape[1], 0, grid_spacing * density.shape[2]]
        ax = axs[i]
        im = ax.imshow(
            log_proj,
            origin="lower",
            cmap="cividis",
            extent=extent,
            interpolation="none",
            aspect="equal",
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(f"{cond_label}\nvmin={vmin:.3f}, vmax={vmax:.3f}", fontsize=10)
        ax.set_xlabel("Y (cMpc/h)")
        if i % 2 == 0:
            ax.set_ylabel("Z (cMpc/h)")
        fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

    fig.suptitle(f"ICS Density Comparison | x∈[100,120] cMpc/h | LOG_OFFSET={LOG_OFFSET}", fontsize=14)
    output_path = os.path.join(IMG_RESULTS_DIR, f"comparison_ics_individual_scaling.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[✓] Saved ICS subplot with individual color scales: {output_path}")

if __name__ == "__main__":
    visualize_all_conditions()
