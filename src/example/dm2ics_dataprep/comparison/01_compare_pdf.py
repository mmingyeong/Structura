import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

# 설정
CANDIDATE_KEYS = ["density", "density_map", "kde_density", "fft_density"]
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DIRS = {
    "z=127": "/caefs/data/IllustrisTNG/densitymap-ics-hdf5",
    "z=0": "/caefs/data/IllustrisTNG/densitymap-99-dm-hdf5",
}

COLORS = ["blue", "orange", "green", "red", "purple", "cyan"]

def log(msg):
    print(f"[log] {msg}")

def safe_log10(x, eps=1e-12):
    return np.log10(np.maximum(x, eps))

def load_density_info(z_label, path):
    fname = os.path.basename(path)
    label = f"{z_label} | {fname.replace('.hdf5', '')}"
    try:
        with h5py.File(path, "r") as f:
            for key in CANDIDATE_KEYS:
                if key in f:
                    data = f[key][:]
                    break
            else:
                raise KeyError(f"No valid dataset key found in {path}")

        box_volume = 205.0 ** 3
        mean_density = np.sum(data) / box_volume
        flat = data.ravel()
        log_density = safe_log10(flat)
        log_overdensity = safe_log10(flat / mean_density)

        stats = {
            "mean": np.mean(log_overdensity),
            "std": np.std(log_overdensity),
            "skew": skew(log_overdensity),
            "kurt": kurtosis(log_overdensity)
        }

        return z_label, label, log_density, log_overdensity, mean_density, stats

    except Exception as e:
        log(f"[✗] Failed to load {label}: {e}")
        return z_label, label, None, None, None, None

# 데이터 수집
density_logs = {"z=127": {}, "z=0": {}}
overdensity_logs = {"z=127": {}, "z=0": {}}
stats_summary = {"z=127": {}, "z=0": {}}
log("▶ Start parallel density map loading...")

with ThreadPoolExecutor(max_workers=6) as executor:
    futures = []
    for z_label, folder in DIRS.items():
        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".hdf5"):
                path = os.path.join(folder, fname)
                futures.append(executor.submit(load_density_info, z_label, path))
    for future in futures:
        z_label, label, log_vals, log_overdelta, mean, stats = future.result()
        if log_vals is not None:
            density_logs[z_label][label] = log_vals
            overdensity_logs[z_label][label] = log_overdelta
            stats_summary[z_label][label] = stats
            log(f"[✓] Loaded: {label} (mean = {mean:.2f})")

def plot_zpair_subplot_with_stats(log_density_dict, log_overdensity_dict, stats_dict,
                                   z_label, filename, peak_window=1):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    bins = 100

    # --- log10(density) subplot ---
    # 전체 범위 히스토그램 계산 (bin edges)
    xs1_full = np.linspace(1, 7, bins + 1)
    for i, (label, log_vals) in enumerate(log_density_dict.items()):
        hist, bin_edges = np.histogram(log_vals, bins=xs1_full, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # peak 찾기 (가장 높은 peak 하나만)
        peaks, _ = find_peaks(hist)
        if len(peaks) == 0:
            peak_idx = np.argmax(hist)  # peak 없으면 최대값 인덱스 사용
        else:
            peak_idx = peaks[np.argmax(hist[peaks])]
        peak_x = bin_centers[peak_idx]

        # peak 기준 ±peak_window 범위 설정
        x_min = peak_x - peak_window
        x_max = peak_x + peak_window

        # 해당 범위 내 bin 선택
        mask = (bin_centers >= x_min) & (bin_centers <= x_max)
        axs[0].plot(bin_centers[mask], hist[mask], label=label, color=COLORS[i % len(COLORS)])

    axs[0].set_title(f"{z_label} | log10(Density) (peak zoomed)")
    axs[0].set_xlabel("log10(Density)")
    axs[0].set_ylabel("Probability Density")
    axs[0].legend(fontsize=8)
    axs[0].set_xlim(x_min, x_max)

    # --- log10(overdensity) subplot ---
    xs2_full = np.linspace(-3, 3, bins + 1)
    stat_lines = []
    for i, (label, log_vals) in enumerate(log_overdensity_dict.items()):
        hist, bin_edges = np.histogram(log_vals, bins=xs2_full, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        peaks, _ = find_peaks(hist)
        if len(peaks) == 0:
            peak_idx = np.argmax(hist)
        else:
            peak_idx = peaks[np.argmax(hist[peaks])]
        peak_x = bin_centers[peak_idx]

        x_min = peak_x - peak_window
        x_max = peak_x + peak_window

        mask = (bin_centers >= x_min) & (bin_centers <= x_max)
        axs[1].plot(bin_centers[mask], hist[mask], label=label, color=COLORS[i % len(COLORS)])

        stats = stats_dict[label]
        line = f"{label}\nμ={stats['mean']:.2f}, σ={stats['std']:.2f}, S={stats['skew']:.2f}, K={stats['kurt']:.2f}"
        stat_lines.append(line)

    axs[1].set_title(f"{z_label} | log10(δ) = log10(density / mean) (peak zoomed)")
    axs[1].set_xlabel("log10(δ)")
    axs[1].legend(fontsize=8)
    axs[1].set_xlim(x_min, x_max)

    full_text = "\n\n".join(stat_lines)
    axs[1].text(1.02, 0.97, full_text, transform=axs[1].transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    fig.suptitle(f"Density and Overdensity Comparison for {z_label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), format="png", bbox_inches="tight")
    plt.close()
    log(f"✅ Saved: {filename}")


# 그림 저장
plot_zpair_subplot_with_stats(
    density_logs["z=127"],
    overdensity_logs["z=127"],
    stats_summary["z=127"],
    z_label="z=127",
    filename="z127_comparison_stats.png",
    peak_window=2
)

plot_zpair_subplot_with_stats(
    density_logs["z=0"],
    overdensity_logs["z=0"],
    stats_summary["z=0"],
    z_label="z=0",
    filename="z0_comparison_stats.png",
    peak_window=3
)
