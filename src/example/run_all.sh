#!/bin/bash

# Exit on any error
set -e

echo "▶ Running 02_compute_final_density_map.py ..."
python /home/users/mmingyeong/structura/Structura/src/example/density_fft_seq_chunks/02_compute_final_density_map.py
echo "✅ Done: 02_compute_final_density_map.py"

echo "▶ Running 03_visualize_kde_hdf5_snapshot99.py ..."
python /home/users/mmingyeong/structura/Structura/src/example/density_kde_seq_chunks/03_visualize_kde_hdf5_snapshot99.py
echo "✅ Done: 03_visualize_kde_hdf5_snapshot99.py"

echo "▶ Running 03_visualize_fft_hdf5_snapshot99.py ..."
python /home/users/mmingyeong/structura/Structura/src/example/density_fft_seq_chunks/03_visualize_fft_hdf5_snapshot99.py
echo "✅ Done: 03_visualize_fft_hdf5_snapshot99.py"

echo "🎉 All scripts executed successfully!"
