#!/bin/bash
#PBS -N z=0
#PBS -q long
#PBS -t 0-599
#PBS -l select=1:ncpus=2:mem=32gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o logs/density_single_${PBS_JOBID}_${PBS_ARRAYID}.out
#PBS -d /caefs/user/mmingyeong/structura/Structura/src/example/density_kde_seq_chunks

source ~/.bashrc
conda activate new_env

# Create log directory if not exists
LOGDIR="logs"
mkdir -p "$LOGDIR"

OUT_LOG="${LOGDIR}/dm_${PBS_JOBID}_${PBS_ARRAYID}.out"
TIME_LOG="${LOGDIR}/dm_${PBS_JOBID}_${PBS_ARRAYID}_time.log"

exec > "$OUT_LOG" 2>&1

START_TIME=$(date +%s)
echo "▶️ [START] Task for PBS_ARRAYID = $PBS_ARRAYID at $(date)"

python 01_compute_densitymap_all.py "$PBS_ARRAYID"

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

echo "✅ [END] Task for PBS_ARRAYID = $PBS_ARRAYID at $(date)"
echo "⏱️ Elapsed time: ${ELAPSED_TIME} seconds" | tee "$TIME_LOG"
