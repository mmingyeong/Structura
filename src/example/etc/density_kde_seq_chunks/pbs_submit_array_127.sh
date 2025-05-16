#!/bin/bash
#PBS -N ics
#PBS -q long
#PBS -l select=1:ncpus=2:mem=32gb
#PBS -l walltime=06:00:00
#PBS -t 0-313
#PBS -j oe
#PBS -o /home/users/mmingyeong/structura/Structura/src/example/density_kde_seq_chunks/logs/density_single_${PBS_JOBID}_${PBS_ARRAYID}.out
#PBS -d /caefs/user/mmingyeong/structura/Structura/src/example/density_kde_seq_chunks

source ~/.bashrc
conda activate new_env

# Create log directory if not exists
LOGDIR="logs"
mkdir -p "$LOGDIR"

OUT_LOG="${LOGDIR}/dm_${PBS_JOBID}_${PBS_ARRAYID}.out"
TIME_LOG="${LOGDIR}/dm_${PBS_JOBID}_${PBS_ARRAYID}_time.log"

exec > "$OUT_LOG" 2>&1

# 확인용 정보 출력
echo "===== Job Started: $(date) ====="
echo "Job ID       : $PBS_JOBID"
echo "Array Index  : $PBS_ARRAYID"
echo "Host         : $(hostname)"
echo "Working Dir  : $(pwd)"
start_time=$(date +%s)

# 스크립트 실행
echo "[INFO] Executing 02_compute_densitymap_ics.py"
python 02_compute_densitymap_ics.py ${PBS_ARRAYID}

# 종료 시간 출력
end_time=$(date +%s)
echo "===== Job Ended: $(date) ====="
echo "⏱️  Total runtime: $((end_time - start_time)) seconds"
