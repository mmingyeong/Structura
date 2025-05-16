#!/bin/bash
#PBS -N ics_density_merge
#PBS -q long
#PBS -t 0-1%50
#PBS -l select=1:ncpus=2:mem=32gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o /home/users/mmingyeong/structura/Structura/src/example/dm2ics_dataprep/output_ics/logs/density_merge_array.o$PBS_ARRAY_INDEX
#PBS -e //home/users/mmingyeong/structura/Structura/src/example/dm2ics_dataprep/output_ics/logs/density_merge_array.e$PBS_ARRAY_INDEX
#PBS -V

# ✅ 환경 설정
source ~/.bashrc
conda activate new_env

# ✅ 로그 디렉토리 설정
LOGDIR="/home/users/mmingyeong/structura/Structura/src/example/dm2ics_dataprep/output_ics/logs"
mkdir -p "$LOGDIR"

OUT_LOG="${LOGDIR}/dm_${PBS_JOBID}_${PBS_ARRAYID}.out"
TIME_LOG="${LOGDIR}/dm_${PBS_JOBID}_${PBS_ARRAYID}_time.log"

# ✅ 로그 출력 리디렉션
exec > "$OUT_LOG" 2>&1

# ✅ 작업 시작
start_time=$(date +%s)
echo "▶️ [START] Task for PBS_ARRAY_INDEX = $PBS_ARRAYID at $(date)"
echo "[DEBUG] PBS_ARRAY_INDEX = $PBS_ARRAYID"

python "/home/users/mmingyeong/structura/Structura/src/example/dm2ics_dataprep/output_ics/02_compute_final_density_map.py" "$PBS_ARRAYID"

# ✅ 종료 시간 출력
end_time=$(date +%s)
echo "===== Job Ended: $(date) ====="
echo "$((end_time - start_time))" > "$TIME_LOG"
echo "⏱️  Total runtime: $((end_time - start_time)) seconds"

