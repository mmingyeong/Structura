#!/bin/bash
#PBS -N z=0calculation
#PBS -q long
#PBS -t 0-599%10
#PBS -l select=1:ncpus=2:mem=32gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o /home/users/mmingyeong/structura/Structura/src/example/dm2ics_dataprep/input_z=0/logs/density_single_${PBS_JOBID}_${PBS_ARRAYID}.out
#PBS -d /home/users/mmingyeong/structura/Structura/src/example/dm2ics_dataprep/input_z=0/

# ✅ 환경 설정
source ~/.bashrc
conda activate new_env

# ✅ 로그 디렉토리 생성
LOGDIR="logs"
mkdir -p "$LOGDIR"

OUT_LOG="${LOGDIR}/dm_${PBS_JOBID}_${PBS_ARRAYID}.out"
TIME_LOG="${LOGDIR}/dm_${PBS_JOBID}_${PBS_ARRAYID}_time.log"

# ✅ 로그 출력 리디렉션
exec > "$OUT_LOG" 2>&1

# ✅ 작업 시작
START_TIME=$(date +%s)
echo "▶️ [START] Task for PBS_ARRAYID = $PBS_ARRAYID at $(date)"

# ✅ 정확한 경로의 코드 실행
python /home/users/mmingyeong/structura/Structura/src/example/dm2ics_dataprep/input_z=0/01_compute_densitymap_all.py "$PBS_ARRAYID"

# ✅ 작업 종료 시간 기록
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

echo "✅ [END] Task for PBS_ARRAYID = $PBS_ARRAYID at $(date)"
echo "⏱️ Elapsed time: ${ELAPSED_TIME} seconds" | tee "$TIME_LOG"
