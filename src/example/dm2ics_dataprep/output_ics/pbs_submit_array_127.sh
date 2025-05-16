#!/bin/bash
#PBS -N ics
#PBS -q long
#PBS -t 0-313%10
#PBS -l select=1:ncpus=2:mem=32gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -d /home/users/mmingyeong/structura/Structura/src/example/dm2ics_dataprep/output_ics

# ✅ Array index를 정수로 안전하게 추출
SAFE_ARRAYID=$(echo "$PBS_ARRAYID" | sed 's/[^0-9]//g')

# ✅ 환경 설정
source ~/.bashrc
conda activate new_env

# ✅ 로그 디렉토리 생성
LOGDIR="logs"
mkdir -p "$LOGDIR"

# ✅ 로그 파일 경로 지정 (여기서부터는 우리가 통제 가능)
OUT_LOG="${LOGDIR}/dm_${PBS_JOBID}_${SAFE_ARRAYID}.out"
TIME_LOG="${LOGDIR}/dm_${PBS_JOBID}_${SAFE_ARRAYID}_time.log"

# ✅ 로그 출력 리디렉션
exec > "$OUT_LOG" 2>&1

# ✅ 작업 시작
start_time=$(date +%s)
echo "▶️ [START] Task for PBS_ARRAYID = $PBS_ARRAYID at $(date)"
echo "[INFO] Logging to $OUT_LOG"

# ✅ 스크립트 실행
echo "[INFO] Executing 01_compute_densitymap_ics.py"
python /home/users/mmingyeong/structura/Structura/src/example/dm2ics_dataprep/output_ics/01_compute_densitymap_ics.py "$PBS_ARRAYID"

# ✅ 종료 시간 출력
end_time=$(date +%s)
echo "===== Job Ended: $(date) ====="
echo "⏱️  Total runtime: $((end_time - start_time)) seconds"
