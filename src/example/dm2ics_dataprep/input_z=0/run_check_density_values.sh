#!/bin/bash
#PBS -N check_density_detailed
#PBS -q long
#PBS -t 0-3%1
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /home/users/mmingyeong/structura/Structura/src/example/dm2ics_dataprep/input_z=0/logs/check_${PBS_JOBID}_${PBS_ARRAYID}.out
#PBS -d /home/users/mmingyeong/structura/Structura/src/example/dm2ics_dataprep/input_z=0/

# ✅ 환경 설정
source ~/.bashrc
conda activate new_env

# ✅ 로그 디렉토리 생성
LOGDIR="logs"
mkdir -p "$LOGDIR"
OUT_LOG="${LOGDIR}/check_${PBS_JOBID}_${PBS_ARRAYID}.out"

# ✅ 로그 출력 리디렉션
exec > "$OUT_LOG" 2>&1

# ✅ 실행
python /home/users/mmingyeong/structura/Structura/src/example/dm2ics_dataprep/input_z=0/01-02_check_density_values_array.py "$PBS_ARRAYID"
