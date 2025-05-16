#!/bin/bash
#PBS -N sum_density_maps
#PBS -q long
#PBS -t 0-3%1
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o /home/users/mmingyeong/structura/Structura/src/example/dm2ics_dataprep/input_z=0/logs/sum_density_${PBS_JOBID}_${PBS_ARRAYID}.out
#PBS -d /home/users/mmingyeong/structura/Structura/src/example/dm2ics_dataprep/input_z=0/

# ✅ 환경 설정
source ~/.bashrc
conda activate new_env

# ✅ 로그 디렉토리 생성
LOGDIR="logs"
mkdir -p "$LOGDIR"
OUT_LOG="${LOGDIR}/sum_density_${PBS_JOBID}_${PBS_ARRAYID}.out"

# ✅ 로그 출력 리디렉션
exec > "$OUT_LOG" 2>&1

# ✅ 처리할 폴더 목록 정의 (res_0.16 제외)
FOLDER_LIST=($(find /caefs/data/IllustrisTNG/densitymap-99-dm-hdf5 -mindepth 1 -maxdepth 1 -type d | grep -v res_0.16 | xargs -n 1 basename))
TARGET_FOLDER=${FOLDER_LIST[$PBS_ARRAYID]}

echo "처리할 폴더: $TARGET_FOLDER"

# ✅ 실행
python /home/users/mmingyeong/structura/Structura/src/example/dm2ics_dataprep/input_z=0/02_compute_final_density_map.py "$TARGET_FOLDER"
