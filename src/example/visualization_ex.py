#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-03
# @Filename: visualization_ex.py

import sys
import os
import time  # ⏳ 실행 시간 측정용

# 🔧 현재 스크립트(example 폴더)의 상위 디렉토리(src)를 Python 모듈 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualization import Visualizer
from data_loader import DataLoader
from config import LBOX_CMPCH, LBOX_CKPCH, RESULTS_FOLDER
from logger import logger
from utils import set_x_range

start_time = time.time()  # 시작 시간 저장

# 설정 (모든 단위를 cMpc/h 기준으로 설정)
npy_folder = "/home/users/mmingyeong/tng_init/250121/npy"
x_center = 100  # X축 중심 (cMpc/h)
x_thickness = 10  # 선택할 X 범위 두께 (cMpc/h)

# X 범위 변환 (cMpc/h → ckpc/h)
x_min, x_max = set_x_range(center_cMpc=x_center, thickness_cMpc=x_thickness, lbox_cMpc=LBOX_CMPCH, lbox_ckpch=LBOX_CKPCH)
logger.info(f"🔹 Filtering X range: {x_min:.2f} - {x_max:.2f} ckpc/h ({x_min / 1000:.3f} - {x_max / 1000:.3f} cMpc/h)")

# 데이터 로딩
logger.info("🔹 Loading data...")
loader = DataLoader(npy_folder)
positions = loader.load_all_chunks(x_min=x_min, x_max=x_max, sampling_rate=1)

# 데이터 로드 완료 로그
logger.info(f"✅ Data loaded successfully. Shape: {positions.shape}")

# 시각화: 2D 히스토그램 (Y-Z 평면)
logger.info(f"🔹 Generating 2D histogram, saving to ../results/...")
viz = Visualizer(bins=500, cmap="cividis", dpi=200)
viz.plot_2d_histogram(
    positions, 
    results_folder=RESULTS_FOLDER,  # 결과 폴더 상대 경로 설정
    x_range=x_thickness, 
    lbox_cMpc=LBOX_CMPCH, 
    lbox_ckpch=LBOX_CKPCH, 
    save_pdf=True
)

# 실행 종료 시간 측정
end_time = time.time()
elapsed_time = end_time - start_time  # 총 실행 시간 계산

# ⏳ 보기 좋은 시간 형식으로 변환 (hh:mm:ss)
formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
logger.info(f"⏳ Total execution time: {formatted_time} ({elapsed_time:.2f} seconds)")
logger.info("✅ Plot saved successfully!")
