#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-03-04
# @Filename: check_system_ex.py

import os
import sys
import time  # ⏳ 실행 시간 측정용

# 🔧 현재 스크립트의 상위 디렉터리(src)를 Python 모듈 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger import logger
from system_checker import SystemChecker, LAST_UPDATE_DATE  # ✅ 최신 업데이트 날짜 추가

if __name__ == "__main__":
    logger.info(f"🔍 Starting system check (SystemChecker last updated: {LAST_UPDATE_DATE})")

    start_time = time.time()  # ⏳ 시작 시간 저장

    # ✅ `SystemChecker` 실행하여 환경 점검 (verbose=True → 상세 분석 포함)
    checker = SystemChecker(verbose=True)
    checker.run_all_checks()
    checker.log_results()

    # ✅ `use_gpu` 값 확인하여 로깅
    use_gpu = checker.get_use_gpu()
    if use_gpu:
        logger.info("✅ GPU is available. Computation will be accelerated.")
    else:
        logger.warning("⚠️ No GPU detected. Using CPU. Computation may be significantly slower.")

    # ⏳ 실행 종료 시간 측정
    end_time = time.time()
    elapsed_time = end_time - start_time

    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    logger.info(f"⏳ System check completed in {formatted_time} ({elapsed_time:.2f} seconds).")

    # ✅ 명확한 종료
    sys.exit(0)
