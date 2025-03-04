#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: logger.py

import logging
import os
import sys
import time
from datetime import datetime

# 현재 logger.py가 위치한 디렉토리(src/)를 BASE_DIR로 설정
BASE_DIR = os.path.dirname(__file__)  # 예: Structura/src
LOG_DIR = os.path.join(BASE_DIR, "log")  # => Structura/src/log

# log 폴더가 없으면 생성
os.makedirs(LOG_DIR, exist_ok=True)

# 실행 중인 스크립트 이름 추출
# 예: python converter_ex.py -> script_name = "converter_ex"
if len(sys.argv) > 0:
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
else:
    script_name = "main"

# 실행 날짜와 시간 기반의 로그 파일 이름 설정
# 예: converter_ex_2025-03-05_00-39-53.log
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"{script_name}_{timestamp}.log"
LOG_FILE = os.path.join(LOG_DIR, log_filename)

# Logger 설정
logger = logging.getLogger("Structura")
logger.setLevel(logging.INFO)

# 로그 포맷 정의
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# 콘솔 핸들러
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# 파일 핸들러 (로그를 실행 시간별 파일에 저장)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(formatter)

# 핸들러 중복 추가 방지
if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# stderr도 stdout으로 리디렉션
sys.stderr = sys.stdout

# 예외 발생 시 로그 파일에도 기록되도록 설정
def log_exception(exc_type, exc_value, exc_traceback):
    """예외 발생 시 로그 파일에 기록"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Unhandled exception occurred:", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = log_exception

# 로거 테스트
if __name__ == "__main__":
    logger.info("Logger is successfully set up!")
