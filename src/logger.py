#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: logger.py

import logging
import os
import sys
from datetime import datetime

# 현재 파일이 속한 Structura 패키지의 루트 디렉토리를 기준으로 log 폴더 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Structura/src 기준
LOG_DIR = os.path.join(BASE_DIR, "log")

# log 폴더가 없으면 생성
os.makedirs(LOG_DIR, exist_ok=True)

# 실행 날짜와 시간 기반의 로그 파일 이름 설정 (YYYY-MM-DD_HH-MM-SS.log)
log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
LOG_FILE = os.path.join(LOG_DIR, log_filename)

# Logger 설정
logger = logging.getLogger("Structura")
logger.setLevel(logging.INFO)

# 로그 포맷 정의
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# 콘솔 핸들러 추가
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# 파일 핸들러 추가 (로그를 실행 시간별 파일에 저장)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(formatter)

# 핸들러 중복 추가 방지
if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# stderr도 stdout으로 리디렉션하여 에러도 로그에 기록되도록 설정
sys.stderr = sys.stdout

# 예외 발생 시 로그 파일에도 기록되도록 설정
def log_exception(exc_type, exc_value, exc_traceback):
    """예외 발생 시 로그 파일에 기록하는 함수"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Unhandled exception occurred:", exc_info=(exc_type, exc_value, exc_traceback))

# 전역 예외 훅 설정 (모든 예외를 로깅)
sys.excepthook = log_exception

# 로거 사용 예시
if __name__ == "__main__":
    logger.info("Logger is successfully set up!")
    
    # 예제: 예외 발생 테스트
    try:
        raise ValueError("🚨 테스트용 에러 발생!")
    except Exception as e:
        logger.exception("🔥 예외 발생: %s", str(e))
