#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang
# @Date: 2025-03-04
# @Filename: system_checker.py

import os
import platform
import psutil
import numpy as np
import cupy as cp
import yaml
from datetime import date  # ✅ 오늘 날짜 추가
from logger import logger

# ✅ PyTorch 예외 처리 추가
try:
    import torch
    torch_version = torch.__version__
except ModuleNotFoundError:
    torch_version = "Not Installed"

# ✅ 최신 업데이트 날짜 (오늘 날짜 자동 설정)
LAST_UPDATE_DATE = date.today().strftime("%Y-%m-%d")

# ✅ 일반적인 GPU 성능 기준 (업그레이드 추천 기준)
RECOMMENDED_GPUS = {
    "Deep Learning": ["NVIDIA H100", "NVIDIA A100", "RTX 4090", "RTX 3090", "Tesla V100"],
    "High Performance Computing": ["NVIDIA H100", "A100", "Tesla P100"],
    "General Computing": ["RTX 4060", "RTX 3060", "GTX 1660 Super"],
}

class SystemChecker:
    """시스템 환경을 체크하여 GPU 사용 가능 여부 및 성능 평가를 수행하는 클래스."""

    def __init__(self, verbose=False):
        """✅ 초기화 시 GPU 사용 가능 여부를 먼저 확인"""
        self.use_gpu = self.check_gpu_availability()
        self.verbose = verbose  # ✅ 상세 분석 활성화 여부
        self.gpu_info = []
        self.recommended_gpus = set()  # ✅ 중복 방지를 위한 set 사용
        self.cpu_cores = None
        self.total_memory = None
        self.python_version = None
        self.numpy_version = None
        self.cupy_version = None
        self.torch_version = torch_version

    def check_gpu_availability(self):
        """✅ GPU가 사용 가능한 환경인지 확인하고 True/False 반환"""
        try:
            if cp.cuda.runtime.getDeviceCount() > 0:
                return True
        except (cp.cuda.runtime.CUDARuntimeError, RuntimeError):
            pass
        return False

    def check_gpu(self):
        """🖥 GPU 성능 분석 및 추천"""
        try:
            num_gpus = cp.cuda.runtime.getDeviceCount()
            if num_gpus > 0:
                for i in range(num_gpus):
                    device = cp.cuda.Device(i)
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    name = props["name"].decode("utf-8")  # ✅ 바이트 문자열을 문자열로 변환
                    total_mem = props["totalGlobalMem"] / (1024 ** 3)  # GB 변환
                    cuda_cores = props["multiProcessorCount"]  # CUDA 코어 수
                    clock_speed = props["clockRate"] / 1e6  # GHz 변환

                    gpu_details = (
                        f"🖥 GPU {i}: {name} ({total_mem:.2f} GB, "
                        f"{cuda_cores} CUDA cores, {clock_speed:.2f} GHz)"
                    )
                    self.gpu_info.append(gpu_details)

                    # ✅ 사용 목적에 따른 GPU 추천 (이미 충분히 좋은 GPU는 추천 안 함)
                    recommended_gpu = self.recommend_gpu(name, total_mem, cuda_cores)
                    if recommended_gpu and recommended_gpu != name:
                        self.recommended_gpus.add(recommended_gpu)  # ✅ 중복 방지

        except (cp.cuda.runtime.CUDARuntimeError, RuntimeError):
            self.gpu_info.append("⚠️ No GPU detected. Using CPU (NumPy fallback mode).")
            self.use_gpu = False

    def recommend_gpu(self, current_gpu, memory_gb, cuda_cores):
        """🛠 현재 사용자의 GPU 성능을 분석하고 추천 GPU 반환"""
        # ✅ A100 40GB 또는 80GB 사용 중이면 업그레이드 추천하지 않음
        if any(a100_variant in current_gpu for a100_variant in ["A100", "A100-PCIE", "A100 80GB"]):
            return None
        
        # ✅ H100 사용 중이면 업그레이드 추천하지 않음
        if "H100" in current_gpu:
            return None

        # ✅ 메모리가 8GB 미만이거나 CUDA 코어가 적으면 A100 이상 추천
        if memory_gb < 8 or cuda_cores < 2500:
            return RECOMMENDED_GPUS["Deep Learning"][0]  # NVIDIA H100 추천
        elif "GTX" in current_gpu or "RTX 2060" in current_gpu:
            return RECOMMENDED_GPUS["General Computing"][0]  # RTX 4060 추천
        elif "Tesla" in current_gpu and memory_gb < 32:
            return RECOMMENDED_GPUS["High Performance Computing"][0]  # H100 추천
        return None

    def check_cpu(self):
        """🖥 CPU 및 RAM 정보 확인"""
        self.cpu_cores = psutil.cpu_count(logical=True)
        self.total_memory = psutil.virtual_memory().total / (1024 ** 3)

    def check_python_libraries(self):
        """🐍 Python 및 주요 라이브러리 버전 확인"""
        self.python_version = platform.python_version()
        self.numpy_version = np.__version__
        self.cupy_version = cp.__version__ if cp else "Not Installed"

    def log_environment_analysis(self):
        """🛠 개발 환경 분석 및 평가 (Verbose 모드 활성화 시)"""
        if not self.verbose:
            return
        
        logger.info(f"📅 Environment analysis based on SystemChecker update: {LAST_UPDATE_DATE}")
        
        if self.python_version < "3.9":
            logger.warning(f"⚠️ Python {self.python_version} is outdated. Upgrade to Python 3.9+ recommended.")
        
        if self.torch_version == "Not Installed":
            logger.warning("⚠️ PyTorch is not installed. Consider installing it for deep learning.")
        
        logger.info(f"🐍 Python Version: {self.python_version}")
        logger.info(f"📦 NumPy Version: {self.numpy_version}")
        logger.info(f"📦 CuPy Version: {self.cupy_version}")
        logger.info(f"📦 PyTorch Version: {self.torch_version}")

    def run_all_checks(self):
        """✅ 모든 시스템 체크 실행"""
        self.check_cpu()
        self.check_gpu()
        self.check_python_libraries()
        self.log_environment_analysis()  # ✅ 환경 분석 추가

    def log_results(self):
        """🔍 시스템 체크 결과 로깅"""
        logger.info("🔍 Running System Check...")

        # ✅ GPU 정보 출력
        if self.gpu_info:
            for gpu in self.gpu_info:
                logger.info(gpu)

        # ✅ GPU 업그레이드 추천
        if self.recommended_gpus:
            for rec_gpu in self.recommended_gpus:
                logger.warning(f"⚠️ Recommended Upgrade: {rec_gpu}")

        logger.info(f"🖥 CPU Cores: {self.cpu_cores}")
        logger.info(f"💾 Total RAM: {self.total_memory:.2f} GB")
        logger.info(f"🚀 Using GPU: {self.use_gpu}")
        logger.info("✅ System check complete.")

    def get_use_gpu(self):
        """✅ 현재 설정된 use_gpu 값을 반환"""
        return self.use_gpu
