#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_verify_density_maps.py

이 스크립트는 테스트 디렉터리에 저장된 모든 .npy 밀도 맵 파일을 검사하여
기본 통계(shape, dtype, min, max, sum)와 NaN/Inf 유무를 리포트합니다.
"""

import os
import sys
import numpy as np
import logging

def main(test_dir):
    logger = logging.getLogger("VerifyDensityMaps")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(ch)

    if not os.path.isdir(test_dir):
        logger.error(f"테스트 디렉터리를 찾을 수 없습니다: {test_dir}")
        sys.exit(1)

    files = sorted(f for f in os.listdir(test_dir) if f.endswith('.npy'))
    if not files:
        logger.error(f"디렉터리에 .npy 파일이 없습니다: {test_dir}")
        sys.exit(1)

    logger.info(f"총 {len(files)}개 파일 검사 시작: {test_dir}")
    for fname in files:
        path = os.path.join(test_dir, fname)
        try:
            arr = np.load(path)
        except Exception as e:
            logger.error(f"{fname} 로딩 실패: {e}")
            continue

        # 기본 통계
        shape = arr.shape
        dtype = arr.dtype
        total = np.sum(arr)
        vmin = np.min(arr) if arr.size else float('nan')
        vmax = np.max(arr) if arr.size else float('nan')
        has_nan = not np.isfinite(arr).all()

        # 리포트
        logger.info(f"\n파일: {fname}")
        logger.info(f"  shape: {shape}, dtype: {dtype}")
        logger.info(f"  min: {vmin:.3e}, max: {vmax:.3e}")
        logger.info(f"  sum: {total:.3e}")
        if total == 0:
            logger.warning("  → sum이 0입니다! 밀도 계산이 올바르게 수행되지 않았을 수 있습니다.")
        if has_nan:
            logger.warning("  → NaN 또는 Inf 값이 포함되어 있습니다!")

    logger.info("검사 완료.")

if __name__ == '__main__':
    # 스크립트 위치 기준으로 test 폴더를 찾도록 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_directory = os.path.join(base_dir, 'test')
    main(test_directory)
