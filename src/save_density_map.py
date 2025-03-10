#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: save_density_map.py
# structura/save_density_map.py

import os
import datetime
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)

def save_density_map(density_map, filename=None, data_name="data", grid_spacing=None, kernel_name="unknown", h=1.0, folder="", file_format="npy"):
    """
    계산된 3차원 밀도 맵을 지정된 파일 포맷과 경로로 저장합니다.
    
    Parameters
    ----------
    density_map : np.ndarray
        저장할 3D 밀도 맵 배열.
    filename : str or None
        저장할 파일 이름 (기본값: 자동 생성).
    data_name : str
        데이터 이름 (예: "TNG300_snapshot99").
    grid_spacing : tuple or None
        격자 해상도 정보.
    kernel_name : str
        사용한 커널 함수의 이름.
    h : float
        사용된 커널 밴드위스.
    folder : str
        저장할 디렉토리 경로 (없으면 현재 디렉토리).
    file_format : str
        저장 포맷. ("npy", "npz", "hdf5", "csv")
    """
    current_dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename is None:
        base_filename = f"density_map_{data_name}_{grid_spacing}_{kernel_name}_h{h:.4f}_{current_dt}"
    else:
        base_filename = filename

    # 확장자 여부를 검사하는 함수
    def ensure_ext(fname, ext):
        return fname if fname.lower().endswith(ext) else fname + ext

    # 저장 폴더 처리
    base_path = folder.rstrip(os.sep) if folder else ""
    if base_path and not os.path.exists(base_path):
        try:
            os.makedirs(base_path)
        except Exception as e:
            logger.error("폴더 생성 실패: %s", e)
            return

    file_format = file_format.lower()
    out_filename = os.path.join(base_path, base_filename)
    
    try:
        if file_format == "npy":
            out_filename = ensure_ext(out_filename, ".npy")
            np.save(out_filename, density_map)
            logger.info("밀도 맵이 npy 형식으로 저장되었습니다: %s", out_filename)
        elif file_format == "npz":
            out_filename = ensure_ext(out_filename, ".npz")
            np.savez_compressed(out_filename, density_map=density_map)
            logger.info("밀도 맵이 npz 형식으로 저장되었습니다: %s", out_filename)
        elif file_format == "hdf5":
            try:
                import h5py
            except ImportError:
                logger.error("h5py 모듈이 필요합니다. hdf5 형식으로 저장하려면 h5py를 설치하세요.")
                return
            out_filename = ensure_ext(out_filename, ".h5")
            with h5py.File(out_filename, "w") as hf:
                hf.create_dataset("density_map", data=density_map, compression="gzip")
            logger.info("밀도 맵이 hdf5 형식으로 저장되었습니다: %s", out_filename)
        elif file_format == "csv":
            out_filename = ensure_ext(out_filename, ".csv")
            flat_data = density_map.flatten()
            np.savetxt(out_filename, flat_data, delimiter=",")
            logger.info("밀도 맵이 csv 형식으로 저장되었습니다 (flattened): %s", out_filename)
        else:
            logger.error("지원하지 않는 파일 포맷입니다: %s", file_format)
    except Exception as e:
        logger.error("밀도 맵 저장 중 오류 발생: %s", e)


def save_parameters_info(info_dict, filename=None, folder=""):
    """
    밀도 계산에 사용된 파라미터 및 데이터 사양 정보를 JSON 파일로 저장합니다.
    
    Parameters
    ----------
    info_dict : dict
        저장할 파라미터 정보를 담은 딕셔너리.
    filename : str or None
        저장할 파일 이름 (기본값: 자동 생성).
    folder : str
        저장할 디렉토리 경로 (없으면 현재 디렉토리).
    """
    if filename is None:
        base_filename = f"parameters_info_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        base_filename = filename

    # 저장 폴더 처리
    base_path = folder.rstrip(os.sep) if folder else ""
    if base_path and not os.path.exists(base_path):
        try:
            os.makedirs(base_path)
        except Exception as e:
            logger.error("폴더 생성 실패: %s", e)
            return

    out_filename = os.path.join(base_path, base_filename)
    out_filename = out_filename if out_filename.lower().endswith(".json") else out_filename + ".json"
    try:
        with open(out_filename, "w") as f:
            json.dump(info_dict, f, indent=4)
        logger.info("파라미터 정보가 JSON 형식으로 저장되었습니다: %s", out_filename)
    except Exception as e:
        logger.error("파라미터 정보 저장 중 오류 발생: %s", e)
