#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess

def detailed_gpu_process_info():
    """
    각 GPU에서 사용 중인 컴퓨팅 애플리케이션에 대해 
    PID, 프로세스 이름, 사용된 메모리 정보를 출력합니다.
    """
    try:
        command = [
            "nvidia-smi", 
            "--query-compute-apps=pid,process_name,used_memory", 
            "--format=csv,noheader,nounits"
        ]
        output = subprocess.check_output(command, universal_newlines=True)
        print("GPU 메모리를 사용 중인 프로세스 목록 (PID, Process Name, Used Memory in MB):")
        print(output)
    except Exception as e:
        print("nvidia-smi 명령 실행 중 오류 발생:", e)

if __name__ == "__main__":
    detailed_gpu_process_info()
