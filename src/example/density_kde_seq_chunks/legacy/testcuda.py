try:
    import cupy as cp
    test_array = cp.array([0])
    print("CUDA 초기화 테스트 성공:", test_array)
except Exception as e:
    print("CUDA 초기화 테스트 실패:", e)
