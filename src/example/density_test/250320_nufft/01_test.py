import numpy as np
import finufft
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------
# 도메인 및 그리드 설정
# -------------------------
nx, ny = 64, 64
Lx, Ly = 2 * np.pi, 2 * np.pi

# 주기적 도메인 [-π, π)에서 균일한 그리드 생성
x_grid = np.linspace(-np.pi, np.pi, nx, endpoint=False)
y_grid = np.linspace(-np.pi, np.pi, ny, endpoint=False)
xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')
points = np.column_stack((xx.ravel(), yy.ravel()))
N = points.shape[0]

# -------------------------
# 테스트 입력: 모든 점에서 상수 1
# -------------------------
c = np.ones(N, dtype=np.complex128)

# finufft 경고에 따라, C-contiguous 배열로 변환
x_scaled = np.ascontiguousarray(points[:, 0])
y_scaled = np.ascontiguousarray(points[:, 1])
c = np.ascontiguousarray(c)

# -------------------------
# NUFFT 계획 생성 (Type 1: nonuniform -> uniform grid)
# -------------------------
plan = finufft.Plan(1, (nx, ny), isign=-1, eps=1e-6, upsampfac=3.0, spread_kerevalmeth=0)
plan.setpts(x_scaled, y_scaled)
F_k = plan.execute(c).reshape((nx, ny))
# F_k는 Fourier 계수가 중앙에 위치한 "centered" 배열일 가능성이 있음

# -------------------------
# 만약 F_k가 centered 형태라면, np.fft.ifftshift로 표준 FFT 순서로 변경
# -------------------------
F_k_shifted = np.fft.ifftshift(F_k)

# -------------------------
# Inverse FFT 수행 및 보정
# -------------------------
# np.fft.ifft2는 1/(nx*ny) 정규화를 포함하므로, 역변환 후 (nx*ny)를 곱해 줍니다.
f_ifft = np.fft.ifft2(F_k_shifted)
f_recovered = np.real(f_ifft) * (nx * ny)

# -------------------------
# 결과 검증
# -------------------------
# 모든 입력이 1이므로, type1 NUFFT의 경우 k=0 성분은 전체 합이 되어야 합니다.
# 복원된 공간 신호는 모두 1이 되어야 하며, (nx*ny)를 곱하면 4096가 되어야 함.
expected_value = nx * ny  # 4096
max_error = np.abs(f_recovered - expected_value).max()
logging.info("왕복 테스트 (ifftshift 적용 후): 최대 오차 = {:.4e}".format(max_error))
logging.info("복원된 상수 값 (기대: {}): {:.4f}".format(expected_value, np.mean(f_recovered)))
