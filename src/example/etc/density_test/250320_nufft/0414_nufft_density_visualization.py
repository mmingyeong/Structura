import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import finufft
from scipy.special import j1

# --- 커널 Fourier 변환 함수들 ---
def gaussian_kernel_ft(r2, h):
    """
    Gaussian kernel의 Fourier 변환:
      K_ft = exp(-0.5 * h^2 * (kx^2+ky^2))
    """
    return np.exp(-0.5 * h**2 * r2)

def tophat_kernel_ft(r2, h):
    """
    Top-hat kernel의 Fourier 변환:
      실공간에서 Top-hat: r <= h일 때 1/(πh^2), 그 외 0
      Fourier 공간에서는: 2 * J1(k*h)/(k*h)
    """
    r = np.sqrt(r2)
    kernel = np.ones_like(r)
    nonzero = (r != 0)
    kernel[nonzero] = 2 * j1(r[nonzero] * h) / (r[nonzero] * h)
    return kernel

def triangle_kernel_ft(kx, ky, h):
    """
    Triangle kernel의 Fourier 변환 (실제 구현은 문제에 따라 다를 수 있음):
      일반적으로 sinc 제곱 형태로 표현됨.
    """
    r = np.sqrt(kx**2 + ky**2)
    kernel = np.ones_like(r)
    nonzero = (r != 0)
    kernel[nonzero] = (np.sin(r[nonzero]*h)/(r[nonzero]*h))**2
    return kernel

def tophat_kernel_ft_softcut(r2, h, method="hann", kmax_factor=0.9):
    """
    Fourier-space top-hat kernel with soft low-pass filter (e.g. Hann, Lanczos)
    - r2: kx^2 + ky^2 (i.e., |k|^2)
    - h: top-hat radius
    - method: 'hann', 'lanczos', 'gaussian'
    - kmax_factor: 최대 cutoff 주파수 비율 (0 ~ 1)
    """
    r = np.sqrt(r2)
    kh = r * h
    kernel = np.ones_like(kh)
    nonzero = (r != 0)
    kernel[nonzero] = 2 * j1(kh[nonzero]) / kh[nonzero]

    # soft low-pass cutoff
    kmax = r.max() * kmax_factor
    if method == "hann":
        mask = r < kmax
        window = np.zeros_like(r)
        window[mask] = np.cos(np.pi * r[mask] / (2 * kmax)) ** 2
    elif method == "lanczos":
        eps = 1e-12
        x = np.pi * r / kmax
        window = np.sinc(x / np.pi)  # sinc(x) = sin(x)/x
    elif method == "gaussian":
        alpha = 4.0  # tuning parameter
        window = np.exp(-alpha * (r / kmax) ** 2)
    else:
        raise ValueError("지원하지 않는 soft cutoff 방식입니다.")

    return kernel * window

def uniform_kernel_ft_2d(r2, h):
    """
    2D Uniform (Top-hat) kernel's Fourier transform for use in NUFFT.
    
    Parameters:
        r2 : ndarray
            kx^2 + ky^2 (squared wave numbers in 2D)
        h : float
            Kernel radius in real space

    Returns:
        kernel_ft : ndarray
            Fourier transform of the uniform (Top-hat) kernel
    """
    r = np.sqrt(r2)
    kh = r * h
    kernel_ft = np.ones_like(kh)

    mask = kh != 0
    kernel_ft[mask] = 2 * j1(kh[mask]) / kh[mask]

    return kernel_ft

def uniform_kernel_ft_2d_softcut(r2, h, method="hann", kmax_factor=0.9):
    """
    2D Uniform (Top-hat) kernel's Fourier transform with soft cutoff.
    
    Parameters:
        r2 : ndarray
            Squared wave number grid (kx^2 + ky^2)
        h : float
            Real-space kernel radius
        method : str
            Cutoff window type: 'hann', 'lanczos', 'gaussian'
        kmax_factor : float
            Maximum k for cutoff window (as fraction of max k)

    Returns:
        kernel_ft : ndarray
            Softly attenuated Fourier-space kernel
    """

    r = np.sqrt(r2)
    kh = r * h

    # base kernel: 2D tophat FT
    kernel_ft = np.ones_like(kh)
    mask = kh != 0
    kernel_ft[mask] = 2 * j1(kh[mask]) / kh[mask]

    # soft low-pass window
    kmax = r.max() * kmax_factor
    if method == "hann":
        window = np.zeros_like(r)
        mask = r < kmax
        window[mask] = np.cos(np.pi * r[mask] / (2 * kmax)) ** 2
    elif method == "lanczos":
        eps = 1e-12  # avoid division by 0
        x = np.pi * r / kmax
        window = np.sinc(x / np.pi)
    elif method == "gaussian":
        alpha = 4.0  # tuning parameter
        window = np.exp(-alpha * (r / kmax) ** 2)
    else:
        raise ValueError("Unsupported softcut method. Choose 'hann', 'lanczos', or 'gaussian'.")

    return kernel_ft * window


# --- 데이터 로드 및 파라미터 설정 ---
data_filename = "particles_periodic.npy"
particles = np.load(data_filename)  # particles.shape = (N, 2)

# 도메인 경계는 데이터의 최소, 최대값으로 결정
xmin, xmax = particles[:, 0].min(), particles[:, 0].max()
ymin, ymax = particles[:, 1].min(), particles[:, 1].max()
grid_bounds = {"x": (xmin, xmax), "y": (ymin, ymax)}

# 격자 해상도: 예를 들어 128x128 격자
nx = 128
ny = 128
grid_spacing = ((xmax - xmin) / nx, (ymax - ymin) / ny)

# Bandwidth와 사용할 커널 종류 선택
h = 1  # 사용자 지정 파라미터 (필요에 따라 조정)
kernel_type = "triangle"  # "gaussian", "tophat", "triangle" 중 선택

# 도메인 길이 계산
Lx = xmax - xmin
Ly = ymax - ymin

# --- 단계별 계산 ---

# (A) 좌표 재조정: 원래 도메인 -> [-π, π]
x = particles[:, 0]
y = particles[:, 1]
x_center = (xmin + xmax) / 2.0
y_center = (ymin + ymax) / 2.0
x_scaled = (x - x_center) * (2 * np.pi / Lx)
y_scaled = (y - y_center) * (2 * np.pi / Ly)

# (B) FINUFFT를 사용한 NUFFT 계산
plan = finufft.Plan(1, (nx, ny), isign=-1, eps=1e-6, upsampfac=3.0, spread_kerevalmeth=0)
plan.setpts(x_scaled, y_scaled)
c = np.ones_like(x_scaled, dtype=np.complex128)
F_k = plan.execute(c)
F_k_2d = F_k.reshape((nx, ny))  # Fourier 계수를 2차원 배열로 재구성

# (C) FFT 주파수를 물리적 파수로 변환
kx = np.fft.fftfreq(nx, d=1.0) * nx  # 기본 FFT 주파수 (정수 계수)
ky = np.fft.fftfreq(ny, d=1.0) * ny
kx_mesh, ky_mesh = np.meshgrid(kx, ky, indexing='ij')
kx_phys = (2 * np.pi / Lx) * kx_mesh
ky_phys = (2 * np.pi / Ly) * ky_mesh

# (D) 커널 Fourier 변환 적용
if kernel_type.lower() == "gaussian":
    kernel_ft = gaussian_kernel_ft(kx_phys**2 + ky_phys**2, h)
elif kernel_type.lower() == "tophat":
    kernel_ft = uniform_kernel_ft_2d_softcut(kx_phys**2 + ky_phys**2, h, kmax_factor=0.7)
    #kernel_ft = tophat_kernel_ft(kx_phys**2 + ky_phys**2, h)
    #kernel_ft = tophat_kernel_ft_softcut(kx_phys**2 + ky_phys**2, h, method="hann")
elif kernel_type.lower() == "triangle":
    kernel_ft = triangle_kernel_ft(kx_phys, ky_phys, h)
else:
    raise ValueError("지원하지 않는 커널 타입입니다. 'gaussian', 'tophat', 'triangle' 중 선택하세요.")

# (E) 커널 곱셈: NUFFT 결과에 커널 적용
G_k = F_k_2d * kernel_ft

# (F) 역 FFT를 통한 공간 영역 밀도 맵 복원 (np.fft.ifftshift 적용)
density_complex = np.fft.ifft2(G_k)
density_map = np.real(density_complex) * (nx * ny)

# ✅ 음수값 제거
density_map = np.clip(density_map, 0, None)

# --- 단계별 결과를 하나의 그림으로 시각화 ---
plt.figure(figsize=(18, 12))

# 1. 원본 입자 분포 (실제 좌표)
plt.subplot(2, 3, 1)
plt.scatter(x, y, s=5, color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Particle Distribution')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# 2. 재조정된 좌표 분포 ([-π, π] 범위)
plt.subplot(2, 3, 2)
plt.scatter(x_scaled, y_scaled, s=5, color='red')
plt.xlabel('x_scaled')
plt.ylabel('y_scaled')
plt.title('Rescaled Coordinates (-π, π)')
plt.xlim(-np.pi, np.pi)
plt.ylim(-np.pi, np.pi)

# 3. NUFFT 결과 F_k (로그 스케일 이미지)
plt.subplot(2, 3, 3)
plt.imshow(np.log10(np.abs(F_k_2d) + 1e-12), origin='lower', cmap='viridis')
plt.colorbar()
plt.title('NUFFT F_k (log10 |F_k|)')

# 4. 커널 Fourier 변환 (kernel_ft)
plt.subplot(2, 3, 4)
plt.imshow(np.log10(np.abs(kernel_ft) + 1e-12), origin='lower', cmap='viridis')
plt.colorbar()
plt.title(f'{kernel_type.capitalize()} kernel_ft (log10 |kernel_ft|)')

# 5. 커널 곱셈 후 Fourier 데이터 (G_k)
plt.subplot(2, 3, 5)
plt.imshow(np.log10(np.abs(G_k) + 1e-12), origin='lower', cmap='viridis')
plt.colorbar()
plt.title('G_k')

# 6. 최종 밀도 맵
plt.subplot(2, 3, 6)
plt.imshow(density_map, origin='lower', cmap='viridis', extent=[xmin, xmax, ymin, ymax], vmin=0)
plt.colorbar(label='Density')
plt.title('Final Density Map')

plt.tight_layout()
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"nufft_density_{timestamp_str}.png"
plt.savefig(output_filename, dpi=300)
plt.close()

print("각 단계별 시각화 그림이 저장되었습니다:", output_filename)
