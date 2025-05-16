import numpy as np

try:
    import cupy as cp
    from cupy.cuda import runtime
    def select_best_gpu():
        num_devices = runtime.getDeviceCount()
        best_device = 0
        best_score = 0
        for device in range(num_devices):
            props = runtime.getDeviceProperties(device)
            # Simple heuristic
            score = props['multiProcessorCount'] * props['clockRate']
            if score > best_score:
                best_score = score
                best_device = device
        return best_device

    GPU_DEVICE = select_best_gpu()
    use_gpu = True
except ImportError:
    use_gpu = False
    GPU_DEVICE = None


class FFTKDE2D:
    """
    Perform 2D kernel density estimation using an FFT-based convolution
    of binned particle data with a user-provided *real-space* kernel.
    We treat the domain as purely periodic.
    """

    def __init__(self, particles, grid_bounds, grid_spacing, kernel_func, h):
        """
        Parameters
        ----------
        particles : ndarray, shape (N,2)
        grid_bounds : {'x':(xmin,xmax),'y':(ymin,ymax)}
        grid_spacing : (dx, dy)
        kernel_func : callable
            A real-space 2D kernel function: kernel_func(r, h) -> scalar
            e.g. 2D Gaussian, Epanechnikov, etc.
        h : float
            Kernel bandwidth.
        """
        self.grid_bounds = grid_bounds
        self.grid_spacing = grid_spacing
        self.kernel_func = kernel_func  # REAL-space kernel
        self.h = h

        (xmin, xmax) = self.grid_bounds['x']
        (ymin, ymax) = self.grid_bounds['y']
        width_x = xmax - xmin
        width_y = ymax - ymin

        p = particles.copy()
        p[:, 0] = np.mod(p[:, 0] - xmin, width_x) + xmin
        p[:, 1] = np.mod(p[:, 1] - ymin, width_y) + ymin
        self.particles = p

    def compute_density(self):
        from numpy.fft import fft2, ifft2

        xmin, xmax = self.grid_bounds['x']
        ymin, ymax = self.grid_bounds['y']
        dx, dy = self.grid_spacing

        # 1) 2D 히스토그램 (bins)
        x_edges = np.arange(xmin, xmax + dx, dx, dtype=np.float64)
        y_edges = np.arange(ymin, ymax + dy, dy, dtype=np.float64)

        H, xed, yed = np.histogram2d(
            self.particles[:, 0], 
            self.particles[:, 1],
            bins=[x_edges, y_edges]
        )
        nx, ny = H.shape
        x_centers = 0.5 * (xed[:-1] + xed[1:])
        y_centers = 0.5 * (yed[:-1] + yed[1:])
        cell_area = dx * dy
        density_hist = H / cell_area  # hist -> density

        # 2) 실공간에서 커널 격자를 만들어 FFT
        #    (중심이 (0,0)인 커널을 nx, ny 크기로 생성)
        kernel_grid = self._make_kernel_grid(nx, ny, dx, dy)
        kernel_fft = fft2(kernel_grid)

        # 3) density_hist를 FFT → 곱 → IFFT
        if use_gpu and GPU_DEVICE is not None:
            with cp.cuda.Device(GPU_DEVICE):
                density_gpu = cp.asarray(density_hist)
                density_fft_gpu = cp.fft.fft2(density_gpu)
                kernel_fft_gpu = cp.asarray(kernel_fft)
                conv_result_gpu = density_fft_gpu * kernel_fft_gpu
                density_ifft_gpu = cp.fft.ifft2(conv_result_gpu).real
                density_fft = cp.asnumpy(density_ifft_gpu)
        else:
            density_fft = ifft2(fft2(density_hist) * kernel_fft).real

        return x_centers, y_centers, density_fft.astype(np.float64)

    def _make_kernel_grid(self, nx, ny, dx, dy):
        """
        실공간 커널(크기: nx×ny)을 생성.
        중심을 (0,0)이라고 가정하고, 각 격자점까지의 거리를 구해
        self.kernel_func(distance, self.h) 값을 저장.

        주의: 
          - nx, ny 격자 전체 범위를 얼마나 잡아야 할지(주기 경계) 
            에 따라 wrap-around가 발생할 수 있습니다.
          - 여기서는 '도메인 전체가 주기적'이라 보고, 
            kernel_grid도 동일 크기로 맞춰서 FFT.
        """
        # 커널 격자를 만들기 위해, 
        # x 좌표: 0 ~ (nx-1), y좌표: 0 ~ (ny-1)을 "격자 인덱스"로 보고
        # 이를 -Nx/2 ~ Nx/2 범위로 shift 하여 중앙을 (0,0)로 배치
        import numpy.fft

        x_indices = np.arange(nx)
        y_indices = np.arange(ny)
        # 주파수 도메인에서처럼 FFT shift를 고려
        x_indices_shifted = (x_indices + nx//2) % nx - nx//2
        y_indices_shifted = (y_indices + ny//2) % ny - ny//2
        # 2D mesh
        XX, YY = np.meshgrid(x_indices_shifted, y_indices_shifted, indexing='ij')

        # 실제 길이 단위로 환산
        # dx, dy 만큼씩 떨어진 격자에서 (0,0)을 중심으로 했을 때의 좌표
        # ex) XX[i,j] * dx = x방향 거리
        RR = np.sqrt((XX * dx) ** 2 + (YY * dy) ** 2)

        kernel_grid = self.kernel_func(RR, self.h)
        return kernel_grid
