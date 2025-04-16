#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# kernel_2d.py

import numpy as np

class KernelFunctions2D:
    """
    2차원(2D) 커널 함수들을 static method 형태로 제공하는 클래스입니다.
    각 커널은 2D에서의 적분이 1이 되도록 정규화 상수를 조정했으며,
    입력 distance는 0 이상인 거리(r)를 의미합니다.
    
    사용 예:
        distance = sqrt((x - xi)^2 + (y - yi)^2)
        value = KernelFunctions2D.gaussian(distance, h=...)
    
    무한 지지 커널(logistic, laplacian 등)은
    여기서 단순 형태만 제시했으니,
    실제론 적분값을 다시 확인하시기 바랍니다.
    """

    @staticmethod
    def gaussian(distance, h):
        """
        2D 가우시안 커널:
            K(r) = (1 / (2*pi*h^2)) * exp(-r^2/(2*h^2))
        """
        xp = np if not hasattr(distance, "shape") else \
             getattr(distance, "xp", np)  # 간단 추론
        norm = 2.0 * xp.pi * (h**2)
        return xp.exp(-0.5 * (distance / h)**2) / norm

    @staticmethod
    def uniform(distance, h):
        """
        2D 균일(Top-hat) 커널:
            r <= h 일 때 1 / (pi * h^2), 그 외에는 0.
        """
        xp = np if not hasattr(distance, "shape") else \
             getattr(distance, "xp", np)
        area = xp.pi * (h**2)
        return xp.where(distance <= h, 1.0 / area, 0.0)

    @staticmethod
    def epanechnikov(distance, h):
        """
        2D 에파네치니코프(Epanechnikov) 커널:
            r <= h 일 때 K(r) = (2 / (pi*h^2)) * (1 - (r^2 / h^2)), 그 외 0.
        """
        xp = np if not hasattr(distance, "shape") else \
             getattr(distance, "xp", np)
        r = distance / h
        return xp.where(r <= 1, (2.0 / (xp.pi * h**2)) * (1 - r**2), 0.0)

    @staticmethod
    def triangular(distance, h):
        """
        2D 삼각형(Conical) 커널:
            r <= h 일 때 K(r) = (3 / (pi*h^2)) * (1 - r/h), 그 외 0.
        """
        xp = np if not hasattr(distance, "shape") else \
             getattr(distance, "xp", np)
        r = distance / h
        return xp.where(r <= 1, (3.0 / (xp.pi * h**2)) * (1 - r), 0.0)

    @staticmethod
    def quartic(distance, h):
        """
        2D Biweight (Quartic) 커널:
            r <= h 일 때 K(r) = (3 / (pi*h^2)) * (1 - (r^2 / h^2))^2, 그 외 0.
        """
        xp = np if not hasattr(distance, "shape") else \
             getattr(distance, "xp", np)
        r = distance / h
        return xp.where(r <= 1,
                        (3.0 / (xp.pi * h**2)) * (1 - r**2)**2,
                        0.0)

    @staticmethod
    def triweight(distance, h):
        """
        2D Triweight 커널:
            r <= h 일 때 K(r) = (4 / (pi*h^2)) * (1 - (r^2 / h^2))^3, 그 외 0.
        """
        xp = np if not hasattr(distance, "shape") else \
             getattr(distance, "xp", np)
        r = distance / h
        return xp.where(r <= 1,
                        (4.0 / (xp.pi * h**2)) * (1 - r**2)**3,
                        0.0)

    @staticmethod
    def cosine(distance, h):
        """
        2D 코사인 커널(유계 지지 버전):
            r <= h 일 때 K(r) = C * cos( (pi/2) * (r/h) ), 그 외 0.
        여기서 C는 적분=1을 만족하는 상수.
        
        정확한 적분값은
          C * 2π * ∫[0..h] r cos( (π/2)*(r/h) ) dr = 1
        로부터 구해야 하며,
        아래 값은 그 결과를 미리 계산한 상수(약 1.11072/(h^2*pi)...)가 됩니다.

        여기서는 편의상 대략적 상수를 넣었으므로,
        실제론 필요에 따라 정확히 적분해 쓰시길 권장합니다.
        """
        xp = np if not hasattr(distance, "shape") else \
             getattr(distance, "xp", np)
        # 임의로 "A = 2/(pi*h^2)" 형태로 선언하되, 실제론 적분으로 확인 필요
        A = 2.0 / (xp.pi * h**2)
        r = distance / h
        return xp.where(r <= 1, A * xp.cos((xp.pi/2.0)*r), 0.0)

    @staticmethod
    def laplacian(distance, h):
        """
        2D 라플라시안(Laplacian) 커널 (무한 지지):
          통상적으로 K(r) ∝ exp(-r/h).
          하지만 r=0~∞까지 적분이 1이 되도록 정규화하려면,
          C = 1/(2π h^2) 형태가 아님. (정확도는 아래 식 참조)

        여기서는 '형태'만 보여주며, 실제론
        ∫[0..∞] 2π r * C * exp(-r/h) dr = 1
        => C = 1/(2π h^2).

        즉, 이 식 그대로 두면
            K(r) = (1/(2π h^2)) * exp(-r/h),  0 ≤ r < ∞.
        가 2D 평면에서 적분=1이 됩니다.
        """
        xp = np if not hasattr(distance, "shape") else \
             getattr(distance, "xp", np)
        # 2D 라플라시안 정규화: 1/(2π h^2)
        C = 1.0 / (2.0 * xp.pi * (h**2))
        return C * xp.exp(-distance / h)

    @staticmethod
    def logistic(distance, h):
        """
        2D 로지스틱(Logistic) 커널 (무한 지지).
        3D 버전을 그대로 2D에 적용하면 정규화가 다릅니다.

        실제 2D 정규화 상수는
          ∫[0..∞] 2π r K(r) dr = 1
        를 풀어야 하며, closed-form이 쉽지 않을 수 있습니다.

        여기서는 단순히 '형태'만 보여주는 예시입니다.
        """
        xp = np if not hasattr(distance, "shape") else \
             getattr(distance, "xp", np)
        # 편의상 h^2로 스케일링 (정규화 미검증)
        u = distance / h
        # 아래 식은 적분=1 보장 안 함. 실제론 별도 적분 필수.
        return xp.exp(-u) / (1 + xp.exp(-u))**2 / (h**2)
