#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: kernel.py
# structura/kernel.py

import numpy as np


class KernelFunctions:
    """
    다양한 커널 함수들을 static method 형태로 제공하는 클래스입니다.
    각 커널은 3차원 밀도 추정에 사용할 수 있도록 벡터화되어 있으며,
    입력 배열이 NumPy 또는 CuPy 배열인지에 따라 자동으로 처리합니다.
    """

    @staticmethod
    def gaussian(distance, h):
        """
        3차원 가우시안 커널 함수.
        K(d, h) = exp(-0.5*(d/h)^2) / ((2*pi)^(3/2) * h^3)
        """
        xp = np.get_array_module(distance) if hasattr(np, "get_array_module") else __import__("cupy").get_array_module(distance)
        norm = (2 * xp.pi) ** (1.5) * h ** 3
        return xp.exp(-0.5 * (distance / h) ** 2) / norm

    @staticmethod
    def uniform(distance, h):
        """
        균일 커널 (Top-hat Kernel).
        d <= h 인 경우 1 / (4/3*pi*h^3), 그 외에는 0.
        """
        xp = np.get_array_module(distance) if hasattr(np, "get_array_module") else __import__("cupy").get_array_module(distance)
        volume = 4.0 / 3.0 * xp.pi * h ** 3
        return xp.where(distance <= h, 1.0 / volume, 0.0)

    @staticmethod
    def epanechnikov(distance, h):
        """
        에파네chnik오프 커널.
        d <= h 인 경우 K(d,h) = (15/(8*pi*h^3))*(1 - (d/h)^2), 그 외에는 0.
        """
        xp = np.get_array_module(distance) if hasattr(np, "get_array_module") else __import__("cupy").get_array_module(distance)
        u = distance / h
        return xp.where(u <= 1, (15.0 / (8.0 * xp.pi * h ** 3)) * (1 - u ** 2), 0.0)

    @staticmethod
    def triangular(distance, h):
        """
        삼각형 커널.
        d <= h 인 경우: K(d,h) = (3/(pi*h^3)) * (1 - d/h), 그 외에는 0.
        """
        xp = np.get_array_module(distance) if hasattr(np, "get_array_module") else __import__("cupy").get_array_module(distance)
        return xp.where(distance <= h, (3.0 / (xp.pi * h ** 3)) * (1 - distance / h), 0.0)

    @staticmethod
    def quartic(distance, h):
        """
        4차 (Biweight) 커널.
        d <= h 인 경우: K(d,h) = (105/(32*pi*h^3)) * (1 - (d/h)^2)^2, 그 외에는 0.
        """
        xp = np.get_array_module(distance) if hasattr(np, "get_array_module") else __import__("cupy").get_array_module(distance)
        u = distance / h
        return xp.where(u <= 1, (105.0 / (32.0 * xp.pi * h ** 3)) * (1 - u ** 2) ** 2, 0.0)

    @staticmethod
    def triweight(distance, h):
        """
        6차 (Triweight) 커널.
        d <= h 인 경우: K(d,h) = (315/(64*pi*h^3)) * (1 - (d/h)^2)^3, 그 외에는 0.
        """
        xp = np.get_array_module(distance) if hasattr(np, "get_array_module") else __import__("cupy").get_array_module(distance)
        u = distance / h
        return xp.where(u <= 1, (315.0 / (64.0 * xp.pi * h ** 3)) * (1 - u ** 2) ** 3, 0.0)

    @staticmethod
    def cosine(distance, h):
        """
        코사인 커널.
        d <= h 인 경우: K(d,h) = (2/(pi*h^3)) * cos(pi/2 * (d/h)), 그 외에는 0.
        """
        xp = np.get_array_module(distance) if hasattr(np, "get_array_module") else __import__("cupy").get_array_module(distance)
        return xp.where(distance <= h, (2.0 / (xp.pi * h ** 3)) * xp.cos((xp.pi / 2) * (distance / h)), 0.0)

    @staticmethod
    def logistic(distance, h):
        """
        로지스틱 커널.
        K(d,h) = exp(-d/h)/(1+exp(-d/h))^2 / (h^3)
        """
        xp = np.get_array_module(distance) if hasattr(np, "get_array_module") else __import__("cupy").get_array_module(distance)
        u = distance / h
        return xp.exp(-u) / (1 + xp.exp(-u)) ** 2 / (h ** 3)

    @staticmethod
    def sigmoid(distance, h):
        """
        시그모이드 커널.
        d <= h 인 경우: K(d,h) = (1/(2*h^3)) * tanh(1 - d/h), 그 외에는 0.
        """
        xp = np.get_array_module(distance) if hasattr(np, "get_array_module") else __import__("cupy").get_array_module(distance)
        return xp.where(distance <= h, (1.0 / (2.0 * h ** 3)) * xp.tanh(1 - distance / h), 0.0)

    @staticmethod
    def laplacian(distance, h):
        """
        라플라시안 커널.
        K(d,h) = exp(-|d/h|) / (2*h^3)
        """
        xp = np.get_array_module(distance) if hasattr(np, "get_array_module") else __import__("cupy").get_array_module(distance)
        return xp.exp(-xp.abs(distance / h)) / (2 * h ** 3)
