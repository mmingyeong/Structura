#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: utils.py
# structura/utils.py

from logger import logger  

def set_x_range(center_cMpc, thickness_cMpc, lbox_cMpch, lbox_ckpch):
    """
    Converts cMpc/h range to ckpc/h for filtering.

    Args:
        center_cMpc (float): Center X position (cMpc/h)
        thickness_cMpc (float): Selection thickness (cMpc/h)
        lbox_cMpc (float): Box size (cMpc/h)
        lbox_ckpch (float): Box size (ckpc/h)

    Returns:
        tuple: (x_min, x_max) in ckpc/h
    """
    conversion_factor = lbox_ckpch / lbox_cMpch
    center_ckpch = center_cMpc * conversion_factor
    thickness_ckpch = thickness_cMpc * conversion_factor
    x_min = center_ckpch - thickness_ckpch / 2
    x_max = center_ckpch + thickness_ckpch / 2

    logger.info(f"✅ X range set: {x_min:.2f} - {x_max:.2f} ckpc/h (Center: {center_cMpc} cMpc/h, Thickness: {thickness_cMpc} cMpc/h)")

    return x_min, x_max
