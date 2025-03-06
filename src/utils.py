#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2025-02-28
# @Filename: utils.py
# structura/utils.py

from logger import logger
import numpy as np


def recommend_bins(
    box_size,
    num_recommendations=5,
    min_resolution=None,
    max_resolution=None,
    scale_type="linear",
):
    """
    Computes recommended bin counts and resolutions for generating 2D histograms at various scales.

    Parameters
    ----------
    box_size : float
        The size of the simulation box (in cMpc/h).
    num_recommendations : int, optional
        The number of recommended bin configurations to generate (default: 5).
    min_resolution : float or None, optional
        The minimum resolution. If None, defaults to box_size / 20.
    max_resolution : float or None, optional
        The maximum resolution. If None, defaults to box_size / 200.
    scale_type : str, optional
        The type of spacing to use for resolutions; either "linear" for linear spacing or "log" for logarithmic spacing.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing:
            - "bins": the computed number of bins for the corresponding resolution,
            - "resolution": the resolution value rounded to two decimal places.
    """
    # Set default resolution values if not provided.
    if min_resolution is None:
        min_resolution = box_size / 20  # Largest scale
    if max_resolution is None:
        max_resolution = box_size / 200  # Smallest scale

    # Generate the list of resolutions based on the selected scale type.
    if scale_type == "linear":
        resolutions = np.linspace(min_resolution, max_resolution, num_recommendations)
    elif scale_type == "log":
        resolutions = np.logspace(
            np.log10(min_resolution), np.log10(max_resolution), num_recommendations
        )
    else:
        raise ValueError("scale_type must be 'linear' or 'log'.")

    # Compute the number of bins for each resolution.
    bins_list = [int(box_size / res) for res in resolutions]

    # Construct and return the recommendations list.
    recommendations = [
        {"bins": bins, "resolution": round(res, 2)}
        for bins, res in zip(bins_list, resolutions)
    ]

    return recommendations


def set_x_range(center_cMpc, thickness_cMpc, lbox_cMpc, lbox_ckpch):
    """
    Converts a spatial range from comoving Mpc/h units to comoving kpc/h units for data filtering purposes.

    Parameters
    ----------
    center_cMpc : float
        The center position in comoving Mpc/h.
    thickness_cMpc : float
        The thickness of the selection range in comoving Mpc/h.
    lbox_cMpc : float
        The box size in comoving Mpc/h.
    lbox_ckpch : float
        The box size in comoving kpc/h.

    Returns
    -------
    tuple of float
        The minimum and maximum x values in comoving kpc/h.
    """
    conversion_factor = lbox_ckpch / lbox_cMpc
    center_ckpch = center_cMpc * conversion_factor
    thickness_ckpch = thickness_cMpc * conversion_factor
    x_min = center_ckpch - thickness_ckpch / 2
    x_max = center_ckpch + thickness_ckpch / 2

    logger.info(
        f"X range set: {x_min:.2f} - {x_max:.2f} ckpc/h "
        f"(Center: {center_cMpc} cMpc/h, Thickness: {thickness_cMpc} cMpc/h)"
    )

    return x_min, x_max


def get_hubble_parameter():
    """
    Retrieves the reduced Hubble parameter (h) from Astropy's Planck15 cosmology.

    Returns
    -------
    float
        The reduced Hubble constant h, defined as h = H0/100.
    """
    try:
        # Lazy import: only import when this function is called.
        from astropy.cosmology import Planck15
    except ImportError:
        logger.warning(
            "Failed to import Planck15 from astropy.cosmology; "
            "using FlatLambdaCDM with Planck15 parameters as fallback."
        )
        try:
            from astropy.cosmology import FlatLambdaCDM

            Planck15 = FlatLambdaCDM(H0=67.7, Om0=0.3089, Tcmb0=2.7255)
        except Exception as e_inner:
            logger.error("Failed to import astropy.cosmology properly.", exc_info=True)
            raise e_inner
    return Planck15.H0.value / 100.0


def cMpc_to_cMpc_h(distance_cMpc, h=None):
    """
    Converts a distance from comoving megaparsecs (cMpc) to comoving megaparsecs per h (cMpc/h).

    Parameters
    ----------
    distance_cMpc : float
        The distance in comoving Mpc.
    h : float, optional
        The reduced Hubble constant. If None, the value is obtained from Astropy's Planck15.

    Returns
    -------
    float
        The distance in comoving Mpc/h.

    Note
    ----
    The conversion is performed via the relation: cMpc/h = cMpc * h.
    """
    if h is None:
        h = get_hubble_parameter()
    return distance_cMpc * h


def cMpc_h_to_cMpc(distance_cMpc_h, h=None):
    """
    Converts a distance from comoving megaparsecs per h (cMpc/h) to comoving megaparsecs (cMpc).

    Parameters
    ----------
    distance_cMpc_h : float
        The distance in comoving Mpc/h.
    h : float, optional
        The reduced Hubble constant. If None, the value is obtained from Astropy's Planck15.

    Returns
    -------
    float
        The distance in comoving Mpc.

    Note
    ----
    The conversion is performed via the relation: cMpc = cMpc/h / h.
    """
    if h is None:
        h = get_hubble_parameter()
    return distance_cMpc_h / h
