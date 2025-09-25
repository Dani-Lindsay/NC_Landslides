#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 16:08:06 2025

@author: daniellelindsay
"""

import numpy as np

def get_design_matrix(los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg):
    """
    Parameters
    ----------
    los_inc_angle_deg : float
        Angle between vertical and incidence angle in degrees.
        0 is vertically down, 30 is a steep angle to the ground, 90 is horizontal.
    los_az_angle_deg : float
        Satellite flight direction, measured anti-clockwise from east.
        Typical values:
            Descending ALOS-2  -100
            Ascending  ALOS-2   100
    slope_deg : float
        Angle between horizontal and the ground surface.
        0 is horizontal, 30 is a shallow slope, 90 is vertical.
    aspect_deg : float
        Downslope direction, measured positive from north.
        350–10   landslide moving downslope towards the north
        80–100   landslide moving eastward (towards a descending satellite)
        170–190  landslide moving downslope towards the south
        260–290  landslide moving downslope towards the west (away from a descending satellite)

    Returns
    -------
    G : float
        Projection of LOS unit into downslope direction (dot product).
        This number will always be bigger than LOS when used to scale from LOS→downslope.
    """
    # Convert aspect from positive clockwise-from-north to positive anti-clockwise-from-east
    slope_direction_deg = (aspect_deg - 90) * -1

    print(
        f" Incidence: {los_inc_angle_deg:.2f}   Azimuth {los_az_angle_deg:.2f} (Anti-CW from East)\n"
        f" Slope {slope_deg:.2f}       Aspect {aspect_deg:.2f}         "
        f"Aspect {slope_direction_deg:.2f} (Anti-CW from East)"
    )

    # LOS components
    los_E = np.sin(np.deg2rad(los_inc_angle_deg)) * np.sin(np.deg2rad(los_az_angle_deg)) * -1
    los_N = np.sin(np.deg2rad(los_inc_angle_deg)) * np.cos(np.deg2rad(los_az_angle_deg))
    los_U = np.cos(np.deg2rad(los_inc_angle_deg))

    # Downslope components
    vector_E = np.cos(np.deg2rad(slope_deg)) * np.cos(np.deg2rad(slope_direction_deg)) 
    vector_N = np.cos(np.deg2rad(slope_deg)) * np.sin(np.deg2rad(slope_direction_deg))
    vector_U = np.sin(np.deg2rad(slope_deg)) * -1

    # Normalize vectors
    L = np.array([los_E, los_N, los_U])
    F = np.array([vector_E, vector_N, vector_U])

    # Compute design matrix (dot product)
    G = np.dot(L, F)

    return G


def project_from_LOS_to_downslope(los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg):
    """
    Parameters
    ----------
    los_inc_angle_deg : float
        Angle between vertical and incidence angle in degrees.
        0 is vertically down, 30 is a steep angle to the ground, 90 is horizontal.
    los_az_angle_deg : float
        Satellite flight direction, measured anti-clockwise from east.
        Typical values:
            Descending ALOS-2  -100
            Ascending  ALOS-2   100
    slope_deg : float
        Angle between horizontal and the ground surface.
        0 is horizontal, 30 is a shallow slope, 90 is vertical.
    aspect_deg : float
        Downslope direction, measured positive from north.
        350–10, 80–100, 170–190, 260–290 describe N/E/S/W examples.

    Returns
    -------
    project_vel : float
        Projection of LOS unit into downslope direction.
        This number will always be bigger than LOS.
    """
    G = get_design_matrix(los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg)
    
    # Divide LOS to get to downslope 
    project_vel = los_value / G if G != 0 else np.nan

    print(
        f" LOS value:          {los_value:.2f} unit\n"
        f" Downslope unit:     {project_vel:.2f}\n"
    )
    return project_vel


def project_from_downslope_to_LOS(ds_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg):
    """
    Parameters
    ----------
    los_inc_angle_deg : float
        Angle between vertical and incidence angle in degrees.
        0 is vertically down, 30 is a steep angle to the ground, 90 is horizontal.
    los_az_angle_deg : float
        Satellite flight direction, measured anti-clockwise from east.
        Typical values:
            Descending ALOS-2  -100
            Ascending  ALOS-2   100
    slope_deg : float
        Angle between horizontal and the ground surface.
        0 is horizontal, 30 is a shallow slope, 90 is vertical.
    aspect_deg : float
        Downslope direction, measured positive from north.

    Returns
    -------
    project_LOS : float
        Projection of downslope unit into LOS direction.
        LOS will always be smaller than downslope.
    """
    G = get_design_matrix(los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg)
    
    # Multiple Downslope value to get to LOS
    project_LOS = ds_value * G if G != 0 else np.nan

    print(
        f" Downslope value:    {ds_value:.2f} unit\n"
        f" LOS value:          {project_LOS:.2f}\n"
    )
    return project_LOS