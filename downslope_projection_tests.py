#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 13:45:27 2025

@author: daniellelindsay

Prepare aspect and slope from geo_geometry.h5

save_gmt.py geo_geometryRadar.h5 height
gmt grdgradient height.grd -Da -Gaspect.grd -fg -Sgradient.grd -V
gmt grdmath gradient.grd ATAN = slope.grd
^^^ read back into HDF5 (REMEMBER to flip the Y-axis on the way back in!)

GMT 6 Documentation:
https://docs.generic-mapping-tools.org/dev/grdgradient.html#d

gmt grdgradient height.grd -Da -Gaspect.grd -fg -Sgradient.grd -V

height.grd      input DEM grid in lat/lon/meters
-Da             find the aspect (i.e., the downslope direction)
-Gaspect.grd    name of outgrid "aspect.grd"
-fg             input grid is in geographic coordinates
-Sgradient.grd  name of output grid with scalar magnitudes of gradient vectors
"""
import landslide_utils as utils


# -----------------------
# Tests / Examples
# -----------------------
los_value = 1
los_inc_angle_deg = 30
los_az_angle_deg = -100
print("Test for ALOS-2 Descending:  LOS to Downslope Projection\n")

print("# Example 1: When landslide and LOS are exactly the same, very steep, moving westward\n")
slope_deg = 60
aspect_deg = 280
expected_return = 1
print(f"Expected return: {expected_return:.2f}")
projected_vel = utils.project_from_LOS_to_downslope(
    los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg
)

print("# Example 2: moderate slope, moving westward exactly in range direction\n")
slope_deg = 30
aspect_deg = 280
expected_return = 1
print(f"Expected return: {expected_return:.2f}")
projected_vel = utils.project_from_LOS_to_downslope(
    los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg
)

print("# Example 3: shallow slope, moving westward exactly in range direction\n")
slope_deg = 15
aspect_deg = 280
expected_return = 1.5
print(f"Expected return: {expected_return:.2f}")
projected_vel = utils.project_from_LOS_to_downslope(
    los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg
)

print("# Example 4: moderate slope, moving northward exactly perpendicular to range direction\n")
slope_deg = 30
aspect_deg = 10
expected_return = 1.5
print(f"Expected return: {expected_return:.2f}")
projected_vel = utils.project_from_LOS_to_downslope(
    los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg
)

print("# Example 5: moderate slope, moving southwest\n")
slope_deg = 30
aspect_deg = 210
expected_return = 2
print(f"Expected return: {expected_return:.2f}")
projected_vel = utils.project_from_LOS_to_downslope(
    los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg
)

print(
    "******************************\n"
    "Test for ALOS-2 Descending:  Downslope to LOS Projection\n"
    "******************************\n"
)

print("# Example 1: When landslide and LOS are exactly the same, very steep, moving westward\n")
slope_deg = 60
aspect_deg = 280
expected_return = 1
print(f"Expected return: {expected_return:.2f}")
projected_vel = utils.project_from_downslope_to_LOS(
    los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg
)

print("# Example 2: moderate slope, moving westward exactly in range direction\n")
slope_deg = 30
aspect_deg = 280
expected_return = 0.9
print(f"Expected return: {expected_return:.2f}")
projected_vel = utils.project_from_downslope_to_LOS(
    los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg
)

print("# Example 3: shallow slope, moving westward exactly in range direction\n")
slope_deg = 15
aspect_deg = 280
expected_return = 0.7
print(f"Expected return: {expected_return:.2f}")
projected_vel = utils.project_from_downslope_to_LOS(
    los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg
)

print("# Example 4: moderate slope, moving northward exactly perpendicular to range direction\n")
slope_deg = 30
aspect_deg = 10
expected_return = 0.4
print(f"Expected return: {expected_return:.2f}")
projected_vel = utils.project_from_downslope_to_LOS(
    los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg
)

print("# Example 5: moderate slope, moving southwest\n")
slope_deg = 30
aspect_deg = 210
expected_return = 0.8
print(f"Expected return: {expected_return:.2f}")
projected_vel = utils.project_from_downslope_to_LOS(
    los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg
)
