#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 13:45:27 2025

@author: daniellelindsay

Prepare aspect and slope from geo_geometry.h5

save_gmt.py geo_geometryRadar.h5 height
gmt grdgradient height.grd -Da -Gaspect.grd -fg -Sgradient.grd -V
gmt grdmath gradient.grd ATAN = slope.grd
^^^ read back into hdf5, REMEMBER to flip the Y-axis on the way back in!) 

GMT 6 Documentaion 
https://docs.generic-mapping-tools.org/dev/grdgradient.html#d

gmt grdgradient height.grd -Da -Gaspect.grd -fg -Sgradient.grd -V

height.grd      input DEM grd in lat/lon/meters
-Da             Find the aspect (i.e., the down-slope direction).
-Gaspect.grd    Name of outgrid is "aspect.grd"
-fg             input grid is in geographic coordinates
-Sgradient.grd  Name of output grid file with scalar magnitudes of gradient vectors.

"""
import numpy as np     


def get_design_matrix(los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg):
    """

    Parameters
    ----------
    los_inc_angle_deg : float
        Angle between vertical and incidence angle in degrees. 0 is vertically down, 30 is a steep angle to the ground, 90 is horizontal. 
    los_az_angle_deg : float
        Satellites flight direciton. Measured anti-clockwise from east. Typical Values:
            Descending ALOS-2 -100
            Ascending ALOS-2 100
    slope_deg : float
        Angle between horizontal and the ground surface. 0 is horizontal, 30 is a shallow slope, 90 is vertical. 
    aspect_deg : float
        downslope angle, measured postive from north. 
        350-10 is landslide moving downslope towards the north
        80-100 is lanslide moving eastward (towards a descending satellite)
        170-190 is landslide moving down slope towards the south
        260-290 is landslide moving downslope towards the west (away from a descending satellite)

    Returns
    -------
    project_vel : float
        Projection of LOS unit into downslope direciton. 
        This number will always be bigger than LOS. 

    """
    # Convert aspect from postive clockwise from north to postive anticlockwise from east
    slope_direction_deg = (aspect_deg - 90) *-1 
    
    print(f" Incidence: {los_inc_angle_deg:.2f}   Azimuth {los_az_angle_deg:.2f} (Anti-CW from East) \n", 
          f"Slope {slope_deg:.2f}       Aspect {aspect_deg:.2f}         Aspect {slope_direction_deg:.2f} (Anti-CW from East)")
        
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
        Angle between vertical and incidence angle in degrees. 0 is vertically down, 30 is a steep angle to the ground, 90 is horizontal. 
    los_az_angle_deg : float
        Satellites flight direciton. Measured anti-clockwise from east. Typical Values:
            Descending ALOS-2 -100
            Ascending ALOS-2 100
    slope_deg : float
        Angle between horizontal and the ground surface. 0 is horizontal, 30 is a shallow slope, 90 is vertical. 
    aspect_deg : float
        downslope angle, measured postive from north. 
        350-10 is landslide moving downslope towards the north
        80-100 is lanslide moving eastward (towards a descending satellite)
        170-190 is landslide moving down slope towards the south
        260-290 is landslide moving downslope towards the west (away from a descending satellite)

    Returns
    -------
    project_vel : float
        Projection of LOS unit into downslope direciton. 
        This number will always be bigger than LOS. 

    """
    
    G = get_design_matrix(los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg)
    
    project_vel = los_value / G if G != 0 else np.nan
    
    print(f" LOS value:          {los_value:.2f} unit \n",
          f"Downslope unit:     {project_vel:.2f} \n")
    
    return project_vel

def project_from_downslope_to_LOS(ds_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg):
    """

    Parameters
    ----------
    los_inc_angle_deg : float
        Angle between vertical and incidence angle in degrees. 0 is vertically down, 30 is a steep angle to the ground, 90 is horizontal. 
    los_az_angle_deg : float
        Satellites flight direciton. Measured anti-clockwise from east. Typical Values:
            Descending ALOS-2 -100
            Ascending ALOS-2 100
    slope_deg : float
        Angle between horizontal and the ground surface. 0 is horizontal, 30 is a shallow slope, 90 is vertical. 
    aspect_deg : float
        downslope angle, measured postive from north. 
        350-10 is landslide moving downslope towards the north
        80-100 is lanslide moving eastward (towards a descending satellite)
        170-190 is landslide moving down slope towards the south
        260-290 is landslide moving downslope towards the west (away from a descending satellite)

    Returns
    -------
    project_vel : float
        Projection of downslope unit into LOS direction
        LOS will always be smaller than downslope. 

    """
    
    G = get_design_matrix(los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg)
    
    project_LOS = ds_value * G if G != 0 else np.nan
    
    print(f" Downslope value:          {ds_value:.2f} unit \n",
          f"LOS value:     {project_LOS:.2f} \n")
    
    return project_LOS


los_value = 1
los_inc_angle_deg = 30
los_az_angle_deg = -100
print("Test for ALOS-2 Descending:  LOS to Downslope Projection \n")

print("# Example 1: When landslide and LOS are exactly the same, very steep, moving westward \n") 
slope_deg = 60
aspect_deg = 280
expected_return = 1
print(f"Expected return: {expected_return:.2f}")
projected_vel = project_from_LOS_to_downslope(los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg)


print("# Example 2: moderate slope, moving westward exactly in range direction \n") 
slope_deg = 30
aspect_deg = 280
expected_return = 1
print(f"Expected return: {expected_return:.2f}")
projected_vel = project_from_LOS_to_downslope(los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg)

print("# Example 3: shallow slope, moving westward exactly in range direction \n") 
slope_deg = 15
aspect_deg = 280
expected_return = 1.5
print(f"Expected return: {expected_return:.2f}")
projected_vel = project_from_LOS_to_downslope(los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg)


print("# Example 4: moderate slope, moving northward exactly perpendicular to range direction \n") 
slope_deg = 30
aspect_deg = 10
expected_return = 1.5
print(f"Expected return: {expected_return:.2f}")
projected_vel = project_from_LOS_to_downslope(los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg)

print("# Example 5: moderate slope, moving southwest \n") 
slope_deg = 30
aspect_deg = 210
expected_return = 2
print(f"Expected return: {expected_return:.2f}")
projected_vel = project_from_LOS_to_downslope(los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg)

print("****************************** \n",
      "Test for ALOS-2 Descending:  Downslope to LOS Projection \n", 
      "******************************")

print("# Example 1: When landslide and LOS are exactly the same, very steep, moving westward \n") 
slope_deg = 60
aspect_deg = 280
expected_return = 1
print(f"Expected return: {expected_return:.2f}")
projected_vel = project_from_downslope_to_LOS(los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg)

print("# Example 2: moderate slope, moving westward exactly in range direction \n") 
slope_deg = 30
aspect_deg = 280
expected_return = 0.9
print(f"Expected return: {expected_return:.2f}")
projected_vel = project_from_downslope_to_LOS(los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg)

print("# Example 3: shallow slope, moving westward exactly in range direction \n") 
slope_deg = 15
aspect_deg = 280
expected_return = 0.7
print(f"Expected return: {expected_return:.2f}")
projected_vel = project_from_downslope_to_LOS(los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg)


print("# Example 4: moderate slope, moving northward exactly perpendicular to range direction \n") 
slope_deg = 30
aspect_deg = 10
expected_return = 0.4
print(f"Expected return: {expected_return:.2f}")
projected_vel = project_from_downslope_to_LOS(los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg)

print("# Example 5: moderate slope, moving southwest \n") 
slope_deg = 30
aspect_deg = 210
expected_return = 0.8
print(f"Expected return: {expected_return:.2f}")
projected_vel = project_from_downslope_to_LOS(los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg)

# print("# Example 2: Moderate, moving westward \n") 
# slope_deg = 30
# aspect_deg = 270
# expected_return = 1
# print(f"Expected return: {expected_return:.2f}")
# projected_vel = project_from_LOS_to_downslope(los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg)


# print("# Example 3: Shallow, moving south \n")
# slope_deg = 15
# aspect_deg = 181
# expected_return = 1
# print(f"Expected return: {expected_return:.2f}")
# projected_vel = project_from_LOS_to_downslope(los_value, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg)


#print("Expected Value for input 1 unit: north facing slope (aspect = 90), slope direction (north = 0), shallow slope (15)")
#print("North unit of 1 = 0.2 ")


# if the slope is 60, and the incidence is 30. in the range direction -100 +90 for downslope.  -->the input of 1, output is 1. 

# if it moves north on 30 slope. north projects ).1 into the look vector. 10% of north in the LOS. for 20 degree slope, 
# the projection into LOS of north moving landlside on 30 slope is 0.5 
# 1 unit of downslope is 0.5 as LOS. 
# How much the east projects to LOS, ... 
# 1/0.8 1 in downslope projects to 0.8 in LOS. 

# 1 unit downslope * LOS 0.8 = 0.8 in LOS. 

# one unit of downslope, is always less in LOS. 

# 1 unit of LOS, will always be more in downslope direction. 

# E = 1, 0, 0.5. 
# relationship between aspet and the 3 componet downslope. 


#print("Expected Value for input 1 unit: ")

#print("Expected Value for input 1 unit: ")

# plot the dot product as a map. 


