#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:09:19 2023

@author: daniellelindsay
"""


import numpy as np
from datetime import datetime
import matplotlib.path as mpath
from scipy.spatial import KDTree
import h5py
import netCDF4 as nc
import pandas as pd

def date_to_decimal_year(date_input, format_str=None):
    """
    Convert a date to a decimal year.

    :param date_input: Date input which can be a string, byte string, or datetime object.
    :param format_str: Format string for parsing the date. If None, the function will attempt to infer the format.
                       Acceptable formats include: "%Y-%m-%d" (e.g., "2021-12-31"), 
                       "%m/%d/%Y" (e.g., "12/31/2021"), and "%Y%m%d" (e.g., "20211231").
    :return: Decimal year corresponding to the input date.
    """

    # If the input is a byte string, decode it to a regular string
    if isinstance(date_input, bytes):
        date_input = date_input.decode('utf-8')

    # If the input is already a datetime object, use it directly
    if isinstance(date_input, datetime):
        date = date_input
    else:
        # Attempt to infer the format if not provided
        if format_str is None:
            if "-" in date_input:
                format_str = "%Y-%m-%d"
            elif "/" in date_input:
                format_str = "%m/%d/%Y"
            elif len(date_input) == 8:
                format_str = "%Y%m%d"
            else:
                raise ValueError("Unknown date format. Please provide a format string.")

        # Parse the date string using the provided format
        date = datetime.strptime(date_input, format_str)

    start_of_year = datetime(year=date.year, month=1, day=1)
    start_of_next_year = datetime(year=date.year+1, month=1, day=1)
    year_length = (start_of_next_year - start_of_year).total_seconds()
    year_progress = (date - start_of_year).total_seconds()
    decimal_year = date.year + year_progress / year_length
    return decimal_year

def decimal_year_to_datetime(decimal_years):
    from datetime import datetime, timedelta

    if isinstance(decimal_years, (float, int)):
        decimal_years = [decimal_years]

    dates = []
    for y in decimal_years:
        year = int(y)
        remainder = y - year
        start = datetime(year, 1, 1)
        if year + 1 <= 9999:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, 12, 31)
        delta = end - start
        date = start + timedelta(seconds=remainder * delta.total_seconds())
        dates.append(date)
    return pd.to_datetime(dates)


def read_insar_data(dic):
    """
    Read InSAR geometry, velocity, and timeseries data from provided file paths.

    Parameters:
    - geo_file: Path to the file containing InSAR geometry data.
    - vel_file: Path to the file containing InSAR velocity data.
    - ts_file: Path to the file containing InSAR timeseries data.

    Returns:
    A dictionary containing:
    - 'lons': Longitudes from the geometry file.
    - 'lats': Latitudes from the geometry file.
    - 'inc': Incidence angles from the geometry file.
    - 'azi': Azimuth angles from the geometry file.
    - 'vel': Velocities from the velocity file.
    - 'ts': Timeseries data from the timeseries file.
    - 'ts_dates': Decimal years for each date in the timeseries.
    """
    print("Loading %s " % dic["Platform"])

    if dic["Platform"] == "ALOS-2":
        with h5py.File(dic["geo_file"], 'r') as hfgeo:
            print("Loading %s " % dic["geo_file"])
            
            lons = np.array(hfgeo["longitude"][:])
            lats = np.array(hfgeo["latitude"][:])
            inc = np.array(hfgeo["incidenceAngle"][:])
            azi = np.array(hfgeo["azimuthAngle"][:])
            
    if dic["Platform"] == "Sentinel-1":
        with h5py.File(dic["geo_file"], 'r') as hfgeo:
            print("Loading %s " % dic["geo_file"])
            inc = np.array(hfgeo["incidenceAngle"][:])
            # Get attributes
            x_start = hfgeo.attrs['X_FIRST']
            y_start = hfgeo.attrs['Y_FIRST']
            x_step = hfgeo.attrs['X_STEP']
            y_step = hfgeo.attrs['Y_STEP']
            length = hfgeo.attrs['LENGTH']
            width = hfgeo.attrs['WIDTH']
            heading = dic["Heading"]

            print(f'inc has shape:  {inc.shape}')
            print(f'attributes has: {length, width}')
            
            if float(length) != inc.shape[0]:
                print(f'Lats : Length = {length} not equal to inc length {inc.shape[0]}')

            if float(width) != inc.shape[1]:
                print(f'Lons: Width = {width} not equal to inc width {inc.shape[1]}')
            
            # Make mesh of Eastings and Northings using linspace to ensure correct array size
            lon = np.linspace(float(x_start), float(x_start) + float(x_step) * (float(width)-1), int(width))
            lat = np.linspace(float(y_start), float(y_start) + float(y_step) * (float(length)-1), int(length))

            lons, lats = np.meshgrid(lon, lat)
            print(f'linspace lons has shape:  {lons.shape}')
            print(f'linsapre lats has shape:  {lats.shape}')

            # Add dataset of azimuthAngle to geometry
            azi = np.full((int(length), int(width)), float(heading))
    
    with h5py.File(dic["vel_file"], 'r') as hfvel:
        print("Loading %s " % dic["vel_file"])
        vel = np.array(hfvel["velocity"][:])
        vel = np.array(hfvel["velocity"][:])
        vel[vel == 0] = np.nan  # Set zero values to nan

    with h5py.File(dic["ts_file"], 'r') as hfvel:
        print("Loading %s " % dic["ts_file"])
        ts = np.array(hfvel["timeseries"][:])
        ts[ts == 0] = np.nan  # Set zero values to nan
        ts_dates_bytes = np.array(hfvel["date"][:])  # Assuming dates are stored as bytes
        print(f'ts has shape:  {ts.shape}')

    # Convert byte strings of dates to decimal years
    ts_dates = [date_to_decimal_year(d.decode('utf-8')) for d in ts_dates_bytes]
    
    # Return a dictionary of the data
    return {
        'lons': lons,
        'lats': lats,
        'inc': inc,
        'azi': azi,
        'vel': vel,
        'ts': ts,
        'ts_dates': ts_dates
    }


def read_gmt_grid(filename):
    with nc.Dataset(filename) as ds:
        # Assume the grid variables are named 'lon', 'lat', and 'z'
        lon = ds.variables['lon'][:]
        lat = ds.variables['lat'][:]
        grid_data = ds.variables['z'][:]
    return lon, lat, grid_data

def read_gmt_polygon(gmt_file):
    coords = []
    with open(gmt_file, 'r') as file:
        for line in file:
            if not line.startswith('>'):
                lon, lat = line.strip().split()
                coords.append((float(lon), float(lat)))
    return coords

def create_mask(lons, lats, coords):
    path = mpath.Path(coords)
    x, y = np.meshgrid(lons, lats)
    points = np.vstack((x.flatten(), y.flatten())).T
    mask = path.contains_points(points).reshape(x.shape)
    return mask


def calc_dist(lon1, lat1, lon2, lat2, ellps='WGS84'):
    """
    Returns distance between two lat and lon in meters
    """
    import pyproj
    geodesic = pyproj.Geod(ellps=ellps)
    fwd_az, back_az, dist = geodesic.inv(lon1, lat1, lon2, lat2)
    return dist 

def calc_azimuth_distance(lon1, lat1, lon2, lat2, ellps='WGS84'):
    """
    Returns distance between two lat and lon in meters
    """
    import pyproj
    geodesic = pyproj.Geod(ellps=ellps)
    fwd_az, back_az, dist = geodesic.inv(lon1, lat1, lon2, lat2)
    return fwd_az, back_az, dist 

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = np.nanargmin(np.abs(array - value))
    return idx

