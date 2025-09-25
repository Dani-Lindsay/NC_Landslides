#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 12:46:02 2025

@author: daniellelindsay
"""

import pandas as pd
import numpy as np
import h5py
import os
from glob import glob

# Directory containing all HDF5 landslide time series files
data_dir = "/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4"
h5_files = glob(os.path.join(data_dir, "*.h5"))

def collect_all_hdf5_data(h5_files):
    all_rows = []

    for file_path in h5_files:
        row = {"filename": os.path.basename(file_path)}

        try:
            with h5py.File(file_path, "r") as f:
                # Collect root-level datasets
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        row[key] = f[key][()] if f[key].shape == () else np.nan

                # Recursively collect datasets and attributes in all groups
                def collect(group, prefix=""):
                    for k, v in group.items():
                        full_key = f"{prefix}{k}"
                        if isinstance(v, h5py.Dataset):
                            try:
                                data = v[()]
                                if isinstance(data, np.ndarray):
                                    if data.ndim == 0:
                                        row[full_key] = data.item()
                                    else:
                                        row[full_key] = np.nan
                                else:
                                    row[full_key] = data
                            except Exception:
                                row[full_key] = np.nan
                        elif isinstance(v, h5py.Group):
                            collect(v, prefix=f"{full_key}/")

                    # Collect group attributes
                    for attr_key, attr_val in group.attrs.items():
                        full_attr_key = f"{prefix[:-1]}__{attr_key}"
                        try:
                            if isinstance(attr_val, np.ndarray) and attr_val.ndim == 0:
                                row[full_attr_key] = attr_val.item()
                            else:
                                row[full_attr_key] = attr_val
                        except Exception:
                            row[full_attr_key] = np.nan

                collect(f)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        all_rows.append(row)

    return pd.DataFrame(all_rows)

# Compile the data
compiled_df = collect_all_hdf5_data(h5_files)

# Save to CSV
csv_path = "/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/compiled_landslide_data.csv"
compiled_df.to_csv(csv_path, index=False)

csv_path
