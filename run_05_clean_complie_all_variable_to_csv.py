#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flatten HDF5 landslide data into a CSV, decoding bytes and replacing b'null' with NaN.
"""

import os
import numpy as np
import pandas as pd
import h5py
from glob import glob

# Directory containing all HDF5 files
data_dir = "/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4"
h5_files = glob(os.path.join(data_dir, "*.h5"))

def clean_value(val):
    """Decode bytes and replace 'null' with NaN."""
    if isinstance(val, bytes):
        val = val.decode("utf-8", errors="ignore")
    if isinstance(val, str) and val.strip().lower() == "null":
        return np.nan
    return val

def collect_all_hdf5_data(h5_files):
    all_rows = []

    for file_path in h5_files:
        row = {"filename": os.path.basename(file_path)}

        try:
            with h5py.File(file_path, "r") as f:

                def collect(group, prefix=""):
                    for key in group:
                        item = group[key]
                        full_key = f"{prefix}{key}"

                        if isinstance(item, h5py.Dataset):
                            try:
                                data = item[()]
                                if isinstance(data, np.ndarray):
                                    if data.ndim == 0:
                                        row[full_key] = clean_value(data.item())
                                    else:
                                        row[full_key] = np.nan  # skip arrays
                                else:
                                    row[full_key] = clean_value(data)
                            except Exception:
                                row[full_key] = np.nan

                        elif isinstance(item, h5py.Group):
                            collect(item, prefix=f"{full_key}/")

                    # Group attributes
                    for attr_key, attr_val in group.attrs.items():
                        attr_name = f"{prefix[:-1]}__{attr_key}"
                        try:
                            if isinstance(attr_val, np.ndarray) and attr_val.ndim == 0:
                                row[attr_name] = clean_value(attr_val.item())
                            else:
                                row[attr_name] = clean_value(attr_val)
                        except Exception:
                            row[attr_name] = np.nan

                collect(f)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        all_rows.append(row)

    return pd.DataFrame(all_rows)

# Compile and save
compiled_df = collect_all_hdf5_data(h5_files)
out_csv = os.path.join(data_dir, "compiled_landslide_data.csv")
compiled_df.to_csv(out_csv, index=False)

print("✓ Saved cleaned CSV to:", out_csv)
