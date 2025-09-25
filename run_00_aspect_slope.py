 # 1034  save_gmt.py geo_geometryRadar.h5 height
 # 1035  gmt grdgradient height.grd -Da -Gaspect.grd -fg -Sgradient.grd -V
 # 1036  gmt grdmath gradient.grd ATAN = slope.grd


#!/usr/bin/env python3
import os
import glob
import subprocess
import h5py
import xarray as xr
import numpy as np


# → adjust these if needed
TS_DIR     = "/Volumes/Seagate/NC_Landslides/Timeseries_2"
TS_DIR     = "/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/170_5_28"
SAVE_GMT   = "save_gmt.py"   # your helper script
GMT        = "gmt"           # must be on PATH

def run(cmd):
    print(" >", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__=="__main__":
    boxes = sorted(glob.glob(os.path.join(TS_DIR, "y*_box")))
    boxes = sorted(glob.glob(os.path.join(TS_DIR, "*")))
    for box in boxes:
        box_id   = os.path.basename(box)
        geo_h5   = os.path.join(box, "geo", "geo_geometryRadar.h5")
        if not os.path.exists(geo_h5):
            print(f"⚠️  missing {geo_h5}, skipping")
            continue

        print(f"\n=== BOX {box_id} ===")

        # temp filenames (in CWD or box folder, your choice)
        height_grd   = os.path.join(box, "geo", "height.grd")
        grad_grd     = os.path.join(box, "geo", "gradient.grd")
        aspect_grd   = os.path.join(box, "geo", "aspect.grd")
        slope_grd    = os.path.join(box, "geo", "slope.grd")

        

        # # 1) export 'height' to GMT grid
        run([SAVE_GMT, geo_h5, "height", "-o", height_grd])

        
        run([
            "gmt", "grdgradient",
            height_grd,
            "-Da",
            f"-G{aspect_grd}",
            "-fg",
            f"-S{grad_grd}",
            "-V",
        ])

        # # 3) compute slope in degrees: slope = atan(gradient)*180/pi
        run([GMT, "grdmath", grad_grd, "ATAN", "=", slope_grd])

        # 4) read back with xarray
        da_slope  = xr.open_dataarray(slope_grd)
        da_aspect = xr.open_dataarray(aspect_grd)

        # # Flip the arrays on the Y‐axis if latitudes run high→low
        slope_arr  = np.flipud(da_slope.values)
        aspect_arr = np.flipud(da_aspect.values)

        # 5) write into the HDF5
        with h5py.File(geo_h5, "r+") as hf:
            for name in ("slope","aspect"):
                if name in hf:
                    del hf[name]
            hf.create_dataset("slope",  data=slope_arr,
                              compression="gzip")
            hf.create_dataset("aspect", data=aspect_arr,
                              compression="gzip")
        print(f"✔  Written slope & aspect into {geo_h5}")

        # 6) clean up
        for fn in (height_grd, grad_grd, aspect_grd, slope_grd):
            try: os.remove(fn)
            except OSError:
                pass
