#!/usr/bin/env python3
import os
import glob
import subprocess
import h5py
import xarray as xr

# → adjust these if needed
TS_DIR     = "/Volumes/Seagate/NC_Landslides/Timeseries"
SAVE_GMT   = "save_gmt.py"   # your helper script
GMT        = "gmt"           # must be on PATH

def run(cmd):
    print(" >", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__=="__main__":
    boxes = sorted(glob.glob(os.path.join(TS_DIR, "y*_box")))
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

        # 1) export 'height' to GMT grid
        run([SAVE_GMT, geo_h5, "height", "-o", height_grd])

        # 2) compute gradient (m/m) and aspect (°)
        run([GMT, "grdgradient", height_grd, 
             "-Da", 
             "-G" + grad_grd,
             "-fg",  # give gradient in true m/m, aspect in degrees
             "-V"])  # verbose

        # 3) compute slope in degrees: slope = atan(gradient)*180/pi
        run([GMT, "grdmath", grad_grd, "ATAN", "=", slope_grd])

        # 4) read back with xarray
        da_slope  = xr.open_dataarray(slope_grd)
        da_aspect = xr.open_dataarray(aspect_grd)

        # 5) write into the HDF5
        with h5py.File(geo_h5, "r+") as hf:
            for name in ("slope","aspect"):
                if name in hf:
                    del hf[name]
            hf.create_dataset("slope",  data=da_slope .values,
                              compression="gzip")
            hf.create_dataset("aspect", data=da_aspect.values,
                              compression="gzip")
        print(f"✔  Written slope & aspect into {geo_h5}")

        # 6) clean up
        for fn in (height_grd, grad_grd, aspect_grd, slope_grd):
            try: os.remove(fn)
            except OSError:
                pass
