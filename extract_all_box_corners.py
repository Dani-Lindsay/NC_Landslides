
"""
Scan a directory tree for */geo/geo_geometryRadar.h5 files, extract TL/TR/BL/BR
corners (lat/lon) for each, and write CSV + JSON + console output.

Methods:
- "edge": find outermost finite samples along the array edges (TL/TR/BL/BR);
          if any corner is NaN, fall back to "bbox".
- "bbox": compute axis-aligned bounding box from finite lon/lat (min/max).

Usage examples:
  python extract_all_box_corners.py --root .
  python extract_all_box_corners.py --root /path/to/Timeseries_2 --out_csv corners.csv --out_json corners.json
  python extract_all_box_corners.py --root . --method edge
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import h5py

def list_boxes(root):
    # recursive glob for */geo/geo_geometryRadar.h5
    pattern = os.path.join(root, "**", "geo", "geo_geometryRadar.h5")
    return sorted(glob.glob(pattern, recursive=True))

def first_finite_from_edge(lon, lat, r, c):
    # scan outward-in from a corner index (r,c) until a finite lon&lat is found
    # we search along a small neighborhood to be resilient to NaN edges
    H, W = lon.shape
    # short-circuit
    if np.isfinite(lon[r, c]) and np.isfinite(lat[r, c]):
        return r, c
    # search a diamond that grows
    max_rad = max(H, W)
    for rad in range(1, min(50, max_rad)):  # up to 50 px radius
        r0 = max(r - rad, 0); r1 = min(r + rad + 1, H)
        c0 = max(c - rad, 0); c1 = min(c + rad + 1, W)
        sub_lon = lon[r0:r1, c0:c1]
        sub_lat = lat[r0:r1, c0:c1]
        finite = np.isfinite(sub_lon) & np.isfinite(sub_lat)
        if finite.any():
            idx = np.argwhere(finite)[0]
            return r0 + idx[0], c0 + idx[1]
    return None

def corners_edge(lon, lat):
    H, W = lon.shape
    tl_idx = first_finite_from_edge(lon, lat, 0,     0)
    tr_idx = first_finite_from_edge(lon, lat, 0,     W-1)
    bl_idx = first_finite_from_edge(lon, lat, H-1,   0)
    br_idx = first_finite_from_edge(lon, lat, H-1,   W-1)
    if None in (tl_idx, tr_idx, bl_idx, br_idx):
        return None
    (r0,c0),(r1,c1),(r2,c2),(r3,c3) = tl_idx, tr_idx, bl_idx, br_idx
    tl = (float(lon[r0,c0]), float(lat[r0,c0]))
    tr = (float(lon[r1,c1]), float(lat[r1,c1]))
    bl = (float(lon[r2,c2]), float(lat[r2,c2]))
    br = (float(lon[r3,c3]), float(lat[r3,c3]))
    return tl, tr, bl, br

def corners_bbox(lon, lat):
    finite = np.isfinite(lon) & np.isfinite(lat)
    if not finite.any():
        return None
    lon_f = np.where(finite, lon, np.nan)
    lat_f = np.where(finite, lat, np.nan)
    min_lon = float(np.nanmin(lon_f))
    max_lon = float(np.nanmax(lon_f))
    min_lat = float(np.nanmin(lat_f))
    max_lat = float(np.nanmax(lat_f))
    # TL, TR, BR, BL from bbox
    tl = (min_lon, max_lat)
    tr = (max_lon, max_lat)
    br = (max_lon, min_lat)
    bl = (min_lon, min_lat)
    return tl, tr, bl, br

def wkt_from_corners(tl, tr, br, bl):
    ring = [tl, tr, br, bl, tl]
    coords = ", ".join([f"{x} {y}" for x,y in ring])
    return f"POLYGON(({coords}))"

def get_lon_lat(hf):
    # try common dataset names
    if "longitude" in hf and "latitude" in hf:
        return hf["longitude"][:], hf["latitude"][:]
    # fallback: scan
    lon = lat = None
    def visitor(name, obj):
        nonlocal lon, lat
        if isinstance(obj, h5py.Dataset):
            if lon is None and obj.shape and len(obj.shape)==2 and "lon" in name.lower():
                lon = obj[:]
            if lat is None and obj.shape and len(obj.shape)==2 and "lat" in name.lower():
                lat = obj[:]
    hf.visititems(visitor)
    return lon, lat

def main(root, out_csv, out_json, method="bbox"):
    files = list_boxes(root)
    rows = []
    for f in files:
        # infer box_id as the directory two levels up (*/geo/file -> */box_id/geo/file)
        head, geo_dir = os.path.split(os.path.dirname(f))  # .../y0_x0_box, geo
        box_id = os.path.basename(head)
        try:
            with h5py.File(f, "r") as hf:
                lon, lat = get_lon_lat(hf)
            if lon is None or lat is None or lon.ndim != 2 or lat.ndim != 2 or lon.shape != lat.shape:
                rows.append({"box_id": box_id, "file": f, "error": "lon/lat not found or bad shape"})
                continue

            # pick method
            corners = None
            if method == "edge":
                corners = corners_edge(lon, lat)
                if corners is None:
                    # fallback
                    corners = corners_bbox(lon, lat)
                    used_method = "bbox_fallback"
                else:
                    used_method = "edge"
            else:
                corners = corners_bbox(lon, lat)
                used_method = "bbox"

            if corners is None:
                rows.append({"box_id": box_id, "file": f, "error": "could not compute corners"})
                continue

            tl, tr, br, bl = corners  # note order returns tl,tr,bl,br for edge/bbox; normalize to tl,tr,br,bl
            # ensure consistent order:
            # if we got (tl,tr,bl,br), swap to (tl,tr,br,bl)
            if len(corners) == 4 and corners[2] == bl and corners[3] == br:
                pass
            # Build WKT
            wkt = wkt_from_corners(tl, tr, br, bl)

            rows.append({
                "box_id": box_id, "file": f, "method": used_method,
                "tl_lon": tl[0], "tl_lat": tl[1],
                "tr_lon": tr[0], "tr_lat": tr[1],
                "bl_lon": bl[0], "bl_lat": bl[1],
                "br_lon": br[0], "br_lat": br[1],
                "wkt_polygon": wkt
            })
        except Exception as e:
            rows.append({"box_id": box_id, "file": f, "error": str(e)})

    if not rows:
        print("No files found under:", root)
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {out_csv} and {out_json} with {len(df)} records.")
    # Also print a compact Python structure for quick paste
    minimal = [
        {
            "box_id": r["box_id"],
            "corners": {
                "tl": (r["tl_lon"], r["tl_lat"]),
                "tr": (r["tr_lon"], r["tr_lat"]),
                "br": (r["br_lon"], r["br_lat"]),
                "bl": (r["bl_lon"], r["bl_lat"]),
            }
        }
        for _, r in df.dropna(subset=["tl_lon","tr_lon","br_lon","bl_lon"]).iterrows()
        if "error" not in r
    ]
    print("corners = ", minimal)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract TL/TR/BL/BR corners from geo_geometryRadar.h5 files")
    ap.add_argument("--root", default=".", help="root directory to scan (recursive)")
    ap.add_argument("--out_csv", default="box_corners_all.csv", help="output CSV path")
    ap.add_argument("--out_json", default="box_corners_all.json", help="output JSON path")
    ap.add_argument("--method", choices=["edge","bbox"], default="bbox", help="corner extraction method")
    args = ap.parse_args()
    main(args.root, args.out_csv, args.out_json, method=args.method)
