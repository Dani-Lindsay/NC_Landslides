import requests
import numpy as np
import io
import h5py
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ShakeMapError(Exception):
    """Custom exception for ShakeMap retrieval/conversion errors."""

def get_pga(event_id: str, lat: float, lon: float):
    """
    Fetch the ShakeMap HDF5 for `event_id` and return:
      - PGA mean (g)
      - PGA σ (g)
    at the given (lat, lon).
    Raises ShakeMapError on any failure.
    """
    # 1. Discover the latest ShakeMap product
    try:
        endpoint = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params   = {"eventid": event_id, "format": "geojson", "producttype": "shakemap"}
        r        = requests.get(endpoint, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        msg = f"Error fetching shakemap list for {event_id}: {e}"
        logger.error(msg)
        raise ShakeMapError(msg)

    try:
        prods = data["properties"]["products"]["shakemap"]
        latest = max(prods, key=lambda p: p["updateTime"])
        hdf_url = latest["contents"]["download/shake_result.hdf"]["url"]
    except Exception as e:
        msg = f"Could not find a valid shake_result.hdf URL in response: {e}"
        logger.error(msg)
        raise ShakeMapError(msg)

    # 2. Download & open the HDF5
    try:
        hdf_bytes = requests.get(hdf_url, timeout=20).content
        bio = io.BytesIO(hdf_bytes)
    except Exception as e:
        msg = f"Error downloading HDF5 from {hdf_url}: {e}"
        logger.error(msg)
        raise ShakeMapError(msg)

    try:
        with h5py.File(bio, "r") as f:
            # 3. Locate the PGA datasets
            if "arrays/imts/GREATER_OF_TWO_HORIZONTAL/PGA/mean" not in f:
                raise KeyError("mean array not found")
            if "arrays/imts/GREATER_OF_TWO_HORIZONTAL/PGA/std" not in f:
                raise KeyError("std array not found")

            ds_mean = f["arrays/imts/GREATER_OF_TWO_HORIZONTAL/PGA/mean"]
            ds_std  = f["arrays/imts/GREATER_OF_TWO_HORIZONTAL/PGA/std"]

            # 4. Read grid attributes
            attrs = ds_mean.attrs
            try:
                nx, ny = int(attrs["nx"]), int(attrs["ny"])
                xmin   = float(attrs["xmin"])
                ymax   = float(attrs["ymax"])
                dx, dy = float(attrs["dx"]), float(attrs["dy"])
            except KeyError as e:
                raise KeyError(f"Missing grid attribute: {e}")

            # 5. Units
            units_mean = ds_mean.attrs.get("units", "")
            units_std  = ds_std.attrs.get("units", "")

            # 6. Compute column/row
            col = int(round((lon - xmin) / dx))
            row = int(round((ymax - lat) / dy))
            if not (0 <= col < nx and 0 <= row < ny):
                raise IndexError(f"Computed (row,col)=({row},{col}) out of bounds")

            # 7. Extract raw values
            pga_raw       = ds_mean[row, col]
            pga_sigma_raw = ds_std[row, col]

            # 8. Convert mean from ln-space if needed
            if "ln" in units_mean.lower():
                pga = float(np.exp(pga_raw))
            else:
                pga = float(pga_raw)

            # 9. Convert σ from ln-space if needed
            if "ln" in units_std.lower():
                pga_sigma = float(pga * pga_sigma_raw)
            else:
                pga_sigma = float(pga_sigma_raw)

            return pga, pga_sigma

    except (OSError, KeyError, IndexError, TypeError) as e:
        msg = f"Error reading/processing HDF5: {e}"
        logger.error(msg)
        raise ShakeMapError(msg)

def get_shakemap_df(event_id: str,
                    locations,
                    imts=('PGA', 'PGV', 'MMI')):
    """
    Fetch the ShakeMap HDF for `event_id`, then for each (lat, lon) in `locations`
    extract the median and standard‐deviation for each IMT in `imts` (default PGA, PGV, MMI).
    Returns a DataFrame with columns:
      lat, lon,
      pga_mean, pga_std,
      pgv_mean, pgv_std,
      mmi_mean, mmi_std
    Raises ShakeMapError on any failure.
    """
    endpoint = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params   = {"eventid": event_id, "format": "geojson", "producttype": "shakemap"}
    try:
        r    = requests.get(endpoint, params=params, timeout=10)
        r.raise_for_status()
        prods = r.json()["properties"]["products"]["shakemap"]
        latest = max(prods, key=lambda p: p["updateTime"])
        hdf_url = latest["contents"]["download/shake_result.hdf"]["url"]
    except Exception as e:
        raise ShakeMapError(f"Error discovering ShakeMap HDF URL: {e}")

    try:
        bio = io.BytesIO(requests.get(hdf_url, timeout=20).content)
    except Exception as e:
        raise ShakeMapError(f"Error downloading HDF5: {e}")

    try:
        with h5py.File(bio, "r") as f:
            base0 = f"/arrays/imts/GREATER_OF_TWO_HORIZONTAL/{imts[0]}/mean"
            attrs = f[base0].attrs
            nx, ny = int(attrs["nx"]), int(attrs["ny"])
            xmin, ymax = float(attrs["xmin"]), float(attrs["ymax"])
            dx, dy = float(attrs["dx"]), float(attrs["dy"])

            ds = {}
            for imt in imts:
                grp = f"/arrays/imts/GREATER_OF_TWO_HORIZONTAL/{imt}"
                m = f[f"{grp}/mean"]
                s = f[f"{grp}/std"]
                ds[imt] = {
                    "mean": m,
                    "std": s,
                    "units_mean": m.attrs.get("units", ""),
                    "units_std":  s.attrs.get("units",  "")
                }

            records = []
            for lat, lon in locations:
                col = int(round((lon - xmin) / dx))
                row = int(round((ymax - lat) / dy))
                if not (0 <= col < nx and 0 <= row < ny):
                    raise ShakeMapError(f"Location ({lat},{lon}) out of bounds")

                rec = {"lat": lat, "lon": lon}
                for imt, H in ds.items():
                    raw_m = H["mean"][row, col]
                    raw_s = H["std"][row, col]

                    mean = float(np.exp(raw_m)) if "ln" in H["units_mean"].lower() else float(raw_m)
                    std  = float(mean * raw_s)    if "ln" in H["units_std"].lower()  else float(raw_s)

                    rec[f"{imt.lower()}_mean"] = mean
                    rec[f"{imt.lower()}_std"]  = std

                records.append(rec)

    except (KeyError, OSError, IndexError, TypeError) as e:
        raise ShakeMapError(f"Error reading/processing HDF5: {e}")

    return pd.DataFrame.from_records(records)

if __name__ == "__main__":
    # try:
    #     eid = "nc73821036"
    #     lat, lon = 40.80194, -124.16361
    #     pga, pga_std = get_pga(eid, lat, lon)
    #     print(f"PGA at ({lat}, {lon}) for {eid}: {pga:.3f} g ± {pga_std:.3f} g")
    # except ShakeMapError as e:
    #     print("Failed to retrieve PGA:", e)
        
    try:
        eid = "nc73821036"
        
        sites = [
    (40.80194, -124.16361),   
    (40.70194, -124.16361),   
    (40.60194, -124.16361),   
]
        df = get_shakemap_df(eid, sites)
    except ShakeMapError as e:
        print("Failed to retrieve PGA:", e)
