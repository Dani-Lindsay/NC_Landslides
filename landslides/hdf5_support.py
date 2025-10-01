#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 12:18:07 2025

@author: andrewmcnab
"""

import os
import json
from datetime import date, datetime
from typing import Any, Union
import h5py
import numpy as np
import pandas as pd

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)

def _is_str(x: Any) -> bool:
    return isinstance(x, str)

def _is_datetime_like(x: Any) -> bool:
    # Supports date, datetime (aware/naive), and pandas timestamp
    return isinstance(x, (date, datetime, pd.Timestamp))

def _to_iso(x: Union[date, datetime, pd.Timestamp]) -> str:
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, datetime):
        # Preserve timezone if present
        return x.isoformat()
    if isinstance(x, date):
        # date -> YYYY-MM-DD
        return x.isoformat()
    raise TypeError(f"Unsupported datetime-like: {type(x)}")

def _from_iso(s: str) -> Union[pd.Timestamp, date]:
    # Try full datetime first, then fallback to date
    ts = pd.to_datetime(s, utc=True, errors="ignore")
    # pd.to_datetime returns Timestamp for datetime-like, str unchanged if parsable? (with errors="ignore")
    if isinstance(ts, pd.Timestamp):
        # If it looks like a pure date (midnight w/o tz info), you can keep it as Timestamp
        return ts
    # If pandas left it as string, try date-only
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return s  # give up, return raw string

def _list_homogeneous_numbers(lst) -> bool:
    return all(_is_number(x) for x in lst)

def _list_datetimes(lst) -> bool:
    return all(_is_datetime_like(x) for x in lst)

def _list_strings(lst) -> bool:
    return all(_is_str(x) for x in lst)

# ---------- HDF5 write (recursive) ----------

def _save_item(h5group: h5py.Group, name: str, obj: Any):
    """
    Save a python object under h5group[name]:
    - dict -> subgroup, recurse
    - list/tuple -> dataset if homogeneous; else JSON blob
    - numpy array -> dataset
    - number/bool/str -> dataset
    - datetime-like -> dataset (string ISO)
    """
    # dict -> subgroup
    if isinstance(obj, dict):
        sub = h5group.create_group(name)
        sub.attrs["__type__"] = "dict"
        for k, v in obj.items():
            _save_item(sub, str(k), v)
        return

    # numpy array -> dataset
    if isinstance(obj, np.ndarray):
        dset = h5group.create_dataset(name, data=obj)
        dset.attrs["__pytype__"] = "ndarray"
        return

    # datetime-like -> ISO string dataset
    if _is_datetime_like(obj):
        iso = _to_iso(obj)
        dt = h5py.string_dtype(encoding="utf-8")
        dset = h5group.create_dataset(name, data=np.array(iso, dtype=object), dtype=dt)
        dset.attrs["__encoding__"] = "datetime-iso"
        return

    # list/tuple
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            # store empty list as JSON to keep type
            dt = h5py.string_dtype(encoding="utf-8")
            dset = h5group.create_dataset(name, data=np.array("[]", dtype=object), dtype=dt)
            dset.attrs["__encoding__"] = "json"
            return

        if _list_homogeneous_numbers(obj):
            # numeric array
            arr = np.asarray(obj, dtype=np.float64)
            h5group.create_dataset(name, data=arr)
            return

        if _list_strings(obj):
            dt = h5py.string_dtype(encoding="utf-8")
            h5group.create_dataset(name, data=np.array(obj, dtype=object), dtype=dt)
            return

        if _list_datetimes(obj):
            iso_list = [_to_iso(x) for x in obj]
            dt = h5py.string_dtype(encoding="utf-8")
            dset = h5group.create_dataset(name, data=np.array(iso_list, dtype=object), dtype=dt)
            dset.attrs["__encoding__"] = "datetime-iso"
            return

        # mixed/complex -> JSON blob
        dt = h5py.string_dtype(encoding="utf-8")
        payload = json.dumps(obj, default=_to_iso)
        dset = h5group.create_dataset(name, data=np.array(payload, dtype=object), dtype=dt)
        dset.attrs["__encoding__"] = "json"
        return

    # scalar number/bool
    if _is_number(obj) or isinstance(obj, bool):
        h5group.create_dataset(name, data=obj)
        return

    # plain string
    if isinstance(obj, str):
        dt = h5py.string_dtype(encoding="utf-8")
        h5group.create_dataset(name, data=np.array(obj, dtype=object), dtype=dt)
        return

    # fallback: JSON-serialize arbitrary object
    dt = h5py.string_dtype(encoding="utf-8")
    dset = h5group.create_dataset(name, data=np.array(json.dumps(obj, default=_to_iso), dtype=object), dtype=dt)
    dset.attrs["__encoding__"] = "json"
    
def save_landslide_supporting(data: dict, out_dir: str):
    """
    Create one HDF5 file per top-level key in `data`.
    Example: data['ls_001'] -> out_dir/ls_001.h5
    """
    os.makedirs(out_dir, exist_ok=True)
    for key, payload in data.items():
        path = os.path.join(out_dir, f"{key}-supporting.h5")
        with h5py.File(path, "w") as f:
            # root group named same as key for clarity
            root = f.create_group(key)
            _save_item(root, "payload", payload)

def _load_item(h5obj: Union[h5py.Group, h5py.Dataset]) -> Any:
    if isinstance(h5obj, h5py.Dataset):
        enc = h5obj.attrs.get("__encoding__", None)

        if enc == "json":
            s = h5obj[()].decode("utf-8") if isinstance(h5obj[()], (bytes, np.bytes_)) else str(h5obj[()])
            return json.loads(s)

        if enc == "datetime-iso":
            arr = h5obj[()]
            # Could be scalar or array of strings
            if isinstance(arr, (bytes, np.bytes_)):
                return _from_iso(arr.decode("utf-8"))
            if isinstance(arr, np.ndarray):
                out = []
                for x in arr:
                    if isinstance(x, (bytes, np.bytes_)):
                        out.append(_from_iso(x.decode("utf-8")))
                    else:
                        out.append(_from_iso(str(x)))
                return out
            # fallback
            return _from_iso(str(arr))

        # No special encoding: return as native numpy/scalar/string
        val = h5obj[()]
        # Convert bytes -> str if string dtype
        if isinstance(val, (bytes, np.bytes_)):
            return val.decode("utf-8")
        # Single-element arrays -> python scalars
        if isinstance(val, np.ndarray) and val.shape == ():
            return val.item()
        # String arrays need decoding
        if isinstance(val, np.ndarray) and val.dtype.kind in ("S", "O", "U"):
            out = []
            for x in val:
                if isinstance(x, (bytes, np.bytes_)):
                    out.append(x.decode("utf-8"))
                else:
                    out.append(str(x))
            return out
        return val

    # Group
    if isinstance(h5obj, h5py.Group):
        # If it was a dict, we expect children with names
        out = {}
        for k in h5obj.keys():
            out[k] = _load_item(h5obj[k])
        # If this group only contains {"payload": ...}, unwrap to that payload for convenience
        if set(out.keys()) == {"payload"}:
            return out["payload"]
        return out

    raise TypeError(f"Unsupported HDF5 object: {type(h5obj)}")

def load_landslide_hdf5(file_path: str) -> dict:
    """
    Read a single landslide HDF5 file produced by save_landslide_supporting_per_key()
    and return the Python dict that was stored at 'payload'.
    """
    with h5py.File(file_path, "r") as f:
        # Expect root group with same name as file stem, but we won't rely on it
        # Find the first group, then load its "payload"
        roots = [k for k, v in f.items() if isinstance(v, h5py.Group)]
        if not roots:
            raise ValueError("No groups found in HDF5")
        # Prefer exact match to stem if present
        base = os.path.splitext(os.path.basename(file_path))[0]
        if base in roots:
            grp = f[base]
        else:
            grp = f[roots[0]]
        return _load_item(grp["payload"])