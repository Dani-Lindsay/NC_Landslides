
"""
Spyder-friendly: Detection rate by source

What it does
------------
- Uses columns: "sources" (string; may contain multiple sources per row) and "selected" (TRUE/FALSE).
- Splits multi-source rows (commas, semicolons, pipes, slashes, ampersands) and explodes to per-source rows.
- Coerces "selected" to boolean (TRUE/True/1/yes/y/t => True; FALSE/False/0/no/n/f => False).
- Computes, per source:
    * total candidates
    * # active (selected==True)
    * # not detected (selected==False)
    * active percent (of total with boolean available)
- Prints clean lines and saves two files next to your CSV:
    * <csv>_source_detection_summary.csv
    * <csv>_source_detection_summary.md

How to use
----------
1) Open in Spyder and Run ▶. Choose your CSV in the file picker (or hardcode in choose_csv_path()).
2) Read console output; see saved CSV/MD next to your CSV.

If your column names differ, set SOURCE_COL and SELECTED_COL below.
"""

import os
import re
import numpy as np
import pandas as pd
from NC_Landslides_paths import *

# --------------- Configuration ---------------
SOURCE_COL   = "sources"
SELECTED_COL = "selected"

# --------------- File picker ---------------
try:
    import tkinter as tk
    from tkinter import filedialog
    _HAS_TK = True
except Exception:
    _HAS_TK = False

def choose_csv_path():
    return  os.path.join(ts_final_dir, "final_selection.csv")

# --------------- Helpers ---------------
TRUE_SET  = {"true","t","1","yes","y","on"}
FALSE_SET = {"false","f","0","no","n","off"}

def to_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().lower()
    if s in TRUE_SET:
        return True
    if s in FALSE_SET:
        return False
    # Try numeric
    try:
        v = float(s)
        if v == 0:
            return False
        if v == 1:
            return True
    except Exception:
        pass
    return np.nan

SPLIT_RE = re.compile(r"[;,&/|]+")

def normalize_sources(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    # Split on common delimiters, also allow commas and semicolons together
    parts = SPLIT_RE.split(str(s))
    # Also split commas (keep as fallback to be safe)
    out = []
    for p in parts:
        for q in str(p).split(","):
            name = q.strip()
            if name:
                out.append(name)
    return out

# --------------- Main ---------------
def main():
    csv_path = choose_csv_path()
    if not csv_path or not os.path.exists(csv_path):
        print("No CSV selected or path not found. Edit choose_csv_path() to hardcode your file or re-run and pick one.")
        return

    df = pd.read_csv(csv_path)

    if SOURCE_COL not in df.columns or SELECTED_COL not in df.columns:
        print(f"ERROR: Required columns not found. Looking for '{SOURCE_COL}' and '{SELECTED_COL}'.")
        print("Columns present:", list(df.columns))
        return

    # Coerce selected to boolean/NaN
    sel = df[SELECTED_COL].apply(to_bool)

    # Build exploded frame: one row per (landslide, source)
    source_lists = df[SOURCE_COL].apply(normalize_sources)
    exploded = df.loc[source_lists.index].copy()
    exploded["__sources_list"] = source_lists
    exploded["__selected_bool"] = sel
    exploded = exploded.explode("__sources_list", ignore_index=True)
    exploded = exploded.rename(columns={"__sources_list": "source"})
    exploded = exploded.dropna(subset=["source"])  # drop rows with no source after split

    # Group by source
    grp = exploded.groupby("source", dropna=False)

    rows = []
    for source, g in grp:
        total = len(g)
        # Consider only rows where selected is boolean (not NaN) for active/not_detected counts
        valid = g["__selected_bool"].dropna()
        n_valid = len(valid)
        n_active = int((valid == True).sum())
        n_inactive = int((valid == False).sum())
        pct_active = (100.0 * n_active / n_valid) if n_valid > 0 else np.nan
        rows.append({
            "source": source,
            "total_candidates": total,
            "with_selected_flag": n_valid,
            "active_count": n_active,
            "not_detected_count": n_inactive,
            "active_percent": None if np.isnan(pct_active) else round(pct_active, 1),
        })

    out = pd.DataFrame(rows).sort_values(["active_percent","total_candidates"], ascending=[False, False])

    # Print clean lines
    print("\n=== Detection by Source ===\n")
    
    print(f"Overall success: {int((df[SELECTED_COL].apply(to_bool)==True).sum())}/{len(df)} = {((df[SELECTED_COL].apply(to_bool)==True).sum()/max(len(df),1))*100:.1f}%")


    for _, r in out.iterrows():
        src = r["source"]
        tot = int(r["total_candidates"])
        nval = int(r["with_selected_flag"])
        act = int(r["active_count"])
        inact = int(r["not_detected_count"])
        pct = r["active_percent"]
        pct_str = "N/A" if pd.isna(pct) else f"{pct:.1f}%"
        print(f"{src}: total={tot}, with_selected={nval}, active={act}, not_detected={inact}, active%={pct_str}")

    # Save outputs
    base, _ = os.path.splitext(csv_path)
    csv_out = base + "_source_detection_summary.csv"
    md_out  = base + "_source_detection_summary.md"
    out.to_csv(csv_out, index=False)

    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Detection by Source\n\n")
        for _, r in out.iterrows():
            src = r["source"]
            tot = int(r["total_candidates"])
            nval = int(r["with_selected_flag"])
            act = int(r["active_count"])
            inact = int(r["not_detected_count"])
            pct = r["active_percent"]
            pct_str = "N/A" if pd.isna(pct) else f"{pct:.1f}%"
            f.write(f"- **{src}**: {act}/{nval} detectable ({pct_str}) out of {tot} candidates\n")
        f.write("\n")

    print(f"\nSaved:\n  - {csv_out}\n  - {md_out}\n")

if __name__ == "__main__":
    main()
