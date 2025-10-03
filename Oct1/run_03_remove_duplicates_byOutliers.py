#!/usr/bin/env python3
import os, glob, shutil
import h5py, numpy as np, pandas as pd, matplotlib.pyplot as plt
from NC_Landslides_paths import *

# ─── USER CONFIG ────────────────────────────────────────────────────────────
SRC_DIR   = ts_out_dir  # e.g., "/Volumes/Seagate/NC_Landslides/Data/LS_Timeseries_4"
DEST_DIR  = "/Volumes/Seagate/NC_Landslides/Data_3/LS_Timeseries2" #ts_final_dir  # e.g., "/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4"
os.makedirs(DEST_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

# 1) find all TS files
all_files = sorted(glob.glob(os.path.join(SRC_DIR, "ls_*_box_q*.h5")))
if not all_files:
    raise RuntimeError(f"No files in {SRC_DIR}")

# 2) read ALL /meta attrs into a list of dicts
records = []
for fn in all_files:
    with h5py.File(fn, "r") as hf:
        rec = {"file": fn}
        for k, v in hf["meta"].attrs.items():
            if isinstance(v, bytes):
                rec[k] = v.decode()
            elif hasattr(v, "tolist"):
                v2 = v.tolist()
                # normalize: scalar or 1-element list -> float
                if isinstance(v2, (list, tuple)) and len(v2) == 1:
                    rec[k] = v2[0]
                else:
                    rec[k] = v2
            else:
                rec[k] = v
        # rename slide‐ID
        if "ID" not in rec:
            raise RuntimeError(f"{fn} missing meta/ID")
        rec["ls_id"] = rec.pop("ID")
        records.append(rec)

df = pd.DataFrame.from_records(records)

# sanity check
for col in ("ls_id", "ls_sign", "ts_rmse_clean_m"):
    if col not in df.columns:
        raise RuntimeError(f"Missing required meta column: {col}")

# 3) drop any ls_id with mixed signs
sign_counts = df.groupby("ls_id")["ls_sign"].nunique()
bad = sign_counts[sign_counts > 1].index.tolist()
if bad:
    print(f"Dropping {len(bad)} slides w/ mixed signs:", bad)
df = df[~df["ls_id"].isin(bad)]

# 4) pick the best per ls_id using Error-Aware Outlier Fraction (EAOF)
#    EAOF = fraction of epochs where |residual from rolling-median trend|
#           > 3×max( (err_high-err_low)/2 , 1.4826×MAD(residuals) )
#    Tie-break: smaller median interquartile width (err_high-err_low)

def _eaof_and_iqrmed(h5_path, k=3.0, win=3):
    """
    Returns (eaof, iqrmed) for a single HDF5 time series file.
    - Robust local trend: centered rolling-median with window=win (preserves steps).
    - Residual scale blends per-epoch IQR with global MAD (converted to sigma).
    """
    with h5py.File(h5_path, "r") as hf:
        ts = hf["clean_ts"][:].astype(float)
        el = hf["err_low"][:].astype(float)
        eh = hf["err_high"][:].astype(float)

    # Robust local trend that preserves steps/time-dependence
    s = pd.Series(ts)
    trend = s.rolling(window=win, center=True, min_periods=1).median().to_numpy()

    # Residuals + robust scale (MAD→σ)
    r = ts - trend
    med = np.nanmedian(r)
    mad = np.nanmedian(np.abs(r - med))
    sigma_mad = 1.4826 * mad if np.isfinite(mad) and mad > 0 else 0.0

    # Epoch-wise tolerance blends formal IQR and robust scatter
    iqr_half = (eh - el) / 2.0
    tau = np.maximum(iqr_half, sigma_mad)

    # Outliers (ignore epochs with NaNs or zero/neg tau)
    mask = np.isfinite(r) & np.isfinite(tau) & (tau > 0)
    if not np.any(mask):
        return 1.0, np.inf  # worst if unusable

    out = np.abs(r[mask]) > (k * tau[mask])
    eaof = float(np.mean(out.astype(float)))
    iqrmed = float(np.nanmedian((eh - el)[mask]))
    return eaof, iqrmed

# Compute metrics for each candidate file
eaof_list, iqrmed_list = [], []
for fn in df["file"]:
    eaof, iqrmed = _eaof_and_iqrmed(fn)
    eaof_list.append(eaof)
    iqrmed_list.append(iqrmed)

df["eaof"] = eaof_list
df["iqrmed"] = iqrmed_list

# Select the best: lowest EAOF, then lowest median IQR
best = (
    df.sort_values(["eaof", "iqrmed"], ascending=[True, True])
      .groupby("ls_id", as_index=False)
      .first()  # keeps all meta cols + 'file'
)

selected_ids = set(best["ls_id"])
print("Final ls_id selected:", sorted(selected_ids))



# 5) read your inventory CSV & merge in best-meta columns + backfill from processing summaries
inv = pd.read_csv(common_paths['ls_inventory'], dtype={"ls_id": str})

# ---- helpers ----
def _load_processing_summaries(summary_dir: str) -> pd.DataFrame:
    """Load and concatenate all processing_summary_Timeseries CSVs from summary_dir.
       Also derives series version (Timeseries_1 vs _2) and base filename.
    """
    pats = [
        os.path.join(summary_dir, "processing_summary_Timeseries_*_y*_x*_box*.csv"),
        os.path.join(summary_dir, "processing_summary_Timeseries_*_y*_x*.csv"),
        os.path.join(summary_dir, "processing_summary_Timeseries_*.csv"),
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        print("⚠️ No processing_summary_Timeseries CSVs found — skipping backfill.")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            tmp = pd.read_csv(f)
            tmp["__source_csv__"] = os.path.basename(f)
            # derive Timeseries version from filename (e.g., Timeseries_2)
            fname = os.path.basename(f)
            ver = None
            if "Timeseries_" in fname:
                try:
                    ver = int(fname.split("Timeseries_")[1].split("_")[0])
                except Exception:
                    ver = None
            tmp["__series_ver__"] = ver
            tmp["__is_ts2__"] = (tmp["__series_ver__"] == 2)
            dfs.append(tmp)
        except Exception as e:
            print(f"⚠️ Could not read {f}: {e}")
    if not dfs:
        return pd.DataFrame()

    ps = pd.concat(dfs, ignore_index=True)

    # normalize ls_id
    if "ls_id" not in ps.columns:
        for alt in ["ID", "slide_id", "SlideID"]:
            if alt in ps.columns:
                ps = ps.rename(columns={alt: "ls_id"})
                break
    if "ls_id" in ps.columns:
        ps["ls_id"] = ps["ls_id"].astype(str)

    # find a column that carries the H5 path
    file_col = None
    for c in ["file", "h5_path", "path", "h5", "ts_file"]:
        if c in ps.columns:
            file_col = c
            break
    if file_col is not None:
        ps["__file_base__"] = ps[file_col].astype(str).apply(
            lambda s: os.path.basename(s) if isinstance(s, str) else s
        )
    else:
        ps["__file_base__"] = np.nan

    # align RMSE column name if needed
    if "ts_rmse_clean_m" not in ps.columns:
        for c in ["rmse_clean_m", "rmse_m", "ts_rmse_m"]:
            if c in ps.columns:
                ps = ps.rename(columns={c: "ts_rmse_clean_m"})
                break

    return ps

def _first_float(x):
    """Return the first numeric value from x. Handles '(0.02,)', '[0.02]', arrays, etc."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return np.nan
    if isinstance(x, (list, tuple, np.ndarray)):
        return _first_float(x[0] if len(x) else np.nan)
    s = str(x).strip()
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
        s = s[1:-1]
    if "," in s:
        s = s.split(",", 1)[0]
    try:
        return float(s)
    except Exception:
        return pd.to_numeric(s, errors="coerce")

def _pick_ps_row(group: pd.DataFrame, selected_base: str) -> pd.Series:
    """Choose one processing-summary row per ls_id by priority:
       Prefer Timeseries_2:
         (1) TS2 & filename match to selected H5, else
         (2) TS2 & minimum ts_rmse_clean_m, else
       Otherwise:
         (3) any & filename match, else
         (4) any & minimum ts_rmse_clean_m, else
         (5) first available row.
    """
    def pick_min_rmse(g):
        if "ts_rmse_clean_m" in g.columns:
            rmse = g["ts_rmse_clean_m"].apply(_first_float)
            if rmse.notna().any():
                return g.loc[rmse.idxmin()]
        return None

    # TS2 subset if present
    ts2 = g_ts2 = group[group.get("__is_ts2__", pd.Series([], dtype=bool)) == True]
    if len(ts2) > 0:
        if selected_base and "__file_base__" in ts2.columns:
            m = ts2["__file_base__"].astype(str) == str(selected_base)
            if m.any():
                return ts2[m].iloc[0]
        best_ts2 = pick_min_rmse(ts2)
        if best_ts2 is not None:
            return best_ts2

    # fall back to full group
    if selected_base and "__file_base__" in group.columns:
        m = group["__file_base__"].astype(str) == str(selected_base)
        if m.any():
            return group[m].iloc[0]
    best_any = pick_min_rmse(group)
    if best_any is not None:
        return best_any

    return group.iloc[0]

# ---- merge best-meta and prep for backfill ----
inv = inv.merge(best.drop(columns=["file"]), on="ls_id", how="left")
sel_map = best.set_index("ls_id")["file"].map(lambda p: os.path.basename(p))
inv["selected"] = inv["ls_id"].isin(set(best["ls_id"]))
inv["selected_file"] = inv["ls_id"].map(lambda L: sel_map.get(L, ""))

processing_summaries = _load_processing_summaries(SRC_DIR)

# choose ONE PS row per ls_id (preferring Timeseries_2) and merge
if not processing_summaries.empty and "ls_id" in processing_summaries.columns:
    picked_rows = []
    for ls, g in processing_summaries.groupby("ls_id", sort=False):
        sel_base = inv.loc[inv["ls_id"] == ls, "selected_file"]
        sel_base = sel_base.iloc[0] if len(sel_base) else ""
        picked_rows.append(_pick_ps_row(g, sel_base))
    ps_picked = pd.DataFrame(picked_rows).reset_index(drop=True)

    # --- Build a TS2-only reject_reason map (join unique reasons) ---
    reject_ts2_map = {}
    if "reject_reason" in processing_summaries.columns:
        for ls, g in processing_summaries.groupby("ls_id", sort=False):
            g2 = g[g.get("__is_ts2__", pd.Series([], dtype=bool)) == True]
            if len(g2) and "reject_reason" in g2.columns:
                vals = (
                    g2["reject_reason"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                vals = [v for v in vals if len(v) > 0]
                if len(vals):
                    # unique, order-preserving
                    seen, uniq = set(), []
                    for v in vals:
                        if v not in seen:
                            seen.add(v)
                            uniq.append(v)
                    reject_ts2_map[ls] = "; ".join(uniq)

    # keep helpers until after we construct reject_reason fallback
    helper_cols = [c for c in ["__file_base__", "__source_csv__", "__series_ver__", "__is_ts2__"]
                   if c in ps_picked.columns]
    # Determine new vs overlapping columns
    new_cols = [c for c in ps_picked.columns if c not in inv.columns and c != "ls_id"]
    overlap = [c for c in ps_picked.columns if c in inv.columns and c not in ["ls_id"]]

    # (a) add new columns
    if new_cols:
        inv = inv.merge(ps_picked[["ls_id"] + new_cols], on="ls_id", how="left")

    # (b) for overlapping columns, only backfill where inv is missing/empty
    if overlap:
        inv = inv.merge(ps_picked[["ls_id"] + overlap], on="ls_id", how="left", suffixes=("", "__ps"))
        for c in overlap:
            ps_col = f"{c}__ps"
            if ps_col in inv.columns:
                if inv[c].dtype == object:
                    inv[c] = inv[c].where(inv[c].astype(str).str.len() > 0, inv[ps_col])
                else:
                    inv[c] = inv[c].fillna(inv[ps_col])
                inv = inv.drop(columns=[ps_col])

    # --- Preserve/Prefer reject_reason from Timeseries_2 ---
    # If inv has no reject_reason or it's empty, fill from TS2 map; else keep inv's value.
    if "reject_reason" not in inv.columns:
        inv["reject_reason"] = ""
    inv["reject_reason"] = inv.apply(
        lambda row: row["reject_reason"] if isinstance(row["reject_reason"], str) and len(row["reject_reason"].strip()) > 0
        else reject_ts2_map.get(row["ls_id"], row["reject_reason"]),
        axis=1
    )

    # finally drop helper cols if any were added via new_cols merge
    drop_helpers = [c for c in helper_cols if c in inv.columns]
    if drop_helpers:
        inv = inv.drop(columns=drop_helpers, errors="ignore")
else:
    print("ℹ️ Processing summaries missing or lack 'ls_id'; skipping backfill merge.")

# 6) write out full & winners‐only CSVs
out_full = os.path.join(DEST_DIR, "final_selection.csv")
out_only = os.path.join(DEST_DIR, "final_selection_only.csv")
inv.to_csv(out_full, index=False)
inv[inv["selected"]].to_csv(out_only, index=False)
print("→ Wrote", out_full)
print("→ Wrote", out_only)

# 7) copy only the selected "best" time series into DEST_DIR
from pathlib import Path

winners = best[["ls_id", "file"]].drop_duplicates(subset=["ls_id"]).reset_index(drop=True)

copied = []
skipped = []

print(f"\nCopying {len(winners)} selected time series to {DEST_DIR} ...")
for _, row in winners.iterrows():
    src = row["file"]
    lsid = str(row["ls_id"])
    if not isinstance(src, str) or not os.path.isfile(src):
        skipped.append((lsid, src))
        print(f"  ⚠️  Missing file for {lsid}: {src}")
        continue
    dst = os.path.join(DEST_DIR, os.path.basename(src))
    try:
        # ensure parent exists (DEST_DIR was already created above, but safe to guard)
        Path(DEST_DIR).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)  # overwrite if exists; preserves mtime
        copied.append((lsid, dst))
    except Exception as e:
        skipped.append((lsid, src))
        print(f"  ❌  Failed to copy {lsid} -> {dst}: {e!r}")

print(f"\n✓ Copied {len(copied)} files to {DEST_DIR}")
if skipped:
    print(f"⚠️ Skipped {len(skipped)} (see messages above)")

# optional: write a manifest of what was copied
manifest_path = os.path.join(DEST_DIR, "final_timeseries_manifest.csv")
pd.DataFrame(copied, columns=["ls_id", "dest_file"]).to_csv(manifest_path, index=False)
print(f"Manifest written: {manifest_path}")

print("✅ Done.")
