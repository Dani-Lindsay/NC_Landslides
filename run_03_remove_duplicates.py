#!/usr/bin/env python3
import os, glob, shutil
import h5py, numpy as np, pandas as pd, matplotlib.pyplot as plt
from NC_Landslides_paths import *

# ─── USER CONFIG ────────────────────────────────────────────────────────────
SRC_DIR   = ts_out_dir # "/Volumes/Seagate/NC_Landslides/Data/LS_Timeseries_4"
DEST_DIR  = ts_final_dir #"/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4"
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
for col in ("ls_id","ls_sign","ts_rmse_clean_m"):
    if col not in df.columns:
        raise RuntimeError(f"Missing required meta column: {col}")

# 3) drop any ls_id with mixed signs
sign_counts = df.groupby("ls_id")["ls_sign"].nunique()
bad = sign_counts[sign_counts>1].index.tolist()
if bad:
    print(f"Dropping {len(bad)} slides w/ mixed signs:", bad)
df = df[~df["ls_id"].isin(bad)]

# # 4) pick the best (min ts_rmse_clean_m) per ls_id
# best = (
#     df
#     .sort_values("ts_rmse_clean_m", ascending=True)
#     .groupby("ls_id", as_index=False)
#     .first()    # retains *all* meta columns + 'file'
# )
# selected_ids = set(best["ls_id"])
# print("Final ls_id selected:", sorted(selected_ids))
# 4) pick the best per ls_id: RMSE primary, NN secondary within a 0.2 m RMSE window
df['rmse_min'] = df.groupby('ls_id')['ts_rmse_clean_m'].transform('min')

# candidate pool = those within 0.2 m of the best RMSE
candidates = df[df['ts_rmse_clean_m'] <= df['rmse_min'] + 0.1]

# for each ls_id, pick the one with the lowest mean‐NN
best = (
    candidates
    .loc[candidates.groupby('ls_id')['ts_mean_nn']
         .idxmin()]
    .reset_index(drop=True)
)

selected_ids = set(best["ls_id"])
print("Final ls_id selected:", sorted(selected_ids))


# 5) read your inventory‐stats CSV & merge in all best‐meta columns
inv = pd.read_csv(common_paths['ls_inventory'], dtype={"ls_id":str})
inv = inv.merge(
    best.drop(columns=["file"]), 
    on="ls_id", how="left"
)
# mark selection and copy file name
inv["selected"] = inv["ls_id"].isin(selected_ids)
inv["selected_file"] = inv["ls_id"].map(
    lambda L: os.path.basename(best.loc[best.ls_id==L,"file"].iat[0])
    if L in selected_ids else ""
)

# 6) write out full & winners‐only CSVs
out_full  = os.path.join(DEST_DIR, "final_selection.csv")
out_only  = os.path.join(DEST_DIR, "final_selection_only.csv")
inv.to_csv(out_full, index=False)
inv[inv["selected"]].to_csv(out_only, index=False)
print("→ Wrote", out_full)
print("→ Wrote", out_only)

# 7) copy winners + plot duplicates
for ls, grp in df.groupby("ls_id"):
    if len(grp)==1:
        src = grp["file"].iat[0]
        dst = os.path.join(DEST_DIR, os.path.basename(src))
        shutil.copyfile(src, dst)
    else:
        # compare all candidates
        plt.figure()
        for _, row in grp.iterrows():
            with h5py.File(row["file"], "r") as hf:
                d = hf["dates"][:] ; ts = hf["clean_ts"][:]
            plt.plot(d, ts - np.nanmean(ts), alpha=0.6,
                     label=os.path.basename(row["file"]))
        # highlight best
        bf = best[best.ls_id==ls].iloc[0]
        with h5py.File(bf["file"], "r") as hf:
            d = hf["dates"][:] ; ts = hf["clean_ts"][:]
        plt.plot(d, ts - np.nanmean(ts), "k-", lw=2, label="BEST")
        plt.title(f"Slide {ls}")
        plt.legend(fontsize="small")
        png = os.path.join(DEST_DIR, f"{fig_dir}/Duplicates/{ls}_comparison.png")
        plt.savefig(png, dpi=200)
        plt.close()
        # copy best file
        dst = os.path.join(DEST_DIR, os.path.basename(bf["file"]))
        shutil.copyfile(bf["file"], dst)

print("✅ Done.")
