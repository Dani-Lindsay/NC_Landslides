
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
CSV_PATH      = r"/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/final_selection.csv"   # path to your inventory CSV
SELECTED_ONLY = True                                # analyze only selected==True
OUT_DIR       = r"/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/"                        # where to write outputs
ID_COL        = "ls_id"

# Column names (edit if yours differ)
ASPECT_COL    = "ls_mean_aspect"            # degrees
SIGN_COL      = "ls_sign"                    # observed sign (+1/-1)
VEL_COL       = "ts_linear_vel_myr"         # LOS velocity (m/yr)
BGSTD_COL     = "ts_background_std_my-1"    # background std (m/yr)
CLUSTER_COL   = "ts_cluster_area_m2"        # cluster area (m^2)
NN_COL        = "ts_mean_nn_dist_m"         # mean NN distance (m)
SELECTED_COL  = "selected"

# Expected sign rule parameters
LOW_DEG  = 10.0
HIGH_DEG = 190.0
# ------------------------------------------

def to_bool(s):
    return s.astype(str).str.lower().isin(["true","t","1","yes","y","on"])

def load_data(csv_path, selected_only=True):
    df = pd.read_csv(csv_path)
    if selected_only and (SELECTED_COL in df.columns):
        df = df[to_bool(df[SELECTED_COL])].copy()
    return df

def expected_sign_from_aspect(aspect_series, low=LOW_DEG, high=HIGH_DEG):
    a = pd.to_numeric(aspect_series, errors="coerce") % 360.0
    return np.where((a >= low) & (a <= high), 1, -1), a

def dist_to_bound(a_deg, low=LOW_DEG, high=HIGH_DEG):
    if pd.isna(a_deg):
        return np.nan
    a = float(a_deg) % 360.0
    return min(abs(a-low), abs(a-high))

def make_mismatch_df(df):
    exp, a_norm = expected_sign_from_aspect(df[ASPECT_COL])
    obs = pd.to_numeric(df[SIGN_COL], errors="coerce")
    valid = (~pd.isna(a_norm)) & (~obs.isna())
    exp_s = pd.Series(exp, index=df.index)
    mismatch_mask = valid & (obs.astype(int) != exp_s.astype(int))
    out = df.loc[mismatch_mask].copy()
    out["aspect_norm_deg"] = a_norm[mismatch_mask]
    out["expected_sign"]   = exp_s[mismatch_mask].astype(int)
    out["observed_sign"]   = obs[mismatch_mask].astype(int)
    # Direction label
    out["mismatch_direction"] = np.where(
        (out["expected_sign"] == -1) & (out["observed_sign"] == 1),
        "expected_neg_observed_pos",
        np.where(
            (out["expected_sign"] == 1) & (out["observed_sign"] == -1),
            "expected_pos_observed_neg",
            "other"
        )
    )
    # Boundary distance
    out["dist_to_boundary_deg"] = out["aspect_norm_deg"].apply(dist_to_bound)
    return out, valid.sum()

def series(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)

def abs_series(df, col):
    return series(df, col).abs()

def summarize(label, a, b, units=""):
    a = a.dropna(); b = b.dropna()
    def q(s): return np.percentile(s, [25,50,75]) if len(s)>0 else [np.nan]*3
    a25,a50,a75 = q(a); b25,b50,b75 = q(b)
    return {
        "label": label,
        "mismatch_n": int(len(a)),
        "mismatch_p25": float(a25) if not np.isnan(a25) else None,
        "mismatch_median": float(a50) if not np.isnan(a50) else None,
        "mismatch_p75": float(a75) if not np.isnan(a75) else None,
        "other_n": int(len(b)),
        "other_p25": float(b25) if not np.isnan(b25) else None,
        "other_median": float(b50) if not np.isnan(b50) else None,
        "other_p75": float(b75) if not np.isnan(b75) else None,
        "units": units
    }

def save_boxplot(a, b, ylabel, title, outpath):
    # a = mismatches series, b = others series
    fig = plt.figure()
    plt.boxplot([b.dropna(), a.dropna()], labels=["others", "mismatches"], whis=1.5, showfliers=False)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def save_hist(series_vals, outpath, bins=None, title="", xlabel="", ylabel="count"):
    series_vals = series_vals.dropna()
    fig = plt.figure()
    if bins is None:
        bins = 30
    plt.hist(series_vals, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data(CSV_PATH, SELECTED_ONLY)
    mism, n_valid = make_mismatch_df(df)

    # Split others as selected minus mismatches
    if (ID_COL in df.columns) and (ID_COL in mism.columns):
        others = df[~df[ID_COL].isin(mism[ID_COL])].copy()
    else:
        others = df.loc[~df.index.isin(mism.index)].copy()

    # Save mismatches
    mism_csv = os.path.join(OUT_DIR, "sign_mismatches.csv")
    mism.to_csv(mism_csv, index=False)

    # Compute metrics
    v_mism, v_other = abs_series(mism, VEL_COL), abs_series(others, VEL_COL)
    bg_mism, bg_other = series(mism, BGSTD_COL), series(others, BGSTD_COL)
    c_mism, c_other = series(mism, CLUSTER_COL), series(others, CLUSTER_COL)
    nn_mism, nn_other = series(mism, NN_COL), series(others, NN_COL)

    # Summary JSON/CSV
    summaries = []
    summaries.append(summarize("|velocity| (m/yr)", v_mism, v_other, "m/yr"))
    summaries.append(summarize("background std (m/yr)", bg_mism, bg_other, "m/yr"))
    summaries.append(summarize("cluster area (m^2)", c_mism, c_other, "m^2"))
    summaries.append(summarize("mean NN distance (m)", nn_mism, nn_other, "m"))
    sum_df = pd.DataFrame(summaries)
    sum_csv = os.path.join(OUT_DIR, "sign_mismatch_summary_stats.csv")
    sum_df.to_csv(sum_csv, index=False)

    # Print a concise summary
    n_mismatch = len(mism)
    pct_mismatch = (100.0 * n_mismatch / n_valid) if n_valid > 0 else np.nan
    near_5 = int((mism["dist_to_boundary_deg"] <= 5).sum())
    near_10 = int((mism["dist_to_boundary_deg"] <= 10).sum())
    near_20 = int((mism["dist_to_boundary_deg"] <= 20).sum())
    print("=== Sign mismatch analysis ===")
    print(f"Valid for test: {n_valid}")
    print(f"Mismatches: {n_mismatch} ({pct_mismatch:.1f}%)")
    print(f"Near boundaries: within 5° = {near_5}/{n_mismatch}, within 10° = {near_10}/{n_mismatch}, within 20° = {near_20}/{n_mismatch}")
    print("\nSummary stats (mismatch vs others):")
    print(sum_df.to_string(index=False))

    # Separate boxplots
    save_boxplot(v_mism,  v_other,  "|velocity| (m/yr)",      "Absolute velocity",           os.path.join(OUT_DIR, "box_velocity_abs.pdf"))
    save_boxplot(bg_mism, bg_other, "background std (m/yr)",  "Background variability",      os.path.join(OUT_DIR, "box_background_std.pdf"))
    save_boxplot(c_mism,  c_other,  "cluster area (m^2)",     "Kinematic cluster area",      os.path.join(OUT_DIR, "box_cluster_area.pdf"))
    save_boxplot(nn_mism, nn_other, "mean NN distance (m)",   "Mean nearest-neighbor dist.", os.path.join(OUT_DIR, "box_nn_distance.pdf"))

    # Histogram of distances to sign-change boundaries (mismatches only)
    dists = pd.to_numeric(mism["dist_to_boundary_deg"], errors="coerce").dropna()
    if len(dists) > 0:
        # bins every 5 degrees up to 90
        bins = np.arange(0, 91, 5)
        save_hist(dists, os.path.join(OUT_DIR, "hist_boundary_distance.pdf"),
                  bins=bins,
                  title="Distance to 10°/190° (mismatches only)",
                  xlabel="degrees")

    # Write a tiny README for the outputs
    readme = os.path.join(OUT_DIR, "sign_mismatch_README.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write("Outputs:\n")
        f.write("  - sign_mismatches.csv : rows where observed sign != expected (with aspect_norm_deg, expected/observed sign, dist_to_boundary)\n")
        f.write("  - sign_mismatch_summary_stats.csv : quartiles for each metric (mismatches vs others)\n")
        f.write("  - box_velocity_abs.pdf : boxplot of |velocity| for mismatches vs others\n")
        f.write("  - box_background_std.pdf : boxplot of background std for mismatches vs others\n")
        f.write("  - box_cluster_area.pdf : boxplot of cluster area for mismatches vs others\n")
        f.write("  - box_nn_distance.pdf : boxplot of mean NN distance for mismatches vs others\n")
        f.write("  - hist_boundary_distance.pdf : histogram of mismatch distance to 10°/190°\n")
        f.write("\nNotes:\n")
        f.write("  - Expected sign: +1 if 10° <= aspect <= 190°, else -1.\n")
        f.write("  - SELECTED_ONLY=True limits analysis to rows marked selected==True in the input CSV.\n")

if __name__ == "__main__":
    main()
