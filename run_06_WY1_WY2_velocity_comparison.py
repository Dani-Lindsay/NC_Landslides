#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 15:00:39 2025

@author: daniellelindsay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dry season velocity comparison with summary stats and plot.
@author: daniellelindsay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from NC_Landslides_paths import *
import pygmt

vel_min_threshold = 2
vel_multiple = 5
active_threshold = 1

# Load data
df = pd.read_csv("/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/compiled_landslide_data.csv")

# Compute absolute velocities in cm/yr
df["vel_dry1"] = np.abs(df["meta__ts_dry1_vel_myr"] * 100)
df["vel_dry2"] = np.abs(df["meta__ts_dry2_vel_myr"] * 100)

df_lowrate = df[(df["vel_dry1"] < vel_min_threshold) | (df["vel_dry2"] < vel_min_threshold)].copy()

# Filter for significant motion (>2 cm/yr in either year)
df = df[(df["vel_dry1"] > vel_min_threshold) | (df["vel_dry2"] > vel_min_threshold)].copy()

# Compute velocity ratio
df["vel_ratio"] = df["vel_dry2"] / df["vel_dry1"]

# Define categories
def categorize_ratio_group(ratio):
    if ratio < 1/vel_multiple:
        return "Much Slower"
    elif 1/vel_multiple <= ratio < 0.83:
        return "Slower"
    elif 0.83 <= ratio <= 1.2:
        return "Similar"
    elif 1.2 < ratio <= vel_multiple:
        return "Faster"
    elif ratio > vel_multiple:
        return "Much Faster"

df["group"] = df["vel_ratio"].apply(categorize_ratio_group)

# Roma-inspired color map (reversed)
roma_colors = {
    "Much Slower": "#3b4cc0",  # dark blue
    "Slower": "#a6bddb",       # light blue
    "Similar": "#fefcbf",        # yellow
    "Faster": "#f4a582",       # light red
    "Much Faster": "#b2182b",  # dark red
    "Low Rate": "#cccccc"
}

# --------- Summary stats ---------
total = len(df)
group_counts = df["group"].value_counts()

similar = group_counts.get("Similar", 0)
faster = group_counts.get("Faster", 0)
much_faster = group_counts.get("Much Faster", 0)
slower = group_counts.get("Slower", 0)
much_slower = group_counts.get("Much Slower", 0)
uncat = group_counts.get("Low Rate", 0)

# Activity state thresholds (1 cm/yr)
remained_active = df[(df["vel_dry1"] > active_threshold) & (df["vel_dry2"] > 1)].shape[0]
became_active = df[(df["vel_dry1"] <= active_threshold) & (df["vel_dry2"] > 1)].shape[0]
became_inactive = df[(df["vel_dry1"] > active_threshold) & (df["vel_dry2"] <= 1)].shape[0]

# Extreme changes
big_accels = df[df["vel_ratio"] > vel_multiple].shape[0]
big_decels = df[df["vel_ratio"] < 1/vel_multiple].shape[0]

# Most extreme cases
max_accel = df.loc[df["vel_ratio"].idxmax()]
max_decel = df.loc[df["vel_ratio"].idxmin()]

# --------- Print Summary ---------
print(f"\nTotal landslides analyzed: {total} greater than {vel_min_threshold} cm/yr\n")

print(f"{similar} landslides ({similar/total:.1%}) maintained similar speeds (±20%) between WY22 and WY23.")
print(f"{faster} landslides ({faster/total:.1%}) became faster in WY23.")
print(f"{much_faster} landslides ({much_faster/total:.1%}) became much faster (>3×) in WY23.")
print(f"{slower} landslides ({slower/total:.1%}) became slower in WY23.")
print(f"{much_slower} landslides ({much_slower/total:.1%}) became much slower (<1/3×) in WY23.")
if uncat:
    print(f"{uncat} landslides could not be categorized.")


print(f"{(faster + much_faster) / total:.1%} became faster in WY23")
print(f"{(slower + much_slower) / total:.1%} became slower in WY23")

print(f"\n{big_accels} landslides accelerated by more than 3× in WY23.")
print(f"{big_decels} landslides slowed by more than 3× in WY23.")

print(f"\nMost extreme acceleration:\n- WY22 velocity: {max_accel['vel_dry1']:.2f} cm/yr"
      f"\n- WY23 velocity: {max_accel['vel_dry2']:.2f} cm/yr"
      f"\n- Ratio: {max_accel['vel_ratio']:.2f}")

print(f"\nMost extreme deceleration:\n- WY22 velocity: {max_decel['vel_dry1']:.2f} cm/yr"
      f"\n- WY23 velocity: {max_decel['vel_dry2']:.2f} cm/yr"
      f"\n- Ratio: {max_decel['vel_ratio']:.2f}")

print(f"\nActivity status:")
print(f"- {remained_active} landslides remained active (>1 cm/yr in both years)")
print(f"- {became_active} landslides became active in WY23")
print(f"- {became_inactive} landslides became inactive in WY23")


# Create the scatter plot
fig, ax = plt.subplots(figsize=(5, 5))

# Add 1:1 line
max_val = 20
ax.plot([0, max_val], [0, max_val], linestyle='--', color='black', linewidth=1, alpha=0.5)

ax.scatter(df_lowrate["vel_dry1"], df_lowrate["vel_dry2"], s=30, color="#cccccc", edgecolor='black', linewidth=0.3)

for group, color in roma_colors.items():
    subset = df[df["group"] == group]
    ax.scatter(subset["vel_dry1"], subset["vel_dry2"],
               label=group, s=30, color=color, edgecolor='black', linewidth=0.3)

# Set equal limits
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_box_aspect(1) 

# Labels and formatting
ax.set_xlabel("Dry Season Velocity WY22 (cm/yr, abs)", fontsize=10)
ax.set_ylabel("Dry Season Velocity WY23 (cm/yr, abs)", fontsize=10)
ax.set_title("Comparison of Dry Season Velocities (Absolute Values)", fontsize=10)
ax.tick_params(labelsize=10)

# Move legend to middle right
ax.legend(fontsize=9, title_fontsize=10, loc='upper right',)

plt.tight_layout()
plt.savefig(f"{fig_dir}/Fig_4_dry_velocity_comparison_x{vel_multiple}_minvel{vel_min_threshold}.png", dpi=300)
plt.savefig(f"{fig_dir}/Fig_4_dry_velocity_comparison_x{vel_multiple}_minvel{vel_min_threshold}.jpeg", dpi=300)
plt.savefig(f"{fig_dir}/Fig_4_dry_velocity_comparison_x{vel_multiple}_minvel{vel_min_threshold}.pdf")
plt.show()



# Set categorical order
group_order = [
    "Much Faster",
    "Faster",
    "Similar",
    "Slower",
    "Much Slower"
]

# Set plot aesthetics
sns.set(style="whitegrid")
plt.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8})

# Variables to plot
top_vars = [
    "support_params/wy23_vs_wy22_rain_ratio",
    "support_params/wy23_vs_wy22_pga_ratio",
    "support_params/wy23_vs_wy22_eq_ratio",
    "meta__ls_mean_slope",
    "meta__ls_axis_ratio",
    "meta__ts_cluster_area_m2"
]

# Transform area to log scale
df["log_area"] = np.log10(df["meta__ts_cluster_area_m2"])

# Rename columns for display
pretty_names = {
    "support_params/wy23_vs_wy22_rain_ratio": "Rainfall WY23/WY22",
    "support_params/wy23_vs_wy22_pga_ratio": "PGA WY23/WY22",
    "support_params/wy23_vs_wy22_eq_ratio": "EQ Count WY23/WY22",
    "meta__ls_mean_slope": "Mean Slope (°)",
    "meta__ls_axis_ratio": "Axis Ratio",
    "log_area": "Log Area (m²)"
}

# Replace original with log-transformed area
plot_vars = top_vars[:-1] + ["log_area"]

# Create the plot grid
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 5), sharex=False)
axes = axes.flatten()

for ax, var in zip(axes, plot_vars):
    sns.violinplot(
        data=df,
        x="group",
        y=var,
        order=group_order,
        palette=roma_colors,
        ax=ax,
        linewidth=1,
        inner="quart",
        scale="width"
    )
    ax.set_title(pretty_names.get(var, var), fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)

subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f']

# Add labels (a)–(f) in the upper-left of each subplot
for ax, lab in zip(axes, subplot_labels):
    ax.text(
        0.02, 1.1, f'{lab})',
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=10
    )
    

plt.tight_layout()
fig.savefig(f"{fig_dir}/Fig_5_violin_plots_x{vel_multiple}_minvel{vel_min_threshold}.png", dpi=300)
fig.savefig(f"{fig_dir}/Fig_5_violin_plots_x{vel_multiple}_minvel{vel_min_threshold}.jpeg", dpi=300)
fig.savefig(f"{fig_dir}/Fig_5_violin_plots_x{vel_multiple}_minvel{vel_min_threshold}.pdf")
plt.show()


# =========================================
#            STATISTICAL TESTS
# =========================================
from scipy import stats
import itertools

rng = np.random.default_rng(42)

# Use same variables you plotted
analysis_vars = plot_vars[:]  # copy

# Helper: print a section header
def _hdr(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

# Helper: bootstrap CI for difference in medians (x - y)
def bootstrap_diff_median(x, y, B=10000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(pd.Series(x).dropna())
    y = np.asarray(pd.Series(y).dropna())
    if x.size == 0 or y.size == 0:
        return np.nan, (np.nan, np.nan)
    diffs = np.empty(B, dtype=float)
    for b in range(B):
        xb = rng.choice(x, size=x.size, replace=True)
        yb = rng.choice(y, size=y.size, replace=True)
        diffs[b] = np.median(xb) - np.median(yb)
    diffs.sort()
    point = np.median(x) - np.median(y)
    lo = diffs[int(0.025 * B)]
    hi = diffs[int(0.975 * B)]
    return float(point), (float(lo), float(hi))

# Helper: pairwise Mann–Whitney with Holm correction
def pairwise_mw_holm(table, groups, var, min_n=5):
    pairs, raw_p = [], []
    for g1, g2 in itertools.combinations(groups, 2):
        x = table.loc[table["group"] == g1, var].dropna()
        y = table.loc[table["group"] == g2, var].dropna()
        if len(x) >= min_n and len(y) >= min_n:
            U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            pairs.append((g1, g2, len(x), len(y), U, p))
            raw_p.append(p)
    # Holm step-down
    m = len(raw_p)
    if m == 0:
        return []
    order = np.argsort(raw_p)          # indices of p in ascending order
    holm_adj = np.empty(m, dtype=float)
    for rank, j in enumerate(order, start=1):
        holm_adj[j] = min((m - rank + 1) * raw_p[j], 1.0)
    results = []
    for (g1, g2, n1, n2, U, p), p_adj in zip(pairs, holm_adj):
        results.append({
            "pair": f"{g1} vs {g2}", "n1": n1, "n2": n2,
            "U": float(U), "p": float(p), "p_holm": float(p_adj)
        })
    return results

# Reporting function for one dataframe (full or filtered)
def run_group_comparisons(df_in, label, fastest_group="Much Faster", slowest_group="Much Slower"):
    _hdr(f"GROUPED COMPARISONS ({label})")
    present_groups = [g for g in group_order if g in df_in["group"].unique()]
    print(f"Groups present (n≥5 considered in tests): {present_groups}")

    for var in analysis_vars:
        print(f"\n--- {var} ---")
        # Per-group medians (and n)
        med = df_in.groupby("group")[var].median().reindex(group_order)
        n = df_in.groupby("group")[var].count().reindex(group_order)
        print("Median by group:")
        for g in group_order:
            if g in med.index and not pd.isna(med.loc[g]):
                print(f"  {g:<12}  median={med.loc[g]:.3g}   n={int(n.loc[g]) if not pd.isna(n.loc[g]) else 0}")

        # Global Kruskal–Wallis across groups with n≥5
        groups_data = [df_in.loc[df_in["group"] == g, var].dropna() for g in present_groups]
        groups_data = [x for x in groups_data if len(x) >= 5]
        if len(groups_data) >= 2:
            H, p_kw = stats.kruskal(*groups_data)
            print(f"Global Kruskal–Wallis: H={H:.3f}, p={p_kw:.3g}")
        else:
            print("Global Kruskal–Wallis: skipped (insufficient n)")

        # Pairwise Mann–Whitney (Holm)
        pw = pairwise_mw_holm(df_in, present_groups, var, min_n=5)
        # Print only a compact subset: (fastest vs slowest) and (Faster vs Slower) if available
        pairs_to_show = []
        if any(r["pair"] == f"{fastest_group} vs {slowest_group}" for r in pw):
            pairs_to_show.append(f"{fastest_group} vs {slowest_group}")
        if "Faster" in present_groups and "Slower" in present_groups:
            pairs_to_show.append("Faster vs Slower")

        if pairs_to_show:
            print("Key pairwise (Mann–Whitney, Holm-corrected):")
            for psel in pairs_to_show:
                for r in pw:
                    if r["pair"] == psel:
                        print(f"  {r['pair']:<24} U={r['U']:.1f}  n=({r['n1']},{r['n2']})  p={r['p']:.3g}  p_holm={r['p_holm']:.3g}")

        # Bootstrap 95% CI for difference in medians (fastest - slowest)
        if (fastest_group in present_groups) and (slowest_group in present_groups):
            x = df_in.loc[df_in["group"] == fastest_group, var].dropna()
            y = df_in.loc[df_in["group"] == slowest_group, var].dropna()
            if len(x) >= 5 and len(y) >= 5:
                point, (lo, hi) = bootstrap_diff_median(x, y, B=10000, seed=123)
                print(f"Δ median ({fastest_group} − {slowest_group}) = {point:.3g}  [95% CI {lo:.3g}, {hi:.3g}]")
            else:
                print(f"Δ median bootstrap: skipped (insufficient n in {fastest_group} or {slowest_group})")

    # Optional: monotonic association with velocity ratio (descriptive)
    print("\nSpearman correlation vs vel_ratio (descriptive):")
    for var in analysis_vars:
        x = df_in[var]
        y = df_in["vel_ratio"]
        mask = x.notna() & y.notna()
        if mask.sum() >= 10:
            rho, p = stats.spearmanr(x[mask], y[mask])
            print(f"  {var:<35}  ρ={rho:.3f}  p={p:.3g}  (n={mask.sum()})")
        else:
            print(f"  {var:<35}  ρ: skipped (n<10)")

# ---------------------------------------------------------
# 1) ANALYSIS INCLUDING ALL GROUPS
run_group_comparisons(df, label="ALL GROUPS (includes 'Much Slower')",
                      fastest_group="Much Faster", slowest_group="Much Slower")

# 2) ANALYSIS EXCLUDING 'MUCH SLOWER' (shutdown group)
df_excl = df[df["group"] != "Much Slower"].copy()
run_group_comparisons(df_excl, label="EXCLUDING 'Much Slower'",
                      fastest_group="Much Faster", slowest_group="Slower")
# =========================================

# =========================================
# SAVE STATISTICS TO CSVs (parameterized filenames)
# =========================================
import os
from datetime import datetime

def _slugify(s):
    return s.lower().replace(" ", "_").replace("'", "").replace("-", "_")

def compile_and_save_stats(df_in, label, fastest_group, slowest_group):
    label_slug = _slugify(label)
    ts = datetime.now().strftime("%Y%m%d")

    # Output directory (under fig_dir for convenience)
    stats_dir = os.path.join(fig_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    # Containers
    med_rows = []
    kw_rows = []
    pw_rows = []
    boot_rows = []
    sp_rows = []

    # --- Per-variable stats (reuse same logic as printers) ---
    present_groups = [g for g in group_order if g in df_in["group"].unique()]
    for var in analysis_vars:
        # Medians
        med = df_in.groupby("group")[var].median()
        n = df_in.groupby("group")[var].count()
        for g in present_groups:
            med_rows.append({
                "label": label,
                "variable": var,
                "group": g,
                "median": med.get(g, np.nan),
                "n": int(n.get(g, 0)),
                "vel_min_threshold": vel_min_threshold,
                "vel_multiple": vel_multiple,
                "date": ts
            })

        # Global Kruskal–Wallis
        groups_data = [df_in.loc[df_in["group"] == g, var].dropna() for g in present_groups]
        groups_data = [x for x in groups_data if len(x) >= 5]
        if len(groups_data) >= 2:
            H, p_kw = stats.kruskal(*groups_data)
            kw_rows.append({
                "label": label,
                "variable": var,
                "H": float(H),
                "p_value": float(p_kw),
                "k_groups": len(groups_data),
                "vel_min_threshold": vel_min_threshold,
                "vel_multiple": vel_multiple,
                "date": ts
            })

        # Pairwise (Holm) — only keep key pairs
        pw = pairwise_mw_holm(df_in, present_groups, var, min_n=5)
        key_pairs = [
            f"{fastest_group} vs {slowest_group}",
            "Faster vs Slower"
        ]
        for r in pw:
            if r["pair"] in key_pairs:
                pw_rows.append({
                    "label": label,
                    "variable": var,
                    "pair": r["pair"],
                    "U": r["U"],
                    "n1": r["n1"],
                    "n2": r["n2"],
                    "p_value": r["p"],
                    "p_value_holm": r["p_holm"],
                    "vel_min_threshold": vel_min_threshold,
                    "vel_multiple": vel_multiple,
                    "date": ts
                })

        # Bootstrap Δ median (fastest - slowest)
        if (fastest_group in present_groups) and (slowest_group in present_groups):
            x = df_in.loc[df_in["group"] == fastest_group, var].dropna()
            y = df_in.loc[df_in["group"] == slowest_group, var].dropna()
            if len(x) >= 5 and len(y) >= 5:
                point, (lo, hi) = bootstrap_diff_median(x, y, B=10000, seed=123)
                boot_rows.append({
                    "label": label,
                    "variable": var,
                    "delta_median": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "fast_group": fastest_group,
                    "slow_group": slowest_group,
                    "n_fast": int(len(x)),
                    "n_slow": int(len(y)),
                    "vel_min_threshold": vel_min_threshold,
                    "vel_multiple": vel_multiple,
                    "date": ts
                })

        # Spearman vs vel_ratio (descriptive)
        x = df_in[var]
        y = df_in["vel_ratio"]
        mask = x.notna() & y.notna()
        if mask.sum() >= 10:
            rho, p = stats.spearmanr(x[mask], y[mask])
            sp_rows.append({
                "label": label,
                "variable": var,
                "spearman_rho": float(rho),
                "p_value": float(p),
                "n": int(mask.sum()),
                "vel_min_threshold": vel_min_threshold,
                "vel_multiple": vel_multiple,
                "date": ts
            })

    # Make DataFrames
    df_med = pd.DataFrame(med_rows)
    df_kw  = pd.DataFrame(kw_rows)
    df_pw  = pd.DataFrame(pw_rows)
    df_bt  = pd.DataFrame(boot_rows)
    df_sp  = pd.DataFrame(sp_rows)

    # Filenames with params
    base = f"{label_slug}_min{vel_min_threshold}_x{vel_multiple}_{ts}"
    f_med = os.path.join(stats_dir, f"{base}_medians.csv")
    f_kw  = os.path.join(stats_dir, f"{base}_kruskal.csv")
    f_pw  = os.path.join(stats_dir, f"{base}_pairwise_keypairs.csv")
    f_bt  = os.path.join(stats_dir, f"{base}_bootstrap_delta_median.csv")
    f_sp  = os.path.join(stats_dir, f"{base}_spearman_vs_vel_ratio.csv")

    # Write CSVs (empty frames still saved for traceability)
    df_med.to_csv(f_med, index=False)
    df_kw.to_csv(f_kw, index=False)
    df_pw.to_csv(f_pw, index=False)
    df_bt.to_csv(f_bt, index=False)
    df_sp.to_csv(f_sp, index=False)

    print(f"\nSaved stats for {label} to:")
    print(" ", f_med)
    print(" ", f_kw)
    print(" ", f_pw)
    print(" ", f_bt)
    print(" ", f_sp)

# Run saves for both analyses
compile_and_save_stats(
    df, label="ALL GROUPS (includes Much Slower)",
    fastest_group="Much Faster", slowest_group="Much Slower"
)

df_excl = df[df["group"] != "Much Slower"].copy()
compile_and_save_stats(
    df_excl, label="EXCLUDING Much Slower",
    fastest_group="Much Faster", slowest_group="Slower"
)
# =========================================

# =========================================
# SIGNIFICANCE SUMMARY PRINTOUTS (place at end)
# =========================================
from scipy import stats

ALPHA = 0.05

def significance_summary(df_in, label, fastest_group, slowest_group, alpha=ALPHA, min_n=5):
    print("\n" + "="*72)
    print(f"SIGNIFICANCE SUMMARY: {label}")
    print("="*72)
    present_groups = [g for g in group_order if g in df_in["group"].unique()]
    # Keep groups with enough data in at least one variable (approx screen by counts)
    var_counts = {g: int((df_in["group"]==g).sum()) for g in present_groups}
    print("Groups present (count of rows, all vars combined):")
    print("  " + ", ".join([f"{g} (n~{var_counts[g]})" for g in present_groups]))

    sig_kw = []
    sig_pair = []

    for var in analysis_vars:
        # --- Kruskal–Wallis across groups with n>=min_n for this var
        data = []
        ok_groups = []
        for g in present_groups:
            vals = df_in.loc[df_in["group"] == g, var].dropna()
            if len(vals) >= min_n:
                data.append(vals)
                ok_groups.append(g)
        kw_sig = False
        H = np.nan
        p_kw = np.nan
        if len(data) >= 2:
            H, p_kw = stats.kruskal(*data)
            kw_sig = (p_kw < alpha)
            if kw_sig:
                sig_kw.append(var)

        # --- Key pairwise (Holm) using helper defined earlier
        pair_res = pairwise_mw_holm(df_in, present_groups, var, min_n=min_n)
        key_pair = f"{fastest_group} vs {slowest_group}"
        rkey = next((r for r in pair_res if r["pair"] == key_pair), None)

        # Direction via medians
        med_fast = df_in.loc[df_in["group"] == fastest_group, var].median()
        med_slow = df_in.loc[df_in["group"] == slowest_group, var].median()
        direction = "↑" if pd.notna(med_fast) and pd.notna(med_slow) and (med_fast > med_slow) else ("↓" if pd.notna(med_fast) and pd.notna(med_slow) and (med_fast < med_slow) else "—")

        pair_sig = False
        if rkey is not None and not np.isnan(rkey["p_holm"]):
            pair_sig = (rkey["p_holm"] < alpha)
            if pair_sig:
                sig_pair.append(var)

        # Print a compact line for each variable
        var_name = pretty_names.get(var, var)
        print(f"- {var_name}: "
              f"KW p={p_kw:.3g} {'*' if kw_sig else ''} | "
              f"{key_pair} p_holm={rkey['p_holm']:.3g} {'*' if pair_sig else ''} | "
              f"median({fastest_group})={med_fast:.3g}, median({slowest_group})={med_slow:.3g} {direction}")

    # Final concise lists
    print("\nSignificant (Kruskal–Wallis, α=0.05):")
    if sig_kw:
        print("  " + ", ".join([pretty_names.get(v, v) for v in sig_kw]))
    else:
        print("  (none)")

    print("Significant (Key pairwise, Holm-adjusted, α=0.05):")
    if sig_pair:
        print("  " + ", ".join([pretty_names.get(v, v) for v in sig_pair]))
    else:
        print("  (none)")
    print("="*72)

# Run summaries
significance_summary(
    df, label="ALL GROUPS (includes 'Much Slower')",
    fastest_group="Much Faster", slowest_group="Much Slower", alpha=ALPHA
)

df_excl = df[df["group"] != "Much Slower"].copy()
significance_summary(
    df_excl, label="EXCLUDING 'Much Slower'",
    fastest_group="Much Faster", slowest_group="Slower", alpha=ALPHA
)

# =========================================
# MANUSCRIPT SENTENCES FOR ALL VARIABLES
# =========================================
import os, glob
import pandas as pd
import math

def _fmt(x, nd=2):
    if pd.isna(x):
        return "nan"
    # use fixed 2 decimals for readability, fall back to 3 sig figs if very small
    if abs(x) >= 0.01:
        return f"{x:.2f}"
    else:
        return f"{x:.3g}"

def _sig_phrase(lo, hi):
    if pd.isna(lo) or pd.isna(hi):
        return " (insufficient data)"
    return " (significant)" if (lo > 0 or hi < 0) else " (not statistically significant; CI includes 0)"

def print_manuscript_sentences_all_variables(stats_dir, vel_min_threshold, vel_multiple,
                                            analysis_vars, pretty_names):
    # Load all bootstrap CSVs produced by compile_and_save_stats()
    boot_files = glob.glob(os.path.join(stats_dir, "*_bootstrap_delta_median.csv"))
    if not boot_files:
        print("\n[manuscript] No bootstrap CSVs found. Run compile_and_save_stats() first.")
        return
    dfb = pd.concat((pd.read_csv(f) for f in boot_files), ignore_index=True)

    # Keep only rows for current parameters and variables actually analyzed
    dfb = dfb[
        (dfb["vel_min_threshold"] == vel_min_threshold) &
        (dfb["vel_multiple"] == vel_multiple) &
        (dfb["variable"].isin(analysis_vars))
    ].copy()
    if dfb.empty:
        print(f"\n[manuscript] No matching bootstrap rows for min={vel_min_threshold}, x{vel_multiple}.")
        return

    # Keep only latest by (label, variable)
    # 'date' was written as YYYYMMDD string
    dfb["date"] = pd.to_datetime(dfb["date"].astype(str), format="%Y%m%d", errors="coerce")
    dfb.sort_values(["label", "variable", "date"], inplace=True)
    latest = dfb.groupby(["label", "variable"], as_index=False).tail(1)

    label_order = [
        "ALL GROUPS (includes Much Slower)",
        "EXCLUDING Much Slower",
    ]

    for label in label_order:
        sub = latest[latest["label"] == label]
        if sub.empty:
            continue
        print("\n" + "="*72)
        print(f"MANUSCRIPT SENTENCES — {label}")
        print("="*72)

        # Order variables as in your plots
        for var in analysis_vars:
            row = sub[sub["variable"] == var]
            if row.empty:
                continue
            r = row.iloc[0]
            name = pretty_names.get(var, var)

            dm  = r.get("delta_median", float("nan"))
            lo  = r.get("ci_low", float("nan"))
            hi  = r.get("ci_high", float("nan"))
            n_f = int(r.get("n_fast", 0)) if not pd.isna(r.get("n_fast", float("nan"))) else 0
            n_s = int(r.get("n_slow", 0)) if not pd.isna(r.get("n_slow", float("nan"))) else 0
            fast = r.get("fast_group", "fast")
            slow = r.get("slow_group", "slow")

            # Build sentence
            # Example:
            # "The median PGA WY23/WY22 was 1.88 higher in Much Faster than Slower (95% CI [0.68, 2.05]; n=71 vs 83), significant."
            sent = (f"The median {name} was {_fmt(dm)} higher in {fast} than {slow} "
                    f"(95% CI [{_fmt(lo)}, {_fmt(hi)}]; n={n_f} vs {n_s})"
                    f"{_sig_phrase(lo, hi)}.")
            print("- " + sent)

# Where the stats were saved
stats_dir = os.path.join(fig_dir, "stats")
print_manuscript_sentences_all_variables(
    stats_dir=stats_dir,
    vel_min_threshold=vel_min_threshold,
    vel_multiple=vel_multiple,
    analysis_vars=analysis_vars,
    pretty_names=pretty_names
)

# =========================================
# SI GRID SUMMARY (LaTeX tables)
# =========================================
import os, glob, re
import pandas as pd
import numpy as np

def _cell_symbol(dm, lo, hi):
    if pd.isna(dm) or pd.isna(lo) or pd.isna(hi):
        return "—"
    if lo > 0:
        return r"$\uparrow^\ast$"
    if hi < 0:
        return r"$\downarrow^\ast$"
    return "ns"

def _make_pivot_table(df_boot, label, analysis_vars, pretty_names):
    sub = df_boot[df_boot["label"] == label].copy()
    if sub.empty:
        return None

    # Column key like "min2_x5"
    sub["colkey"] = sub.apply(lambda r: f"min{int(r['vel_min_threshold'])}_x{int(r['vel_multiple'])}", axis=1)

    # Keep most recent date per (variable, colkey)
    sub["date"] = pd.to_datetime(sub["date"].astype(str), errors="coerce")
    sub.sort_values(["variable","colkey","date"], inplace=True)
    sub = sub.groupby(["variable","colkey"], as_index=False).tail(1)

    # Build grid
    col_order = sorted(sub["colkey"].unique(),
                       key=lambda s: (int(re.search(r"min(\d+)", s).group(1)),
                                      int(re.search(r"_x(\d+)", s).group(1))))
    row_order = [v for v in analysis_vars if v in sub["variable"].unique()]
    grid = pd.DataFrame(index=[pretty_names.get(v, v) for v in row_order],
                        columns=col_order)
    for _, r in sub.iterrows():
        name = pretty_names.get(r["variable"], r["variable"])
        grid.loc[name, r["colkey"]] = _cell_symbol(r["delta_median"], r["ci_low"], r["ci_high"])

    # Replace missing with en-dash
    grid = grid.fillna("—")
    return grid

def _to_latex_table(grid: pd.DataFrame, title: str, label: str):
    # Escape LaTeX specials in row labels (index)
    idx = grid.index.to_series().str.replace("_", r"\_", regex=False)
    grid2 = grid.copy()
    grid2.index = idx

    # Escape underscores in column headers too (min1_x3 -> min1\_x3)
    cols_escaped = [str(c).replace("_", r"\_") for c in grid2.columns]

    header = (
        r"\begin{table}[p]" "\n"
        r"\centering" "\n"
        r"%\scriptsize" "\n"
        r"\setlength{\tabcolsep}{3pt}" "\n"
        r"\renewcommand{\arraystretch}{0.95}" "\n"
        r"\caption{" + title + r"}" "\n"
        r"\label{" + label + r"}" "\n"
        r"\begin{adjustbox}{max width=\linewidth}" "\n"
        r"\begin{tabular}{l" + "c"*grid2.shape[1] + r"}" "\n"
        r"\toprule" "\n"
    )

    colhdr = "Variable & " + " & ".join(cols_escaped) + r" \\" "\n" r"\midrule" "\n"

    body_lines = []
    for row in grid2.index:
        row_cells = " & ".join(grid2.loc[row].tolist())
        body_lines.append(f"{row} & {row_cells} \\\\")
    body = "\n".join(body_lines) + "\n"

    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{adjustbox}" "\n"
        r"\begin{flushleft}\footnotesize "
        r"Symbols denote direction and significance of the bootstrap $\Delta$-median (fast$-$slow): "
        r"$\uparrow^\ast$ = significant positive (95\% CI $>0$), "
        r"$\downarrow^\ast$ = significant negative (95\% CI $<0$), "
        r"ns = not significant (95\% CI spans 0). "
        r"Columns show parameter settings $\mathrm{min}$ (cm/yr) and $\times$ threshold."
        r"\end{flushleft}" "\n"
        r"\end{table}" "\n"
    )
    return header + colhdr + body + footer

def build_and_save_si_tables(fig_dir, analysis_vars, pretty_names):
    stats_dir = os.path.join(fig_dir, "stats")
    boots = glob.glob(os.path.join(stats_dir, "*_bootstrap_delta_median.csv"))
    if not boots:
        print("[SI] No bootstrap CSVs found.")
        return
    dfb = pd.concat([pd.read_csv(f) for f in boots], ignore_index=True)

    # Two labels
    labels = [
        "EXCLUDING Much Slower",
        "ALL GROUPS (includes Much Slower)",
    ]
    titles = {
        "EXCLUDING Much Slower":
            "Significance grid across parameter settings (excluding 'Much Slower')",
        "ALL GROUPS (includes Much Slower)":
            "Significance grid across parameter settings (including 'Much Slower')",
    }
    latex_names = {
        "EXCLUDING Much Slower": "tab:si_grid_excl_muchslower",
        "ALL GROUPS (includes Much Slower)": "tab:si_grid_incl_muchslower",
    }

    os.makedirs(stats_dir, exist_ok=True)
    for lab in labels:
        grid = _make_pivot_table(dfb, lab, analysis_vars, pretty_names)
        if grid is None or grid.empty:
            print(f"[SI] No data for label: {lab}")
            continue
        latex = _to_latex_table(grid, titles[lab], latex_names[lab])
        fn = "SI_table_excluding_much_slower.tex" if lab.startswith("EXCLUDING") else "SI_table_including_much_slower.tex"
        outpath = os.path.join(stats_dir, fn)
        with open(outpath, "w") as f:
            f.write(latex)
        print(f"[SI] Wrote {outpath}")
        print(latex)

# Run it
build_and_save_si_tables(fig_dir, analysis_vars=analysis_vars, pretty_names=pretty_names)

# =========================================
# SPEARMAN (CONTINUOUS) — manuscript-ready lines (+ optional LaTeX)
# =========================================
from scipy import stats
import numpy as np
import pandas as pd

ALPHA = 0.05

def _p_fmt(p):
    if pd.isna(p):
        return "p=nan"
    if p < 1e-3:
        return "p<0.001"
    return f"p={p:.3f}"

def _dir_phrase(rho, p, alpha):
    if pd.isna(rho) or pd.isna(p):
        return "no estimate (insufficient data)"
    if p < alpha:
        return "increased with" if rho > 0 else "decreased with"
    return "showed no clear monotonic association with"

def _fdr_bh(pvals):
    """Benjamini–Hochberg FDR adjustment; returns array of q-values (same shape)."""
    pvals = np.asarray(pvals, dtype=float)
    m = np.sum(~np.isnan(pvals))
    order = np.argsort(np.nan_to_num(pvals, nan=np.inf))
    q = np.full_like(pvals, np.nan, dtype=float)
    prev = 1.0
    rank = 1
    for idx in order:
        p = pvals[idx]
        if np.isnan(p): 
            continue
        qv = p * m / rank
        prev = min(prev, qv)
        q[idx] = prev
        rank += 1
    return q

def spearman_manuscript_lines(df_in, label, analysis_vars, pretty_names, alpha=ALPHA, use_fdr=False, emit_latex=False):
    print("\n" + "="*72)
    print(f"SPEARMAN MONOTONIC TRENDS — {label}")
    print("="*72)

    rows = []
    for var in analysis_vars:
        x = df_in[var]
        y = df_in["vel_ratio"]
        mask = x.notna() & y.notna()
        n = int(mask.sum())
        if n >= 10:
            rho, p = stats.spearmanr(x[mask], y[mask])
        else:
            rho, p = (np.nan, np.nan)
        rows.append({
            "variable": var,
            "name": pretty_names.get(var, var),
            "rho": rho,
            "p": p,
            "n": n
        })

    res = pd.DataFrame(rows)

    # Optional FDR correction across variables
    if use_fdr:
        res["q"] = _fdr_bh(res["p"].values)
        # use q for significance decision
        sig_col = "q"
        thr_label = "q"
    else:
        res["q"] = np.nan
        sig_col = "p"
        thr_label = "p"

    # Print manuscript-style sentences
    for _, r in res.iterrows():
        name = r["name"]
        rho = r["rho"]
        p = r[sig_col]
        n = r["n"]
        if np.isnan(rho) or np.isnan(p):
            print(f"- {name}: insufficient data (n={n}).")
            continue
        direction_text = _dir_phrase(rho, p, alpha)
        ptext = _p_fmt(p)
        print(f"- Velocity ratio {direction_text} {name} (Spearman ρ={rho:.2f}, {ptext}; n={n}).")

    # Optional LaTeX table
    if emit_latex:
        # Build compact grid with rho and p (or q)
        res_disp = res.copy()
        res_disp["metric"] = res_disp.apply(
            lambda r: fr"{r['rho']:.2f} ({thr_label}={r[sig_col]:.3g})", axis=1
        )
        # Order variables as in plots
        res_disp = res_disp.set_index("name").loc[[pretty_names.get(v, v) for v in analysis_vars]]
        # Assemble LaTeX
        latex = []
        latex.append(r"\begin{table}[p]")
        latex.append(r"\centering")
        latex.append(r"\setlength{\tabcolsep}{6pt}")
        latex.append(r"\renewcommand{\arraystretch}{0.95}")
        latex.append(r"\caption{Spearman correlations between velocity ratio and covariates (" + label.replace("_","\_") + r").}")
        latex.append(r"\label{tab:spearman_" + label.lower().replace(" ","_").replace("(","").replace(")","").replace("'","") + r"}")
        latex.append(r"\begin{tabular}{l c r}")
        latex.append(r"\toprule")
        latex.append(r"Variable & Spearman $\rho$ (" + thr_label + r") & n \\")
        latex.append(r"\midrule")
        for nm, row in res_disp.iterrows():
            latex.append(fr"{nm.replace('_', r'\_')} & {row['metric']} & {int(row['n'])} \\")
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        print("\n".join(latex))

# ---- Run Spearman summaries ----
# Primary: exclude 'Much Slower'
df_excl = df[df["group"] != "Much Slower"].copy()
spearman_manuscript_lines(
    df_excl, label="EXCLUDING Much Slower",
    analysis_vars=analysis_vars,
    pretty_names=pretty_names,
    alpha=ALPHA,
    use_fdr=False,        # set True if you want BH–FDR across variables
    emit_latex=False      # set True to also print a LaTeX table
)

# Optional: including all groups
spearman_manuscript_lines(
    df, label="ALL GROUPS (includes Much Slower)",
    analysis_vars=analysis_vars,
    pretty_names=pretty_names,
    alpha=ALPHA,
    use_fdr=False,
    emit_latex=False
)

# =========================================
# MAIN RESULTS TABLE (min=2, x5) from CSVs
# =========================================
import os, glob
import pandas as pd
import numpy as np

def _fmt_num(x, nd=2):
    if pd.isna(x): return "nan"
    return f"{x:.{nd}f}"

def _arrow_from_sign(x):
    if pd.isna(x): return "—"
    return r"$\uparrow$" if x > 0 else (r"$\downarrow$" if x < 0 else "0")

def _sig_from_ci(lo, hi):
    if pd.isna(lo) or pd.isna(hi): return "no"
    return "yes" if (lo > 0 or hi < 0) else "no"

def _spearman_dir_sig(rho, p, alpha=0.05):
    if pd.isna(rho) or pd.isna(p):
        return "—, no", rho, p
    direc = r"$\uparrow$" if rho > 0 else r"$\downarrow$" if rho < 0 else "0"
    sig = "yes" if p < alpha else "no"
    return f"{direc}, {sig}", rho, p

def build_main_results_table_from_csvs(
    stats_dir,
    vel_min_threshold=2,
    vel_multiple=5,
    label="EXCLUDING Much Slower",
    analysis_vars=None,
    pretty_names=None,
    alpha=0.05
):
    # Load bootstrap delta-median files
    boot_files = glob.glob(os.path.join(stats_dir, "*_bootstrap_delta_median.csv"))
    if not boot_files:
        print("[table] No bootstrap CSVs found in:", stats_dir); return
    dfb = pd.concat([pd.read_csv(f) for f in boot_files], ignore_index=True)

    # Load Spearman files
    sp_files = glob.glob(os.path.join(stats_dir, "*_spearman_vs_vel_ratio.csv"))
    if not sp_files:
        print("[table] No spearman CSVs found in:", stats_dir); return
    dfs = pd.concat([pd.read_csv(f) for f in sp_files], ignore_index=True)

    # Filter to parameters & label
    dfb = dfb[
        (dfb["vel_min_threshold"] == vel_min_threshold) &
        (dfb["vel_multiple"] == vel_multiple) &
        (dfb["label"] == label)
    ].copy()
    dfs = dfs[
        (dfs["vel_min_threshold"] == vel_min_threshold) &
        (dfs["vel_multiple"] == vel_multiple) &
        (dfs["label"] == label)
    ].copy()

    if dfb.empty:
        print(f"[table] No bootstrap rows for label={label}, min={vel_min_threshold}, x{vel_multiple}."); return
    if dfs.empty:
        print(f"[table] No Spearman rows for label={label}, min={vel_min_threshold}, x{vel_multiple}."); return

    # Keep latest date per variable
    if "date" in dfb.columns:
        dfb["date"] = pd.to_datetime(dfb["date"].astype(str), errors="coerce")
        dfb.sort_values(["variable","date"], inplace=True)
        dfb = dfb.groupby("variable", as_index=False).tail(1)
    if "date" in dfs.columns:
        dfs["date"] = pd.to_datetime(dfs["date"].astype(str), errors="coerce")
        dfs.sort_values(["variable","date"], inplace=True)
        dfs = dfs.groupby("variable", as_index=False).tail(1)

    # Build rows in the order of your analysis_vars
    rows = []
    for var in (analysis_vars or sorted(dfb["variable"].unique())):
        nm = pretty_names.get(var, var) if pretty_names else var
        b = dfb[dfb["variable"] == var].iloc[0] if (dfb["variable"] == var).any() else None
        s = dfs[dfs["variable"] == var].iloc[0] if (dfs["variable"] == var).any() else None

        if b is not None:
            dm, lo, hi = b.get("delta_median", np.nan), b.get("ci_low", np.nan), b.get("ci_high", np.nan)
            dm_txt = f"{_fmt_num(dm)} [{_fmt_num(lo)}, {_fmt_num(hi)}]"
            dm_sig = _sig_from_ci(lo, hi)
            dm_dir = _arrow_from_sign(dm)
        else:
            dm_txt, dm_sig, dm_dir = "—", "no", "—"

        if s is not None:
            rho, p = s.get("spearman_rho", np.nan), s.get("p_value", np.nan)
            sp_txt, rho_val, p_val = _spearman_dir_sig(rho, p, alpha=alpha)
            # Also keep a numeric display with rho and p if you want:
            sp_detail = f"{_fmt_num(rho_val)} ({'p<0.001' if p_val < 1e-3 else f'p={_fmt_num(p_val,3)}'})" if not pd.isna(rho_val) else "—"
        else:
            sp_txt, sp_detail = "—, no", "—"

        rows.append({
            "name": nm,
            "dm_txt": dm_txt,
            "dm_sig": dm_sig + f" ({dm_dir})",
            "sp_txt": sp_txt,
            "sp_detail": sp_detail
        })

    # Emit LaTeX
    caption = (r"Summary of contrasts (min $=2$ cm yr$^{-1}$, $\times 5$). Bootstrap rows report the "
               r"difference in medians between \emph{Much Faster} and \emph{Slower} among active slides "
               r"with percentile 95\% CIs; “sig?” indicates whether the CI excludes 0. Spearman rows "
               r"report monotonic association between the continuous velocity ratio (WY23/WY22) and each "
               r"variable; arrows denote direction; “sig?” indicates $p<0.05$.")

    # Table header
    latex = []
    latex.append(r"\begin{table}[p]")
    latex.append(r"\centering")
    latex.append(r"\setlength{\tabcolsep}{6pt}")
    latex.append(r"\renewcommand{\arraystretch}{1.05}")
    latex.append(r"\caption{" + caption + r"}")
    latex.append(r"\label{tab:main_results_min2_x5}")
    latex.append(r"\begin{tabular}{lccc}")
    latex.append(r"\toprule")
    latex.append(r"Variable & Bootstrap $\Delta$ median [95\% CI] & sig? & Spearman trend (dir., sig?) \\")
    latex.append(r"\midrule")

    for r in rows:
        nm = r["name"].replace("_", r"\_")
        latex.append(fr"{nm} & {r['dm_txt']} & {r['dm_sig']} & {r['sp_txt']} \\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    print("\n".join(latex))

# ---- choose where to read from ----
# if your script saved to fig_dir/stats, use that; otherwise use the uploaded /mnt/data in this session
stats_dir = os.path.join(fig_dir, "stats") if 'fig_dir' in globals() else "/mnt/data"

build_main_results_table_from_csvs(
    stats_dir=stats_dir,
    vel_min_threshold=2,
    vel_multiple=5,
    label="EXCLUDING Much Slower",
    analysis_vars=analysis_vars,      # uses your existing list to control row order
    pretty_names=pretty_names,
    alpha=0.05
)
