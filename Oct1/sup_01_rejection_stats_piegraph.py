#!/usr/bin/env python3
"""
Donut chart of selection outcomes + reject reasons (seaborn-styled).

Inner ring: Selected vs Rejected (overall fraction).
Outer ring: Reject reasons among the rejected. For rows with multiple reasons
            separated by ';', we take the FIRST reason to keep totals consistent.
            (Optionally switch to fractional allocation — see NOTE below.)

Also writes:
- reject_reason_counts.csv (per-reason counts among rejected)
- selection_outcome_counts.csv (selected vs rejected totals)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from NC_Landslides_paths import *

# ─── USER CONFIG (defaults; override via CLI) ────────────────────────────────
DEFAULT_INPUT  = csv_path = os.path.join(ts_final_dir, "final_selection.csv")
DEFAULT_OUTDIR = fig_dir
DEFAULT_TITLE  = "Selection Outcomes & Reject Reasons"
PALETTE_INNER  = "Set2"   # seaborn palette for inner ring
PALETTE_OUTER  = "tab10"  # seaborn palette for outer ring
MIN_FRACTION_FOR_OWN_SLICE = 0.03  # group rare reasons into "Other" (<3%)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Donut chart of selection outcomes + reject reasons.")
    ap.add_argument("--input",  default=DEFAULT_INPUT,  help="Path to final_selection.csv")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Directory to write outputs")
    ap.add_argument("--title",  default=DEFAULT_TITLE,  help="Figure title")
    ap.add_argument("--inner_palette", default=PALETTE_INNER, help="Seaborn palette for inner ring")
    ap.add_argument("--outer_palette", default=PALETTE_OUTER, help="Seaborn palette for outer ring")
    ap.add_argument("--min_frac", type=float, default=MIN_FRACTION_FOR_OWN_SLICE,
                    help="Group reasons with fraction < this into 'Other' (0–1)")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ── Load
    df = pd.read_csv(args.input, dtype={"ls_id": str})

    # Require columns
    for col in ["selected", "reject_reason"]:
        if col not in df.columns:
            raise RuntimeError(f"'{col}' missing from {args.input}")

    # Normalize types/strings
    df["selected"] = df["selected"].astype(bool)
    df["reject_reason"] = df["reject_reason"].astype(str).fillna("").str.strip()

    # ── Inner ring: Selected vs Rejected
    sel_count = int((df["selected"] == True).sum())
    rej_count = int((df["selected"] == False).sum())
    inner_labels = ["Selected", "Rejected"]
    inner_counts = np.array([sel_count, rej_count], dtype=float)
    inner_total = inner_counts.sum()

    # Save outcome table
    out_sel_csv = os.path.join(args.outdir, "selection_outcome_counts.csv")
    pd.DataFrame({"outcome": inner_labels,
                  "count": inner_counts.astype(int),
                  "fraction": inner_counts / max(inner_total, 1.0)}
                 ).to_csv(out_sel_csv, index=False)
    print(f"→ Wrote {out_sel_csv}")

    # ── Outer ring: reasons among the Rejected
    rej = df[df["selected"] == False].copy()
    # Take FIRST reason only to keep totals consistent with #rejected
    # (NOTE: if you prefer fractional allocation across multiple reasons,
    #        comment the 'first reason' block and use the commented
    #        'fractional share' block below.)
    rej["reason_first"] = (
        rej["reject_reason"]
        .fillna("")
        .astype(str)
        .str.split(";")
        .apply(lambda parts: parts[0].strip() if len(parts) else "")
    )

    # Count reasons; drop empties
    reason_counts = (
        rej["reason_first"]
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="count")
    )
    # If no reasons at all, create a single "Unspecified" slice
    if reason_counts.empty and rej_count > 0:
        reason_counts = pd.DataFrame({"reason": ["Unspecified"], "count": [rej_count]})

    # Fractions among rejected
    reason_counts["fraction"] = reason_counts["count"] / max(rej_count, 1)

    # Group rare reasons into "Other"
    if args.min_frac > 0 and not reason_counts.empty:
        big = reason_counts["fraction"] >= args.min_frac
        small = reason_counts[~big]
        if not small.empty:
            other_row = pd.DataFrame({
                "reason":   ["Other"],
                "count":    [small["count"].sum()],
                "fraction": [small["fraction"].sum()]
            })
            reason_counts = pd.concat([reason_counts[big], other_row], ignore_index=True)

    # Sort for stable order
    reason_counts = reason_counts.sort_values("count", ascending=False).reset_index(drop=True)

    # Save reason table
    out_reason_csv = os.path.join(args.outdir, "reject_reason_counts.csv")
    reason_counts.to_csv(out_reason_csv, index=False)
    print(f"→ Wrote {out_reason_csv}")

    # ── Plot (two-ring donut)
    sns.set_context("talk")
    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(9, 9))

    # Inner ring
    inner_colors = sns.color_palette(args.inner_palette, n_colors=len(inner_labels))
    def autopct_inner(pct):
        count = int(round(pct/100.0 * inner_total))
        return f"{pct:.1f}%\n(n={count})"

    wedges_inner, *_ = ax.pie(
        inner_counts / max(inner_total, 1.0),
        radius=1.0,
        labels=inner_labels,
        labeldistance=0.5,
        autopct=autopct_inner,
        startangle=90,
        colors=inner_colors,
        wedgeprops=dict(width=0.30, edgecolor="white"),
        textprops={"fontsize": 12}
    )

    # Outer ring (reasons) — only if there are rejects
    if rej_count > 0 and not reason_counts.empty:
        outer_labels = reason_counts["reason"].tolist()
        outer_fracs  = reason_counts["count"].to_numpy(dtype=float) / float(rej_count)
        outer_colors = sns.color_palette(args.outer_palette, n_colors=len(outer_labels))

        def autopct_outer(pct):
            # fraction relative to rejected only
            count = int(round(pct/100.0 * rej_count))
            return f"{pct:.1f}%\n(n={count})"

        wedges_outer, *_ = ax.pie(
            outer_fracs,
            radius=1.0,
            labels=outer_labels,
            labeldistance=1.1,
            autopct=autopct_outer,
            pctdistance=0.85,
            startangle=90,
            colors=outer_colors,
            wedgeprops=dict(width=0.30, edgecolor="white"),
            textprops={"fontsize": 11}
        )

    ax.set(aspect="equal")
    ax.set_title(args.title, pad=16)

    # Center hole label: overall success rate
    success_rate = 100.0 * sel_count / max(inner_total, 1.0)
    ax.text(0, 0, f"{success_rate:.1f}%\nSelected", ha="center", va="center", fontsize=14)

    out_png = os.path.join(args.outdir, "selection_and_reject_reasons_donut.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"→ Wrote {out_png}")

    # ── NOTE: fractional share option (advanced; comment 'first reason' block above)
    # If you later want to allocate a landslide's weight across all its reasons so
    # that the *sum of outer slices equals #rejected*, you can do:
    #
    # reasons = (
    #     rej["reject_reason"].fillna("").astype(str).str.split(";").apply(lambda L: [r.strip() for r in L if r.strip()])
    # )
    # exploded = reasons.explode().dropna()
    # counts = exploded.groupby(level=0).size()  # per-row number of reasons
    # weights = 1.0 / counts
    # weighted = exploded.to_frame("reason").join(weights.rename("w"), how="left")
    # reason_counts = weighted.groupby("reason")["w"].sum().reset_index(name="count")
    # reason_counts["fraction"] = reason_counts["count"] / max(rej_count, 1)
    # (…then proceed with grouping small categories and plotting like above.)
    #

if __name__ == "__main__":
    main()
