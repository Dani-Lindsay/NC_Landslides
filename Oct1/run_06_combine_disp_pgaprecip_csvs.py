#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 00:46:44 2025

@author: daniellelindsay
"""

#!/usr/bin/env python3
import pandas as pd
from NC_Landslides_paths import *

# ─── USER CONFIG ───────────────────────────────────────────────
# wy_pga_precip_path = common_paths["ls_pga_precip"]
# final_selection_path = common_paths["ls_mapped2support"]
# out_path_inner = common_paths["ls_disp_pga_precip"]
# #out_path_outer = "final_selection_with_wy_pga_precip_all.csv"
# ───────────────────────────────────────────────────────────────
# ─── USER CONFIG ───────────────────────────────────────────────
wy_pga_precip_path = os.path.join(ts_final_dir, "final_selection_only_mapped_pga_precip.csv")
final_selection_path = os.path.join(ts_final_dir, "final_selection_only_mapped.csv")
out_path_inner = os.path.join(ts_final_dir, "final_selection_only_with_pga_precip.csv")
#out_path_outer = "final_selection_with_wy_pga_precip_all.csv"
# ───────────────────────────────────────────────────────────────

# 1) Load both CSVs
wy_pga_precip = pd.read_csv(wy_pga_precip_path)
final_selection = pd.read_csv(final_selection_path)

# 2) Align column names (wy_pga_precip uses 'id', final_selection uses 'ls_id')
if "id" in wy_pga_precip.columns:
    wy_pga_precip = wy_pga_precip.rename(columns={"id": "ls_id"})

# 3) Inner join → only landslides that exist in both
merged_inner = pd.merge(final_selection, wy_pga_precip, on="ls_id", how="inner")
merged_inner.to_csv(out_path_inner, index=False)
print(f"Inner merge saved to {out_path_inner} with shape {merged_inner.shape}")

# 4) Outer join → keep all, fill missing with NaN
merged_outer = pd.merge(final_selection, wy_pga_precip, on="ls_id", how="outer", indicator=True)
#merged_outer.to_csv(out_path_outer, index=False)
#print(f"Outer merge saved to {out_path_outer} with shape {merged_outer.shape}")

# 5) Optional: check what’s missing
missing_in_final = merged_outer.loc[merged_outer["_merge"] == "right_only", "ls_id"].tolist()
missing_in_wy    = merged_outer.loc[merged_outer["_merge"] == "left_only", "ls_id"].tolist()

print("Missing in final_selection:", missing_in_final[:10], "..." if len(missing_in_final) > 10 else "")
print("Missing in wy_pga_precip:", missing_in_wy[:10], "..." if len(missing_in_wy) > 10 else "")

