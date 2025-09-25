#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 12:26:55 2025

@author: daniellelindsay
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from NC_Landslides_paths import *

# Load the data
df = pd.read_csv('/Volumes/Seagate/NC_Landslides/Data/LS_Final_TS_4/final_selection_only.csv')

# Select and clean the variables
vars = [ 'ls_mean_aspect', 'ls_mean_slope', 'ts_linear_vel_myr', 'ls_area_m2', 'axis_ratio', ]
df_sub = df[vars].replace([np.inf, -np.inf], np.nan).dropna()

# Create the pairplot
sns.pairplot(df_sub)
plt.suptitle('Landslide Inventory Variables', y=1.02)
plt.tight_layout()
plt.savefig(f'{fig_dir}/Figure_S1_Landslide_Inventory_Scatter.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{fig_dir}/Figure_S1_Landslide_Inventory_Scatter.jpeg', dpi=300, bbox_inches='tight')
plt.savefig(f'{fig_dir}/Figure_S1_Landslide_Inventory_Scatter.pdf', bbox_inches='tight')
plt.show()

