#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 11:13:55 2025

@author: daniellelindsay
"""

import glob
import os

fig_dir = "/Volumes/Seagate/NC_Landslides/Figures/Sept29"
data_dir = "/Volumes/Seagate/NC_Landslides/Data_2"
inputs_dir = "/Volumes/Seagate/NC_Landslides/Inputs"
ts_dir = "/Volumes/Seagate/NC_Landslides/Timeseries_1"
ls_input_dir = os.path.join(data_dir, "NorCal_Slides")
ts_out_dir = os.path.join(data_dir, "All_LS_Timeseries")
ts_final_dir = os.path.join(data_dir, "LS_Timeseries")
supporting_dir = "/Volumes/Seagate/NC_Landslides/Inputs/landslide_supporting"

# ------------------------
# Common Paths
# ------------------------
common_paths = {
    "ls_inventory": os.path.join(data_dir, "landslide_inventory.csv"),
    "ls_pga_precip": os.path.join(ts_final_dir, "landslides_wy_pga_precip_summary.csv"),
    "ls_complied": os.path.join(ts_final_dir, "final_selection_only.csv"),
    "ls_mapped2support": os.path.join(ts_final_dir, "final_selection_mapped.csv"),
    "ls_disp_pga_precip" : os.path.join(ts_final_dir, "final_selection_with_wy_pga_precip.csv"),
    
    "fault_files": glob.glob(os.path.join(inputs_dir, "qfaults", "*.txt")),
    "box_poly": glob.glob(os.path.join(inputs_dir, "box_polygons", "*.gmt")),
    "wildfire_poly": glob.glob(os.path.join(inputs_dir, "wildfire_perimeters", "*.gmt")),
    "pb_file":  os.path.join(inputs_dir, "transform.gmt"),
    "pb2_file": os.path.join(inputs_dir, "trench.gmt"),
    "pb3_file": os.path.join(inputs_dir, "ridge.gmt"),
    "170_2800_frame": os.path.join(inputs_dir, "Frame_170_2800.txt"),
    "ls_kmls": {
        "Bennett_Active": os.path.join(ls_input_dir, "Bennettetal_2016_Active_earthflows.kml"),
        "Bennett_Debris": os.path.join(ls_input_dir, "Bennettetal_2016_Debris_slides.kml"),
        "Bennett_Dormant": os.path.join(ls_input_dir, "Bennettetal_2016_Dormant_earthflows.kml"),
        "Handwerger_2015": os.path.join(ls_input_dir, "Handwergeretal_2015.kml"),
        "Handwerger_WY2016": os.path.join(ls_input_dir, "Handwergeretal_2019_WY2016landslides.kml"),
        "Handwerger_WY2017": os.path.join(ls_input_dir, "Handwergeretal_2019_WY2017landslides.kml"),
        "Handwerger_WY2018": os.path.join(ls_input_dir, "Handwergeretal_2019_WY2018landslides.kml"),
        "Mackey_2011":     os.path.join(ls_input_dir, "MackeyandRoering2011_inventory.kml"),
        "Xu_2021":         os.path.join(ls_input_dir, "Xu_2021_wcSlides.kml"),
        "Lindsay_Coast":         os.path.join(ls_input_dir, "Coast_LS_DL.kml"),
        "Lindsay_Inland":         os.path.join(ls_input_dir, "InLand_LS_DL.kml"),
        # add more sources here if needed…
    },
    "ls_geojson": {
        "Bennett_Active": os.path.join(ls_input_dir, "Bennettetal_2016_Active_earthflows.geojson"),
        "Bennett_Debris": os.path.join(ls_input_dir, "Bennettetal_2016_Debris_slides.geojson"),
        "Bennett_Dormant": os.path.join(ls_input_dir, "Bennettetal_2016_Dormant_earthflows.geojson"),
        "Handwerger_2015": os.path.join(ls_input_dir, "Handwergeretal_2015.geojson"),
        "Handwerger_WY2016": os.path.join(ls_input_dir, "Handwergeretal_2019_WY2016landslides.geojson"),
        "Handwerger_WY2017": os.path.join(ls_input_dir, "Handwergeretal_2019_WY2017landslides.geojson"),
        "Handwerger_WY2018": os.path.join(ls_input_dir, "Handwergeretal_2019_WY2018landslides.geojson"),
        "Mackey_2011":     os.path.join(ls_input_dir, "MackeyandRoering2011_inventory.geojson"),
        "Xu_2021":         os.path.join(ls_input_dir, "Xu_2021_wcSlides.geojson"),
        "Lindsay_Coast":  os.path.join(ls_input_dir, "Lindsay_Coast.geojson"),
        "Lindsay_Inland": os.path.join(ls_input_dir, "Lindsay_Inland.geojson"),
    },
    "ls_gmt": {
        "all_polygons": os.path.join(ls_input_dir, "all_polygons.txt"),
        "Bennett_Active": os.path.join(ls_input_dir, "Bennettetal_2016_Active_earthflows.txt"),
        "Bennett_Debris": os.path.join(ls_input_dir, "Bennettetal_2016_Debris_slides.txt"),
        "Bennett_Dormant": os.path.join(ls_input_dir, "Bennettetal_2016_Dormant_earthflows.txt"),
        "Handwerger_2015": os.path.join(ls_input_dir, "Handwergeretal_2015.txt"),
        "Handwerger_WY2016": os.path.join(ls_input_dir, "Handwergeretal_2019_WY2016landslides.txt"),
        "Handwerger_WY2017": os.path.join(ls_input_dir, "Handwergeretal_2019_WY2017landslides.txt"),
        "Handwerger_WY2018": os.path.join(ls_input_dir, "Handwergeretal_2019_WY2018landslides.txt"),
        "Mackey_2011":     os.path.join(ls_input_dir, "MackeyandRoering2011_inventory.txt"),
        "Xu_2021":         os.path.join(ls_input_dir, "Xu_2021_wcSlides.txt"),
        "Lindsay_Coast":         os.path.join(ls_input_dir, "Lindsay_Coast.txt"),
        "Lindsay_Inland":         os.path.join(ls_input_dir, "Lindsay_Inland.txt"),
    },
    "handwerger22_sup3": os.path.join(ls_input_dir, "2022gl099499-sup-0003-table si-s02.xlsx"),
    "handwerger22_sup4": os.path.join(ls_input_dir, "2022gl099499-sup-0004-table si-s03.xlsx"),
    #"ls_inventory_stats": os.path.join(data_dir, "landslide_inventory_stats.csv"),
    "track_170_bbox": [(-125.249, 42.389), (-121.018, 41.843), (-121.833, 38.672), (-125.864, 39.22), (-125.249, 42.389)],
}

    
#     "data_dir": data_dir,
#     "fig_dir": fig_dir,
#     "ls_input_dir": ls_input_dir,
#     "pb_file": os.path.join(ls_input_dir, "transform.gmt"),
#     "pb2_file": os.path.join(ls_input_dir, "trench.gmt"),
#     "pb3_file": os.path.join(ls_input_dir, "ridge.gmt"),
#     "canals_file": os.path.join(ls_input_dir, "ca_canals.txt"),
#     "Tehama-Colusa_file": os.path.join(ls_input_dir, "Tehama-Colusa_canal_cumdisp22.txt"),
#     "Artois_file": os.path.join(ls_input_dir, "Artois_canal_cumdisp22.txt"),
#     "Arbuckle_file": os.path.join(ls_input_dir, "Arbuckle_canal_cumdisp22.txt"),
#     "aquifer_file": os.path.join(ls_input_dir, "aquifer_boundaries.txt"),
#     "roads_major": os.path.join(ls_input_dir, "Roads_Major_Highways.txt"),
#     "roads_primary": os.path.join(ls_input_dir, "Roads_Primary_Secondary.txt"),
#     "roads_local": os.path.join(ls_input_dir, "Roads_Residential_Local.txt"),
#     "roads_tertiary": os.path.join(ls_input_dir, "Roads_Tertiary_Unclassified.txt"),
    
#     "fault_files": glob.glob(os.path.join(ls_input_dir, "qfaults", "*.txt")),
#     "frames": {
#         "170_2800": os.path.join(ls_input_dir, "Frame_170_2800.txt"),
#         "170_2850": os.path.join(ls_input_dir, "Frame_170_2850.txt"),
#         "169_2800": os.path.join(ls_input_dir, "Frame_169_2800.txt"),
#         "169_2850": os.path.join(ls_input_dir, "Frame_169_2850.txt"),
#         "068_0800": os.path.join(ls_input_dir, "Frame_068_0800.txt"),
#         "068_0750": os.path.join(ls_input_dir, "Frame_068_0750.txt"),
#         "169_170_overlap": os.path.join(ls_input_dir, "Frame_169_170_overlap.txt"),
#     },
#     "network": {
#         "baseline": {
#             "068": os.path.join(data_dir, "068", "baseline_center.txt"),
#             "169": os.path.join(data_dir, "169", "baseline_center.txt"),
#             "170": os.path.join(data_dir, "170", "baseline_center.txt"),
#         },
#         "coherence": {
#             "068": os.path.join(data_dir, "068", "coherenceSpatialAvg.txt"),
#             "169": os.path.join(data_dir, "169", "coherenceSpatialAvg.txt"),
#             "170": os.path.join(data_dir, "170", "coherenceSpatialAvg.txt"),
#         },
#     },
#     "bbox": {"w": "-124.63", "e": "-119.22", "s": "36.17", "n": "42.41"},
    
#     "dist" : 0.004, #~1km in each direction 0.008 = 1km. 
#     "ref_station" : "CASR",
#     "ref_lat" : 38.43978,
#     "ref_lon" : -122.74691,
#     "lat_step" : str(0.002), #str(0.000925926*2),# track 068 orginal "-0.0012702942", 0.000925926 = ~100m
#     "lon_step" : str(0.002) # str(0.000925926*2), # track 068 orginal "0.001449585", 0.000925926 = ~100m

# }
