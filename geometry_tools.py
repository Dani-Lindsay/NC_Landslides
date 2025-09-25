import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
from typing import Optional, Tuple
import requests
import pandas as pd


def distance_to_ocean(lat: float, lon: float) -> float:
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    pt_merc = pt.to_crs(epsg=3857).iloc[0]
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except (AttributeError, IOError):
        ne_url = (
            "https://github.com/nvkelso/natural-earth-vector/raw/master/"
            "110m_cultural/ne_110m_admin_0_countries.shp"
        )
        world = gpd.read_file(ne_url)

    land_union = world.geometry.unary_union

    coastline = land_union.boundary

    coast_gs = gpd.GeoSeries([coastline], crs="EPSG:4326")
    coast_merc = coast_gs.to_crs(epsg=3857).iloc[0]

    return pt_merc.distance(coast_merc)


def distance_to_nearest_road(lat: float, lon: float, search_radius: float = 5000) -> Optional[float]:
    center_point = (lat, lon)

    try:
        G = ox.graph_from_point(center_point, dist=search_radius, network_type="drive")
    except ValueError:
        return None
    except Exception:
        return None

    if not G:
        return None

    G_proj = ox.project_graph(G)

    proj_crs = G_proj.graph.get("crs")
    if proj_crs is None:
        try:
            proj_crs = ox.graph_to_gdfs(G_proj, nodes=True, edges=False).crs
        except Exception:
            return None

    point_geo = (
        gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
        .to_crs(proj_crs)
        .iloc[0]
    )
    px, py = point_geo.x, point_geo.y

    try:
        u, v, key = ox.distance.nearest_edges(G_proj, X=px, Y=py)
    except Exception:
        return None

    edge_data = G_proj.edges[u, v, key]
    if "geometry" in edge_data:
        edge_geom = edge_data["geometry"]
    else:
        edge_geom = ox.utils_graph.get_route_edge_geometry(G_proj, [u, v])[0]

    return point_geo.distance(edge_geom)

def get_elevation(lat: float, lon: float) -> Optional[float]:
    url = "https://api.open-elevation.com/api/v1/lookup"
    params = {"locations": f"{lat},{lon}"}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    
    data = resp.json()
    results = data.get("results", [])
    if not results:
        return None
    
    elev = results[0].get("elevation")
    return elev



if __name__ == "__main__":
    # Example: San Francisco coordinates
    lat0, lon0 = 37.7749, -122.4194

    d_ocean = distance_to_ocean(lat0, lon0)
    print(f"Distance to ocean: {d_ocean:.1f} m")

    d_road = distance_to_nearest_road(lat0, lon0, search_radius=300)
    if d_road is None:
        print("No roads found within 3 m.")
    else:
        print(f"Distance to nearest road: {d_road:.1f} m")
        

