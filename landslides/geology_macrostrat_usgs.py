import requests

def get_geology(lat: float, lon: float):
    """
    Retrieve geology at a point (lat, lon). Tries Macrostrat first, then SGMC.
    Returns a dict with keys:
      - source: 'macrostrat' or 'sgmc'
      - name / unit_name
      - lithology
      - additional metadata (map_id, description, rule_id, gDdId)
    Or None if no data found.
    """
    # 1) Macrostrat stratigraphic-​column lookup
    try:
        ms_url = "https://macrostrat.org/api/v2/geologic_units/map"
        ms_resp = requests.get(ms_url, params={"lat": lat, "lng": lon})
        ms_resp.raise_for_status()
        ms_data = ms_resp.json().get("data") or ms_resp.json().get("success", {}).get("data")
        if ms_data:
            top = ms_data[0]
            return {
                "source": "macrostrat",
                "name": top.get("name") or top.get("strat_name"),
                "lithology": top.get("lith"),
                "description": top.get("descrip"),
                "map_id": top.get("map_id"),
            }
    except requests.RequestException:
        pass

    try:
        sgmc_query = "https://ngmdb.usgs.gov/arcgis/rest/services/mapview/sgmcGddIndex/MapServer/0/query"
        params = {
            "f": "json",
            "geometry": f"{lon},{lat}",
            "geometryType": "esriGeometryPoint",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "UNIT_NAME,RuleID,gDdId",
        }
        sgmc_resp = requests.get(sgmc_query, params=params)
        sgmc_resp.raise_for_status()
        feats = sgmc_resp.json().get("features", [])
        if feats:
            attrs = feats[0]["attributes"]
            # fetch coded values for RuleID → lithology
            meta = requests.get(
                "https://ngmdb.usgs.gov/arcgis/rest/services/mapview/sgmcGddIndex/MapServer/0",
                params={"f": "json"},
            ).json()
            rule_fld = next(f for f in meta["fields"] if f["name"] == "RuleID")
            coded = {cv["code"]: cv["name"] for cv in rule_fld["domain"]["codedValues"]}
            return {
                "source": "sgmc",
                "unit_name": attrs.get("UNIT_NAME"),
                "lithology": coded.get(attrs.get("RuleID")),
                "rule_id": attrs.get("RuleID"),
                "gDdId": attrs.get("gDdId"),
            }
    except requests.RequestException:
        pass

    # nothing found
    return None


# ── example ──
if __name__ == "__main__":
    pt = get_geology(40.80194, -124.14061)  
    if pt:
        print(f"{pt['source']} →", pt)
    else:
        print("No geology found at that location.")
