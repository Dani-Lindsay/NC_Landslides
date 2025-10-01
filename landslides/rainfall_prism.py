import pandas as pd
from pvlib.iotools import get_acis_prism

def prism_cumulative_rainfall(latitude, longitude, timestamps):
    ts = sorted(pd.to_datetime(timestamps))

    results = []
    for start, end in zip(ts[:-1], ts[1:]):
        df, meta = get_acis_prism(latitude, longitude, start, end)
        total_rain = df['precipitation'].sum()
        results.append({
            'start':   start.date(),
            'end':     end.date(),
            'rain_mm': total_rain
        })

    return pd.DataFrame(results)

def prism_daily_rainfall(latitude, longitude, starttime, endtime):
    start = pd.to_datetime(starttime).normalize()
    end   = pd.to_datetime(endtime).normalize()

    df_all, meta = get_acis_prism(latitude, longitude, start, end + pd.Timedelta(days=1))
    
    if not isinstance(df_all.index, pd.DatetimeIndex):
        if "date" in df_all.columns:
            df_all.index = pd.to_datetime(df_all["date"])
        elif "datetime" in df_all.columns:
            df_all.index = pd.to_datetime(df_all["datetime"])
        else:
            raise ValueError(
                "Returned DataFrame must have a DatetimeIndex or a 'date'/'datetime' column."
            )
    
    df_all = df_all.copy()
    df_all["calendar_date"] = df_all.index.date
    
    daily_sums = (
        df_all
        .groupby("calendar_date")["precipitation"]
        .sum()
    )  
    full_dates = pd.date_range(start, end, freq="D").date
    full_index = pd.Index(full_dates, name="calendar_date")

    daily_sums = daily_sums.reindex(full_index, fill_value=0.0)
    
    out_df = daily_sums.reset_index().rename(columns={
        "calendar_date": "date",
        "precipitation": "rain_mm"
    })
    out_df.columns = ["date", "rain_mm"]
    
    return out_df

if __name__ == "__main__":
    loc_lat, loc_lon = 39.4096, -123.3556
    dates = ["2023-05-01", "2024-05-15", "2025-05-29"]
    #rain_df = prism_cumulative_rainfall(loc_lat, loc_lon, dates)
    
    daily_rain_df = prism_daily_rainfall(loc_lat, loc_lon, "1990-10-01", "2024-10-01")