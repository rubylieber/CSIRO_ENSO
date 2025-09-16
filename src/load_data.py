# New functions to correctly load monthly ACCESS-S2 seasonal rainfall ensemble for a specified start date and lead time

import xarray as xr
from pathlib import Path
import datetime as dt
import pandas as pd

CALIBRATED_PR_DIR = Path('/g/data/ux62/access-s2/hindcast/calibrated/atmos/pr/monthly/')

def generate_lag_dates(month_day_string):
    """
    For a given start date expressed as a month-day-string (e.g. 0901), 
    generate a list of timestamp objects across the entire hindcast period 1981-2018 for the nine-day lagged ensemble valid from this start date
    """

    years = range(1981,2019)

    dates = []
    for year in years:
        start_date = dt.datetime.strptime(str(year)+month_day_string,'%Y%m%d')

        date_range = pd.date_range(end=start_date,periods=9)

        dates.extend(date_range)

    return dates