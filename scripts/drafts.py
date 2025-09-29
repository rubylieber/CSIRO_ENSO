import xarray as xr
import datetime as dt
import pandas as pd
import itertools
import pytest
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.resolve())

from config import *
from src import logger

from src.load_data import generate_lag_dates, generate_file_list, generate_lag_dates_year

LOG = logger.get_logger(__name__)

month_day_string='0901'
ens='e01'

year='2000'

dates = generate_lag_dates_year(month_day_string,year)
ens_list = ['e01','e02','e03']
a=list(itertools.product(ens_list,dates[::-1]))

patterns = [f"{a[0]}/maq5_pr_{a[1].strftime('%Y%m%d')}_{a[0]}.nc" for a in a ]

matched = list(
    itertools.chain.from_iterable(
        CALIBRATED_PR_DIR.glob(pattern) for pattern in patterns
    )
)

data = []

for i,filename in enumerate(matched[:5]):
    ds = xr.open_dataset(filename).pr


    # Follow this example
    # https://docs.xarray.dev/en/stable/examples/monthly-means.html#

    month_length = ds.time.dt.days_in_month
    year = ds.time.dt.year[0] #The year when the forecast is valid from
    weights = (
        month_length.groupby("time.season") / month_length.groupby("time.season").sum()
    )

    # Always select the Sep-Oct-Nov season
    ds_season = (ds * weights).groupby("time.season").sum(dim="time").sel(season='SON')

    # Assign the year to the seasonal dataset
    #ds_season = ds_season.expand_dims("year").assign_coords(year=year)

    # Assign the ensemble number to the seasonal data
    ds_season = ds_season.expand_dims("ens").assign_coords(ens=i+1)

    data.append(ds_season)

