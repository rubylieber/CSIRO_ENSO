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

from src.load_data import generate_lag_dates, generate_file_list

LOG = logger.get_logger(__name__)

month_day_string='0901'
ens='e01'
dates = generate_lag_dates(month_day_string)
files = generate_file_list(ens,dates)

ds = xr.open_dataset(files[0]).pr

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
ds_season = ds_season.expand_dims("year").assign_coords(year=year)