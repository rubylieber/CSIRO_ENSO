import xarray as xr
import datetime as dt
import pandas as pd
import itertools
import pytest
import sys
from pathlib import Path

sys.path.append('/home/548/pag548/code/CSIRO_ENSO/src')
sys.path.append('/home/548/pag548/code/CSIRO_ENSO/config')
from config import *
from src import logger

from src.load_data import compute_hindcast_data

LOG = logger.get_logger(__name__)

month_day_string='0901'
ens='e01'

data = compute_hindcast_data(2000,'0901')

# A dictionary to hold a datarray of SON ensemble data for every year
year_dict = {}

year=2000

dates = generate_lag_dates_year(month_day_string,str(year))
ens_list = ['e01','e02','e03']
dates_ens_list=list(itertools.product(ens_list,dates[::-1]))

#Add the ensemble number
lagged_ens_list = [ d+(i+1,) for i,d in enumerate(dates_ens_list) ]


# A list of data for every lagged ensemble member
data = []

for ens in lagged_ens_list:
    
    # Construct file pattern for this
    pattern = f"{ens[0]}/maq5_pr_{ens[1].strftime('%Y%m%d')}_{ens[0]}.nc"

    # Find a file matching this pattern
    filename = list(CALIBRATED_PR_DIR.glob(pattern))

    if filename: # Has the glob found a matching file in this directory
        ds = xr.open_dataset(filename[0]).pr
        LOG.info(f'Opening {filename[0].name}')
    else:
        LOG.error(f'No file matching {pattern} found in {CALIBRATED_PIR_DIR}')
        LOG.error(f'Check definition of start dates and source directories')
        sys.exit()

    # Follow this example to compute a seasonal mean
    # https://docs.xarray.dev/en/stable/examples/monthly-means.html#

    month_length = ds.time.dt.days_in_month

    weights = (
        month_length.groupby("time.season") / month_length.groupby("time.season").sum()
    )

    # Always select the Sep-Oct-Nov season
    ds_season = (ds * weights).groupby("time.season").sum(dim="time").sel(season='SON')

    # Assign the year to the seasonal dataset
    #ds_season = ds_season.expand_dims("year").assign_coords(year=year.data)

    # Assign the ensemble number to the seasonal data
    ds_season = ds_season.expand_dims(["year","ens"]).assign_coords(year=[year],ens=[ens[-1]])

    data.append(ds_season)

# Now concatenate all ensembles for this year
year_dict[year] = xr.concat(data,dim='ens')
