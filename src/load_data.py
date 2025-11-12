# New functions to correctly load monthly ACCESS-S2 seasonal rainfall ensemble for a specified start date and lead time

import xarray as xr
from pathlib import Path
import datetime as dt
import calendar
import pandas as pd
import itertools
import os
import sys

SRC_DIR=Path(__file__).parent.resolve()
PROJECT_ROOT_DIR = SRC_DIR.parent.resolve()

sys.path.insert(0, str(PROJECT_ROOT_DIR / 'config') )
sys.path.insert(0, str(SRC_DIR ))

from config import ENS_LIST, CALIBRATED_PR_DIR, YEAR_RANGE
import logger

LOG = logger.get_logger(__name__)

def generate_lag_dates_year(month_day_string,year):
    """
    For a given start date expressed as a month-day-string (e.g. 0901), 
    generate a list of timestamp objects across the entire hindcast period 1981-2018 
    for the nine-day lagged ensemble valid from this start date
    Contains extra logic for leap years, as there are no hindcast files for February 29
    """

    dates = []

    start_date = dt.datetime.strptime(str(year)+month_day_string,'%Y%m%d')

    if calendar.isleap(int(year)) and month_day_string == '0301':
        # Logic to handle the lack of February 29th hindcast files
        date_range = pd.date_range(end=start_date,periods=10)

        # Remove Feb 29th
        date_range = date_range.drop(pd.to_datetime(str(year) + '-02-29'))

    else:
        date_range = pd.date_range(end=start_date,periods=9)

    dates.extend(date_range)

    return dates


def generate_file_list(month_day_string, 
                       year):
    """
    Generate a list of tuples containing the ensemble member, valid timesmap and index (1-27) 
    for a ACCESS-S2 hindcast valid at the prescribed year 
    (e.g. '2010') and month-day-string (e.g. '0901')
    """

    dates = generate_lag_dates_year(month_day_string,
                                    str(year)) 
    dates_ens_list=list(itertools.product(ENS_LIST,dates[::-1]))

    #Add the ensemble number to this data sturcture
    lagged_ens_list = [ d+(i+1,) for i,d in enumerate(dates_ens_list) ]

    return lagged_ens_list


def seasonal_mean(ds):
    """
    Compute a weighted seasonal mean (valid for September-October-November) of a dataset, 
    following the example provided in the xarray docs
    https://docs.xarray.dev/en/stable/examples/monthly-means.html
    """
    
    month_length = ds.time.dt.days_in_month

    weights = (
            month_length.groupby("time.season") / month_length.groupby("time.season").sum()
        )

    # Always select the Sep-Oct-Nov season
    ds_season = (ds * weights).groupby("time.season").sum(dim="time").sel(season='SON')
        
    return ds_season


def load_yearly_lagged_ensemble(month_day_string, 
                                year):
    """
    Load a full 27-member ensemble ACCESS-S2 hindcast a given year 
    (e.g. '2010') and month-day-string (e.g. '0901'). The hindcast is 
    valid for the September-October-November ('SON') season
    """
    # A list of dataarrays for every lagged ensemble hindcast
    data = []
    
    # Create a list of tuples which contains the ensemble string, timestamp
    # and and index (1-27) for every member of the lagged ensemble
    lagged_ens_list = generate_file_list(month_day_string, 
                                            year)

    for ens in lagged_ens_list:
        # Construct file pattern for this
        pattern = f"{ens[0]}/maq5_pr_{ens[1].strftime('%Y%m%d')}_{ens[0]}.nc"
                                            
        # Find a single file matching this pattern
        filename = list(CALIBRATED_PR_DIR.glob(pattern))

        if filename: # Has the glob found a matching file in this directory
            try:
                ds = xr.open_dataset(filename[0]).pr
                LOG.info(f'Opening {filename[0].name}')
            except:
                LOG.error(f'Cannot open {filename[0]}. Does it contain "pr" data?')
                LOG.error(f'Check defintion of CALIBRATED_PR_DIR : {CALIBRATED_PR_DIR}')
                sys.exit()
        else:
            LOG.error(f'No file matching {pattern} found in {CALIBRATED_PR_DIR}')
            LOG.error(f'Check definition of start dates and source directories')
            sys.exit()

        # Compute seasonal mean for September-October-November
        ds_season = seasonal_mean(ds)

        # Assign the year and ensemble number the seasonal dataarray
        ds_season = ds_season.expand_dims(["year","ens"]).assign_coords(year=[year],ens=[ens[-1]])

        # Append to the list of datararray
        data.append(ds_season)

    # Now concatenate every ensemble member to create a single dataarray 
    ds_year = xr.concat(data,dim='ens')

    return ds_year


def compute_hindcast_data(month_day_string):                                
    """
    Return a data array of dimension (38, 691, 886, 27))
    38 years
    691 latitude points
    886 longitude points
    27 ensemble members
    of ACCESS-S2 September-October-November seasonal precipitation hindcasts,
    valid from 'month_day_string'
    """
    
    # A list of dataframes for every year of the hindcast period
    data = []

    for year in YEAR_RANGE:

        # Load the 27-member lagged seasonal ensemble valid for this year
        ds = load_yearly_lagged_ensemble(month_day_string, 
                                    year)
        LOG.info(f'Computed a seasonal precipitation hindcast for {year} with dimensions {ds.shape}')

        # Append to the list of dataarrays
        data.append(ds)

    # Now concatenate every year data array to create a single dataarray
    ds_hindcast = xr.concat(data,dim='year')

    return ds_hindcast