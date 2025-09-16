# New functions to correctly load monthly ACCESS-S2 seasonal rainfall ensemble for a specified start date and lead time

import xarray as xr
from pathlib import Path
import datetime as dt
import pandas as pd
import itertools
import os
import sys

sys.path.append(Path(__file__).parent.resolve())

from config import *
from . import logger

LOG = logger.get_logger(__name__)

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

def generate_file_list(ens,dates):
    """
    Generate a list of ACCESS-S2 hindcast filepaths for a give ensemble string 
    (e.g. 'e01') and list of dates
    """

    allowed_ens = {'e01', 'e02', 'e03'} #Allowed values of ensemble string


    if ens not in allowed_ens:
        LOG.error(f'{ens} is not an allowed value : {allowed_ens}')
        LOG.error('Specify an allowed ensemble string')
        sys.exit()

    patterns = [f"{ens}/maq5_pr_{dt.datetime.strftime(date,'%Y%m%d')}_{ens}.nc" for date in dates]

    matched = list(
        itertools.chain.from_iterable(
            CALIBRATED_PR_DIR.glob(pattern) for pattern in patterns
        )
    )

    return matched