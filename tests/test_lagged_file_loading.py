# Some simple tests to check the handling of lagged ensemble 
# valid times
 
import xarray as xr
import datetime as dt
import pandas as pd
import itertools
import pytest
from src.load_data import generate_lag_dates, generate_file_list


def test_lag_dates():
   """
   Return the start dates required to generate a 9-day lagged ensemble over the entire hindcast period (1981-2018) for specified dates
   """

   # Test first for 1st September
   start_date_string = '0901'
   dates = generate_lag_dates(start_date_string)

   # We should have 38 years * 9 dates = 342
   assert len(dates) == 342

   # Test for year 2000, the are nine dates valid from August 24- Sep 1
   test_dates = [ d for d in dates if d.year == 2000 ]

   assert len(test_dates) == 9

   # Check their days and months are valid from August 24 to September 1st
   days =  [ d.day for d in test_dates ]
   months =  [ d.month for d in test_dates ]
   assert days == [24, 25, 26, 27, 28, 29, 30, 31, 1]
   assert months == [8, 8, 8, 8, 8, 8, 8, 8, 9]

   # Repeat for March 1st 2004 (A leap year)
   start_date_string = '0301'
   dates = generate_lag_dates(start_date_string)

   test_dates = [ d for d in dates if d.year == 2004 ]

   assert len(test_dates) == 9

   # Check their days and months are valid from February 22 to March 1st
   days =  [ d.day for d in test_dates ]
   months =  [ d.month for d in test_dates ]
   assert days == [22, 23, 24, 25, 26, 27, 28, 29, 1]
   assert months == [2, 2, 2, 2, 2, 2, 2, 2, 3]


def test_file_list():
   """
   Checks the number of files generated for a given list of dates
   """
   # Test first for 1st September
   start_date_string = '0901'
   dates = generate_lag_dates(start_date_string)

   ens = 'e04'
   with pytest.raises(SystemExit) as pytest_exit:  
      files = generate_file_list(ens,dates)
   assert pytest_exit.type == SystemExit

   ens = 'e03'

   file_list = generate_file_list(ens,dates)

   # We should have 38 years * 9 dates = 342
   assert len(file_list) == 342