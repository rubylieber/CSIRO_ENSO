from pathlib import Path

# Location of ACCESS-S2 hindcast data
CALIBRATED_PR_DIR = Path('/g/data/ux62/access-s2/hindcast/calibrated/atmos/pr/monthly/')

# List of ensembles to analyse for every forecast
ENS_LIST = ['e01','e02','e03']

# List of years to compute hindcast statistics
YEAR_RANGE = range(1981,2019)

# Data storage directory
OUT_PATH = Path('/g/data/gb02/pag548/CSIRO_ENSO')