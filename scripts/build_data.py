from pathlib import Path
import sys

SCRIPTS_DIR=Path(__file__).parent.resolve()
PROJECT_ROOT_DIR = SCRIPTS_DIR.parent.resolve()
SRC_DIR=PROJECT_ROOT_DIR / 'src'

print(f'SCRIPTS_DIR={SCRIPTS_DIR}')
print(f'SRC_DIR={SRC_DIR}')
print(f'PROJECT_ROOT_DIR={PROJECT_ROOT_DIR}')

sys.path.insert(0, str(PROJECT_ROOT_DIR / 'config') )
sys.path.insert(0, str(SRC_DIR ))

print (sys.path)

from config import ENS_LIST, CALIBRATED_PR_DIR, YEAR_RANGE, OUT_PATH
from load_data import compute_hindcast_data
#import logger

#LOG = logger.get_logger(__name__)

month_day_string='0301'

ds = compute_hindcast_data(month_day_string)

out_file = f'{month_day_string}.nc'

ds.to_netcdf( OUT_PATH / out_file)