def empirical_cdf(data):
    """Computes the empirical cumulative distribution function (ECDF).

    Parameters
    ----------
    data : array-like
        A 1D array of numerical values.

    Returns
    -------
    tuple of (ndarray, ndarray)
        sorted_data : ndarray
            The input data sorted in ascending order.
        cdf : ndarray
            The empirical cumulative distribution values corresponding to `sorted_data`.
    """
    import numpy as np
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf

def get_probability_exceed(cdf, data_values, threshold):
    """Computes the probability P(X ≥ threshold) using the empirical CDF.

    Parameters
    ----------
    cdf : ndarray
        The empirical cumulative distribution function values.
    data_values : ndarray
        The sorted data values corresponding to the CDF.
    threshold : float
        The threshold value for which the exceedance probability is computed.

    Returns
    -------
    float or ndarray
        The probability that a random variable X is greater than or equal to `threshold`.
    """
    import numpy as np
    idx = np.searchsorted(data_values, threshold, side='right') - 1
    idx = np.clip(idx, 0, len(data_values) - 1)  # Ensure valid index
    return 1 - cdf[..., idx] 

def get_probability_deficit(cdf, data_values, threshold):
    """Computes the probability P(X ≤ threshold) using the empirical CDF.

    Parameters
    ----------
    cdf : ndarray
        The empirical cumulative distribution function values.
    data_values : ndarray
        The sorted data values corresponding to the CDF.
    threshold : float
        The threshold value for which the deficit probability is computed.

    Returns
    -------
    float or ndarray
        The probability that a random variable X is less than or equal to `threshold`.
    """
    import numpy as np
    idx = np.searchsorted(data_values, threshold, side='right') - 1
    idx = np.clip(idx, 0, len(data_values) - 1)  # Ensure valid index
    return cdf[..., idx]

def spatial_probability_exceed(cdf, anom_data, threshold):
    """Computes the probability P(X ≥ threshold) for spatial data using the empirical CDF.

    This function applies `get_probability_exceed` across a spatial dataset using `xarray.apply_ufunc`,
    allowing for efficient computation across multiple grid points.

    Parameters
    ----------
    cdf : xarray.DataArray
        The empirical cumulative distribution function values with a `time` dimension.
    anom_data : xarray.DataArray
        The anomaly data corresponding to the CDF, with a `time` dimension.
    threshold : float or xarray.DataArray
        The threshold value(s) for which the exceedance probability is computed.

    Returns
    -------
    xarray.DataArray
        The probability that a random variable X is greater than or equal to `threshold`
        at each spatial grid point.
    """
    import xarray as xr
    P = xr.apply_ufunc(
        get_probability_exceed, cdf, anom_data, threshold,
        input_core_dims=[["time"], ["time"], []],  # Match time dimension
        output_core_dims=[[]],  # Remove time dim
        vectorize=True,
        dask="parallelized"
    )
    return P

def spatial_probability_deficit(cdf, anom_data, threshold):
    """Computes the probability P(X ≤ threshold) for spatial data using the empirical CDF.

    This function applies `get_probability_deficit` across a spatial dataset using `xarray.apply_ufunc`,
    allowing for efficient computation across multiple grid points.

    Parameters
    ----------
    cdf : xarray.DataArray
        The empirical cumulative distribution function values with a `time` dimension.
    anom_data : xarray.DataArray
        The anomaly data corresponding to the CDF, with a `time` dimension.
    threshold : float or xarray.DataArray
        The threshold value(s) for which the deficit probability is computed.

    Returns
    -------
    xarray.DataArray
        The probability that a random variable X is less than or equal to `threshold`
        at each spatial grid point.
    """
    import xarray as xr
    P = xr.apply_ufunc(
        get_probability_deficit, cdf, anom_data, threshold,
        input_core_dims=[["time"], ["time"], []],  # Match time dimension
        output_core_dims=[[]],  # Remove time dim
        vectorize=True,
        dask="parallelized"
    )
    return P 

def load_first3_timesteps(file_path):
    """Loads the first three timesteps from a NetCDF file using xarray.

    Parameters
    ----------
    file_path : str
        Path to the NetCDF file.

    Returns
    -------
    xarray.Dataset
        The dataset containing only the first three timesteps, with original time coordinates preserved.
    """
    import xarray as xr
    with xr.open_dataset(file_path, chunks="auto") as ds:
        ds = ds.isel(time=slice(0, 3)) #.load()  # Load only first three timesteps
        ds = ds.assign_coords(time=("time", ds.time.values))   # Keep original time coordinate
    return ds

def load_last_timestep(file_path):
    """Loads the last timestep from a NetCDF file using xarray.

    Parameters
    ----------
    file_path : str
        Path to the NetCDF file.

    Returns
    -------
    xarray.Dataset
        The dataset containing only the last timestep, with the original time coordinate preserved.
    """
    import xarray as xr
    with xr.open_dataset(file_path, chunks="auto") as ds:
        ds = ds.isel(time=-1) #.load()  # Load only last timestep
        ds = ds.assign_coords(time=[ds.time.values])  # Keep original time coordinate
    return ds

def load_3rd_timestep(file_path):
    """Loads the third timestep from a NetCDF file using xarray.

    Parameters
    ----------
    file_path : str
        Path to the NetCDF file.

    Returns
    -------
    xarray.Dataset
        The dataset containing only the third timestep.
    """
    import xarray as xr
    with xr.open_dataset(file_path, chunks="auto") as ds:
        ds = ds.isel(time=2) #.load()  
        #ds = ds.assign_coords(time=("time", ds.time.values)) # Keep original time coordinate
    return ds

def reshape_to_ensemble(ds):
    """Reshapes the dataset to include ensemble members and months.

    This function assumes the dataset has 9 ensemble members and reshapes it to have a structure
    where the time dimension is reorganized into a (year, ensemble, month) format.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset with a time dimension that is divisible by 9 (for the 9 ensemble members).

    Returns
    -------
    xarray.Dataset
        The reshaped dataset with the time dimension replaced by ensemble and month.
        The time dimension is dropped, and ensemble members are indexed along a new dimension.
    """
    import numpy as np
    
    ds = ds.assign_coords(ensemble=("time", np.tile(np.arange(9), len(ds.time) // 9)))
    
    # Reshape time -> (year, ensemble, month)
    ds = ds.groupby("ensemble").map(lambda x: x.coarsen(time=3, boundary="trim").mean())
    
    return ds.swap_dims({"time": "ensemble"}).drop_vars("time")

def read_access_lag0(file_pattern):
    """Reads ACCESS-S2 data for lag 0 from multiple files and reshapes it.

    This function generates file paths for each year (1981-2018) and each base date for September,
    extracts the first three timesteps from each file, concatenates them along the time axis, 
    and reshapes the data into ensemble format by grouping the data by year. This function is
    designed to extract SON averages for the variable "pr". 

    Parameters
    ----------
    file_pattern : str
        The file path pattern with placeholders for year and date, which is used to generate 
        the full file paths. The placeholder is expected to be in the format `{year}{date}`.

    Returns
    -------
    xarray.Dataset
        The reshaped dataset with ensemble data, grouped by year and reshaped to include months 
        and ensemble members.
    """
    import xarray as xr

    years = range(1981, 2019)

    # Define the base dates for sep, oct, and nov with lag 0
    base_dates = ['0824', '0825', '0826', '0827', '0828', '0829', '0830', '0831', '0901']

    # Generate file paths for each year from 1981 to 2018
    file_paths = [file_pattern.format(f"{year}{date}") for year in years for date in base_dates]
    
    # Load datasets with correct time from each file
    datasets = [load_first3_timesteps(file) for file in file_paths]
    
    # Concatenate along the time axis
    ds_final = xr.concat(datasets, dim="time").pr
    
    # Extract years from the time coordinate
    ds_grouped = ds_final.groupby("time.year")
    
    # Apply function to each year separately
    ds_ensemble = ds_grouped.map(reshape_to_ensemble)

    return ds_ensemble

def read_access_lag9(file_pattern):
    """Reads ACCESS-S2 data for lag 9 from multiple files and reshapes it.

    This function generates file paths for each year (1981-2018) and each base date for January to March,
    extracts the last timestep from each file, concatenates them along the time axis, 
    and reshapes the data into ensemble format by grouping the data by year. This function is 
    designed to extract SON averages for the variable "pr". 

    Parameters
    ----------
    file_pattern : str
        The file path pattern with placeholders for year and date, which is used to generate 
        the full file paths. The placeholder is expected to be in the format `{year}{date}`.

    Returns
    -------
    xarray.Dataset
        The reshaped dataset with ensemble data, grouped by year and reshaped to include
        months and ensemble members.
    """
    import xarray as xr

    years = range(1981, 2019)

    # Define the base dates for sep, oct, and nov with lag 0
    base_dates = ['0121', '0124', '0125', '0126', '0127', '0128', '0129', '0130', '0131',
                  '0216', '0221', '0222', '0223', '0224', '0225', '0226', '0227', '0228',
                  '0321', '0324', '0325', '0326', '0327', '0328', '0329', '0330', '0331']

    # Generate file paths for each year from 1981 to 2018
    file_paths = [file_pattern.format(f"{year}{date}") for year in years for date in base_dates]
    
    # Load datasets with correct time from each file
    datasets = [load_last_timestep(file) for file in file_paths]
    
    # Concatenate along the time axis
    ds_final = xr.concat(datasets, dim="time").pr
    
    # Extract years from the time coordinate
    ds_grouped = ds_final.groupby("time.year")
    
    # Apply function to each year separately
    ds_ensemble = ds_grouped.map(reshape_to_ensemble)

    return ds_ensemble

def read_access_lag3(file_pattern):
    """Reads ACCESS data for lag 3 from multiple files and reshapes it.

    This function generates file paths for each year (1981-2018) and each base date for June to September,
    extracts the third timestep from each file, concatenates them along the time axis, 
    and reshapes the data into ensemble format by grouping the data by year. This function is 
    designed to extract SON averages for the variable "pr". 

    Parameters
    ----------
    file_pattern : str
        The file path pattern with placeholders for year and date, which is used to generate 
        the full file paths. The placeholder is expected to be in the format `{year}{date}`.

    Returns
    -------
    xarray.Dataset
        The reshaped dataset with ensemble data, grouped by year and reshaped to include months and ensemble members.
    """
    import xarray as xr

    years = range(1981, 2019)

    # Define the base dates for sep, oct, and nov with lag 3
    base_dates = ['0623', '0624', '0625', '0626', '0627', '0628', '0629', '0630', '0701',
                  '0724', '0725', '0726', '0727', '0728', '0729', '0730', '0731', '0801',
                  '0824', '0825', '0826', '0827', '0828', '0829', '0830', '0831', '0901']

    # Generate file paths for each year from 1981 to 2018
    file_paths = [file_pattern.format(f"{year}{date}") for year in years for date in base_dates]

    # Load datasets with correct time from each file
    datasets = [load_3rd_timestep(file) for file in file_paths]
    # Concatenate along the time axis
    ds_final = xr.concat(datasets, dim="time").pr
    
    # Extract years from the time coordinate
    ds_grouped = ds_final.groupby("time.year")
    
    # Apply function to each year separately
    ds_ensemble = ds_grouped.map(reshape_to_ensemble)

    return ds_ensemble

def pattern_cor(d1, d2):
    """Computes the Pearson correlation coefficient between two datasets, masking non-finite values.

    This function flattens the input datasets into 1D arrays, applies a mask to remove non-finite values,
    and computes the Pearson correlation coefficient between the two datasets.

    Parameters
    ----------
    d1 : xarray.DataArray
        The first input dataset.
    d2 : xarray.DataArray
        The second input dataset.

    Returns
    -------
    float
        The Pearson correlation coefficient between the two input datasets, after masking non-finite values.
    """
    import numpy as np
    from scipy.stats import pearsonr
    # Make 1D
    d1_1d = d1.values.flatten()
    d2_1d = d2.values.flatten()
    # Mask
    mask = d1_1d*d2_1d
    d1_1d = d1_1d[np.isfinite(mask)]
    d2_1d = d2_1d[np.isfinite(mask)]
    # Correlation
    return pearsonr(d1_1d, d2_1d)[0]

def corr_along_time(da1, da2, n=0):
    """Computes the Pearson correlation (or p-value) at each grid point along the time dimension.

    This function calculates the Pearson correlation between two datasets along the time axis
    for each grid point. If `n=0`, the function returns the Pearson correlation coefficient,
    while if `n=1`, it returns the p-value. The function applies a masking operation for non-finite
    values in the input datasets.

    Parameters
    ----------
    da1 : xarray.DataArray
        The first input dataset, with dimensions that include a 'year' or time axis.
    da2 : xarray.DataArray
        The second input dataset, with the same dimensions as `da1` for the time axis.
    n : int, optional
        If 0 (default), returns the Pearson correlation coefficient. If 1, returns the p-value.

    Returns
    -------
    xarray.DataArray
        A DataArray containing the Pearson correlation coefficient or p-value for each grid point.
        The resulting array has no time dimension, only the spatial dimensions.
    """
    import numpy as np
    from scipy.stats import pearsonr
    import xarray as xr 
    
    def pearson_corr(x, y):
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            return pearsonr(x, y)[n]
        return np.nan

    return xr.apply_ufunc(
        pearson_corr,
        da1, da2,
        input_core_dims=[["year"], ["year"]],
        output_core_dims=[[]],  # Ensures a scalar output per grid point
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )