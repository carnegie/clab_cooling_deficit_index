import xarray as xr
import rioxarray as rxr
import numpy as np
from historicalT import get_yearly_degree_time



def compute_degree_time_prediction(year, T_file):
    futureT = xr.open_dataset(T_file)['tas']
    futureT = futureT.sel(time=futureT['time.year'] == year)
    yearly_deg_time = get_yearly_degree_time(futureT, None)
    return yearly_deg_time
    


futureT_file = '/groups/carnegie_poc/leiduan_memex/shared/tas_day_CanESM5_ssp585_r1i1p1f1_gn_20150101-21001231.nc'

future_yearly_deg_time = compute_degree_time_prediction(2030, futureT_file)



def compute_pop_gdp_prediction(filename, coarse_grid_factor=(180/64.)/(30./3600)):
    """
    Compute population and GDP prediction for a given year
    """

    future_val = rxr.open_rasterio(filename)
    future_val_ds = future_val.to_dataset(name='val')
    future_val = future_val_ds.rename({'x': 'longitude', 'y': 'latitude'})

    future_val = future_val.coarsen(latitude=int(coarse_grid_factor), longitude=int(coarse_grid_factor), boundary='trim').sum()
    # Round latitude and longitude values to 1 decimal place
    future_val = future_val.assign_coords(latitude=([round(x, 1) for x in future_val.latitude.values]))
    future_val = future_val.assign_coords(longitude=([round(x, 1) for x in future_val.longitude.values]))
    # Create new latitude values
    new_lat_values = np.arange(88.2, -88.2, -2.8)

    # Create an empty DataArray with the desired dimensions
    empty_data = np.zeros((1, len(new_lat_values), len(future_val['longitude'])))
    coords = {
        'band': future_val['band'],
        'latitude': new_lat_values,
        'longitude': future_val['longitude']
    }
    empty_ds = xr.DataArray(empty_data, coords=coords, dims=['band', 'latitude', 'longitude'])
    empty_ds = empty_ds.to_dataset(name='val')

    # Copy the values from the original dataset (future_val) to the new DataArray (empty_ds)
    for lat_val in future_val['latitude'].values:
        nearest_lat = empty_ds['latitude'].sel(latitude=lat_val, method='nearest').values
        empty_ds.loc[dict(latitude=nearest_lat)] = future_val.sel(latitude=lat_val)

    return future_val


# 30 arc sec resolution
# T resolution is 64x128

year = 2030
future_pop_file = 'data_experiencedT/SSP5_population/SSP5_{}.tif'.format(2030)
future_pop = compute_pop_gdp_prediction(future_pop_file)


future_gdp_file = 'data_experiencedT/GDP_SSP5_1km/GDP{}_ssp5.tif'.format(year)
future_gdp = compute_pop_gdp_prediction(future_gdp_file)

print("cdd", future_yearly_deg_time)
print ("cdd sum", future_yearly_deg_time.sum())
print("pop", future_pop)
print("pop sum", future_pop.sum())
print("gdp", future_gdp)
print("gdp sum", future_gdp.sum())

weighted_cdd = future_yearly_deg_time * future_pop * future_gdp
print(weighted_cdd)



