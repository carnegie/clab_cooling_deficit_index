import xarray as xr
import rioxarray as rxr
import numpy as np
from historicalT import get_yearly_degree_time
import xesmf as xe



def compute_degree_time_prediction(year, T_file, grid):
    futureT = xr.open_dataset(T_file)['tas']
    futureT = futureT.sel(time=futureT['time.year'] == year)
    futureT = futureT.rename({'lon': 'longitude', 'lat': 'latitude'})
    
    # Shift from 0-360 to -180-180
    futureT = futureT.assign_coords(longitude=(((futureT.longitude + 178.6) % 360) - 180)).sortby('longitude')
    
    # Regrid
    ds_out = xr.Dataset({"longitude": (["longitude"], np.arange(grid[1][0], grid[1][1], grid[2])),
                         "latitude": (["latitude"], np.arange(grid[0][0], grid[0][1], grid[2])),})     

    regridder = xe.Regridder(futureT, ds_out, "bilinear")
    futureT = regridder(futureT)

    yearly_deg_time = get_yearly_degree_time(futureT, None)
    yearly_deg_time = yearly_deg_time.transpose("latitude", "longitude")

    # Flip the values along the latitude axis so that 90 is at the top
    yearly_deg_time = yearly_deg_time.reindex(latitude=list(reversed(yearly_deg_time['latitude'].values)))

    return yearly_deg_time
    

def compute_pop_gdp_prediction(filename, grid):
    """
    Compute population and GDP prediction for a given year
    """
    future_val = rxr.open_rasterio(filename)
    # Set nodata values to NaN
    future_val = future_val.where(future_val != future_val.rio.nodata)
    future_val_ds = future_val.to_dataset(name='val')
    future_val = future_val_ds.rename({'x': 'longitude', 'y': 'latitude'})
    
    # Regrid
    future_val = future_val.coarsen(latitude=120, longitude=120, boundary='trim').sum()

    # Create new latitude values
    new_lat_values = np.arange(89.5, -90.5, -1.)

    # Create an empty DataArray with the desired dimensions
    empty_data = np.zeros((1, len(future_val['longitude']), len(new_lat_values)))
    coords = {
        'band': future_val['band'],
        'longitude': future_val['longitude'],
        'latitude': new_lat_values
    }
    empty_ds = xr.DataArray(empty_data, coords=coords, dims=['band', 'longitude', 'latitude'])
    empty_ds = empty_ds.to_dataset(name='val')

    # Copy the values from the original dataset (future_val) to the new DataArray (empty_ds)
    for lat_val in future_val['latitude'].values:
        nearest_lat = empty_ds['latitude'].sel(latitude=lat_val, method='nearest').values
        empty_ds.loc[dict(latitude=nearest_lat)] = future_val.sel(latitude=lat_val)
        
    future_values = empty_ds.sel(band=1)["val"]
    future_values = future_values.transpose("latitude", "longitude")

    future_values = future_values.where(future_values != 0)

    return future_values






