import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_map, plot_time_curve


def get_pop_data(year):
    """
    Get population data for a given year
    """

    # Population
    # Source: https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-count-rev11
    # Resolution: 1 degree
    # 2000, 2005, 2010, 2015, 2020
    pop_file = 'data_experiencedT/gpw_v4_population_count_rev11_1_deg.nc'
    pop_dataset = xr.open_dataset(pop_file)

    year_index = [2000, 2005, 2010, 2015, 2020].index(year)
    pop_data_year = pop_dataset["Population Count, v4.11 (2000, 2005, 2010, 2015, 2020): 1 degree"][year_index, :, :]
    # Drop the raster attribute
    pop_data_year = pop_data_year.drop('raster')
    return pop_data_year


def get_gdp_data(year):
    """
    Get gdp data for a given year
    """
    # GDP
    # Source: https://datadryad.org/stash/dataset/doi:10.5061/dryad.dk1j0
    # Resolution: 5 arc min = 1/12 degree
    gdp_file = 'data_experiencedT/GDP_per_capita_PPP_1990_2015_v2.nc'
    gdp_dataset = xr.open_dataset(gdp_file)
    year_index = gdp_dataset['time'].values.tolist().index(year)
    gdp_data_year = gdp_dataset["GDP_per_capita_PPP"][year_index, :, :]
    
    # Drop the time attribute
    gdp_data_year = gdp_data_year.drop('time')
    # Make grid coarser and change resolution to 1 degree
    gdp_data_year = gdp_data_year.coarsen(latitude=12, longitude=12, boundary='trim').mean()

    return gdp_data_year

def get_exposure_factor(gdp_data, deg_days, gdp_year):
    """
    Get AC penetration factor for given GDP and cooling degree time
    """
    # Compute availability as a function of GDP
    # Factor 0.68 between 1995 USD and 2011 USD: Function derived with 1995 dollars and historical data in 2011 dollars
    if gdp_year < 2020:
        inflation_factor = 0.68
    else:
        # Predictions of GDP are in 2005 dollars
        inflation_factor = 0.78
    availability = 1/(1+np.exp(4.152)*np.exp(-0.237*inflation_factor*gdp_data/1000))
    # Compute AC saturation as a function of cooling degree days
    saturation = 1. - 0.949*np.exp(-0.00187*deg_days)

    new_latitude = np.arange(89.5, -90.5, -1)
    new_longitude = np.arange(-179.5, 180.5, 1)
    availability = availability.interp(longitude=new_longitude, latitude=new_latitude)

    exposure_factor = 1. - (availability * saturation)

    exposure_factor = exposure_factor.astype(np.float32)
    exposure_factor['latitude'] = exposure_factor['latitude'].astype(np.float32)
    exposure_factor['longitude'] = exposure_factor['longitude'].astype(np.float32)

    return exposure_factor

def get_yearly_degree_time(surface_temp, yearly_degree_time, base_temp = 18.0):
    """
    Compute yearly degree time
    """
    # Compute degree time for each grid cell and each time step

    # Loop over year if all days contained in one file (for projections)
    if hasattr(surface_temp, 'time'):
        
        degree_days = None
        for i in range(len(surface_temp.time)):
            degree_time = surface_temp[i, :, :] - (base_temp + 273.15)
            # Set all negative values to zero
            degree_time = degree_time.where(degree_time > 0, 0)
            if degree_days is None:
                degree_days = degree_time
            else:
                degree_days += degree_time
        # Drop the time attribute
        degree_days = degree_days.drop('time')
    # Each day separate file (for historical data)
    else:
        degree_days = surface_temp - (base_temp + 273.15)
        # Set all negative values to zero
        degree_days = degree_days.where(degree_days > 0, 0)

    # Sum over all grid cells
    if yearly_degree_time is None:
        yearly_degree_time = degree_days
    else:
        yearly_degree_time += degree_days
    
    return yearly_degree_time

def compute_degree_time(year, era5_path):
    """
    Compute degree time for a given year
    """
    yearly_degree_time = None
    for month in range(1, 13):
        month = str(month).zfill(2)
        for day in range(1, 32):
            day = str(day).zfill(2)
            if not os.path.exists(os.path.join(era5_path, 'ERA5_{0}_{1}_{2}.nc'.format(year, month, day))):
                continue
                
            era5_file = os.path.join(era5_path, 'ERA5_{0}_{1}_{2}.nc'.format(year, month, day))
            # Open the NetCDF file
            temp_dataset = xr.open_dataset(era5_file)

            # Extract the surface temperature
            surface_temp = temp_dataset['t2m']

            # Make grid coarser and change resolution to 1 degree
            surface_temp = surface_temp.coarsen(latitude=4, longitude=4, boundary='trim').mean()
            # Shift grid to go from -180 to 180 degrees and 90 to -90 degrees
            surface_temp = surface_temp.assign_coords(longitude=([(x + 0.125) for x in surface_temp.longitude.values if x <= 180] + [(x - 359.875) for x in surface_temp.longitude.values if x > 180]))
            surface_temp = surface_temp.assign_coords(latitude=(surface_temp.latitude - 0.125))
            # Sort grid by longitude
            surface_temp = surface_temp.sortby(surface_temp.longitude)
          
            # Compute degree time for each day
            # Get daily mean temperature
            surface_temp = surface_temp.mean(dim='time', skipna=True)
            yearly_degree_time = get_yearly_degree_time(surface_temp, yearly_degree_time)
    
    return yearly_degree_time



def get_deg_time_data(year, era5_path):
    """
    Get degree time data for a given year
    """
    if not os.path.exists('yearly_deg_time/'):
        os.mkdir('yearly_deg_time/')
    if not os.path.exists('yearly_deg_time/yearly_degree_time_{0}.nc'.format(year)):
        yearly_degree_time = compute_degree_time(year, era5_path)
        yearly_degree_time.to_netcdf('yearly_deg_time/yearly_degree_time_{0}.nc'.format(year))
    else:
        yearly_degree_time = xr.open_dataset('yearly_deg_time/yearly_degree_time_{0}.nc'.format(year))['t2m']
    return yearly_degree_time
 

