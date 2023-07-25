import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_map, plot_time_curve

#######################################
# Load gridded datasets

data_path = 'data_experiencedT/'

# Population
# Source: https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-count-rev11
# Resolution: 1 degree
# 2000, 2005, 2010, 2015, 2020
pop_file = os.path.join(data_path, 'gpw_v4_population_count_rev11_1_deg.nc')
pop_dataset = xr.open_dataset(pop_file)

# GDP
# Source: https://datadryad.org/stash/dataset/doi:10.5061/dryad.dk1j0
# Resolution: 5 arc min = 1/12 degree
gdp_file = os.path.join(data_path, 'GDP_per_capita_PPP_1990_2015_v2.nc')

# Temperature
# Source: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview
# Resolution: 0.1 degree
# data on 
era5_path = '/groups/carnegie_poc/leiduan_memex/lustre_scratch/MERRA2_data/ERA5_original_data/'
#######################################

def get_pop_data(year):
    """
    Get population data for a given year
    """
    year_index = [2000, 2005, 2010, 2015, 2020].index(year)
    pop_data_year = pop_dataset["Population Count, v4.11 (2000, 2005, 2010, 2015, 2020): 1 degree"][year_index, :, :]
    # Drop the raster attribute
    pop_data_year = pop_data_year.drop('raster')
    return pop_data_year

def compute_degree_days(year, era5_path):
    """
    Compute degree days for a given year
    """
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
            # Shift grid to go from 0 to 360 degrees and 90 to -90 degrees
            surface_temp = surface_temp.assign_coords(longitude=(surface_temp.longitude - 179.875))
            surface_temp = surface_temp.assign_coords(latitude=(surface_temp.latitude - 0.125))

            # Take mean over 24 hours
            surface_temp = surface_temp.mean(dim='time')
            
            # Compute degree days for each grid cell
            base_temp = 18.0
            degree_days = surface_temp - (base_temp + 273.15)
            # Set all negative values to zero
            degree_days = degree_days.where(degree_days > 0, 0)

            # Sum over all grid cells
            if yearly_degree_days is None:
                yearly_degree_days = degree_days
            else:
                yearly_degree_days += degree_days

def get_deg_day_data(year, era5_path):
    """
    Get degree day data for a given year
    """
    yearly_degree_days = None
    if not os.path.exists('yearly_deg_days/'):
        os.mkdir('yearly_deg_days/')
    if not os.path.exists('yearly_deg_days/yearly_degree_days_{0}.nc'.format(year)):
        yearly_degree_days = compute_degree_days(year, era5_path)
        yearly_degree_days.to_netcdf('yearly_deg_days/yearly_degree_days_{0}.nc'.format(year))
    else:
        yearly_degree_days = xr.open_dataset('yearly_deg_days/yearly_degree_days_{0}.nc'.format(year))['t2m']
    return yearly_degree_days
 

#######################################
# Compute population weighted cooling degree days

def main():
    """
    Compute population weighted cooling degree days, separating effects of population and temperature change
    """
    all_data_dict = {}
    for case in ["population_effect", "temperature_effect", "both_effects"]:

        all_years = [2000, 2005, 2010, 2015, 2020]
        ref_year = [2000]

        if case == "population_effect":
            pop_years = all_years
            temp_years = ref_year

        elif case == "temperature_effect":
            pop_years = ref_year
            temp_years = all_years
        
        else:
            pop_years = all_years
            temp_years = "same"


        deg_day_dict = {}
        for pop_year in pop_years:

            # Population data
            pop_data_year = get_pop_data(pop_year)
 

            # Temperature to cooling degree days
            for temp_year in temp_years:

                if temp_years == "same":
                    temp_year = pop_year

                # Degree day data
                yearly_degree_days = get_deg_day_data(temp_year, era5_path)


                # Multiply degree day grid by population grid
                pop_weighted_cdd = yearly_degree_days * pop_data_year

                # Compute global average and store in dictionary
                global_average = pop_weighted_cdd.mean()
                deg_day_dict["pop{0}_temp{1}".format(pop_year, temp_year)] = global_average

                # Plot the population weighted cooling degree days for 2020 for all effects separately
                if pop_year == 2020 or temp_year == 2020:
                    plot_map(pop_weighted_cdd, pop_year, temp_year)

        # Accumulate data in dictionary                
        all_data_dict[case] = deg_day_dict

    # Plot the time curves for the different effects
    plot_time_curve(all_data_dict)


if __name__ == '__main__':
    main()


