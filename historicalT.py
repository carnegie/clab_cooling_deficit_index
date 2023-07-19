import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


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

for year in [2000]:

    year_index = [2000, 2005, 2010, 2015, 2020].index(year)
    pop_data_year = pop_dataset["Population Count, v4.11 (2000, 2005, 2010, 2015, 2020): 1 degree"][year_index, :, :]
    # Drop the raster attribute
    pop_data_year = pop_data_year.drop('raster')
    print("Population data for {0}".format(year))
    print(pop_data_year)

    # Temperature to cooling degree days
    yearly_degree_days = None
    if not os.path.exists('yearly_degree_days.nc'):
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
                # surface_temp = surface_temp.astype(np.uint8)

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
                print("Cooling degree days for {0}/{1}/{2}".format(year, month, day))
                print(degree_days)
                print(degree_days.max())
                # Add to yearly degree days
                if yearly_degree_days is None:
                    yearly_degree_days = degree_days
                else:
                    yearly_degree_days += degree_days
                print("Yearly cooling degree days for {0}".format(year))
                print(yearly_degree_days)
                print(yearly_degree_days.max())    
        yearly_degree_days.to_netcdf('yearly_degree_days.nc')
    else:
        yearly_degree_days = xr.open_dataset('yearly_degree_days.nc')

    print("Final yearly cooling degree days for {0}".format(year))
    print(yearly_degree_days)
    print(yearly_degree_days.max())

    # Multiply by population grid
    print("Population weighted cooling degree days for {0}".format(year))
    pop_weighted_cdd = yearly_degree_days["t2m"] * pop_data_year
    print(pop_weighted_cdd)
    print(pop_weighted_cdd.max())


    # Plot the population weighted cooling degree days
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the data
    pop_plot_2000 = ax.imshow(pop_weighted_cdd, origin='lower', aspect='auto', cmap='viridis')

    # Add a colorbar
    cbar = plt.colorbar(pop_plot_2000, ax=ax, orientation='vertical')
    cbar.set_label('Population Count')

    # Set the labels for the x and y axes
    ax.set_xlabel('Longitude Index')
    ax.set_ylabel('Latitude Index')

    # Set the title of the plot
    ax.set_title('Population Count for the Year 2000')

    plt.savefig('population_count_2000.png')
    plt.show()



