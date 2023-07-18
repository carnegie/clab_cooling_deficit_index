import os
import xarray as xr


# Load gridded datasets

data_path = 'data_experiencedT/'

# Population
# Source: https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-count-rev11
# Resolution: 1 degree
# 2000, 2005, 2010, 2015, 2020
pop_file = os.path.join(data_path, 'gpw_v4_population_count_rev11_1_deg.nc')

# GDP
# Source: https://datadryad.org/stash/dataset/doi:10.5061/dryad.dk1j0
# Resolution: 5 arc min = 1/12 degree
gdp_file = os.path.join(data_path, 'GDP_per_capita_PPP_1990_2015_v2.nc')

# Temperature
# Source: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview
# Resolution: 0.1 degree
era5_path = '/groups/carnegie_poc/leiduan_memex/lustre_scratch/MERRA2_data/ERA5_original_data/'

for year in ["2005"]:
    for month in ["01"]:
        for day in ["01"]:
            era5_file = os.path.join(era5_path, 'ERA5_{0}_{1}_{2}.nc'.format(year, month, day))
            # Open the NetCDF file
            dataset = xr.open_dataset(file_path)

            # Get an overview of the dataset
            dataset.info()            

