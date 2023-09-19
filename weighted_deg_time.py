from futureT import *
from historicalT import *
from plotting import *
import yaml

def main():
    """
    Compute population weighted cooling degree time, separating effects of population and temperature change
    """

    # Load config file
    with open("config.yaml", 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    #######################################
    # ERA5 data set

    # Temperature at 2m
    # Source: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview
    # Resolution: 0.25 degree
    era5_path = config['era5_path']
    futureT_file = config['cmip6_path']
    #######################################

    all_data_dict = {}
    all_years = [2000, 2005, 2010, 2015, 2030, 2050, 2100]
    ref_year = [2000]
    # Grid is 1 deg x 1 deg
    grid = [[-89.5,90.5],[-179.5,180.5],1]
    
    ref_year_scenario_run = False
    effects = ["temperature_effect", "gdp_effect", "population_effect", "all_effects"]
    for case in effects:
        print ("case: ", case)

        if case == "population_effect":
            pop_years = all_years
            temp_years = ref_year
            gdp_years = ref_year

        elif case == "temperature_effect":
            pop_years = ref_year
            temp_years = all_years
            gdp_years = ref_year

        elif case == "gdp_effect":
            pop_years = ref_year
            temp_years = ref_year
            gdp_years = all_years
        
        else:
            pop_years = all_years
            temp_years = "same"
            gdp_years = "same"


        deg_time_dict = {}
        for pop_year in pop_years:

            # Population data
            if pop_year < 2020:
                pop_data_year = get_pop_data(pop_year)
            else:
                future_pop_file = config['pop_path'] + 'SSP5_{}.tif'.format(pop_year)
                pop_data_year = compute_pop_gdp_prediction(future_pop_file, grid)

            # Temperature to cooling degree time
            if temp_years == "same":
                run_temp_years = [pop_year]
            else:
                run_temp_years = temp_years
            for temp_year in run_temp_years:

                # Degree time data
                if temp_year < 2020:
                    yearly_degree_time = get_deg_time_data(temp_year, era5_path)
                else:
                    yearly_degree_time = compute_degree_time_prediction(temp_year, futureT_file, grid)
                

                if gdp_years == "same":
                    run_gdp_years = [pop_year]
                else:
                    run_gdp_years = gdp_years

                for gdp_year in run_gdp_years:

                    if pop_year == ref_year[0] and temp_year == ref_year[0] and gdp_year == ref_year[0]:
                        if ref_year_scenario_run:
                            continue
                        else:
                            ref_year_scenario_run = True

                    print("pop_year: ", pop_year)
                    print("temp_year: ", temp_year)
                    print("gdp_year: ", gdp_year)                            
                    
                    if gdp_year < 2020:
                        gdp_data_year = get_gdp_data(gdp_year)
                        yearly_deg_time_AC = get_deg_time_data(gdp_year, era5_path)
                        pop_data_year_AC = get_pop_data(gdp_year)
                    else:
                        future_gdp_file = config['gdp_path'] + 'GDP{}_ssp5.tif'.format(gdp_year)
                        gdp_data_year = compute_pop_gdp_prediction(future_gdp_file, grid)
                        yearly_deg_time_AC = compute_degree_time_prediction(gdp_year, futureT_file, grid)
                        future_pop_file_AC = config['pop_path'] + 'SSP5_{}.tif'.format(gdp_year)
                        pop_data_year_AC = compute_pop_gdp_prediction(future_pop_file_AC, grid)

                    # Set all to type float32
                    modified_arrays = []
                    for data in [yearly_degree_time, yearly_deg_time_AC, pop_data_year, pop_data_year_AC, gdp_data_year]:
                        data = data.astype(np.float32)
                        data['latitude'] = data['latitude'].astype(np.float32)
                        data['longitude'] = data['longitude'].astype(np.float32)
                        modified_arrays.append(data)
                    yearly_degree_time, yearly_deg_time_AC, pop_data_year, pop_data_year_AC, gdp_data_year = modified_arrays

                    # Compute exposure factor
                    if gdp_year > 2020:
                        gdp_data_year = gdp_data_year/pop_data_year_AC
                        gdp_data_year = xr.where(pop_data_year == 0, np.nan, gdp_data_year)
                    exposure_factor = get_exposure_factor(gdp_data_year, yearly_deg_time_AC, gdp_year)

                    # Multiply degree time grid by population grid and gdp grid
                    weighted_cdd = yearly_degree_time * pop_data_year * exposure_factor
                    
                    # Set 0 to NaN
                    weighted_cdd = weighted_cdd.where(weighted_cdd != 0)

                    # Save yearly degree time for 2030
                    if temp_year == 2030 and pop_year == 2030 and gdp_year == 2030:
                        weighted_cdd.to_netcdf('data_experiencedT/weighted_cdd_2030.nc')
      
                    # Compute global average and store in dictionary    
                    global_average = weighted_cdd.sum(dim=['latitude', 'longitude'], skipna=True) / pop_data_year.sum(dim=['latitude', 'longitude'], skipna=True)
                    deg_time_dict["pop{0}_temp{1}_gdp{2}".format(pop_year, temp_year, gdp_year)] = global_average

                    # Plot the population weighted cooling degree time for 2015 for all effects separately
                    if pop_year == 2100 or temp_year == 2100 or gdp_year == 2100 or (pop_year == 2000 and temp_year == 2000 and gdp_year == 2000):
                        plot_map(weighted_cdd, pop_year, temp_year, gdp_year)

                    # Control plots
                    if (pop_year == 2000 and temp_year == 2000 and gdp_year == 2000) or (pop_year == 2100 and temp_year == 2100 and gdp_year == 2100):
                        plot_map(yearly_degree_time, pop_year, temp_year, gdp_year, control='degree_time')
                        plot_map(pop_data_year, pop_year, temp_year, gdp_year, control='population')
                        plot_map(gdp_data_year, pop_year, temp_year, gdp_year, control='gdp')
                        plot_map(exposure_factor, pop_year, temp_year, gdp_year, control='exposure_factor')

                        # Plot without population weighting
                        weighted_cdd_no_pop = yearly_degree_time * exposure_factor
                        plot_map(weighted_cdd_no_pop, pop_year, temp_year, gdp_year, control='degree_time*exposure_factor')

        # Accumulate data in dictionary                
        all_data_dict[case] = deg_time_dict

    # Copy reference year data to all cases
    for case in effects[1:]:
        all_data_dict[case]["pop2000_temp2000_gdp2000"] = all_data_dict[effects[0]]["pop2000_temp2000_gdp2000"]
    # Plot the time curves for the different effects
    plot_time_curve(all_data_dict, all_years)


if __name__ == '__main__':
    main()
