from futureT import *
from historicalT import *
from plotting import *
import xesmf as xe
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
    effects = ["all_effects"] # "gdp_effect", "temperature_effect","population_effect", 
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

                print("pop year", pop_year)
                print("temp year", temp_year)

                # Degree time data
                if temp_year < 2020:
                    yearly_degree_time = get_deg_time_data(temp_year, era5_path)
                else:
                    yearly_degree_time = compute_degree_time_prediction(temp_year, futureT_file, grid)
                

                # Multiply degree time grid by population grid and gdp grid
                # print("degree time", yearly_degree_time)
                # print("population", pop_data_year)

                # 1. Build the regridder
                regridder = xe.Regridder(pop_data_year, yearly_degree_time, 'bilinear')

                # 2. Apply the regridder to the 'Population 2100' dataset
                regridded_pop_data_year = regridder(pop_data_year)

                
                # Write to file
                # yearly_degree_time.to_netcdf("degree_time_{}.nc".format(temp_year))
                # pop_data_year.to_netcdf("pop_{}.nc".format(pop_year))
                weighted_cdd = yearly_degree_time * regridded_pop_data_year
                weighted_cdd.to_netcdf("weighted_cdd_{}_{}.nc".format(pop_year, temp_year))
                
                # Set 0 to NaN
                # weighted_cdd = weighted_cdd.where(weighted_cdd != 0)
        
                gdp_year = 2000
                # Compute global average and store in dictionary    
                global_average = weighted_cdd.sum(dim=['latitude', 'longitude'], skipna=True) / pop_data_year.sum(dim=['latitude', 'longitude'], skipna=True)
                if pop_year == 2000 and temp_year == 2000 and gdp_year == 2000:
                    # Copy global average 
                    weighted_cdd_2000 = global_average
                    weighted_cdd_2000_map = weighted_cdd.copy()
                deg_time_dict["pop{0}_temp{1}_gdp{2}".format(pop_year, temp_year, gdp_year)] = global_average
                weighted_cdd_inv = weighted_cdd_2000 * (weighted_cdd_2000 / global_average)
                weighted_cdd_inv_map = (weighted_cdd_2000_map / weighted_cdd)
                deg_time_dict["pop{0}_temp{1}_gdp{2}_inv".format(pop_year, temp_year, gdp_year)] = weighted_cdd_inv

                # Plot the population weighted cooling degree time for 2015 for all effects separately
                if pop_year == 2100 or temp_year == 2100 or gdp_year == 2100 or (pop_year == 2000 and temp_year == 2000 and gdp_year == 2000):
                    plot_map(weighted_cdd, pop_year, temp_year)
                    plot_map(weighted_cdd_inv_map, pop_year, temp_year, control='exposure_factor')

                # # Control plots
                if (pop_year == 2000 and temp_year == 2000) or (pop_year == 2100 and temp_year == 2100):
                    plot_map(yearly_degree_time, pop_year, temp_year, control='degree_time')
                    plot_map(pop_data_year, pop_year, temp_year, control='population')
                #     plot_map(gdp_data_year, pop_year, temp_year, gdp_year, control='gdp')
                #     plot_map(exposure_factor, pop_year, temp_year, gdp_year, control='exposure_factor')

                    # # Plot without population weighting
                    # weighted_cdd_no_pop = yearly_degree_time * exposure_factor
                    # plot_map(weighted_cdd_no_pop, pop_year, temp_year, gdp_year, control='degree_time*exposure_factor')

        # Accumulate data in dictionary                
        all_data_dict[case] = deg_time_dict


    # Copy reference year data to all cases
    for case in effects[1:]:
        all_data_dict[case]["pop2000_temp2000_gdp2000"] = all_data_dict[effects[0]]["pop2000_temp2000_gdp2000"]
    # Plot the time curves for the different effects
    plot_time_curve(all_data_dict, all_years)


if __name__ == '__main__':
    main()
