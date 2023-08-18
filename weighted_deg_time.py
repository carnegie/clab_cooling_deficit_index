from futureT import *
from historicalT import *
from plotting import *

def main():
    """
    Compute population weighted cooling degree time, separating effects of population and temperature change
    """

    # Path to ERA5 data; change this to the path on your machine
    #######################################
    # ERA5 data set stored on Caltech server

    # Temperature
    # Source: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview
    # Resolution: 0.1 degree
    # data on 
    era5_path = '/groups/carnegie_poc/leiduan_memex/lustre_scratch/MERRA2_data/ERA5_original_data/'
    #######################################

    all_data_dict = {}
    all_years = [2000, 2005, 2010, 2015]
    ref_year = [2000]
    
    for case in ["population_effect", "temperature_effect", "gdp_effect", "all_effects"]:
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
            pop_data_year = get_pop_data(pop_year)

            # Temperature to cooling degree time
            if temp_years == "same":
                run_temp_years = [pop_year]
            else:
                run_temp_years = temp_years
            for temp_year in run_temp_years:

                # Degree time data
                yearly_degree_time = get_deg_time_data(temp_year, era5_path)
                print("yearly_degree_time: ", yearly_degree_time.sum(dim=['latitude', 'longitude'], skipna=True))

                if gdp_years == "same":
                    run_gdp_years = [pop_year]
                else:
                    run_gdp_years = gdp_years
                for gdp_year in run_gdp_years:

                    gdp_data_year = get_gdp_data(gdp_year)
                    yearly_deg_time_AC = get_deg_time_data(gdp_year, era5_path)
                    exposure_factor = get_exposure_factor(gdp_data_year, yearly_deg_time_AC)

                    # Multiply degree time grid by population grid and gdp grid
                    weighted_cdd = yearly_degree_time * pop_data_year * exposure_factor
      
                    # Compute global average and store in dictionary    
                    # global_average = weighted_cdd.mean(dim=['latitude', 'longitude'], skipna=True)
                    global_average = weighted_cdd.sum(dim=['latitude', 'longitude'], skipna=True) / pop_data_year.sum(dim=['latitude', 'longitude'], skipna=True)
                    deg_time_dict["pop{0}_temp{1}_gdp{2}".format(pop_year, temp_year, gdp_year)] = global_average

                    # Plot the population weighted cooling degree time for 2015 for all effects separately
                    if pop_year == 2015 or temp_year == 2015 or gdp_year == 2015 or (pop_year == 2000 and temp_year == 2000 and gdp_year == 2000):
                        plot_map(weighted_cdd, pop_year, temp_year, gdp_year)

        # Accumulate data in dictionary                
        all_data_dict[case] = deg_time_dict

    # Plot the time curves for the different effects
    plot_time_curve(all_data_dict, all_years)


if __name__ == '__main__':
    main()
