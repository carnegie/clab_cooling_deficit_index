import numpy as np
import pandas as pd
import logging

def read_ac_data(data_file):
    """
    Read AC data from file created with derive_exposure_functions.ipynb
    """
    ac_data = pd.read_csv(data_file)
    # Reindex
    ac_data = ac_data.reset_index(drop=True)
    return ac_data

def read_cdd_data(year, cdd_data_path):
    dd_data = pd.read_csv(cdd_data_path)

    # For all countries get the Year 2018 and DD_type CDD18
    dd_data = dd_data.loc[(dd_data['Year'] == year) & (dd_data['DD_type'] == 'CDD18')]
    # Drop all columns except ISO3 and DD_mean
    dd_data = dd_data[['ISO3', 'DD_mean']]
    #Make ISO3 as index
    dd_data = dd_data.set_index('ISO3')

    return dd_data

def read_gdp_data(year, gdp_data_path):
    """
    Read in GDP data from file
    """
    # Open GDP PP file
    # GDP in current $
    gdp_data = pd.read_csv(gdp_data_path, skiprows=4)
    # Drop all columns except Country Code and year
    gdp_data = gdp_data[['Country Code', year]]
    # Rename Country Code to ISO3
    gdp_data = gdp_data.rename(columns={'Country Code': 'ISO3'})
    # Rename year to GDP
    gdp_data = gdp_data.rename(columns={year: 'GDP'})
    # Convert from 2017$ to 2018$
    gdp_data['GDP'] = gdp_data['GDP'] * 1.02

    return gdp_data

def read_projections(configurations, data_type, isin_df):
    """
    Read in projections from file
    """
    projection_df_all = pd.read_csv(configurations['{0}_projections_file'.format(data_type)])

    # Select scenarios
    output_df = pd.DataFrame()
    for scenario in configurations['future_scenarios']:
        for year in configurations['future_years']:
           
            ssp = scenario.split('_')[0].upper()
            rcp = scenario.split('_')[1]

            projection_df = projection_df_all[projection_df_all['year'] == year]

            if data_type == 'cdd':
                projection_df = projection_df[(projection_df['ssp'] == ssp) & (projection_df['rcp'] == rcp) & (projection_df['stat'] == 'mean')]
                projection_df = projection_df[['ISO', 'value']]
                projection_df = projection_df.rename(columns={'ISO': 'ISO3', 'value': 'CDD_{0}_{1}'.format(scenario, year)})
            elif data_type == 'gdp':
                projection_df = projection_df[projection_df['scenario'] == ssp]
                projection_df = projection_df[['countrycode', 'gdppc']]
                projection_df = projection_df.rename(columns={'countrycode': 'ISO3', 'gdppc': 'GDP_{0}_{1}'.format(ssp, year)})

            if output_df.empty:
                output_df = projection_df
            else:
                output_df = pd.merge(output_df, projection_df, on='ISO3', how='outer')
    
    output_df = output_df[output_df['ISO3'].isin(isin_df['ISO3'])]

    return output_df

def saturation(cdd, a, b, c):
    return (a - b*np.exp(-c*cdd))

def availability(gdp, a, b):
    return (1/(1 + np.exp(a)* np.exp(b*gdp)))

def exposure_function(gdp, a_avail, b_avail, cdd, a_sat, b_sat, c_sat):
    """
    This function calculates the exposure of a country to outside temperature
    """
    avail = availability(gdp, a_avail, b_avail)
    sat = saturation(cdd, a_sat, b_sat, c_sat)
    return (1 - avail*sat) 

def gdp_from_cdd_exposure(exposure_cdd, cdd, loaded_parameters):
    """
    This function calculates the GDP per capita assuming fixed exposure * cooling degree days
    """
    sat = saturation(cdd, loaded_parameters['sat_a'], loaded_parameters['sat_b'], loaded_parameters['sat_c'])
    exp_overcdd = exposure_cdd[exposure_cdd.index == 62]/cdd[cdd.index == 62]
    return (np.log((1./np.exp(loaded_parameters['av_a']))*((sat/(1 - exposure_cdd/cdd)) - 1))/(loaded_parameters['av_b']))

def calculate_average_gdp_growth(gdp_year_n, gdp_year_0, n_years):
    """
    This function calculates the average annual GDP growth
    """
    return (gdp_year_n / gdp_year_0) ** (1./n_years) - 1

def fill_missing_country_gdp_data(start_year, data_frame, configs):
    """
    This function fills in missing data for a country, by finding the closest year with data
    for the historical reference year, we find the next latest year with data
    for the present day year, we find the next earliest year with data
    """
    # Loop through countries and find empty GDP data
    start_year = int(start_year)
    if start_year == configs['past_year']:
        add_value = 1
    elif start_year == configs['ref_year']:
        add_value = -1
    for country_name in data_frame['ISO3']:
        data_year = start_year
        if data_frame[data_frame['ISO3'] == country_name]['GDP'].isnull().values.any():
            logging.info("No data for {0} in {1}".format(country_name, start_year))
            gdp_data_historical_country = None
            while gdp_data_historical_country == None:
                data_year += add_value
                # GDP data available from 1960 to 2022
                if data_year > 2022 or data_year < 1960:
                    logging.info("No data for {0} in any year".format(country_name))
                    break
                gdp_data_historical = read_gdp_data(str(data_year), configs['gdp_historical_file'])
                if not gdp_data_historical[gdp_data_historical['ISO3'] == country_name]['GDP'].isnull().values.any():
                    gdp_data_historical_country = gdp_data_historical[gdp_data_historical['ISO3'] == country_name]['GDP'].values[0]
                    logging.info("Found data for {0} in {1}".format(country_name, data_year))
            # Add GDP data to gdp_data
            data_frame.loc[data_frame['ISO3'] == country_name, 'GDP'] = gdp_data_historical_country 

    return data_frame

def add_historical_gdp_growth(gdp_cdd_data, configurations):
    """
    This function adds historical GDP growth to the df
    """
    # Set loggin level from configurations
    logging.basicConfig(level=configurations['logging_level'])

    # Add historic GDP growth
    gdp_data = read_gdp_data(str(configurations['past_year']), configurations['gdp_historical_file'])
    #Show all rows
    pd.set_option('display.max_rows', None)

    # Get GDP from next year until there is data   
    gdp_data = fill_missing_country_gdp_data(configurations['past_year'], gdp_data, configurations)  

    # Print countries that still have no data
    logging.info("Countries with still no GDP data:")
    logging.info(gdp_data[gdp_data['GDP'].isnull()]['ISO3'].unique())

    # Rename GDP column
    gdp_data = gdp_data.rename(columns={'GDP': 'GDP_historical_ref_year'})
    # Merge with AC data
    gdp_cdd_data = pd.merge(gdp_cdd_data, gdp_data, on='ISO3', how='outer')

    # Average annual GDP growth
    gdp_cdd_data['gdp_historical_factor'] = calculate_average_gdp_growth(gdp_cdd_data['GDP'], gdp_cdd_data['GDP_historical_ref_year'], configurations['ref_year'] - configurations['past_year'])
    return gdp_cdd_data
