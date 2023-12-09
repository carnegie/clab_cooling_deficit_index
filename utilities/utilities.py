import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import country_converter as coco
import seaborn as sns


def read_ac_data(data_file):
    """
    Read AC data from file created with derive_exposure_functions.ipynb
    """
    ac_data = pd.read_csv(data_file)
    # Remove rows with missing data
    # ac_data = ac_data.dropna()
    # Reindex
    ac_data = ac_data.reset_index(drop=True)
    return ac_data

def read_gdp_data(year):
    """
    Read in GDP data from file
    """
    # Open GDP PP file
    # from https://data.worldbank.org/indicator/NY.GDP.PCAP.PP.KD
    # Purchasing power parity
    # gdp_data = pd.read_csv('data_experiencedT/API_NY.GDP.PCAP.PP.KD_DS2_en_csv_v2_5873868.csv', skiprows=4)
    # GDP in current $
    gdp_data = pd.read_csv('data_experiencedT/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_5871588.csv')
    # Drop all columns except Country Code and year
    gdp_data = gdp_data[['Country Code', year]]
    # Rename Country Code to ISO3
    gdp_data = gdp_data.rename(columns={'Country Code': 'ISO3'})
    # Make ISO3 as index
    gdp_data = gdp_data.set_index('ISO3')
    # Rename year to GDP
    gdp_data = gdp_data.rename(columns={year: 'GDP'})
    # Convert from 2017$ to 2018$
    gdp_data['GDP'] = gdp_data['GDP'] * 1.02

    return gdp_data

def read_projections(configurations, data_type):
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
                projection_df = projection_df[(projection_df_all['ssp'] == ssp) & (projection_df['rcp'] == rcp) & (projection_df['stat'] == 'mean')]
                projection_df = projection_df[['ISO', 'value']]
                projection_df = projection_df.rename(columns={'ISO': 'ISO3', 'value': 'CDD_{0}_{1}'.format(scenario, year)})
            elif data_type == 'gdp':
                projection_df = projection_df[projection_df_all['scenario'] == ssp]
                projection_df = projection_df[['countrycode', 'gdppc']]
                projection_df = projection_df.rename(columns={'countrycode': 'ISO3', 'gdppc': 'GDP_{0}_{1}'.format(ssp, year)})

            if output_df.empty:
                output_df = projection_df
            else:
                output_df = pd.merge(output_df, projection_df, on='ISO3', how='outer')
        
    return output_df


def exposure_new(gdp, cdd):
    """
    This function calculates the exposure of a country to climate change
    """
    avail_new = (1/(1 + np.exp(3.2)* np.exp(-0.00011*gdp)))
    saturation_new = (0.93 - 0.93*np.exp(-0.005*cdd))
    return (1 - avail_new*saturation_new)


def gdp_from_cdd_exposure(exposure_cdd, cdd):
    """
    This function calculates the GDP per capita assuming fixed exposure * cooling degree days
    """
    sat = (0.93 - 0.93*np.exp(-0.005*cdd))
    sat.index = exposure_cdd.index
    cdd.index = exposure_cdd.index
    return (np.log((1./np.exp(3.2))*((sat/(1 - exposure_cdd/cdd)) - 1))/(-0.00011))

def calculate_average_gdp_growth(gdp_year_n, gdp_year_0, n_years):
    """
    This function calculates the average annual GDP growth
    """
    return (gdp_year_n / gdp_year_0) ** (1./n_years) - 1


def add_historical_gdp_growth(gdp_cdd_data):
    """
    This function adds GDP growth to the df, historical and future
    """

    # Add historic GDP growth
    gdp_data = read_gdp_data('1980')
    gdp_data = gdp_data.rename(columns={'GDP': 'GDP_1980'})
    # Merge with AC data
    gdp_cdd_data = pd.merge(gdp_cdd_data, gdp_data, left_on='ISO3', right_index=True)
    # Average annual GDP growth
    gdp_cdd_data['gdp_historical_factor'] = calculate_average_gdp_growth(gdp_cdd_data['GDP'], gdp_cdd_data['GDP_1980'], 2018 - 1980)
    return gdp_cdd_data
