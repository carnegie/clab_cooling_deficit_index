import numpy as np
import pandas as pd
import logging
import country_converter as coco

def saturation(cdd, c):
    return (1. - np.exp(-c*cdd))

def exp_avail(gdp, a, b):
    return (1./(1. + np.exp(a)* np.exp(b*gdp)))

def power_avail(gdp, alpha, k):
    return (gdp**alpha / (gdp**alpha + k**alpha))

def exposure_combined_exponential(xdata, cdd_scale, alpha, k):
    cdd, gdp = xdata
    return (np.exp(-cdd_scale * ((cdd/1000.)**alpha * (gdp/100000.)**k)))

def exposure_function(xdata, avail_func, cdd_scale, alpha, k):
    """
    This function calculates the exposure of a country to outside temperature
    """
    cdd, gdp = xdata
    avail = avail_func(gdp, alpha, k)
    sat = saturation(cdd, cdd_scale)
    return (1 - avail*sat) 


def read_ac_data(infile, year=None, skip=0):
    """
    Read the air conditioning data from the given file
    """
    # Add AC column to merged_data
    ac_data = pd.read_csv(infile, skiprows=skip)
    # Name first column ISO3 and second column AC
    if year is not None:
        ac_data = format_eia_data(ac_data, year)
    else:
        ac_data = format_gdl_data(ac_data)
    # Rename country names to ISO3
    ac_data["ISO3"] = coco.convert(names=ac_data["country"].tolist(), to='ISO3')
    ac_data["country"] = coco.convert(names=ac_data["ISO3"].tolist(), to='name_short')
    # AC Given in percent
    ac_data["AC"] /= 100.
    return ac_data

def format_eia_data(ac_data, year):
    """
    Format the EIA data to match the format of the other data
    """
    ac_data = ac_data.rename(columns={"Unnamed: 0": "country", "Share of households with AC": "AC"})
    # Add year column
    ac_data["Year"] = year
    return ac_data

def format_gdl_data(ac_data):
    """
    Format the GDL data to match the format of the other data
    """
    # Drop columns popsize and numhh
    ac_data = ac_data.drop(columns=["popsize", "numhh"])
    # Remove empty rows
    ac_data = ac_data.dropna()
    # Rename column airco to AC
    ac_data = ac_data.rename(columns={"airco": "AC", "year": "Year"})
    # Year column as integer
    ac_data["Year"] = ac_data["Year"].astype(int)
    return ac_data

def add_gdp_cdd_data(ac_df, config):
    """
    Add GDP and CDD data to the ac_df dataframe
    """
    # Read the cooling degree days data
    cdd_data = read_cdd_data(config['cdd_historical_file'])
    # Read the GDP data
    gdp_data = read_gdp_data(config['gdp_historical_file'])

    # Add GDP and CDD data to the ac_df dataframe for the respective year and country
    ac_df = ac_df.merge(cdd_data, on=["ISO3", "Year"], how="left")
    ac_df = ac_df.merge(gdp_data, on=["ISO3", "Year"], how="left")
    return ac_df

def read_cdd_data(cdd_data_path):
    """
    Read in CDD data from file
    """
    cdd_data = pd.read_csv(cdd_data_path)
    # Drop unit column
    cdd_data = cdd_data.drop(columns=['Unit'])
    # Rename columns
    cdd_data = cdd_data.rename(columns={'Country': 'ISO3', 'Territory': 'Country', 'CDD18dailybypop': 'CDD', 'Date': 'Year'})
    # Round CDD to 5 decimal places
    cdd_data['CDD'] = cdd_data['CDD'].round(5)

    return cdd_data

def read_gdp_data(gdp_data_path):
    """
    Read in GDP data from file
    """
    # Open GDP PP file
    gdp_data = pd.read_csv(gdp_data_path, skiprows=4)

    # Drop columns
    gdp_data = gdp_data.drop(columns=['Indicator Name', 'Indicator Code', "Country Name", 'Unnamed: 67'])

    # Convert data to long format
    gdp_data = pd.melt(gdp_data, id_vars=["Country Code"], var_name='Year', value_name='GDP per capita PPP')
    
    # Rename Country Code to ISO3 and GDP per capita PPP to GDP
    gdp_data = gdp_data.rename(columns={'Country Code': 'ISO3','GDP per capita PPP': 'GDP'})

    # Convert year to integer
    gdp_data['Year'] = gdp_data['Year'].astype(int)

    return gdp_data

def fill_missing_country_gdp_data(start_year, data_frame, configs):
    """
    This function fills in missing data for a country, by finding the closest year with data
    for the historical reference year, we find the next latest year with data
    for the present day year, we find the next earliest year with data
    """
    # Loop through countries and find empty GDP data
    start_year = int(start_year)
    # Take later year if past year is missing, and earlier year if ref year is missing
    if start_year == configs['analysis_years']['past_year']:
        add_value = 1
    elif start_year == configs['analysis_years']['ref_year']:
        add_value = -1
    for country_name in data_frame['ISO3']:
        data_year = start_year
        if data_frame[data_frame['ISO3'] == country_name]['GDP'].isnull().values[0]:
            logging.info("No data for {0} in {1}".format(country_name, start_year))
            gdp_data_historical_country = None
            while gdp_data_historical_country == None:
                data_year += add_value
                # GDP data available from 1960 to 2022
                if data_year > 2022 or data_year < 1960:
                    logging.info("No data for {0} in any year".format(country_name))
                    break
                gdp_data_historical = read_gdp_data(configs['gdp_historical_file'])
                gdp_data_historical = gdp_data_historical[gdp_data_historical['Year'] == data_year]
                if not gdp_data_historical[gdp_data_historical['ISO3'] == country_name]['GDP'].isnull().values[0]:
                    gdp_data_historical_country = gdp_data_historical[gdp_data_historical['ISO3'] == country_name]['GDP'].values[0]
                    new_data_year = data_year
                    logging.info("Found data for {0} in {1}".format(country_name, data_year))
                else:
                    new_data_year = start_year
            # Add GDP data to gdp_data
            data_frame.loc[data_frame['ISO3'] == country_name, 'GDP'] = gdp_data_historical_country
            data_frame.loc[data_frame['ISO3'] == country_name, 'Year'] = int(new_data_year)

    return data_frame


def read_projections(configurations, data_type, isin_df):
    """
    Read in projections from file
    """
    projection_df_all = pd.read_csv(configurations['{0}_projections_file'.format(data_type)])

    # Select scenarios
    output_df = pd.DataFrame()
    for scenario in configurations['future_scenarios']:
        year = configurations['analysis_years']['future_year']
           
        ssp = scenario.split('_')[0].upper()
        rcp = scenario.split('_')[1]

        projection_df = projection_df_all[projection_df_all['year'] == year]

        if data_type == 'cdd':
            projection_df = projection_df[(projection_df['ssp'] == ssp) & (projection_df['rcp'] == rcp) & (projection_df['stat'] == 'PopWeightAv')]
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


def calculate_average_gdp_growth(gdp_year_n, gdp_year_0, n_years):
    """
    This function calculates the average annual GDP growth
    """
    return (gdp_year_n / gdp_year_0) ** (1./n_years) - 1


def gdp_from_cdd_exposure(exposure_cdd, cdd, loaded_parameters, av_func_type):
    """
    This function calculates the GDP per capita assuming fixed exposure * cooling degree days
    """
    cdd_scale, alpha, k = loaded_parameters['cdd_scale'], loaded_parameters['alpha'], loaded_parameters['k']
    if av_func_type == 'power_law':
        gdp_const = ((k**alpha) * (cdd-exposure_cdd)/ (saturation(cdd,cdd_scale)*cdd - cdd + exposure_cdd))**(1./alpha)
    elif av_func_type == 'exponential':
        gdp_const = np.log((1./np.exp(alpha))*((saturation(cdd,cdd_scale)/(1 - exposure_cdd/cdd)) - 1))/k
    elif av_func_type == 'combined_exponential':
        gdp_const = ( (-1./(cdd_scale*(cdd/1000.)**alpha)) * np.log(exposure_cdd/cdd) )**(1./k) * 100000.
    else:
        logging.error("Invalid availability function type")
    return gdp_const
    


def calculate_gdp_const(df, configurations, parameters):
    """
    This function calculates the GDP to keep heat exposure constant
    """
    for scenario in configurations['future_scenarios']:
        gdp_const = gdp_from_cdd_exposure(df['exposure_times_cdd'], 
                    df['CDD_{0}_{1}'.format(scenario, configurations['analysis_years']['future_year'])], 
                    parameters, configurations['availability_function'])
        # Replace inf with NaN
        logging.info("Infinite GDP calculated for {0}, possibly 0 CDD".format(gdp_const[gdp_const == np.inf].index))
        gdp_const[gdp_const == np.inf] = np.nan
        df['gdp_const_{}'.format(scenario)] = calculate_average_gdp_growth(gdp_const, df['GDP'], 
                configurations['analysis_years']['future_year'] - df['Year_ref'])
    return df