import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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



def exposure_contour(exposure_function, ac_data, multiply_cdd=False, add_data=True, contour_lines=False, name_tag='exposure_contour', future_scenario=''):
    """
    Conntour plot of penetration of air conditioning as a function of GDP per capita and cooling degree days
    """
    plt.figure()
    cdd_x = np.linspace(0, 4500, 100)
    gdp_x = np.linspace(0, 200000, 100)
    if multiply_cdd:
        level_max = 4500.
        add_label = ' multiplied by CDD'
        z = (1.-ac_data['AC'])*ac_data['DD_mean']
        name_tag += '_multipliedCDD'
    else:
        level_max = 1.
        add_label = ''
        z = (1.-ac_data['AC'])
    levels = np.linspace(0, level_max, 21)
    cdd_x, gdp_x = np.meshgrid(cdd_x, gdp_x)
    if multiply_cdd:
        plt.contourf(gdp_x, cdd_x, exposure_function(gdp_x, cdd_x)*cdd_x, levels=levels, cmap='YlOrRd')
    else:
        plt.contourf(gdp_x, cdd_x, exposure_function(gdp_x, cdd_x), levels=levels, cmap='YlOrRd')
    plt.colorbar(label='Exposure to outside temperatures{0}'.format(add_label), ticks=np.linspace(0, level_max, 11))

    if contour_lines:
        # Add contour lines
        if multiply_cdd:
            clines = plt.contour(gdp_x, cdd_x, exposure_function(gdp_x, cdd_x)*cdd_x, levels=levels, colors='k', linewidths=0.)
            label_prec = '%d'
        else:
            clines = plt.contour(gdp_x, cdd_x, exposure_function(gdp_x, cdd_x), levels=levels, colors='k', linewidths=0.)
            label_prec = '%.2f'
        plt.clabel(clines, levels[::2], fmt=label_prec, fontsize=8, colors='black')

    plt.xlabel('GDP per capita in 2018 USD')
    # GDP log scale

    plt.xscale('log')
    # plt.xlim(1500, 100000) 
    plt.ylabel('Cooling degree days')
    # color bar range is 0 to 1
    plt.clim(0, level_max)
    # Add label in red and bold
    plt.title('Exposure to outside temperatures{0}\n as a function of GDP and CDD{1}'.format(add_label, future_scenario))
    # if "new" in name_tag:
    #     plt.text(0.02, 0.02, 'New exposure function', color='red', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
     
    if add_data:
        # Overlay AC data
        # Plot AC access as a function of GDP per capita and cooling degree days
        ac_data_2100 = ac_data[ac_data['ISO3'].str.contains('2100')]

        plt.scatter(ac_data['GDP'], ac_data['DD_mean'], c=z, cmap='YlOrRd', label='AC access', vmin=0., vmax=level_max)
        plt.scatter(ac_data_2100['GDP'], ac_data_2100['DD_mean'], c='orange', s=12)
        
        # Label points with country names
        ac_data_scatter = ac_data.reset_index(drop=True)
        for i, txt in enumerate(ac_data_scatter['ISO3'].values):
            if i in z.index and not np.isnan(z[i]):
                if multiply_cdd:
                    z_label = str(int(z[i]))
                else:
                    z_label = str(round(z[i],2))
            else:
                z_label = ''

            if not '2100' in txt:
                color = 'black'
            else:
                color = 'orange'
            plt.annotate(txt+"\n"+z_label, (ac_data_scatter['GDP'][i]*1.05, ac_data_scatter['DD_mean'][i]-100), fontsize=8.5, color=color)

    plt.savefig('Figures/exposure_funct_analysis/{0}.png'.format(name_tag), dpi=300)


def plot_gdp_const_warming(gdp_cdd_data, geo_df, ssp_scenario, rcp_scenario):
    """
    Plot difference in exposure times CDD as a function of GDP per capita
    """
    # Add historic GDP growth
    gdp_data = read_gdp_data('1980')
    gdp_data = gdp_data.rename(columns={'GDP': 'GDP_1980'})
    # Merge with AC data
    gdp_cdd_data = pd.merge(gdp_cdd_data, gdp_data, left_on='ISO3', right_index=True)
    gdp_cdd_data['gdp_historical_factor'] = gdp_cdd_data['GDP']/gdp_cdd_data['GDP_1980']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    geo_df['gdp_development'] = gdp_cdd_data['gdp_const']/gdp_cdd_data['gdp_historical_factor']
    geo_df.plot(column='gdp_development', ax=ax, legend=True, cmap='inferno', vmin=0, vmax=5, legend_kwds={'label': 'GDP increase relative to historical'})
    plt.savefig('Figures/exposure_funct_analysis/gdp_const_ssp_{0}_rcp_{1}_map.png'.format(ssp_scenario, rcp_scenario), dpi=300)


    # New figure
    plt.figure()
    # Make a selection of countries
    countries = ['CHN', 'MEX', 'BRA', 'SAU', 'IND', 'IDN', 'ZAF', 'USA', 'KOR', 'JPN', 'GER', 'EGY', 'KEN', 'NGA', 'RUS', 'ZMB', 'ISL']
    ac_data_new_plot = gdp_cdd_data[gdp_cdd_data['ISO3'].isin(countries)]
    ac_data_new_plot = ac_data_new_plot.reset_index()
    
    
    plt.scatter(ac_data_new_plot['diff_cdd'], ac_data_new_plot['gdp_const']/ac_data_new_plot['gdp_historical_factor'], label='SSP{0} RCP {1}, 2018-2100'.format(ssp_scenario, int(rcp_scenario)/10))
    plt.xlabel('CDD 2100 - CDD 2018'.format(ssp_scenario, int(rcp_scenario)/10))
    # plt.ylabel(r'$\frac{ \mathrm{GDP \ needed \ to \ keep \ exposure \ times \ CDD \ constant}}{\mathrm{2018 \ GDP}}$')
    plt.ylabel('GDP increase needed to keep exposure times CDD constant 2018-2100\n divided by historical GDP increase 1980-2018')

    # plt.scatter(ac_data_new_plot['diff_cdd'], ac_data_new_plot['gdp_historical_factor'], c='cyan', label='Historic GDP growth, 1980-2018')


    # Label each point with country name
    gdp_dev = ac_data_new_plot['gdp_const']/ac_data_new_plot['gdp_historical_factor']
    for i, txt in enumerate(ac_data_new_plot['ISO3'].values):
        plt.annotate(txt, (ac_data_new_plot['diff_cdd'][i]+0.5, gdp_dev[i]), fontsize=8)
        # plt.annotate(txt, (ac_data_new_plot['diff_cdd'][i]+10, ac_data_new_plot['gdp_historical_factor'][i]), fontsize=8)

    # Horizontal dashed line at 1
    plt.axhline(y=1, color='k', linestyle='--')

    # plt.legend()
    plt.savefig('Figures/exposure_funct_analysis/gdp_const_ssp_{0}_rcp_{1}_vsCDDdiff.png'.format(ssp_scenario, rcp_scenario), dpi=300)