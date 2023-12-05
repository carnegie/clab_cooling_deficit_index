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
    cdd_x = np.linspace(0, 5000, 100)
    gdp_x = np.linspace(0, 200000, 100)
    if multiply_cdd:
        level_max = 5000.
        add_label = ' multiplied by CDD'
        # z = (1.-ac_data['AC'])*ac_data['DD_mean']
        z = exposure_function(ac_data['GDP'], ac_data['DD_mean'])*ac_data['DD_mean']
        name_tag += '_multipliedCDD'
    else:
        level_max = 1.
        add_label = ''
        # z = (1.-ac_data['AC'])
        z = exposure_function(ac_data['GDP'], ac_data['DD_mean'])
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
            clines = plt.contour(gdp_x, cdd_x, exposure_function(gdp_x, cdd_x)*cdd_x, levels=levels, colors='k', linewidths=0.3)
            label_prec = '%d'
        else:
            clines = plt.contour(gdp_x, cdd_x, exposure_function(gdp_x, cdd_x), levels=levels, colors='k', linewidths=0.2)
            label_prec = '%.2f'
        plt.clabel(clines, levels[::2], fmt=label_prec, fontsize=8, colors='black')

    plt.xlabel('GDP per capita in 2018 USD')
    # GDP log scale

    plt.xscale('log')
    # plt.xlim(1500, 100000) 
    plt.ylabel('Cooling degree days')
    # color bar range is 0 to 1
    plt.clim(0, level_max)
 
    if add_data:
        # Plot AC access as a function of GDP per capita and cooling degree days for some countries
        countries_highest_pop = ['CHN', 'IND', 'USA', 'IDN', 'PAK', 'BRA', 'NGA']
        ac_data_2100 = ac_data[ac_data['ISO3'].str.contains('2100')]
        ac_data_sel = ac_data[ac_data['ISO3'].isin(countries_highest_pop)]
        ac_data_2100_sel = ac_data_2100[ac_data_2100['ISO3'].isin([country + '_2100' for country in countries_highest_pop])]

        # Use marker that is not filled and black edge
        plt.scatter(ac_data_sel['GDP'], ac_data_sel['DD_mean'], c=exposure_function(ac_data_sel['GDP'], ac_data_sel['DD_mean'])*ac_data_sel['DD_mean'], 
                    cmap='YlOrRd', vmin=0., vmax=level_max, s=12, edgecolors='purple')
        plt.scatter(ac_data_2100_sel['GDP'], ac_data_2100_sel['DD_mean'], c=exposure_function(ac_data_2100_sel['GDP'], ac_data_2100_sel['DD_mean'])*ac_data_2100_sel['DD_mean'], 
                    cmap='YlOrRd', vmin=0., vmax=level_max,  s=12, edgecolors='green', label='SSP3, RCP4.5')
        
        ac_data_scatter = pd.concat([ac_data_sel, ac_data_2100_sel]).reset_index(drop=True)
        # Draw lines between 2018 and 2100 for each country
        for country in countries_highest_pop:
            plt.plot([ac_data_scatter[ac_data_scatter['ISO3'] == country]['GDP'].values[0], ac_data_scatter[ac_data_scatter['ISO3'] == country + '_2100']['GDP'].values[0]], 
                    [ac_data_scatter[ac_data_scatter['ISO3'] == country]['DD_mean'].values[0], ac_data_scatter[ac_data_scatter['ISO3'] == country + '_2100']['DD_mean'].values[0]], 
                    c='green', linewidth=0.5)
        
        # Label points with country names
        for i, txt in enumerate(ac_data_scatter['ISO3'].values):
            if i in z.index and not np.isnan(z[i]):
                if multiply_cdd:
                    z_label = str(int(z[i]))
                else:
                    z_label = str(round(z[i],2))
            else:
                z_label = ''

            if not '2100' in txt:
                color = 'purple'
            else:
                color = 'green'
                txt = txt.split('_')[0] + '\n' + txt.split('_')[1]
            plt.annotate(txt, (ac_data_scatter['GDP'][i]*1.05, ac_data_scatter['DD_mean'][i]-100), fontsize=8.5, color=color)

        plt.legend()

    plt.savefig('Figures/exposure_funct_analysis/{0}.png'.format(name_tag), dpi=300)


def plot_gdp_increase_map(gdp_cdd_data, geo_df, ssp_scenario, rcp_scenario):
    """
    Plot difference in exposure times CDD as a function of GDP per capita
    """
    # Add historic GDP growth
    gdp_data = read_gdp_data('1980')
    gdp_data = gdp_data.rename(columns={'GDP': 'GDP_1980'})
    # Merge with AC data
    gdp_cdd_data = pd.merge(gdp_cdd_data, gdp_data, left_on='ISO3', right_index=True)
    # Average annual GDP growth
    gdp_cdd_data['gdp_historical_factor'] =  (gdp_cdd_data['GDP'] / gdp_cdd_data['GDP_1980']) ** (1./(2018 - 1980)) - 1 
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.15)
    # No axes ticks
    plt.xticks([])
    plt.yticks([])
    # Add title above first subplot
    ax[0].set_title('Average annual GDP growth')
    # Add white space after title
    ax[0].title.set_position([.5, 1.5])
    # Add labels a and b to subplots
    ax[0].text(0.01, .9, 'a', transform=ax[0].transAxes, size=15, weight='bold')
    ax[1].text(0.01, .88, 'b', transform=ax[1].transAxes, size=15, weight='bold')

    geo_df['gdp_const'] = gdp_cdd_data['gdp_const'] * 100.
    geo_df['gdp_historical_factor'] = gdp_cdd_data['gdp_historical_factor'] * 100.
    # color bar in log scale

    geo_df.plot(column='gdp_historical_factor', ax=ax[0], legend=True, cmap='inferno_r', 
            vmin=0, vmax=8,
            legend_kwds={'label': 'historical (%)'})
    geo_df.plot(column='gdp_const', ax=ax[1], legend=True, cmap='inferno_r',
            vmin=0, vmax=8,
            legend_kwds={'label': 'to balance warming (%)'})



    plt.savefig('Figures/exposure_funct_analysis/gdp_const_ssp_{0}_rcp_{1}_map.png'.format(ssp_scenario, rcp_scenario), dpi=300)
    return gdp_cdd_data

def plot_gdp_increase_scatter(gdp_cdd_data, ssp_scenario, rcp_scenario):
    """
    Plot difference in exposure times CDD as a function of GDP per capita
    """
    plt.figure()
    ac_data_new_plot = gdp_cdd_data.reset_index()

    # Add continent column were country is in
    cc = coco.CountryConverter()
    ac_data_new_plot['continent'] = ac_data_new_plot['ISO3'].apply(lambda x: cc.convert(names=x, to='continent'))
    # Assign a number for each continent in ac_data_new_plot['continent']
    continent_code = {'Africa': 0, 'Asia': 1, 'Europe': 2, 'America': 3, 'Oceania': 4}    
    ac_data_new_plot['continent'] = ac_data_new_plot['continent'].apply(lambda x: continent_code[x])
    color_map = mcolors.ListedColormap(['red', 'green', 'blue', 'yellow', 'orange'])

    plt.scatter(ac_data_new_plot['gdp_historical_factor']*100, ac_data_new_plot['gdp_const']*100, 
                label='SSP{0} RCP {1}, 2018-2100'.format(ssp_scenario, int(rcp_scenario)/10), s=9, 
                c=ac_data_new_plot['continent'], cmap=color_map)
    # Add color bar with label
    plt.colorbar(label='Continent')
    
    plt.xlabel('Average historical GDP growth (annual %)')
    plt.ylabel('Average GDP growth for constant experienced CDD (annual %)')

    # Label each point with country name if it has a value that is not NaN
    for i, txt in enumerate(ac_data_new_plot['ISO3'].values):
        if not np.isnan(ac_data_new_plot['gdp_const'][i]) and not np.isnan(ac_data_new_plot['gdp_historical_factor'][i]):
            plt.annotate(txt, (ac_data_new_plot['gdp_historical_factor'][i]*100, ac_data_new_plot['gdp_const'][i]*100), fontsize=6)

    # Dashed diagonal line
    plt.plot([np.min(ac_data_new_plot['gdp_historical_factor']*100), np.max(ac_data_new_plot['gdp_historical_factor']*100)], 
             [np.min(ac_data_new_plot['gdp_const']*100), np.max(ac_data_new_plot['gdp_const']*100)],
             '--', c='grey', label='No change in GDP')

    # plt.legend()
    plt.savefig('Figures/exposure_funct_analysis/gdp_const_ssp_{0}_rcp_{1}_vsCDDdiff.png'.format(ssp_scenario, rcp_scenario), dpi=300)