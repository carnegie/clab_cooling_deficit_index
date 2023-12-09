import matplotlib.pyplot as plt
import geopandas as gpd
import country_converter as coco
import numpy as np
import pandas as pd
from utilities import exposure_new, read_gdp_data, gdp_from_cdd_exposure

def plot_exposure_map(ac_data_historical):
    # Plot exposure map
    # Load world geometries
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Copy your data
    ac_data_map = ac_data_historical.copy()
    # Convert ISO3 codes in your data to match the world dataframe
    cc = coco.CountryConverter()
    ac_data_map['ISO3'] = cc.convert(names=ac_data_map['ISO3'].values.tolist(), to='ISO3', not_found=None)
    # Merge your data with the world geometries
    merged_data = ac_data_map.merge(world[['geometry', 'iso_a3']], left_on='ISO3', right_on='iso_a3', how='left')
    # Convert the merged data to a GeoDataFrame
    ac_data_map_geo = gpd.GeoDataFrame(merged_data, geometry='geometry')

    ac_data_map_geo['exposure_calculated'] = exposure_new(ac_data_map_geo['GDP'], ac_data_map_geo['DD_mean'])
    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot the data
    ac_data_map_geo.plot(column='exposure_calculated', ax=ax, cmap='viridis', vmin=0, vmax=1)

    # Color bar same height as map
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.1)
    plt.colorbar(ax.collections[0], cax=cax, label='Exposure')
    
    # Remove axes ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig('Figures/exposure_map.png', dpi=300)

    return ac_data_map_geo

def plot_exposure_contour(configurations, exposure_function, ac_data, add_data=True, name_tag='exposure_contour'):
    """
    Conntour plot of penetration of air conditioning as a function of GDP per capita and cooling degree days
    """
    plt.figure()
    if not "NGA" in name_tag:
        cdd_x = np.linspace(100, 5000, 100)
        gdp_x = np.linspace(0, 200000, 100)
        cdd_x, gdp_x = np.meshgrid(cdd_x, gdp_x)
        contour_function = exposure_function(gdp_x, cdd_x)*cdd_x
        plt.xscale('log')
        
        plt.xlabel('GDP per capita in 2018 USD')
        plt.ylabel('Cooling degree days')
        color_map = 'YlOrRd'
    else:
        cdd_x = np.linspace(0, 0.5, 100)
        gdp_x = np.linspace(0, .05, 100)
        cdd_x, gdp_x = np.meshgrid(cdd_x, gdp_x)
        # Get GDP for NGA
        gdp_NGA = ac_data[ac_data['ISO3']=='NGA']['GDP'].values[0]
        cdd_NGA = ac_data[ac_data['ISO3']=='NGA']['DD_mean'].values[0]
        contour_function = exposure_function(gdp_NGA*((1+gdp_x)**(configurations['future_years'][-1]-configurations['ref_year'])), cdd_NGA*(1+cdd_x))*cdd_NGA*(1+cdd_x)

        plt.xlabel('Average GDP growth (annual %)')
        plt.ylabel('Cooling degree days increase by 2100 (%)')
        color_map = 'cool'
        gdp_x = gdp_x*100
        cdd_x = cdd_x*100
        plt.title('Nigeria')

    level_max = 5000.
    levels = np.linspace(0, level_max, 21)

    plt.contourf(gdp_x, cdd_x, contour_function, levels=levels, cmap=color_map)
    plt.colorbar(label='Exposure to outside temperatures multiplied by CDD', ticks=np.linspace(0, level_max, 11))

    # Add contour lines
    clines = plt.contour(gdp_x, cdd_x, contour_function, levels=levels, colors='k', linewidths=0.3)
    label_prec = '%d'
    plt.clabel(clines, levels[::2], fmt=label_prec, fontsize=8, colors='black')

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
        
        ac_data_scatter = pd.concat([ac_data_sel, ac_data_2100_sel]).reset_index(drop=True)

        scenario_colors = ['green', 'blue', 'red']
        # Draw lines between 2018 and 2100 for each country
        for country in countries_highest_pop:

            ac_data_sel_country = ac_data_scatter[ac_data_scatter['ISO3'] == country]
            for isc,scenario in enumerate(configurations['future_scenarios']):
                # Collect GDP and CDD values for line plot
                gdps, cdds = [ac_data_sel_country['GDP'].values[0]], [ac_data_sel_country['DD_mean'].values[0]]

                for year in configurations['future_years']:
                    
                    future_gdp = ac_data_sel_country['GDP_{0}_{1}'.format(scenario.split("_")[0].upper(), year)].values[0]
                    future_cdd = ac_data_sel_country['CDD_{0}_{1}'.format(scenario, year)].values[0]

                    plt.scatter(ac_data_sel_country['GDP_{0}_{1}'.format(scenario.split("_")[0].upper(), year)], ac_data_sel_country['CDD_{0}_{1}'.format(scenario, year)], 
                    c=exposure_function(ac_data_sel_country['GDP_{0}_{1}'.format(scenario.split("_")[0].upper(), year)], ac_data_sel_country['GDP_{0}_{1}'.format(scenario.split("_")[0].upper(), year)])*ac_data_sel_country['CDD_{0}_{1}'.format(scenario, year)], 
                    cmap='YlOrRd', vmin=0., vmax=level_max,  s=12, edgecolors=scenario_colors[isc], label=scenario)

                    gdps.append(future_gdp)
                    cdds.append(future_cdd)

                plt.plot(gdps, cdds, c=scenario_colors[isc], linewidth=0.5)
                # Label last point with future_years[-1]
                plt.annotate(configurations['future_years'][-1], (gdps[-1], cdds[-1]+10), fontsize=6, color=scenario_colors[isc])
        
        # Label points with country names
        for i, txt in enumerate(ac_data_scatter['ISO3'].values):
            plt.annotate(txt, (ac_data_scatter['GDP'][i]*1.05, ac_data_scatter['DD_mean'][i]-100), fontsize=8.5, color="purple")

        
        # Only show unique labels once
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper left')
        # plt.legend()

    plt.savefig('Figures/exposure_funct_analysis/{0}.png'.format(name_tag), dpi=300)


def plot_gdp_increase_map(configurations, gdp_cdd_data, geo_df, future_scenario):
    """
    Plot annual average GDP growth for historical and constant experienced CDD
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.15)
    # No axes ticks
    plt.xticks([])
    plt.yticks([])
    # Add title above first subplot
    fig.suptitle('Average annual GDP growth', fontsize=14)
    # Add labels a and b to subplots
    ax[0].text(0.01, 1.02, 'a Historical', transform=ax[0].transAxes, size=10, weight='bold')
    ax[1].text(0.01, 1.02, 'b To balance warming {0}'.format(future_scenario), transform=ax[1].transAxes, size=10, weight='bold')

    geo_df['gdp_const_{0}'.format(future_scenario)] = gdp_cdd_data['gdp_const_{0}'.format(future_scenario)] * 100.
    geo_df['gdp_historical_factor'] = gdp_cdd_data['gdp_historical_factor'] * 100.

    geo_df.plot(column='gdp_historical_factor', ax=ax[0], cmap='inferno_r', vmin=0, vmax=8)
    geo_df.plot(column='gdp_const_{0}'.format(future_scenario), ax=ax[1], cmap='inferno_r', vmin=0, vmax=8)
    
    # Add one shared colorbar which is half the height of the figure
    fig.colorbar(ax[1].collections[0], ax=ax, shrink=0.75, label='GDP growth (annual %)')

    plt.savefig('Figures/exposure_funct_analysis/gdp_const_{0}_map.png'.format(future_scenario), dpi=300)


def plot_gdp_increase_scatter(gdp_cdd_data, future_scenario):
    """
    Plot difference in exposure times CDD as a function of GDP per capita
    """
    plt.figure()
    ac_data_new_plot = gdp_cdd_data.reset_index()

    # Add continent column were country is in
    cc = coco.CountryConverter()
    ac_data_new_plot['continent'] = ac_data_new_plot['ISO3'].apply(lambda x: cc.convert(names=x, to='continent'))
    colors = ['red', 'green', 'blue', 'yellow', 'orange']

    for ic, continent in enumerate(['Africa', 'Asia', 'Europe', 'America', 'Oceania']):
        plt.scatter(ac_data_new_plot[ac_data_new_plot['continent'] == continent]['gdp_historical_factor']*100,
                    ac_data_new_plot[ac_data_new_plot['continent'] == continent]['gdp_const_{0}'.format(future_scenario)]*100,
                    label=continent, c=colors[ic], s=10, marker='o')
    ###
    # plt.scatter(ac_data_new_plot['gdp_historical_factor']*100,
    #             ac_data_new_plot['gdp_const_{0}'.format(future_scenario)]*100,
    #             c=ac_data_new_plot['diff_cdd'], cmap='viridis', s=10, marker='o')
    
    # # Add colorbar
    # plt.colorbar(label='Difference in exposure times CDD')
    ###
    
    plt.xlabel('Average historical GDP growth (annual %)')
    plt.ylabel('Average GDP growth for constant experienced CDD\nunder {0} (annual %)'.format(future_scenario))

    # Label each point with country name if it has a value that is not NaN
    count = 0
    for i, txt in enumerate(ac_data_new_plot['ISO3'].values):
        if not np.isnan(ac_data_new_plot['gdp_const_{0}'.format(future_scenario)][i]) and not np.isnan(ac_data_new_plot['gdp_historical_factor'][i]):
            plt.annotate(txt, (ac_data_new_plot['gdp_historical_factor'][i]*100, ac_data_new_plot['gdp_const_{0}'.format(future_scenario)][i]*100), fontsize=6)
            count += 1
    print('Number of countries with data: {0}'.format(count))

    # Dashed diagonal line
    min = np.min([np.min(ac_data_new_plot['gdp_historical_factor']*100), np.min(ac_data_new_plot['gdp_const_{0}'.format(future_scenario)]*100)])
    max = np.max([np.max(ac_data_new_plot['gdp_historical_factor']*100), np.max(ac_data_new_plot['gdp_const_{0}'.format(future_scenario)]*100)])
    plt.plot([min, max], [min, max], '--', c='grey', label='equal growth')

    # Legend with two columns
    plt.legend(ncol=2, fontsize=10)
    plt.savefig('Figures/exposure_funct_analysis/gdp_const_vs_historical_{0}.png'.format(future_scenario), dpi=300)