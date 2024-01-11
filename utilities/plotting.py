import os
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
import country_converter as coco
import numpy as np
import pandas as pd
from utilities.utilities import exposure_new, read_gdp_data, gdp_from_cdd_exposure

def plot_exposure_map(ac_data_historical):
    """
    Plot exposure map
    """

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

    plt.savefig('Figures/paper/exposure_map.png', dpi=300)

    return ac_data_map_geo


class ContourPlot:
    def __init__(self, configurations, ac_data, name_tag, country=None):
        self.configurations = configurations
        self.ac_data = ac_data
        self.name_tag = name_tag
        self.country = country

        matplotlib.rcParams['font.family'] = 'Helvetica'

    def contour_grid(self, cdd_range, gdp_range):
        """
        Create grid for contour plot
        """
        cdd_x = np.linspace(cdd_range[0], cdd_range[1], cdd_range[2])
        gdp_x = np.linspace(gdp_range[0], gdp_range[1], gdp_range[2])
        self.cdd_x, self.gdp_x = np.meshgrid(cdd_x, gdp_x)

    def calculate_contour(self, exposure_function):
        """
        Calculate contour function
        """
        if self.country == None:
            self.contour_function = exposure_function(self.gdp_x, self.cdd_x)*self.cdd_x
        else:
            self.gdp_country = self.ac_data[self.ac_data['ISO3']==self.country]['GDP'].values[0]
            self.cdd_country = self.ac_data[self.ac_data['ISO3']==self.country]['DD_mean'].values[0]
            self.contour_function = exposure_function(self.gdp_country*((1+self.gdp_x/100.)**(self.configurations['future_years'][-1]-self.configurations['ref_year'])), self.cdd_country*(1+self.cdd_x/100.))*self.cdd_country*(1+self.cdd_x/100.)
    
    def plot_contour(self):
        """
        Contour plot of exposure times CDD as a function of GDP per capita and cooling degree days
        """
        plt.figure()
        self.level_max = 5000.
        self.levels = np.linspace(0, self.level_max, 21)
        color_map = 'YlOrRd'

        plt.contourf(self.gdp_x, self.cdd_x, self.contour_function, levels=self.levels, cmap=color_map)
        plt.colorbar(label='Exposure times CDD [°C days]', ticks=np.linspace(0, self.level_max, 11))

    def set_x_log(self):
        """
        Set x-axis to log scale
        """
        plt.xscale('log')
        
    def add_contour_lines(self):
        """
        Add contour lines
        """
        clines = plt.contour(self.gdp_x, self.cdd_x, self.contour_function, levels=self.levels, colors='k', linewidths=0.3)
        plt.clabel(clines, self.levels[::2], fmt='%d', fontsize=8, colors='black')
        plt.clim(0, self.levels[-1])

    def add_data(self, exposure_function):
        """
        Add data points
        """           
        # Collect data points
        for country in self.configurations['countries_highest_pop']:
            gdps, cdds, exposures_times_cdd = [], [], [] 
            if self.country != None and country!=self.country: continue

            ac_data_sel_country = self.ac_data[self.ac_data['ISO3'] == country]

            # Add historical data point "today"
            if self.country == None:
                gdps.append(ac_data_sel_country['GDP'].values[0])
                cdds.append(ac_data_sel_country['DD_mean'].values[0])
                exposures_times_cdd.append(ac_data_sel_country['exposure_times_cdd'].values[0])

            else:
                gdps.append((ac_data_sel_country['GDP']-self.gdp_country)/self.gdp_country)
                cdds.append((ac_data_sel_country['DD_mean']-self.cdd_country)/self.cdd_country)
                exposures_times_cdd.append(ac_data_sel_country['exposure_times_cdd'].values[0])
            plt.annotate(self.configurations['ref_year'], (gdps[0], cdds[0]), fontsize=6, color=self.configurations['scenario_colors'][1], rotation=80)

            # Add future data points
            for year in self.configurations['future_years']:
                future_gdp = ac_data_sel_country['GDP_{0}_{1}'.format(self.configurations['base_future_scenario'].split("_")[0].upper(), year)]
                future_cdd = ac_data_sel_country['CDD_{0}_{1}'.format(self.configurations['base_future_scenario'], year)]
                if self.country == None:
                    gdps.append(future_gdp.values[0])
                    cdds.append(future_cdd.values[0])
                else:
                    gdps.append(((future_gdp/self.gdp_country)**(1./(self.configurations['future_years'][-1]-self.configurations['ref_year']))-1).values[0]*100.)
                    cdds.append(((future_cdd-self.cdd_country)/self.cdd_country).values[0]*100.)    
                
                exposures_times_cdd.append(exposure_function(future_gdp, future_cdd)*future_cdd)

            plt.scatter(gdps, cdds, c=exposures_times_cdd,
                    cmap='YlOrRd', vmin=0., vmax=self.level_max,  s=12, edgecolors=self.configurations['scenario_colors'][1],
                      label=self.configurations['base_future_scenario'])  
            
            # Label last point with year, tilt by 45 degrees
            plt.annotate(self.configurations['future_years'][-1], (gdps[-1], cdds[-1]), fontsize=6, 
                         color=self.configurations['scenario_colors'][1], rotation=80) 
            
            # Connect points with line
            plt.plot(gdps, cdds, c='blue', linewidth=0.5)          



def plot_exposure_contour(configurations, exposure_function, ac_data, x_y_ranges, country=None, name_tag='exposure_contour'):
    """
    Contour plot of penetration of air conditioning as a function of GDP per capita and cooling degree days
    """
    contour_plot = ContourPlot(configurations, ac_data, name_tag, country=country)
    contour_plot.contour_grid(x_y_ranges[0], x_y_ranges[1])
    contour_plot.calculate_contour(exposure_function)
    contour_plot.plot_contour()
    contour_plot.add_contour_lines()
    contour_plot.add_data(exposure_function)

    if not "country" in name_tag:
        contour_plot.set_x_log()

    """
            for year in configurations['future_years']:
                
                future_gdp = ac_data_sel_country['GDP_{0}_{1}'.format(scenario.split("_")[0].upper(), year)]
                future_cdd = ac_data_sel_country['CDD_{0}_{1}'.format(scenario, year)]
                if "NGA" in name_tag:
                    plt.scatter(((future_gdp/gdp_NGA)**(1./(configurations['future_years'][-1]-configurations['ref_year']))-1)*100, ((future_cdd-cdd_NGA)/cdd_NGA)*100, 
                    c=exposure_function(future_gdp, future_cdd)*future_cdd, 
                    cmap='YlOrRd', vmin=0., vmax=level_max,  s=12, edgecolors=scenario_colors[1], label=scenario)
                    
                    gdps.append(((future_gdp/gdp_NGA)**(1./(configurations['future_years'][-1]-configurations['ref_year']))-1).values[0]*100.)
                    cdds.append(((future_cdd-cdd_NGA)/cdd_NGA).values[0]*100.)
                else:    
                    plt.scatter(ac_data_sel_country['GDP_{0}_{1}'.format(scenario.split("_")[0].upper(), year)], ac_data_sel_country['CDD_{0}_{1}'.format(scenario, year)], 
                    c=exposure_function(ac_data_sel_country['GDP_{0}_{1}'.format(scenario.split("_")[0].upper(), year)], ac_data_sel_country['CDD_{0}_{1}'.format(scenario, year)])*ac_data_sel_country['CDD_{0}_{1}'.format(scenario, year)], 
                    cmap='YlOrRd', vmin=0., vmax=level_max,  s=12, edgecolors=scenario_colors[1], label=scenario)

                    gdps.append(future_gdp.values[0])
                    cdds.append(future_cdd.values[0])
                
            # Label points with year, tilt by 45 degrees
            plt.annotate(configurations['future_years'][-1], (gdps[-1], cdds[-1]+y_space), fontsize=6, color=scenario_colors[1], rotation=80)


            plt.plot(gdps, cdds, c='blue', linewidth=0.5)
        
            # Add horizontal dashed line for 2100 CDD increase
            if "NGA" in name_tag:
                for isc,scenario in enumerate(configurations['future_scenarios']):
                    future_cdd_scenario = (((ac_data_sel_country['CDD_{0}_{1}'.format(scenario, configurations['future_years'][-1])]-cdd_NGA)/cdd_NGA)*100).values[0]
                    plt.axhline(future_cdd_scenario, linestyle='--', c=scenario_colors[isc], linewidth=0.5)
                    # Label line with RCP scenario in the format RCP 4.5
                    plt.annotate(scenario.split("_")[1].upper().replace("P", "P ").replace("0", ".0").replace("5", ".5"), (4.5, future_cdd_scenario+0.5), fontsize=6, color=scenario_colors[isc])
        # Label points with country names
        for i, txt in enumerate(ac_data_scatter['ISO3'].values):
            plt.annotate(txt, (ac_data_scatter['GDP'][i]*1.05, ac_data_scatter['DD_mean'][i]-100), fontsize=8.5, color="blue")

        
        # Only show unique labels once
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper left')
        # plt.legend()

    if not os.path.exists('Figures/paper'):
        os.makedirs('Figures/paper')
    plt.savefig('Figures/paper/{0}.png'.format(name_tag), dpi=300)
    """

def plot_gdp_increase_map(configurations, gdp_cdd_data, geo_df, future_scenario):
    """
    Plot annual average GDP growth for historical and constant experienced CDD
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.15)
    # No axes ticks
    plt.xticks([])
    plt.yticks([])

    # Add title
    fig.suptitle('Economic development', fontsize=14)

    # Add labels a and b to subplots
    ax[0].text(0.01, 1.02, 'a Historical', transform=ax[0].transAxes, size=10, weight='bold')
    
    parts = future_scenario.split('_')
    result = parts[-1][:3].upper() + ' ' + parts[-1][3:4] + '.' + parts[-1][4:]

    ax[1].text(0.01, 1.02, 'b To avoid increased heat exposure under {0}'.format(result), transform=ax[1].transAxes, size=10, weight='bold')

    geo_df['gdp_const_{0}'.format(future_scenario)] = gdp_cdd_data['gdp_const_{0}'.format(future_scenario)] * 100.
    geo_df['gdp_historical_factor'] = gdp_cdd_data['gdp_historical_factor'] * 100.

    geo_df.plot(column='gdp_historical_factor', ax=ax[0], cmap='inferno_r', vmin=0, vmax=8)
    geo_df.plot(column='gdp_const_{0}'.format(future_scenario), ax=ax[1], cmap='inferno_r', vmin=0, vmax=8)
    
    # Add one shared colorbar which is half the height of the figure
    fig.colorbar(ax[1].collections[0], ax=ax, shrink=0.75, label='Mean GDP growth (annual %)')

    plt.savefig('Figures/paper/gdp_const_{0}_map.png'.format(future_scenario), dpi=300)


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
    
    plt.xlabel('Mean historical GDP growth (annual %)')
    plt.ylabel('Economic development to avoid increased heat exposure\nunder {0} (annual %)'.format(future_scenario))

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
    plt.plot([min, max], [min, max], '--', c='grey', label='1:1 line')

    # Legend with two columns
    plt.legend(ncol=2, fontsize=10)
    plt.savefig('Figures/paper/gdp_const_vs_historical_{0}.png'.format(future_scenario), dpi=300)