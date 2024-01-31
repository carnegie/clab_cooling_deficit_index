import os
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
import country_converter as coco
import numpy as np
import re
from utilities.utilities import read_gdp_data, gdp_from_cdd_exposure
import pandas as pd

class ExperiencedTPlot:
    def __init__(self, configurations, ac_data, name_tag, country=None):
        self.configurations = configurations
        self.ac_data = ac_data
        self.name_tag = name_tag
        self.country = country

        matplotlib.rcParams['font.family'] = 'Helvetica'
        plt.figure()

    def add_x_y_labels(self, xlabel, ylabel):
        """
        Add x and y labels
        """
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    
    def add_title(self, title, fontsize=10, make_sup_title=False):
        """
        Add title
        """
        if make_sup_title:
            plt.suptitle(title, fontsize=fontsize)
        else:
            plt.title(title, fontsize=fontsize)

    def add_legend(self, columns=1):
        """
        Add legend
        """
        # Only show unique labels once
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc='upper left', ncol=columns)
    
    def save_figure(self):
        """
        Save figure
        """
        if not os.path.exists('Figures/paper'):
            os.makedirs('Figures/paper')
        plt.savefig('Figures/paper/{0}.png'.format(self.name_tag), dpi=400)

    def set_x_log(self):
        """
        Set x-axis to log scale
        """
        plt.xscale('log')
    
    def remove_axes_ticks(self):
        """
        Remove axes ticks
        """
        plt.xticks([])
        plt.yticks([])

class ExposurePlot(ExperiencedTPlot):
    def __init__(self, configurations, ac_data, name_tag, country=None):
        # Call parent class constructor
        super().__init__(configurations, ac_data, name_tag, country=country)

    def create_exposure_map(self, exposure_func):
        """
        Create exposure map
        """
        # Load world geometries
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        # Copy your data
        ac_data_map = self.ac_data.copy()
        # Convert ISO3 codes in your data to match the world dataframe
        cc = coco.CountryConverter()
        ac_data_map['ISO3'] = cc.convert(names=ac_data_map['ISO3'].values.tolist(), to='ISO3', not_found=None)
        # Merge your data with the world geometries
        merged_data = ac_data_map.merge(world[['geometry', 'iso_a3']], left_on='ISO3', right_on='iso_a3', how='left')
        # Convert the merged data to a GeoDataFrame
        ac_data_map_geo = gpd.GeoDataFrame(merged_data, geometry='geometry')
        # Add exposure column to geo map
        ac_data_map_geo['exposure_calculated'] = exposure_func(ac_data_map_geo['GDP'], ac_data_map_geo['DD_mean'])
        self.ac_data_map_geo = ac_data_map_geo

    def plot_exposure(self):
        """
        Plot exposure map
        """
        # Plot the data
        
        self.plot = self.ac_data_map_geo.plot(column='exposure_calculated', cmap='viridis', vmin=0, vmax=1)

    def add_colorbar(self):
        """
        Add colorbar
        """
        # Color bar same height as map
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="3.5%", pad=0.1)
        plt.colorbar(self.plot.collections[0], cax=cax, label='Exposure')


class ContourPlot(ExperiencedTPlot):
    def __init__(self, configurations, ac_data, name_tag, country=None):
        # Call parent class constructor
        super().__init__(configurations, ac_data, name_tag, country=country)

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
        self.level_max = 5000.
        self.levels = np.linspace(0, self.level_max, 21)
        color_map = 'YlOrRd'

        plt.contourf(self.gdp_x, self.cdd_x, self.contour_function, levels=self.levels, cmap=color_map)
        plt.colorbar(label='Exposure times CDD [°C days]', ticks=np.linspace(0, self.level_max, 11))
        
    def add_contour_lines(self, exposure_function):
        """
        Add contour lines
        """
        clines = plt.contour(self.gdp_x, self.cdd_x, self.contour_function, levels=self.levels, colors='k', linewidths=0.3)
        plt.clabel(clines, self.levels[::2], fmt='%d', fontsize=8, colors='black')
        plt.clim(0, self.levels[-1])

        if self.country:
            # Add constant exposure times CDD line
            self.const_level = [exposure_function(self.gdp_country*(1.**(self.configurations['future_years'][-1]-self.configurations['ref_year'])), self.cdd_country)*self.cdd_country]
            const_line = plt.contour(self.gdp_x, self.cdd_x, self.contour_function, 
                        levels= self.const_level, colors='black', linewidths=1.2)
            plt.clabel(const_line, self.const_level, fmt='const exposure times CDD', fontsize=8, colors='black')

    def add_data(self, exposure_function):
        """
        Add data points
        """           
        # Collect data points
        for country in self.configurations['countries_highest_pop']:
            gdps, cdds, exposures_times_cdd = [], [], [] 
            if self.country != None and country!=self.country: continue

            ac_data_sel_country = self.ac_data[self.ac_data['ISO3'] == country]

            for isc,scenario in enumerate(self.configurations['future_scenarios']):
                # Add data points
                for iy,year in enumerate([self.configurations['ref_year']]+self.configurations['future_years']):
                    if year != self.configurations['ref_year']:
                        gdp = ac_data_sel_country['GDP_{0}_{1}'.format(scenario.split("_")[0].upper(), year)].values[0]
                        cdd = ac_data_sel_country['CDD_{0}_{1}'.format(scenario, year)].values[0]
                    else:
                        gdp = ac_data_sel_country['GDP'].values[0]
                        cdd = ac_data_sel_country['DD_mean'].values[0]
                    
                    if self.country == None:
                        gdps.append(gdp)
                        cdds.append(cdd)
                    else:
                        gdps.append(((gdp/self.gdp_country)**(1./(self.configurations['future_years'][-1]-self.configurations['ref_year']))-1)*100.)
                        cdds.append(((cdd-self.cdd_country)/self.cdd_country)*100.)
                        
                    exposures_times_cdd.append((exposure_function(gdp, cdd)*cdd))

                    # Label points with year for one country
                    if country == self.configurations['countries_highest_pop'][0]:
                        plt.annotate(year, (gdps[iy], cdds[iy]+75), fontsize=8,
                                color='purple', rotation=60)

                plt.scatter(gdps, cdds, c=exposures_times_cdd,
                        cmap='YlOrRd', vmin=0., vmax=self.level_max,  s=12, edgecolors=self.configurations['scenario_colors'][isc],
                        label=scenario)  
                print("scenario color", self.configurations['scenario_colors'][isc])
                # Connect points with line
                plt.plot(gdps, cdds, c=self.configurations['scenario_colors'][isc], linewidth=0.75)

    def add_control_data(self, color):
        """
        Add AC data for control plots
        """
        # Print AC data when not null
        plt.scatter(self.ac_data['GDP'], self.ac_data['DD_mean'], c=(1.-self.ac_data['AC'])*self.ac_data['DD_mean'], cmap='YlOrRd', 
                    vmin=0., vmax=self.level_max,  s=12, edgecolors=color, label='AC data')

    def add_cdd_predictions(self):
        """
        Add CDD predictions as horizontal dashed lines
        """   
        if not self.country:
            print('No country selected')
            return
        
        ac_data_sel_country = self.ac_data[self.ac_data['ISO3'] == self.country]
        for isc,scenario in enumerate(self.configurations['future_scenarios']):
            future_cdd_scenario = (((ac_data_sel_country['CDD_{0}_{1}'
                .format(scenario, self.configurations['future_years'][-1])]-ac_data_sel_country['DD_mean'])/ac_data_sel_country['DD_mean'])*100).values[0]
            # GDP increase to avoid increased heat exposure for given scenario
            gdp_increase_const = ac_data_sel_country['gdp_const_{0}'.format(scenario)].values[0]*100.
            # Plot dashed line from 0 to gdp_increase_const, then vertical line down
            plt.plot([0, gdp_increase_const], [future_cdd_scenario, future_cdd_scenario], '--', 
                     c=self.configurations['scenario_colors'][isc], linewidth=0.8)
            plt.plot([gdp_increase_const, gdp_increase_const], [0, future_cdd_scenario], '--', 
                     c=self.configurations['scenario_colors'][isc], linewidth=0.8)
            # Label line with scenario ssp2_rcp45 in the format RCP 4.5
            formatted_string = re.sub(r'rcp(\d)(\d)', r'RCP \1.\2', scenario.split('_')[1])
            plt.annotate(formatted_string, (self.gdp_x[0][0]+0.1, future_cdd_scenario+0.5), 
                         fontsize=8.5, color=self.configurations['scenario_colors'][isc])
    
    def add_country_labels(self, countries, color):
        """
        Label points with country names
        """
        for txt in countries:
            country_index = self.ac_data[self.ac_data['ISO3'] == txt].index[0]
            plt.annotate(txt, (self.ac_data['GDP'][country_index]*1.05, self.ac_data['DD_mean'][country_index]-100), fontsize=8.5, color=color)


class GDPIncreaseMap(ExposurePlot):
    def __init__(self, configurations, ac_data, name_tag, scenario, country=None):
        super().__init__(configurations, ac_data, name_tag, country=country)
        self.scenario = scenario
        # Format scenario name
        parts = self.scenario.split('_')
        self.formatted_scenario = parts[-1][:3].upper() + ' ' + parts[-1][3:4] + '.' + parts[-1][4:]

    def add_subplots(self):
        """
        Add subplots
        """
        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
        self.fig.subplots_adjust(hspace=0.15)
    
    def add_subtitles(self):
        """
        Add subtitles
        """
        # Add labels a and b to subplots
        self.ax[0].text(0.01, 1.02, 'a Historical ({0}-{1})'.format(self.configurations['past_year'], self.configurations['ref_year']), transform=self.ax[0].transAxes, size=10, weight='bold')
        self.ax[1].text(0.01, 1.02, 'b To avoid increased heat exposure under {0} ({1}-{2})'.format(self.formatted_scenario, self.configurations['ref_year'], self.configurations['future_years'][-1]), transform=self.ax[1].transAxes, size=10, weight='bold')
    
    def plot_maps(self):
        """
        Plot maps
        """
        self.ac_data_map_geo['gdp_historical_factor'] = self.ac_data['gdp_historical_factor'] * 100.
        self.ac_data_map_geo['gdp_const_{0}'.format(self.scenario)] = self.ac_data['gdp_const_{0}'.format(self.scenario)] * 100.

        self.ac_data_map_geo.plot(column='gdp_historical_factor', ax=self.ax[0], cmap='inferno_r', vmin=0, vmax=8)
        self.ac_data_map_geo.plot(column='gdp_const_{0}'.format(self.scenario), ax=self.ax[1], cmap='inferno_r', vmin=0, vmax=8)
        
        # Add one shared colorbar which is half the height of the figure
        self.fig.colorbar(self.ax[1].collections[0], ax=self.ax, shrink=0.75, label='Mean GDP growth (annual %)')


class GDPIncreaseScatter(GDPIncreaseMap):
    def __init__(self, configurations, ac_data, name_tag, scenario, country=None):
        super().__init__(configurations, ac_data, name_tag, scenario, country=country)

    def add_continent_info(self):
        """
        Add continent column were country is in
        """
        cc = coco.CountryConverter()
        self.ac_data['continent'] = self.ac_data['ISO3'].apply(lambda x: cc.convert(names=x, to='continent'))
    
    def plot_scatter(self, cscale='continent'):
        """
        Plot scatter plot, color coded by continent
        """
        if cscale == 'continent':
            for continent in self.configurations['continent_colors']:
                plt.scatter(self.ac_data[self.ac_data['continent'] == continent]['gdp_historical_factor']*100,
                        self.ac_data[self.ac_data['continent'] == continent]['gdp_const_{0}'.format(self.scenario)]*100,
                        label=continent, c=self.configurations['continent_colors'][continent], s=8, marker='o')
        elif cscale == 'gdp':
            plt.scatter(self.ac_data['gdp_historical_factor']*100, self.ac_data['gdp_const_{0}'.format(self.scenario)]*100,
                        c=self.ac_data['GDP'], s=8, marker='o', cmap='viridis', norm=matplotlib.colors.LogNorm())
            # Add colorbar in greyscale and log
            plt.colorbar(label='GDP per capita (current US$)')
        elif cscale == 'vs_gdp':
            # Plot gdp_const vs current gdp
            plt.scatter(self.ac_data['GDP'], self.ac_data['gdp_const_{0}'.format(self.scenario)]*100,
                        c='black', s=8, marker='o')
    
    def label_countries(self, cscale):
        """
        Label countries with ISO3 code
        """
        for i, txt in enumerate(self.ac_data['ISO3'].values):
            if not np.isnan(self.ac_data['gdp_const_{0}'.format(self.scenario)][i]) and not np.isnan(self.ac_data['gdp_historical_factor'][i]):
                if cscale != 'vs_gdp':
                    plt.annotate(txt, (self.ac_data['gdp_historical_factor'][i]*100+0.1, self.ac_data['gdp_const_{0}'.format(self.scenario)][i]*100), fontsize=6)
                else:
                    # Only print every 5th country
                    if i % 5 == 0:
                        plt.annotate(txt, (self.ac_data['GDP'][i]+100, self.ac_data['gdp_const_{0}'.format(self.scenario)][i]*100+0.1),
                                     fontsize=6, rotation=40)

    def add_1_to_1_line(self):
        """
        Add line where historical and constant GDP growth are equal
        """
        # Dashed diagonal line
        min = np.min([np.min(self.ac_data['gdp_historical_factor']*100), np.min(self.ac_data['gdp_const_{0}'.format(self.scenario)]*100)])
        max = np.max([np.max(self.ac_data['gdp_historical_factor']*100), np.max(self.ac_data['gdp_const_{0}'.format(self.scenario)]*100)])
        plt.plot([min, max], [min, max], '--', c='grey')
        plt.annotate('1:1 line', (max-1.5, max-1), fontsize=12, color='grey', rotation=40)