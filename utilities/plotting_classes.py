import os
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
import country_converter as coco
import numpy as np
import re
from utilities.utilities import read_gdp_data, gdp_from_cdd_exposure
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm
from statsmodels.graphics.plot_grids import scatter_ellipse


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
        plt.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper left', ncol=columns)
    
    def save_figure(self):
        """
        Save figure
        """
        if not os.path.exists('Figures/paper'):
            os.makedirs('Figures/paper')
        plt.savefig('Figures/paper/{0}.eps'.format(self.name_tag), dpi=400, bbox_inches='tight')
        plt.savefig('Figures/paper/{0}.png'.format(self.name_tag), dpi=400, bbox_inches='tight')


    def set_y_log(self):
        """
        Set y-axis to log scale
        """
        plt.yscale('log')
    
    def remove_axes_ticks(self):
        """
        Remove axes ticks
        """
        plt.xticks([])
        plt.yticks([])

class ExposurePlot(ExperiencedTPlot):
    def __init__(self, configurations, ac_data, name_tag, scenario=None, country=None):
        # Call parent class constructor
        super().__init__(configurations, ac_data, name_tag, country=country)
        self.scenario = scenario

    def create_data_map(self):
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
        # Save geo map in self
        self.ac_data_map_geo = ac_data_map_geo

    def plot_map(self, column, colormap, vmin, vmax):
        """
        Plot exposure map
        """
        # Log scale for GDP
        if 'GDP' in column:
            self.plot = self.ac_data_map_geo.plot(column=column, cmap=colormap, vmin=vmin, vmax=vmax, norm=LogNorm())
        else:
            self.plot = self.ac_data_map_geo.plot(column=column, cmap=colormap, vmin=0, vmax=vmax)

        
    def grey_empty_countries(self, column_name):
        """
        Grey out countries with no data
        """
        # Plot countries with no data in grey
        empty_countries = self.ac_data_map_geo[self.ac_data_map_geo[column_name].isnull()]
        if empty_countries.shape[0] > 0:
            empty_countries = empty_countries[empty_countries['geometry'].notnull()]
            empty_countries.plot(ax=plt.gca(), color='lightgrey')
        plt.gca().set_aspect('equal', adjustable='box')

    def color_countries(self, color, group_label):
        """
        Color countries that have data
        """
        self.ac_data_map_geo.plot(ax=plt.gca(), edgecolor='grey', color='white')
        income_group_countries = self.ac_data_map_geo[self.ac_data_map_geo['income_group'] == group_label]
        income_group_countries.plot(ax=plt.gca(), color=color)
        # Remove black frame around plot
        for spine in plt.gca().spines.values():
            spine.set_visible(False)


        
    def add_colorbar(self, label, colormap, colorbar_max, colorbar_min=0):
        """
        Add colorbar
        """
        if 'GDP per' in label:
            # Logarithmic colorbar
            mappable = ScalarMappable(cmap=colormap, norm=LogNorm(vmin=colorbar_min, vmax=colorbar_max))
        else:
            norm = Normalize(vmin=colorbar_min, vmax=colorbar_max)
            mappable = ScalarMappable(norm=norm, cmap=colormap)
            mappable._A = []  # This line is necessary for ScalarMappable to work with colorbar
        plt.colorbar(mappable, label=label, shrink=0.45)

    def plot_histogram(self, column):
        """
        Plot PDF
        """
        from scipy.stats import gaussian_kde
        offset = np.zeros(1000)
        # Draw a probability density function for each income group and for all countries bound in the range of the data
        for income_group in self.configurations['income_groups_colors'].keys():
            # Fill distribution with color
            kde = gaussian_kde(self.ac_data[self.ac_data['income_group'] == income_group][column].dropna())
            # Plot the distribution
            x = np.linspace(0, 5000, 1000)
            plt.fill_between(x, offset, offset + kde(x), edgecolor='none', alpha=0.5,
                             color=self.configurations['income_groups_colors'][income_group])
            offset += kde(x)

            # self.ac_data[self.ac_data['income_group'] == income_group][column].dropna().plot(kind='kde', 
            #     label=income_group, color=self.configurations['income_groups_colors'][income_group])
        # self.ac_data[column].dropna().plot(kind='kde', label='All countries', color='black')
        # Set x range from 0 to 5000
        plt.xlim(0, 4000)
        if not 'diff' in column:
            plt.ylim(0, 0.003)

        else:
            plt.ylim(0, 0.01)        


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
            self.contour_function = exposure_function(self.cdd_x, self.gdp_x)*self.cdd_x
        else:
            self.gdp_country = self.ac_data[self.ac_data.index==self.country]['GDP'].values[0]
            self.cdd_country = self.ac_data[self.ac_data.index==self.country]['CDD'].values[0]
            self.contour_function = exposure_function(self.gdp_country*((1+self.gdp_x/100.)**(self.configurations['analysis_years']['future_year']-self.configurations['analysis_years']['ref_year'])), self.cdd_country*(1+self.cdd_x/100.))*self.cdd_country*(1+self.cdd_x/100.)
    
    def plot_contour(self):
        """
        Contour plot of exposure times CDD as a function of GDP per capita and cooling degree days
        """
        self.level_max = self.configurations['plotting']['experiencedT_max']
        if self.country:
            n_levels = self.configurations['plotting']['contour_levels'][self.country]
        else:
            n_levels = self.configurations['plotting']['contour_levels']['absolute']
        self.levels = np.linspace(0, self.level_max, n_levels)
        color_map = self.configurations['plotting']['exposure_times_cdd_cmap']
        
        plt.contourf(self.cdd_x, self.gdp_x, self.contour_function, levels=self.levels, cmap=color_map)
        plt.colorbar(label=self.configurations['plotting']['exposure_times_cdd_label'], ticks=np.linspace(0, self.level_max, 11))
        
    def add_contour_lines(self, exposure_function):
        """
        Add contour lines
        """
        clines = plt.contour(self.cdd_x, self.gdp_x, self.contour_function, levels=self.levels, colors='k', linewidths=0.3)
        plt.clabel(clines, self.levels[::2], fmt='%d', fontsize=8, colors='black')
        plt.clim(0, self.levels[-1])

    def add_const_heat_exposure_lines(self, exposure_function, const_heat_exposure):
        # Add constant exposure times CDD line
        # self.const_level = [exposure_function(self.gdp_country*(1.**(self.configurations['analysis_years']['future_year']-self.configurations['analysis_years']['ref_year'])), self.cdd_country)*self.cdd_country]
        for line in const_heat_exposure.keys():
            const_level = self.ac_data[self.ac_data.index == line]['exposure_times_cdd']
            const_line = plt.contour(self.cdd_x, self.gdp_x, self.contour_function, 
                    levels= const_level, colors=const_heat_exposure[line], linewidths=1.2)
            # keep label parallel to line
            plt.clabel(const_line, const_level, fmt=line, fontsize=9, colors=const_heat_exposure[line])

    def add_data(self):
        """
        Add data points
        """   
        for income_group in self.configurations['income_groups_colors'].keys():     
            x = self.ac_data[self.ac_data.index == income_group]['CDD']
            y = self.ac_data[self.ac_data.index == income_group]['GDP']
            plt.scatter(x, y, label=income_group, c=self.configurations['income_groups_colors'][income_group], s=8, marker='o')


    def add_control_data(self, color):
        """
        Add AC data for control plots
        """
        # Print AC data when not null
        plt.scatter(self.ac_data['CDD'], self.ac_data['GDP'], c=(1.-self.ac_data['AC']) * self.ac_data['CDD'], 
                    cmap=self.configurations['plotting']['exposure_times_cdd_cmap'], 
                    vmin=0., vmax=self.level_max,  s=16, edgecolors=color, label='AC data')

    def add_cdd_predictions(self):
        """
        Add CDD predictions as horizontal dashed lines
        """   
        if not self.country:
            print('No country selected')
            return
        print('group: {0}'.format(self.country))
        ac_data_sel_country = self.ac_data[self.ac_data.index == self.country]
        for isc,scenario in enumerate(self.configurations['future_scenarios']):
            future_cdd_scenario = (((ac_data_sel_country['CDD_{0}_{1}'
                .format(scenario, self.configurations['analysis_years']['future_year'])]-ac_data_sel_country['CDD'])/ac_data_sel_country['CDD'])*100).values[0]
            # GDP increase to avoid increased heat exposure for given scenario
            gdp_increase_const = ac_data_sel_country['gdp_const_{0}'.format(scenario)].values[0]*100.
            print(scenario)
            print('future cdd increase: {0}'.format(future_cdd_scenario))
            print('gdp increase: {0}'.format(gdp_increase_const))

            
            if future_cdd_scenario > 0:
                y1, y2 = [0, future_cdd_scenario]
                x1, x2 = [gdp_increase_const, gdp_increase_const]
            else:
                x1, x2 = [0, 0]

            plt.plot([0, gdp_increase_const], [future_cdd_scenario, future_cdd_scenario], '--', 
                     c=self.configurations['scenario_colors'][isc], linewidth=0.8)

            plt.plot([x1,x2], [0, future_cdd_scenario], '--', 
                     c=self.configurations['scenario_colors'][isc], linewidth=0.8)
            # Label line with scenario ssp2_rcp45 in the format RCP 4.5
            formatted_string = re.sub(r'rcp(\d)(\d)', r'RCP \1.\2', scenario.split('_')[1])
            plt.annotate(formatted_string, (self.gdp_x[0][0], future_cdd_scenario), 
                         fontsize=8.5, color=self.configurations['scenario_colors'][isc])
    
    def add_country_labels(self, countries, colors, control):
        """
        Label points with country names in control plot and year in main plot
        """ 
        for txt in countries:
            if control:
                color = colors
                label = txt
                fontsize=6
                id = self.ac_data['ISO3']
            else:
                color = colors[txt]
                label = txt
                fontsize=9
                id = self.ac_data.index

            country_index = self.ac_data[id == txt].index

            plt.annotate(label, (self.ac_data['CDD'][country_index]+50, self.ac_data['GDP'][country_index]*0.9),
                         fontsize=fontsize, color=color)


class GDPIncreaseMap(ExposurePlot):
    def __init__(self, configurations, ac_data, name_tag, scenario, country=None):
        super().__init__(configurations, ac_data, name_tag, scenario, country=country)
        # Format scenario name
        parts = self.scenario.split('_')
        self.formatted_scenario = parts[-1][:3].upper() + ' ' + parts[-1][3:4] + '.' + parts[-1][4:]
    
    def plot_growth_map(self):
        """
        Plot maps
        """
        if self.scenario == 'historical':
            mean_gdp_growth = 'gdp_historical_growth'
        else:
            mean_gdp_growth = 'gdp_const_{0}'.format(self.scenario)
        self.column_name = mean_gdp_growth
        
        self.ac_data_map_geo[mean_gdp_growth] = self.ac_data[mean_gdp_growth] * 100.

        self.ac_data_map_geo.plot(column=mean_gdp_growth, cmap=self.configurations['plotting']['gdp_growth_cmap'],
                                  vmin=self.configurations['plotting']['gdp_growth_min'], vmax=self.configurations['plotting']['gdp_growth_max'])

        
class GDPIncreaseScatter(GDPIncreaseMap):
    def __init__(self, configurations, ac_data, name_tag, scenario, country=None):
        super().__init__(configurations, ac_data, name_tag, scenario, country=country)

    def plot_scatter(self, data, groups):
        """
        Plot scatter plot, color coded by continent
        """

        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal', adjustable='box')

        for income_group in groups:
            x = self.ac_data[self.ac_data['income_group'] == income_group][data[0]]
            y = self.ac_data[self.ac_data['income_group'] == income_group][data[1]]
            if 'gdp' in data[0]:
                x = x * 100
                y = y * 100
            self.ax.scatter(x, y, label=income_group, s=10, marker='o',
                    c=self.configurations['income_groups_colors'][income_group])
            
            if len(groups) == 1:
                self.ax.set_xlim(-2., 10.)
                self.ax.set_ylim(-2., 10.)
            else:
                self.ax.set_xlim(0., 4000.)
                self.ax.set_ylim(0., 4000.)

            # Plot GDP growth to obtain high income group average exposure*CDD
            if data[2] is not None:
                # Only show for countries with lower exposure*CDD than high income group average
                y2 = self.ac_data[self.ac_data['income_group'] == income_group][data[2]].clip(lower=0)
                if 'gdp' in data[2]:
                    y2 = y2 * 100
                self.ax.scatter(x, y2, label=income_group, s=10, marker='o', linewidth=0.5,
                        edgecolors=self.configurations['income_groups_colors'][income_group], facecolors='none')


    def confidence_ellipse_median(self, x_val, y_val, ax, n_std=2.4477, facecolor='none'):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*,
        centered around the median of the data instead of the mean.
        """
        from matplotlib.patches import Ellipse
        median_x = np.median(x_val)
        median_y = np.median(y_val)

        cov = np.cov(x_val, y_val)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvalues = np.sqrt(eigenvalues)
        
        ellipse = Ellipse(xy=(median_x, median_y),
                    width=eigenvalues[0]*n_std*2, height=eigenvalues[1]*n_std*2,
                    angle=np.rad2deg(np.arccos(eigenvectors[0, 0])), alpha=0.3, facecolor=facecolor)
        
        ax.add_artist(ellipse)
        
    

    def add_1_to_1_line(self, data, min=None, max=None):
        """
        Add line where historical and constant GDP growth are equal
        """
        # Dashed diagonal line
        x_all = self.ac_data[data[0]]
        y_all = self.ac_data[data[1]]
        if 'gdp' in data[0]:
            x_all = x_all*100
            y_all = y_all*100
        if not max:
            min = np.min([np.min(x_all), np.min(y_all)])
            max = np.max([np.max(x_all), np.max(y_all)])
        plt.plot([min, max], [min, max], '--', c='grey')
        plt.annotate('1:1 line', (max*0.8, max*0.85), fontsize=12, color='grey', rotation=40)