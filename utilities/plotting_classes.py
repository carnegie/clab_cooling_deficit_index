import os
import geopandas as gpd
import country_converter as coco
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colors import LinearSegmentedColormap, to_rgb


class BasicPlot:
    def __init__(self, configurations, ac_data, name_tag, country=None):
        self.configurations = configurations
        self.ac_data = ac_data
        self.name_tag = name_tag
        self.country = country

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
    
    def save_figure(self):
        """
        Save figure
        """
        if not os.path.exists('figures/paper'):
            os.makedirs('figures/paper')
        plt.savefig('figures/paper/{0}.pdf'.format(self.name_tag), dpi=500, bbox_inches='tight')

    def show_close_figure(self):
        """
        Close figure
        """
        plt.show()
        plt.close()


class ExposurePlot(BasicPlot):
    def __init__(self, configurations, ac_data, name_tag, scenario=None, country=None):
        # Call parent class constructor
        super().__init__(configurations, ac_data, name_tag, country=country)
        self.scenario = scenario

    def create_data_map(self):
        """
        Create exposure map
        """
        # Get the path to the naturalearth countries shapefile
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        # Copy data
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
            self.plot = self.ac_data_map_geo.plot(column=column, cmap=colormap, vmin=vmin, vmax=vmax)

        
    def grey_empty_countries(self, column_name):
        """
        Grey out countries with no data
        """
        # Plot countries with no data in grey
        empty_countries = self.ac_data_map_geo[self.ac_data_map_geo[column_name].isnull()]
        # Also grey countries if future CDD is less than today
        if 'CDD_' in column_name and 'diff' in column_name:
            cdd_decrease_countries = self.ac_data_map_geo[self.ac_data_map_geo[column_name] < 0.]
        else:
            cdd_decrease_countries = pd.DataFrame()
        grey_countries = pd.concat([empty_countries, cdd_decrease_countries])
        if grey_countries.shape[0] > 0:
            grey_countries = grey_countries[grey_countries['geometry'].notnull()]
            grey_countries.plot(ax=plt.gca(), color='lightgrey')
        plt.gca().set_aspect('equal', adjustable='box')

    def color_countries(self, color, group_label, col=None, cmin=None, cmax=None):
        """
        Color countries that have data
        """
        income_group_countries = self.ac_data_map_geo[self.ac_data_map_geo['income_group'] == group_label]
        if not col:
            income_group_countries.plot(ax=plt.gca(), color=color)
        else:
            rgb = np.array(to_rgb(color))

            light_rgb = rgb + 0.85 * (1 - rgb)
            custom_cmap = LinearSegmentedColormap.from_list("cmap_{0}_income".format(group_label), [light_rgb, rgb])

            lighter_rgb = rgb + 0.9 * (1 - rgb)       # lighter
            dark_rgb  = rgb * 0.25                   # darker

            custom_cmap_fine = LinearSegmentedColormap.from_list(
                f"cmap_{group_label}_income_fine", 
                [lighter_rgb, rgb, dark_rgb]
            )

            try:
                # Try to retrieve simple and extended colormap for later
                plt.cm.get_cmap("cmap_{0}_income".format(group_label))
            except ValueError:
                plt.register_cmap(cmap=custom_cmap)
            try:
                plt.cm.get_cmap("cmap_{0}_income_fine".format(group_label))
            except ValueError:
                plt.register_cmap(cmap=custom_cmap_fine)

            income_group_countries.plot(ax=plt.gca(), column=col, cmap=custom_cmap_fine, vmin=cmin, vmax=cmax)

        # Plot all other countries in white
        other_countries = self.ac_data_map_geo[self.ac_data_map_geo['income_group'] != group_label]
        other_countries.plot(ax=plt.gca(), color='white')

        self.ac_data_map_geo.plot(ax=plt.gca(), edgecolor='grey', color='none')

        # Remove black frame around plot
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        
    def add_colorbar(self, label, colormap, colorbar_min, colorbar_max):
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




class ContourPlot(BasicPlot):
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

    def calculate_contour(self, exposure_function, multiply_cdd):
        """
        Calculate contour function
        """
        self.contour_function = exposure_function(self.cdd_x, self.gdp_x)
        if multiply_cdd:
            self.contour_function *= self.cdd_x
        else:
            self.contour_function = 1. - self.contour_function
  
    def plot_contour(self, multiply_cdd=True):
        """
        Contour plot of exposure times CDD as a function of GDP per capita and cooling degree days
        """
        if multiply_cdd:
            self.level_max = self.configurations['plotting']['cooling_deficit_index_max']
            label = self.configurations['plotting']['exposure_times_cdd_label']
            color_map = self.configurations['plotting']['exposure_times_cdd_cmap']
        else:
            self.level_max = 1.
            label = 'AC adoption'
            color_map = self.configurations['plotting']['ac_adoption_cmap']
        n_levels = self.configurations['plotting']['contour_levels']
        self.levels = np.linspace(0, self.level_max, n_levels)
        
        plt.contourf(self.cdd_x, self.gdp_x, self.contour_function, levels=self.levels, cmap=color_map)
        plt.colorbar(label=label, ticks=np.linspace(0, self.level_max, 11))
        
    def add_contour_lines(self, multiply_cdd=True):
        """
        Add contour lines
        """
        clines = plt.contour(self.cdd_x, self.gdp_x, self.contour_function, levels=self.levels, colors='k', linewidths=0.3)
        if multiply_cdd:
            format = '%d'
        else:
            format = '%.1f'
        plt.clabel(clines, self.levels[::2], fmt=format, fontsize=8, colors='black')
        plt.clim(0, self.levels[-1])

    def add_data(self):
        """
        Add data points
        """   
        for income_group in self.configurations['income_groups_colors'].keys():     
            x = self.ac_data[self.ac_data.index == income_group]['CDD']
            y = self.ac_data[self.ac_data.index == income_group]['GDP']
            plt.scatter(x, y, label=income_group, c=self.configurations['income_groups_colors'][income_group], s=8, marker='o')

    def add_labels(self, labels, colors):
        """
        Label points
        """ 
        for txt in labels:
            color = colors[txt]
            label = txt
            fontsize=9
            id = self.ac_data.index
            label_index = self.ac_data[id == txt].index

            plt.annotate(label, (self.ac_data['CDD'][label_index]+50, self.ac_data['GDP'][label_index]*0.9),
                         fontsize=fontsize, color=color)


        
class ScatterPlot(ExposurePlot):
    def __init__(self, configurations, ac_data, name_tag, scenario, country=None):
        super().__init__(configurations, ac_data, name_tag, scenario, country=country)

        # Format scenario name
        parts = self.scenario.split('_')
        self.formatted_scenario = parts[-1][:3].upper() + ' ' + parts[-1][3:4] + '.' + parts[-1][4:]

    def plot_scatter(self, data, groups, ratio_cdd=False):
        """
        Plot scatter plot, color coded by continent
        """

        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        for income_group in groups:
            x = self.ac_data[self.ac_data['income_group'] == income_group][data[0]]
            y = self.ac_data[self.ac_data['income_group'] == income_group][data[1]]
            income_color = self.configurations['income_groups_colors'][income_group] 
            income_color_map = "cmap_{0}_income".format(income_group)

            if 'gdp' in data[0]:
                x = x * 100
                y = y * 100
                axes_range = [self.configurations['plotting']['gdp_growth_min'], self.configurations['plotting']['gdp_growth_max']]
            else:
                min = self.configurations['plotting']['cdd_min']
                if 'diff' in data[1]:
                    if not ratio_cdd:
                        axes_range_y = [min, self.configurations['plotting']['cdd_diff_max']]
                    else:
                        axes_range_y = [0, 6]
                        y = y / x
                axes_range = [min, self.configurations['plotting']['cdd_max']]
                sns.kdeplot(x=x, y=y, cmap=income_color_map, fill=True, bw_adjust=1.2, alpha=0.75)
                
            self.ax.scatter(x, y, label=income_group, s=42, marker='o',
                    c=income_color)
            
            self.ax.set_xlim(axes_range[0], axes_range[1])
            if 'diff' in data[1]:
                self.ax.set_ylim(axes_range_y[0], axes_range_y[1])
            else:
                self.ax.set_ylim(axes_range[0], axes_range[1])

            # Plot GDP growth to obtain high income group average exposure*CDD
            if data[2] is not None:
                # Only show for countries with lower exposure*CDD than high income group average
                y2 = self.ac_data[self.ac_data['income_group'] == income_group][data[2]].clip(lower=0)
                if 'gdp' in data[2]:
                    y2 = y2 * 100
                self.ax.scatter(x, y2, label=income_group, s=42, marker='o', linewidth=0.5,
                        edgecolors=income_color, facecolors='none')

    
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

    
    def label_countries(self, countries, data):
        """
        Label points with income group
        """ 
        for txt in countries:
            if txt == 'China':
                country_name = "People's Republic of China"
            else:
                country_name = txt

            if country_name in self.ac_data['Country'].values:
                country_data = self.ac_data[self.ac_data['Country'] == country_name]
                x = country_data[data[0]]
                y = country_data[data[1]]
                if 'gdp' in data[0]:
                    x = x * 100
                    y = y * 100
                plt.annotate(txt, (x, y), fontsize=8, color='black', rotation=20)


class SensitivityPlot(BasicPlot):
    def __init__(self, configurations, ac_data, name_tag, country=None):
        # Call parent class constructor
        super().__init__(configurations, ac_data, name_tag, country=country)

    def plot_sensitivity(self, data, income_group, column, color):
        """
        Plot sensitivity of exposure to GDP growth
        """
        # list all files in data with cooling_deficit_index_calculations
        outdir = self.configurations['output_path'].split('/')[0]
        sensitivity_paths = [f for f in os.listdir(outdir) if f.startswith('cooling_deficit_index_calculations_') and f.endswith('.csv')]
        sensitivity_paths = ['cooling_deficit_index_calculations.csv'] + sensitivity_paths
        line_styles = ['-', '--', '-.', ':']

        # Plot vertical black line at historical median growth per income group
        historical_med = self.ac_data[self.ac_data['income_group'] == income_group]['gdp_historical_growth'].median()*100

        fig, ax = plt.subplots()

        label_dict = {}
        for sens_path in sensitivity_paths:
            sensitivity = sens_path.split('.')[0].replace('cooling_deficit_index_calculations_','')            
            sensitivity_df = pd.read_csv(os.path.join(outdir, sens_path))
            sensitivity_df = sensitivity_df[sensitivity_df['income_group'] == income_group]

            linestyle = line_styles[sensitivity_paths.index(sens_path)]
            if not sensitivity=='cooling_deficit_index_calculations':
                label = sensitivity
            else:
                label = income_group + ' base'
            # Plot kernel density estimate of needed GDP growth to avoid increased exposure
            kde = sns.kdeplot(sensitivity_df[column]*100, color=color, bw_adjust=1.2, alpha=0.75, 
                    linestyle=linestyle, label=label)

            # Extract KDE x and y values
            x_kde = kde.get_lines()[-1].get_xdata()
            y_kde = kde.get_lines()[-1].get_ydata()

            # Ensure x_kde and y_kde are NumPy arrays for indexing
            x_kde = np.array(x_kde)
            y_kde = np.array(y_kde)

            # Calculate area above the historical median
            mask = x_kde >= historical_med
            area = np.trapz(y_kde[mask], x_kde[mask]) / np.trapz(y_kde, x_kde) * 100

            # Add percentage to label of line
            new_label = f"{label}\n({area:.1f}% above historical)"
            label_dict[label] = new_label
            
            # Axes ranges
            ax.set_xlim(-2, 10)
            ax.set_ylim(0, 1.25)
        
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [label_dict[label] for label in labels]
        
        # Add vertical line at historical median growth
        plt.axvline(x=historical_med, color='black', linestyle='--', label='Historical median')

        ax.legend(handles, new_labels, loc='best')
