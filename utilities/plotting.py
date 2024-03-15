import country_converter as coco
from utilities.plotting_classes import *

def plot_variable_map(configurations, heat_exposure_df, plotting_variable):
    """
    Plot map of the variable (CDD or GDP)
    """
    map_plot = ExposurePlot(configurations, heat_exposure_df, '{0}_map'.format(plotting_variable))
    map_plot.create_data_map()
    colormap = configurations['plotting']['{0}_cmap'.format(plotting_variable.split('_')[0].lower())]
    map_plot.plot_map(plotting_variable, colormap=colormap, 
                      vmin=configurations['plotting'][plotting_variable.split('_')[0].lower()+'_min'],
                      vmax=configurations['plotting'][plotting_variable.split('_')[0].lower()+'_max'])
    map_plot.grey_empty_countries(plotting_variable)
    map_plot.remove_axes_ticks()
    map_plot.add_colorbar(label=configurations['plotting'][plotting_variable.split('_')[0].lower()+'_label'], 
                          colormap=colormap,
                          colorbar_max=configurations['plotting'][plotting_variable.split('_')[0].lower()+'_max'],
                          colorbar_min=configurations['plotting'][plotting_variable.split('_')[0].lower()+'_min'])
    map_plot.add_title(plotting_variable, fontsize=12)
    map_plot.save_figure()

def plot_variable_histogram(configurations, heat_exposure_df, plotting_variable):
    """
    Plot histogram of the variable
    """
    histogram_plot = ExposurePlot(configurations, heat_exposure_df, '{0}_histogram'.format(plotting_variable))
    histogram_plot.plot_histogram(plotting_variable)
    histogram_plot.add_x_y_labels(configurations['plotting'][plotting_variable.split('_')[0].lower()+'_label'], 'Frequency')
    histogram_plot.add_title(plotting_variable, fontsize=12)
    histogram_plot.save_figure()
    

def plot_exposure_map(configurations, ac_data_historical, exposure_func, scenario):
    """
    Plot exposure map
    """
    exposure_map = ExposurePlot(configurations, ac_data_historical, 'exposure_map_{0}'.format(scenario), scenario, country=None)
    exposure_map.create_exposure_map(exposure_func, scenario)
    exposure_map.plot_exposure()
    exposure_map.remove_axes_ticks()
    exposure_map.add_colorbar(label=configurations['plotting']['exposure_times_cdd_label'], 
                colormap='inferno_r', colorbar_max=configurations['plotting']['experiencedT_max'])
    exposure_map.add_title(scenario, fontsize=12)
    exposure_map.save_figure()

def plot_exposure_contour(configurations, exposure_function, ac_data, x_y_ranges, scenario=None, country=None, name_tag='exposure_contour', control=False):
    """
    Contour plot of penetration of air conditioning as a function of GDP per capita and cooling degree days
    """
    contour_plot = ContourPlot(configurations, ac_data, name_tag, scenario, country=country)
    contour_plot.contour_grid(x_y_ranges[0], x_y_ranges[1])
    contour_plot.calculate_contour(exposure_function)
    contour_plot.plot_contour()
    contour_plot.add_contour_lines(exposure_function, configurations['income_groups_colors'])
    if not control:
        data_points = configurations['income_groups_colors'].keys()
        data_color = configurations['income_groups_colors']
        if not country:
            contour_plot.add_data()

    else:
        data_color = 'grey'
        contour_plot.add_control_data(data_color)
        # ISO3 values where 'AC' has a value in ac_data
        data_points = ac_data[ac_data['AC'].notnull()]['ISO3'].unique()

    if not country:
        contour_plot.set_y_log()
        contour_plot.add_country_labels(data_points, data_color)
        contour_plot.add_x_y_labels(configurations['plotting']['cdd_label'], configurations['plotting']['gdp_label'])
        
    else:
        contour_plot.add_cdd_predictions()
        contour_plot.add_x_y_labels('Mean GDP growth (annual %)', 'Cooling degree days increase (%)')
        # Replace ISO3 with full name
        cc = coco.CountryConverter()
        country = cc.convert(names=country, to='name_short', not_found=None)
        contour_plot.add_title(country, fontsize=14)

    contour_plot.save_figure()

def plot_gdp_increase_map(configurations, gdp_cdd_data, future_scenario):
    """
    Plot annual average GDP growth for historical and constant experienced CDD
    """
    if not future_scenario == 'historical':
        rcp_scenario = future_scenario.split('_')[1]
    else:
        rcp_scenario = future_scenario
    gdp_increase_plot = GDPIncreaseMap(configurations, gdp_cdd_data, 'gdp_increase_map_{0}'.format(rcp_scenario), future_scenario)
    gdp_increase_plot.create_data_map()
    gdp_increase_plot.plot_growth_map()
    gdp_increase_plot.grey_empty_countries(gdp_increase_plot.column_name)
    gdp_increase_plot.remove_axes_ticks()
    gdp_increase_plot.add_title(rcp_scenario, fontsize=12)
    gdp_increase_plot.add_colorbar(label='Mean annual GDP growth (%)', colormap=configurations['plotting']['gdp_growth_cmap'],
                                   colorbar_max=configurations['plotting']['gdp_growth_max'],
                                   colorbar_min=configurations['plotting']['gdp_growth_min'])
    gdp_increase_plot.save_figure()

def plot_gdp_increase_scatter(configurations, gdp_cdd_data, future_scenario):
    """
    Plot difference in exposure times CDD as a function of GDP per capita
    """
    gdp_increase_scatter = GDPIncreaseScatter(configurations, gdp_cdd_data, 'gdp_increase_scatter_{0}'.format(future_scenario), future_scenario)
    gdp_increase_scatter.plot_scatter(['gdp_historical_growth', 'gdp_const_{0}'.format(future_scenario)])
    gdp_increase_scatter.add_x_y_labels('Mean annual historical GDP growth (%)', 'Mean annual GDP growth to avoid\nincreased heat exposure nunder {0} (%)'.format(gdp_increase_scatter.formatted_scenario))
    gdp_increase_scatter.add_1_to_1_line(['gdp_historical_growth', 'gdp_const_{0}'.format(future_scenario)])
    gdp_increase_scatter.save_figure()

def plot_cdd_scatter(configurations, gdp_cdd_data, future_scenario):
    """
    Plot difference in exposure times CDD as a function of GDP per capita
    """
    gdp_increase_scatter = GDPIncreaseScatter(configurations, gdp_cdd_data, 'cdd_scatter_{0}'.format(future_scenario), future_scenario)
    gdp_increase_scatter.plot_scatter(['CDD', 'CDD_{0}_2100'.format(future_scenario)])
    gdp_increase_scatter.add_x_y_labels('CDD in 2020 (°C days)', 'CDD in 2100\nunder {0} (°C days)'.format(gdp_increase_scatter.formatted_scenario))
    print("Draw 1:1")
    gdp_increase_scatter.add_1_to_1_line(['CDD', 'CDD_{0}_2100'.format(future_scenario)])
    gdp_increase_scatter.save_figure()