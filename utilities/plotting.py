import country_converter as coco
from utilities.plotting_classes import *

def plot_exposure_map(ac_data_historical, exposure_func):
    """
    Plot exposure map
    """
    exposure_map = ExposurePlot(None, ac_data_historical, 'exposure_map')
    exposure_map.create_exposure_map(exposure_func)
    exposure_map.plot_exposure()
    exposure_map.remove_axes_ticks()
    exposure_map.add_colorbar()
    exposure_map.save_figure()

def plot_exposure_contour(configurations, exposure_function, ac_data, x_y_ranges, country=None, name_tag='exposure_contour', control=False):
    """
    Contour plot of penetration of air conditioning as a function of GDP per capita and cooling degree days
    """
    contour_plot = ContourPlot(configurations, ac_data, name_tag, country=country)
    contour_plot.contour_grid(x_y_ranges[0], x_y_ranges[1])
    contour_plot.calculate_contour(exposure_function)
    contour_plot.plot_contour()
    contour_plot.add_contour_lines(exposure_function)
    if not control:
        data_countries = configurations['countries_highest_pop']
        data_color = 'purple'
    else:
        data_color = 'grey'
        contour_plot.add_control_data(data_color)
        # ISO3 values where 'AC' has a value in ac_data
        data_countries = ac_data[ac_data['AC'].notnull()]['ISO3'].unique()

    if not country:
        contour_plot.add_data(exposure_function)
        contour_plot.set_x_log()
        contour_plot.add_country_labels(data_countries, data_color)
        contour_plot.add_x_y_labels('GDP per capita (2018 USD)', 'Cooling degree days (°C days)')
        
    else:
        contour_plot.add_cdd_predictions()
        contour_plot.add_x_y_labels('Mean GDP growth (annual %)', 'Cooling degree days increase (%)')
        # Replace ISO3 with full name
        cc = coco.CountryConverter()
        country = cc.convert(names=country, to='name_short', not_found=None)
        contour_plot.add_title(country, fontsize=14)

    contour_plot.add_legend()
    contour_plot.save_figure()

def plot_gdp_increase_map(configurations, gdp_cdd_data, future_scenario, exposure_func):
    """
    Plot annual average GDP growth for historical and constant experienced CDD
    """
    gdp_increase_plot = GDPIncreaseMap(configurations, gdp_cdd_data, 'gdp_increase_map_{0}'.format(future_scenario), future_scenario)
    gdp_increase_plot.add_subplots()
    gdp_increase_plot.remove_axes_ticks()
    gdp_increase_plot.add_title('Economic development', fontsize=12, make_sup_title=True)
    gdp_increase_plot.add_subtitles()
    gdp_increase_plot.create_exposure_map(exposure_func)
    gdp_increase_plot.plot_maps()
    gdp_increase_plot.save_figure()

def plot_gdp_increase_scatter(configurations, gdp_cdd_data, future_scenario, color_scale):
    """
    Plot difference in exposure times CDD as a function of GDP per capita
    """
    gdp_increase_scatter = GDPIncreaseScatter(configurations, gdp_cdd_data, 'gdp_increase_scatter_{0}'.format(future_scenario), future_scenario)
    gdp_increase_scatter.add_continent_info()
    gdp_increase_scatter.plot_scatter(color_scale)
    if color_scale != 'vs_gdp':
        gdp_increase_scatter.add_x_y_labels('Mean historical GDP growth (annual %)', 'Economic development to avoid increased heat exposure\nunder {0} (annual %)'.format(gdp_increase_scatter.formatted_scenario))
        gdp_increase_scatter.add_1_to_1_line()
    else:
        gdp_increase_scatter.add_x_y_labels('Current day GDP ($)', 'Economic development to avoid increased heat exposure\nunder {0} (annual %)'.format(gdp_increase_scatter.formatted_scenario))
    gdp_increase_scatter.label_countries(color_scale)
    gdp_increase_scatter.add_legend(columns=3)
    gdp_increase_scatter.save_figure()