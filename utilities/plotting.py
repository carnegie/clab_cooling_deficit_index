from utilities.plotting_classes import ExposurePlot, ContourPlot, ScatterPlot

def plot_variable_map(configurations, heat_exposure_df, plotting_variable):
    """
    Plot map of the variable (CDD or GDP)
    """
    map_plot = ExposurePlot(configurations, heat_exposure_df, '{0}_map'.format(plotting_variable))
    map_plot.create_data_map()
    var_name = plotting_variable.split('_')[0].lower() if len(plotting_variable.split('_')) == 1 else plotting_variable.split('_')[0].lower()+'_'+plotting_variable.split('_')[-1].lower()
    colormap = configurations['plotting']['{0}_cmap'.format(var_name)]
    map_plot.plot_map(plotting_variable, colormap=colormap, 
                      vmin=configurations['plotting'][var_name+'_min'],
                      vmax=configurations['plotting'][var_name+'_max'])
    map_plot.grey_empty_countries(plotting_variable)
    map_plot.remove_axes_ticks()
    map_plot.add_colorbar(label=configurations['plotting'][plotting_variable.split('_')[0].lower()+'_label'], 
                          colormap=colormap,
                          colorbar_min=configurations['plotting'][var_name+'_min'],
                                                    colorbar_max=configurations['plotting'][var_name+'_max'])
    map_plot.add_title(plotting_variable, fontsize=12)
    map_plot.save_figure()
    map_plot.show_close_figure()

def plot_income_groups(configurations, heat_exposure_df, col=None):
    """
    Highlight countries in each income group
    """
    for income_gr in configurations['income_groups_colors'].keys():
        map_plot_countries = ExposurePlot(configurations, heat_exposure_df, '{0}_{1}map'.format(income_gr, "_"+col if col else ""))
        map_plot_countries.create_data_map()
        color_min = 0
        color_max = 5
        map_plot_countries.color_countries(configurations['income_groups_colors'][income_gr], income_gr, col, cmin=color_min, cmax=color_max)
        if col:
            map_plot_countries.add_colorbar(label=col, colormap="cmap_{0}_income".format(income_gr), colorbar_min=color_min, colorbar_max=color_max)
        map_plot_countries.remove_axes_ticks()
        map_plot_countries.save_figure()
        map_plot_countries.show_close_figure()

def plot_exposure_contour(configurations, exposure_function, ac_data, x_y_ranges, country=None, name_tag='exposure_contour', multiply_cdd=True):
    """
    Contour plot of penetration of air conditioning as a function of GDP per capita and cooling degree days
    """
    contour_plot = ContourPlot(configurations, ac_data, name_tag, country=country)
    contour_plot.contour_grid(x_y_ranges[0], x_y_ranges[1])
    contour_plot.calculate_contour(exposure_function, multiply_cdd)
    contour_plot.plot_contour(multiply_cdd)
    contour_plot.add_contour_lines(multiply_cdd)
    data_points = configurations['income_groups_colors'].keys()
    data_color = configurations['income_groups_colors']
    contour_plot.add_data()
    contour_plot.set_y_log()
    contour_plot.add_labels(data_points, data_color)
    contour_plot.add_x_y_labels(configurations['plotting']['cdd_label'], configurations['plotting']['gdp_label'])
    contour_plot.save_figure()
    contour_plot.show_close_figure()

def plot_gdp_increase_scatter(configurations, gdp_cdd_data, future_scenario):
    """
    Plot difference in exposure times CDD as a function of GDP per capita
    """
    for income_group in configurations['income_groups_colors'].keys():
        gdp_increase_scatter = ScatterPlot(configurations, gdp_cdd_data[gdp_cdd_data['income_group'] == income_group],
                            'gdp_increase_scatter_{0}_{1}'.format(future_scenario, income_group), future_scenario)
        gdp_increase_scatter.plot_scatter(['gdp_historical_growth', 'gdp_const_{0}'.format(future_scenario), 
                                    'gdp_const_{0}_custom_exp_cdd'.format(future_scenario)], [income_group])
        gdp_increase_scatter.add_x_y_labels('Mean annual historical GDP growth (%)', 'Mean annual GDP growth to avoid\nincreased heat exposure nunder {0} (%)'.format(gdp_increase_scatter.formatted_scenario))
        gdp_increase_scatter.label_countries(configurations['label_countries'], ['gdp_historical_growth', 'gdp_const_{0}'.format(future_scenario), 
                                    'gdp_const_{0}_custom_exp_cdd'.format(future_scenario)])
        gdp_increase_scatter.add_1_to_1_line(['gdp_historical_growth', 'gdp_const_{0}'.format(future_scenario)], min=-2., max=10.)
        gdp_increase_scatter.save_figure()
        gdp_increase_scatter.show_close_figure()

def plot_cdd_scatter(configurations, gdp_cdd_data, future_scenario):
    """
    Plot difference in exposure times CDD as a function of GDP per capita
    """
    gdp_cdd_data = gdp_cdd_data.dropna(subset=['CDD', 'CDD_{0}_2100_diff'.format(future_scenario), 'GDP'])
    for income_group in configurations['income_groups_colors'].keys():
        cdd_scatter = ScatterPlot(configurations, gdp_cdd_data[gdp_cdd_data['income_group'] == income_group], 
                                                  'cdd_scatter_{0}_{1}'.format(future_scenario, income_group), future_scenario)
        cdd_scatter.plot_scatter(['CDD', 'CDD_{0}_2100_diff'.format(future_scenario), None],
                            configurations['income_groups_colors'].keys())
        cdd_scatter.label_countries(configurations['label_countries'], ['CDD', 'CDD_{0}_2100_diff'.format(future_scenario)])
        cdd_scatter.add_x_y_labels('CDD in {0} (°C days)'.format(configurations['analysis_years']['ref_year']), 'CDD diff in 2100\nunder {0} (°C days)'.format(cdd_scatter.formatted_scenario))
        cdd_scatter.save_figure()
        cdd_scatter.show_close_figure()