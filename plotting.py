# Plotting scripts for experienced temperature project

import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_map(pop_weighted_cdd, pop_year, temp_year, gdp_year):
    """
    Plot the population weighted cooling degree days
    """
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the data
    # Use imshow to plot the data with the latitude values on the y-axis and longitude values on the x-axis
    # color bar same height as plot
    pop_plot = ax.imshow(pop_weighted_cdd, extent=[-180, 180, 90, -90], origin='lower', cmap='coolwarm', norm=LogNorm(vmin=1e6, vmax=3e9))
    # Add a colorbar
    # Create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)

    # Add a colorbar in the created axes
    cbar = plt.colorbar(pop_plot, cax=cax, orientation='vertical')
    cbar.set_label('Population and GDP weighted degree hours')
    
    # Flip the y-axis so that 0 is at the top
    ax.invert_yaxis()


    # Set the labels for the x and y axes
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')

    # Set the title of the plot
    ax.set_title('Population and GDP weighted cooling degree hours for the Year pop {0}, temp {1}, gdp {2}'.format(pop_year, temp_year, gdp_year), pad=20)

    # Save the figure
    if not os.path.exists('Figures'):
        os.makedirs('Figures')
    plt.savefig('Figures/population_degree_days_{0}_{1}_{2}.png'.format(pop_year, temp_year, gdp_year))


def plot_time_curve(data_dict, plot_years):
    """
    Plot all curves in one plot to show the effect of population and temperature on cooling degree days
    """
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    # Show x labels as integers
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    # Plot the data
    pop_effect = [data_dict["population_effect"]["pop{0}_temp2000_gdp2000".format(year)] for year in plot_years]
    temp_effect = [data_dict["temperature_effect"]["pop2000_temp{0}_gdp2000".format(year)] for year in plot_years]
    gdp_effect = [data_dict["gdp_effect"]["pop2000_temp2000_gdp{0}".format(year)] for year in plot_years]
    all_effects = [data_dict["all_effects"]["pop{0}_temp{0}_gdp{0}".format(year, year)] for year in plot_years]
    ax.plot(plot_years, pop_effect, label='Population effect')
    ax.plot(plot_years, temp_effect, label='Temperature effect')
    ax.plot(plot_years, gdp_effect, label='GDP effect/AC penetration')
    ax.plot(plot_years, all_effects, label='All effects')

    # Add a legend
    ax.legend()
    ax.set_xlabel('Year')
    # x tick labels at plot_years
    ax.set_xticks(plot_years)
    ax.set_xlim([plot_years[0], plot_years[-1]])
    ax.set_ylabel('Population and GDP weighted cooling degree hours, base T=18°C')
    # Dashed horizontal line at reference year
    ax.axhline(data_dict["all_effects"]["pop2000_temp2000_gdp2000"], color='black', linestyle='--')
    ax.set_title('Effects of population and temperature change on population-weighted cooling degree days', pad=20)

    # Save the figure
    if not os.path.exists('Figures'):
        os.makedirs('Figures')
    plt.savefig('Figures/population_temperature_effect.png')