# Plotting scripts for experienced temperature project

import os
import matplotlib.pyplot as plt

def plot_map(pop_weighted_cdd, pop_year, temp_year):
    """
    Plot the population weighted cooling degree days
    """
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the data
    pop_plot_2000 = ax.imshow(pop_weighted_cdd, origin='lower', aspect='auto', cmap='viridis')

    # Add a colorbar
    cbar = plt.colorbar(pop_plot_2000, ax=ax, orientation='vertical')
    cbar.set_label('Population weighted degree days')
    
    # Flip the y-axis so that 0 is at the top
    ax.invert_yaxis()


    # Set the labels for the x and y axes
    ax.set_xlabel('Longitude Index')
    ax.set_ylabel('Latitude Index')

    # Set the title of the plot
    ax.set_title('Population weighted degree days for the Year pop {0}, temp {1}'.format(pop_year, temp_year))

    # Save the figure
    if not os.path.exists('Figures'):
        os.makedirs('Figures')
    plt.savefig('Figures/population_degree_days_{0}_{1}.png'.format(pop_year, temp_year))


def plot_time_curve(data_dict):
    """
    Plot all curves in one plot to show the effect of population and temperature on cooling degree days
    """
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = [2000, 2005, 2010, 2015, 2020]
    # Show x labels as integers
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    pop_effect = [data_dict["population_effect"]["pop{0}_temp2000".format(year)] for year in x]
    temp_effect = [data_dict["temperature_effect"]["pop2000_temp{0}".format(year)] for year in x]
    both_effect = [data_dict["both_effects"]["pop{0}_temp{0}".format(year, year)] for year in x]
    ax.plot(x, pop_effect, label='Population effect')
    ax.plot(x, temp_effect, label='Temperature effect')
    ax.plot(x, both_effect, label='Both effects')
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel('Population weighted cooling degree days')
    # Some white space between plot and title
    plt.subplots_adjust(top=0.85)
    ax.set_title('Effects of population and temperature change on population-weighted cooling degree days')

    # Save the figure
    if not os.path.exists('Figures'):
        os.makedirs('Figures')
    plt.savefig('Figures/population_temperature_effect.png')