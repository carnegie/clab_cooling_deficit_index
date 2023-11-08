import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_ac_data(data_file):
    """
    Read AC data from file created with derive_exposure_functions.ipynb
    """
    ac_data = pd.read_csv(data_file)
    # Remove rows with missing data
    ac_data = ac_data.dropna()
    # Reindex
    ac_data = ac_data.reset_index(drop=True)
    return ac_data


def exposure_new(gdp, cdd):
    """
    This function calculates the exposure of a country to climate change
    """
    avail_new = (1/(1 + np.exp(3.2)* np.exp(-0.00011*gdp)))
    saturation_new = (0.93 - 0.93*np.exp(-0.005*cdd))
    return (1 - avail_new*saturation_new)


def gdp_from_cdd_exposure(exposure_cdd, cdd):
    """
    This function calculates the GDP per capita assuming fixed exposure * cooling degree days
    """
    sat = (0.93 - 0.93*np.exp(-0.005*cdd))
    sat.index = exposure_cdd.index
    cdd.index = exposure_cdd.index
    return (np.log((1./np.exp(3.2))*((sat/(1 - exposure_cdd/cdd)) - 1))/(-0.00011))



def exposure_contour(exposure_function, ac_data, multiply_cdd=False, add_data=True, contour_lines=False, name_tag='exposure_contour', future_scenario=''):
    """
    Conntour plot of penetration of air conditioning as a function of GDP per capita and cooling degree days
    """
    plt.figure()
    cdd_x = np.linspace(0, 4500, 100)
    gdp_x = np.linspace(0, 200000, 100)
    if multiply_cdd:
        level_max = 4500.
        add_label = ' multiplied by CDD'
        z = (1.-ac_data['AC'])*ac_data['DD_mean']
        name_tag += '_multipliedCDD'
    else:
        level_max = 1.
        add_label = ''
        z = (1.-ac_data['AC'])
    levels = np.linspace(0, level_max, 21)
    cdd_x, gdp_x = np.meshgrid(cdd_x, gdp_x)
    if multiply_cdd:
        plt.contourf(gdp_x, cdd_x, exposure_function(gdp_x, cdd_x)*cdd_x, levels=levels)
    else:
        plt.contourf(gdp_x, cdd_x, exposure_function(gdp_x, cdd_x), levels=levels)
    plt.colorbar(label='Exposure to outside temperatures{0}'.format(add_label), ticks=np.linspace(0, level_max, 11))

    if contour_lines:
        # Add contour lines
        if multiply_cdd:
            clines = plt.contour(gdp_x, cdd_x, exposure_function(gdp_x, cdd_x)*cdd_x, levels=levels, colors='k', linewidths=0.)
            label_prec = '%d'
        else:
            clines = plt.contour(gdp_x, cdd_x, exposure_function(gdp_x, cdd_x), levels=levels, colors='k', linewidths=0.)
            label_prec = '%.2f'
        plt.clabel(clines, levels[::2], fmt=label_prec, fontsize=8, colors='black')

    plt.xlabel('GDP per capita in 2018 USD')
    # GDP log scale

    plt.xscale('log')
    # plt.xlim(1500, 100000) 
    plt.ylabel('Cooling degree days')
    # color bar range is 0 to 1
    plt.clim(0, level_max)
    # Add label in red and bold
    plt.title('Exposure to outside temperatures{0}\n as a function of GDP and CDD{1}'.format(add_label, future_scenario))
    # if "new" in name_tag:
    #     plt.text(0.02, 0.02, 'New exposure function', color='red', fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
     
    if add_data:
        # Overlay AC data
        # Plot AC access as a function of GDP per capita and cooling degree days
        ac_data_2100 = ac_data[ac_data['ISO3'].str.contains('2100')]

        plt.scatter(ac_data['GDP'], ac_data['DD_mean'], c=z, cmap='viridis', label='AC access', vmin=0., vmax=level_max)
        plt.scatter(ac_data_2100['GDP'], ac_data_2100['DD_mean'], c='orange', s=12)
        
        # Label points with country names
        for i, txt in enumerate(ac_data['ISO3'].values):
            if not np.isnan(z[i]):
                if multiply_cdd:
                    z_label = str(int(z[i]))
                else:
                    z_label = str(round(z[i],2))
            else:
                z_label = ''
            if not '2100' in txt:
                color = 'white'
            else:
                color = 'orange'
            plt.annotate(txt+"\n"+z_label, (ac_data['GDP'][i]*1.05, ac_data['DD_mean'][i]-100), fontsize=8.5, color=color)

    plt.savefig('Figures/exposure_funct_analysis/{0}.png'.format(name_tag), dpi=300)