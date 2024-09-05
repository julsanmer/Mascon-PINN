import numpy as np
import pickle as pck
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib import rc

# ------ COMPONENTS & SUBPLOT HANDLING ------ #
rad2deg = 180 / np.pi
sec2day = 1.0 / (3600*24)
m2km = 1e-3
m2mm = 1e3
m2mu = 1e6
err2perc = 1e2
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

font = 25
font_label = 27
font_legend = 21
font_map = 19
font_maplabel = 20
color_asteroid = [105/255, 105/255, 105/255]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
dpi = 600


# This function plots gravity errors w.r.t. altitude
def plot_gravity3D(h_bins, aErrAlt_mascon, aErrAlt_pinn, title=''):
    # Make figure
    plt.gcf()
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Plots list
    mascon_lines = []
    pinn_lines = []

    # Plots
    for i in range(len(aErrAlt_mascon)):
        mascon_line, = ax.plot(h_bins/R, aErrAlt_mascon[i] * err2perc,
                               linestyle='--', color=colors[i], linewidth=2.5)
        pinn_line, = ax.plot(h_bins/R, aErrAlt_pinn[i] * err2perc,
                             linestyle='-', color=colors[i], linewidth=3.0,
                             label='$n_M$=' + str(n_M[i]))
        mascon_lines.append(mascon_line)
        pinn_lines.append(pinn_line)

    ax.set_yscale('log')

    # Make addons
    plt.xlabel('$h/R$ [-]', fontsize=font_label)
    plt.ylabel('Average gravity error [\%]', fontsize=font_label)
    ax.set_xlim(h_bins[0]/R, h_bins[-1]/R)
    ax.set_ylim([1e-5, 1.5*1e1])
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
    legend1 = plt.legend(pinn_lines, [f'$n_M$={n}' for n in n_M],
                         loc='upper right', fontsize=font_legend)
    ax.add_artist(legend1)
    legend2 = plt.legend([pinn_lines[0], mascon_lines[0]], ['Mascon-PINN', 'Mascon'],
                         loc='upper left', fontsize=font_legend)
    ax.tick_params(axis='both', labelsize=font)
    ax.grid()
    ax.set_title(title, fontsize=font)

    # Save figure
    plt.savefig('Plots/gravityerror_nM_' + model + '.png', format='png',
                dpi=dpi)


# This function plots a error map of the surface
def plot_surfacemap(mapsurf_mascon, mapsurf_pinn):
    def plot_map(mapsurf, idx, title=''):
        # Grid
        lon_grid = mapsurf.lon
        lat_grid = mapsurf.lat
        aErr = mapsurf.aErr_surf

        # Compute max and average
        aErr_max = np.max(aErr)
        aErr_mean = np.mean(aErr)

        # Clip to min and max
        np.clip(aErr, aErrmin_plot, aErrmax_plot, out=aErr)

        # Make min and max reflect to appear on corners
        aErr[0, 0], aErr[-1, 0] = aErrmin_plot, aErrmax_plot

        # Plot the data on the first subplot
        ax = axs[idx]
        surf = ax.contourf(lon_grid * rad2deg, lat_grid * rad2deg,
                           aErr * err2perc, cmap=mpl.colormaps['viridis'],
                           norm=lognorm)
        clines = ax.contour(lon_grid * rad2deg, lat_grid * rad2deg,
                            aErr * err2perc, levels=[1e-2, 1e-1, 1e0, 1e1, 1e2],
                            colors='white', linewidths=0.35)
        ax.set_xticks([-180, -135, -90, -45, 0,
                       45, 90, 135, 180])
        ax.set_yticks([-90, -45, 0, 45, 90])
        ax.tick_params(axis='both', labelsize=font_maplabel)
        ax.set_title(title, fontsize=font_map)

        # Add a textbox to the first subplot
        textstr = 'Mean: ' + f'{aErr_mean*err2perc:.2f}' + '\%' + '\n' \
                  + 'Max: ' + f'{aErr_max*err2perc:.2f}' + '\%'
        props = dict(boxstyle='round', facecolor='black', alpha=0.8)
        ax.text(0.825, 0.29, textstr, transform=ax.transAxes, fontsize=font_map,
                verticalalignment='top', bbox=props, color='white')

        # Plot xlabel
        if idx == 1:
            ax.set_xlabel('Longitude [$^{\circ}$]', fontsize=font_maplabel)
            fig.supylabel('Latitude [$^{\circ}$]', fontsize=font_maplabel)
            fig.tight_layout()
            cbar = fig.colorbar(surf, ax=axs, shrink=1, pad=0.05)
            cbar.ax.tick_params(labelsize=font_map)
            cbar.set_label('Gravity error [\%]', rotation=90, fontsize=font_maplabel)

    # Min and max
    aErrmin_plot = 1.00000001 * 1e-4
    aErrmax_plot = 0.99999999 * 1e0

    # Create the figure and two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    # Logarithmic norm and lines labels
    lognorm = mpl.colors.LogNorm(vmin=aErrmin_plot * err2perc,
                                 vmax=aErrmax_plot * err2perc,
                                 clip=True)
    lines_labels = {1e-2: '0.01\%',
                    1e-1: '0.1\%',
                    1e0: '1\%',
                    1e1: '10\%',
                    1e2: '100\%'}

    # Plot maps
    plot_map(mapsurf_mascon, 0, title='Mascon $n_M=100$')
    plot_map(mapsurf_pinn, 1, title='Mascon $n_M=100$ + PINN SIREN 6x40')

    # Save figure
    plt.savefig('Plots/gravityerror_surface_' + model + '.png', format='png',
                dpi=dpi)


# Load files
file_path = os.path.dirname(os.getcwd())
filepinn_list = []

# Change working path
os.chdir('/Users/julio/Desktop/python_scripts/THOR/scripts')

# Scenario
model = 'poly'
#model = 'polyheterogeneous'
faces = '200700faces'

#asteroid = 'polyheterogeneous200700faces'
if model == 'poly':
    title = 'Constant density polyhedron'
elif model == 'polyheterogeneous':
    title = 'Heterogeneous polyhedron'

# Number of masses
n_M = np.array([8,
                20,
                50,
                100,
                1000])
n_neurons = np.repeat(40, len(n_M))
file_path = 'Results/eros/results/' + model + faces + '/ideal/dense_alt50km_100000samples'

# Lists
aErrAlt_mascon = []
aErrAlt_pinn = []

mascon_list = []
pinn_list = []

# Train
for i in range(len(n_M)):
    # Files
    file_mascon = file_path + '/mascon' + str(n_M[i]) + '_muxyz_quadratic_octantrand0.pck'
    file_pinn = file_path + '/pinn6x' + str(n_neurons[i]) + 'SIREN_linear_mascon' + str(n_M[i]) + '.pck'
    filepinn_list.append(file_pinn)

    # Scenarios
    scenario_mascon = pck.load(open(file_mascon, "rb"))
    scenario_pinn = pck.load(open(file_pinn, "rb"))

    mascon_list.append(scenario_mascon)
    pinn_list.append(scenario_pinn)

    # Asteroid radius
    R = 16*1e3

    # Altitude and error bins
    h_bins = scenario_mascon.estimation.gravmap.intervals.h_bins
    aErrAlt_mascon.append(scenario_mascon.estimation.gravmap.intervals.aErrAlt_bins)
    aErrAlt_pinn.append(scenario_pinn.estimation.gravmap.intervals.aErrAlt_bins)

# Plot gravity 3D
plot_gravity3D(h_bins, aErrAlt_mascon, aErrAlt_pinn, title=title)
plot_surfacemap(mascon_list[3].estimation.gravmap.map_surf2D,
                pinn_list[3].estimation.gravmap.map_surf2D)
plt.show()
