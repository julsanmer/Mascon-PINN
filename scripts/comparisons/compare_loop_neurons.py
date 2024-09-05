import numpy as np
import pickle as pck
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

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
color_asteroid = [105/255, 105/255, 105/255]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
dpi = 600


# Plot gravity errors w.r.t. altitude
def plot_gravity3D(h_bins, aErrAlt_mascon, aErrAlt_pinn, title=''):
    # Make figure
    plt.gcf()
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 1, 1)

    # List for plots
    mascon_lines = []
    pinn_lines = []

    # Plots
    for i in range(len(aErrAlt_mascon)):
        mascon_line, = ax.plot(h_bins/R, aErrAlt_mascon[i] * err2perc,
                               linestyle='--', color=colors[i], linewidth=2.5)
        pinn_line, = ax.plot(h_bins/R, aErrAlt_pinn[i] * err2perc,
                             linestyle='-', color=colors[i], linewidth=3.0,
                             label='SIREN 6x' + str(n_neurons[i]))
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
    legend1 = plt.legend(handles=pinn_lines, loc='upper right', fontsize=font_legend)
    ax.add_artist(legend1)
    legend2 = plt.legend([pinn_lines[0], mascon_lines[0]], ['Mascon-PINN', 'Mascon'],
                         loc='upper left', fontsize=font_legend)
    ax.tick_params(axis='both', labelsize=font)
    ax.grid()
    ax.set_title(title, fontsize=font)

    # Save figure
    plt.savefig('Plots/gravityerror_neurons_' + model + '.png', format='png',
                dpi=dpi)


# Load files
file_path = os.path.dirname(os.getcwd())
filepinn_list = []

# Change working path
os.chdir('/Users/julio/Desktop/python_scripts/THOR/scripts')

# Number of masses
n_neurons = np.array([10,
                      20,
                      40,
                      80,
                      160])
n_M = np.repeat(100, len(n_neurons))

# Scenario
model = 'poly'
#model = 'polyheterogeneous'
faces = '200700faces'
file_path = 'Results/eros/results/' + model + faces + '/ideal/dense_alt50km_100000samples'

if model == 'poly':
    title = 'Constant density polyhedron'
elif model == 'polyheterogeneous':
    title = 'Heterogeneous polyhedron'

# Lists
aErrAlt_mascon = []
aErrAlt_pinn = []

# Train
for i in range(len(n_neurons)):
    # Files
    file_mascon = file_path + '/mascon' + str(n_M[i]) + '_muxyz_quadratic_octantrand0.pck'
    file_pinn = file_path + '/pinn6x' + str(n_neurons[i]) + 'SIREN_linear_mascon' + str(n_M[i]) + '.pck'
    filepinn_list.append(file_pinn)

    # Scenarios
    scenario_mascon = pck.load(open(file_mascon, "rb"))
    scenario_pinn = pck.load(open(file_pinn, "rb"))

    # Asteroid radius
    R = 16*1e3

    # Altitude and error bins
    h_bins = scenario_mascon.estimation.gravmap.intervals.h_bins
    aErrAlt_mascon.append(scenario_mascon.estimation.gravmap.intervals.aErrAlt_bins)
    aErrAlt_pinn.append(scenario_pinn.estimation.gravmap.intervals.aErrAlt_bins)

# Plot gravity 3D
plot_gravity3D(h_bins, aErrAlt_mascon, aErrAlt_pinn, title=title)
plt.show()
