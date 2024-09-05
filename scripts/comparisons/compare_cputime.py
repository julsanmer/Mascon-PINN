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

font = 20
font_legend = 15
font_map = 15
color_asteroid = [105/255, 105/255, 105/255]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# Plot gravity errors w.r.t. altitude
def plot_tcpu(tcpu_gt, tcpu_mascon, tcpu_pinn):
    def remove_nan(t_cpu):
        t_cpu = t_cpu[~np.isnan(t_cpu)]

        return t_cpu

    # Make figure
    plt.gcf()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Plots list
    mascon_lines = []
    pinn_lines = []

    # Plots
    for i in range(len(tcpu_mascon)):
        tcpu_mascon[i] = remove_nan(tcpu_mascon[i])

        mascon_line, = ax.plot(tcpu_mascon[i]*1e3, linestyle='',
                               marker='^', markersize=4, color=colors[i],
                               linewidth=1.0, alpha=.2)
        mascon_lines.append(mascon_line)
        ax.plot(-100, -100, linestyle='',
                marker='^', color=colors[i], markersize=4,
                label='Mascon $n_M$=' + str(n_M[i]))

    # Plots
    for i in range(len(tcpu_pinn)):
        tcpu_pinn[i] = remove_nan(tcpu_pinn[i])

        pinn_line, = ax.plot(tcpu_pinn[i]*1e3, linestyle='',
                             marker='.', color=colors[i],
                             alpha=.2)
        mascon_lines.append(pinn_line)
        ax.plot(-100, -100, linestyle='',
                marker='.', color=colors[i],
                label='PINN 6x' + str(n_neurons[i]) + ' SIREN')

    tcpu_gt[0] = remove_nan(tcpu_gt[0])
    gt_line1, = ax.plot(tcpu_gt[0]*1e3, linestyle='',
                        marker='s', color='k',
                        alpha=.1, markersize=2)
    ax.plot(-100, -100, linestyle='',
            marker='s', color='k', markersize=2,
            label='Poly. 200700 faces')
    tcpu_gt[1] = remove_nan(tcpu_gt[1])
    gt_line2, = ax.plot(tcpu_gt[1]*1e3, linestyle='',
                        marker='s', color='r', linewidth=1.0,
                        alpha=.1, markersize=2)
    ax.plot(-100, -100, linestyle='',
            marker='s', color='r', markersize=2,
            label='Poly. 7790 faces')

    ax.set_yscale('log')
    ax.set_xlim(0, len(tcpu_gt[0]))
    ax.set_ylim(1e-3, 100)
    plt.xlabel('Gravity evaluations [-]', fontsize=font)
    plt.ylabel('Computation time [ms]', fontsize=font)
    ax.tick_params(axis='both', labelsize=font)
    ax.legend(fontsize=font_legend, ncol=3, loc='upper right')
    ax.grid()

    plt.savefig('Plots/tcpugravity.png', format='png',
                dpi=400)


# Load files
file_path = os.path.dirname(os.getcwd())
filepinn_list = []

# Change working path
os.chdir('/Users/julio/Desktop/python_scripts/THOR/scripts')

# Scenario
model = 'poly'
faces = '200700faces'
#asteroid = 'polyheterogeneous'

# Number of masses
n_M = np.array([20,
                100,
                1000])
#n_neurons = np.repeat(40, len(n_M))
filepath_results = 'Results/eros/results/' + model + faces + '/ideal/dense_alt50km_100000samples'
filepath_gt = 'Results/eros/groundtruth/'

# Loop mascon to fill its list
tcpu_mascon = []
for i in range(len(n_M)):
    # Files
    file_mascon = filepath_results + '/mascon' + str(n_M[i]) + '_muxyz_quadratic_octantrand0.pck'

    # Scenarios
    scenario_mascon = pck.load(open(file_mascon, "rb"))
    tcpu_mascon.append(np.ravel(scenario_mascon.estimation.gravmap.map_3D.t_cpu))

# Variables
n_neurons = np.array([20, 40, 80])
n_M1 = np.repeat(100, len(n_neurons))

# Loop pinn to fill its list
tcpu_pinn = []
for i in range(len(n_neurons)):
    file_pinn = filepath_results + '/pinn6x' + str(n_neurons[i]) + 'SIREN_linear_mascon' + str(n_M1[i]) + '.pck'

    # Scenarios
    scenario_pinn = pck.load(open(file_pinn, "rb"))

    tcpu_pinn.append(np.ravel(scenario_pinn.estimation.gravmap.map_3D.t_cpu))

# Groundtruth
filepoly_200k = filepath_gt + model + faces + '/' + 'dense_alt50km_100000samples.pck'
filepoly_7k = filepath_gt + model + '7790faces' + '/' + 'dense_alt50km_100000samples.pck'
scenario_poly200k = pck.load(open(filepoly_200k, "rb"))
scenario_poly7k = pck.load(open(filepoly_7k, "rb"))
tcpu_gt_200k = np.ravel(scenario_poly200k.groundtruth.gravmap.map_3D.t_cpu)
tcpu_gt_7k = np.ravel(scenario_poly7k.groundtruth.gravmap.map_3D.t_cpu)
tcpu_gt = [tcpu_gt_200k,
           tcpu_gt_7k]

# Plot gravity 3D
plot_tcpu(tcpu_gt, tcpu_mascon, tcpu_pinn)
plt.show()
