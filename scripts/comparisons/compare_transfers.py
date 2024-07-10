import matplotlib.pyplot as plt
import numpy as np
import pickle as pck
import os
import matplotlib.gridspec as gridspec
from matplotlib import rc

from Basilisk import __path__
bsk_path = __path__[0] + '/supportData/LocalGravData/'

m2km = 1e-3
rad2deg = 180/np.pi

font = 20
font_legend = 15
font_map = 15
color_asteroid = [105/255, 105/255, 105/255]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# This function plots 2 transfers
def plot_2transfers(transfers1, transfers2):
    def plot_transfer(transfers, idx, title=''):
        # Plot the first subplot
        for i in range(len(transfers.pos_list)):
            if transfers.pos_list[i] is not None:
                pos = transfers.pos_list[i]
                posf = transfers.posf[i]
                if i % 2:
                    ax[idx].plot(pos[0, 0] * m2km, pos[0, 1] * m2km, pos[0, 2] * m2km,
                                 marker='.', color=colors[0])
                    ax[idx].plot(pos[:, 0] * m2km, pos[:, 1] * m2km, pos[:, 2] * m2km,
                                 color=colors[0], linewidth=0.5)
                    ax[idx].plot(posf[0] * m2km, posf[1] * m2km, posf[2] * m2km,
                                 marker='s', color='k', markersize=2, zorder=10)

        # Plot the asteroid in the first subplot
        ax[idx].plot_trisurf(shape.xyz_vert[:, 0] * m2km,
                             shape.xyz_vert[:, 1] * m2km,
                             shape.xyz_vert[:, 2] * m2km,
                             triangles=shape.order_face - 1,
                             color=color_asteroid,
                             zorder=0, alpha=.7)
        ax[idx].set_xlim([-40, 40])
        ax[idx].set_ylim([-40, 40])
        ax[idx].set_zlim([-40, 40])
        ax[idx].set_title(title, fontsize=font_map, pad=0)
        ax[idx].set_xlabel('$x$ [km]', fontsize=font-5)
        ax[idx].set_ylabel('$y$ [km]', fontsize=font-5)
        ax[idx].set_zlabel('$z$ [km]', fontsize=font-5)
        ax[idx].tick_params(axis='both', labelsize=font-5)

    # Prepare for 3D plotting with 2x1 plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5),
                           subplot_kw={'projection': '3d'})

    # Plot the first subplot
    idx = 0
    for i in range(len(transfers1.pos_list)):
        if transfers1.pos_list[i] is not None:
            pos = transfers1.pos_list[i]
            posf = transfers1.posf[i]
            if i % 2:
                ax[0].plot(pos[0, 0] * m2km, pos[0, 1] * m2km, pos[0, 2] * m2km,
                           marker='.', color=colors[0])
                ax[0].plot(pos[:, 0] * m2km, pos[:, 1] * m2km, pos[:, 2] * m2km,
                           color=colors[0], linewidth=0.5)
                ax[0].plot(posf[0] * m2km, posf[1] * m2km, posf[2] * m2km,
                           marker='s', color='k', markersize=2, zorder=10)
            idx += 1

    # Plot the corresponding transfers
    plot_transfer(transfers1, 0, title='Psyche crater transfers')
    plot_transfer(transfers2, 1, title='North pole transfers')
    if model == 'poly':
        fig.suptitle('Constant density polyhedron', fontsize=font-5)
    elif model == 'polyheterogeneous':
        fig.suptitle('Heterogeneous polyhedron', fontsize=font-5)
    plt.tight_layout()
    plt.savefig('Plots/transfers_' + model + '.pdf',
                format='pdf', bbox_inches='tight')


# This function plot transfer errors
def plot_2errors(transfers1, transfers2):
    def plot_errors(transfer1, transfer2, idx, title='', map='xy'):
        # Get number of transfers
        n = len(transfer1.posf)
        err1 = np.ones(n) * np.nan
        err2 = np.ones(n) * np.nan
        line_list = []

        # Plot the first subplot
        for i in range(n):
            if transfer2.sol_list[i] is not None:
                # Errors for the first set of data
                posf_truth = transfer1.pos_list[i][-1, 0:3]
                posf = transfer1.posf[i, 0:3]
                err1[i] = np.linalg.norm(posf_truth - posf)
                if map == 'xy':
                    line, = ax[idx].plot(posf_truth[0], posf_truth[1],
                                         marker='.', color=colors[0], linestyle='',
                                         label='Mascon', zorder=10)
                elif map == 'xz':
                    line, = ax[idx].plot(posf_truth[0], posf_truth[2],
                                         marker='.', color=colors[0], linestyle='',
                                         label='Mascon', zorder=10)
                if i == 0:
                    line_list.append(line)

                # Errors for the second set of data
                posf_truth = transfer2.pos_list[i][-1, 0:3]
                posf = transfer2.posf[i, 0:3]
                err2[i] = np.linalg.norm(posf_truth - posf)
                if map == 'xy':
                    line, = ax[idx].plot(posf_truth[0], posf_truth[1],
                                         marker='.', color=colors[1], linestyle='',
                                         zorder=10)
                elif map == 'xz':
                    line, = ax[idx].plot(posf_truth[0], posf_truth[2],
                                         marker='.', color=colors[1], linestyle='',
                                         zorder=10)
                if i == 0:
                    line_list.append(line)

        # Create accuracy circles for the first subplot
        rad = [0.5, 1, 1.5, 2, 2.5, 3]
        nu = np.linspace(0, 2 * np.pi, 100)
        for i in range(len(rad)):
            r = rad[i]
            ax[idx].plot(r * np.cos(nu), r * np.sin(nu), color='k', linewidth=0.25)

        # Fanciness for the first subplot
        ax[idx].set_xlim([-2.5, 2.5])
        ax[idx].set_ylim([-2.5, 2.5])
        if map == 'xy':
            xlabel = '$\delta x$ [m]'
            ylabel = '$\delta y$ [m]'
        elif map == 'xz':
            xlabel = '$\delta x$ [m]'
            ylabel = '$\delta z$ [m]'
        legend = plt.legend(line_list, ['Mascon', 'Mascon-PINN'], loc='upper right', fontsize=font_legend)
        #ax[idx].add_artist(legend)
        ax[idx].set_xlabel(xlabel, fontsize=font)
        ax[idx].set_ylabel(ylabel, fontsize=font)
        ax[idx].tick_params(axis='both', labelsize=font)
        ax[idx].set_title(title, fontsize=font_map, pad=1.5)
        ax[idx].grid()

    # Prepare for 3D plotting with 1x2 plots
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    plot_errors(transfers1[0], transfers2[0], 0,
                title='Psyche crater transfers', map='xz')
    plot_errors(transfers1[1], transfers2[1], 1,
                title='North pole transfers', map='xy')
    if model == 'poly':
        fig.suptitle('Constant density polyhedron', fontsize=font-5)
    elif model == 'polyheterogeneous':
        fig.suptitle('Heterogeneous polyhedron', fontsize=font-5)
    plt.tight_layout()
    plt.savefig('Plots/errortransfers_' + model + '.pdf',
                format='pdf', bbox_inches='tight')


# Change working path
current_path = os.getcwd()
new_path = os.path.dirname(current_path)
os.chdir(new_path)

# Scenario
model = 'poly'
#model = 'polyheterogeneous'
type_list = ['equatorial', 'polar']
faces = '200700faces'

#asteroid = 'polyheterogeneous200700faces'
if model == 'poly':
    title = 'Constant density polyhedron'
elif model == 'polyheterogeneous':
    title = 'Heterogeneous polyhedron'

# Groundtruth orbits
file_gt = 'Results/eros/groundtruth/' + model + faces + '/propagation.pck'
orbits_gt = pck.load(open(file_gt, "rb"))
shape = orbits_gt.asteroid.shape
shape.create_shape()

# Number of masses
n_M = 100
n_neurons = 40
file_path = 'Results/eros/results/' + model + faces + '/ideal/dense_alt50km_100000samples'

# Lists
mascon_list = []
pinn_list = []

# Loop through results
for type in type_list:
    # Files
    file_mascon = file_path + '/mascon' + str(n_M) \
                  + '_muxyz_quadratic_octantrand0_transfers' + type + '.pck'
    file_pinn = file_path + '/pinn6x' + str(n_neurons) \
                + 'SIREN_linear_mascon' + str(n_M) + '_transfers' + type + '.pck'

    # Scenarios
    transfers_mascon = pck.load(open(file_mascon, "rb"))
    transfers_pinn = pck.load(open(file_pinn, "rb"))

    # List of orbits
    mascon_list.append(transfers_mascon)
    pinn_list.append(transfers_pinn)

#plot_transfers(pinn_list[0])
plot_2transfers(pinn_list[0], pinn_list[1])
plot_2errors(mascon_list, pinn_list)
plt.show()
