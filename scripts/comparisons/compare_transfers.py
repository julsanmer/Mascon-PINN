import matplotlib.pyplot as plt
import numpy as np
import pickle as pck
import os
import matplotlib.gridspec as gridspec
from matplotlib import rc

from Basilisk import __path__
bsk_path = __path__[0] + '/supportData/LocalGravData/'

m2km = 1e-3
km2R = 1/16
rad2deg = 180/np.pi

font = 22
font_label = 23
font_legend = 17
color_asteroid = [105/255, 105/255, 105/255]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
red = colors[3]
orange = colors[1]
colors[1] = red
colors[3] = orange
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
dpi = 600


# This function plots 2 transfers
def plot_transfers(transfers1, transfers2):
    def plot_transfer(transfers, idx, title=''):
        # Plot the first subplot
        for i in range(len(transfers.pos_list)):
            if transfers.pos_list[i] is not None:
                pos = transfers.pos_list[i]
                posf = transfers.posf[i]
                if i % 2:
                    ax[idx].plot(pos[0, 0] * m2km*km2R, pos[0, 1] * m2km*km2R, pos[0, 2] * m2km*km2R,
                                 marker='.', color=colors[0])
                    ax[idx].plot(pos[:, 0] * m2km*km2R, pos[:, 1] * m2km*km2R, pos[:, 2] * m2km*km2R,
                                 color=colors[0], linewidth=1.5)
                    ax[idx].plot(posf[0] * m2km*km2R, posf[1] * m2km*km2R, posf[2] * m2km*km2R,
                                 marker='s', color='k', markersize=3, zorder=10)

        # Plot the asteroid in the first subplot
        ax[idx].plot_trisurf(shape.xyz_vert[:, 0] * m2km*km2R,
                             shape.xyz_vert[:, 1] * m2km*km2R,
                             shape.xyz_vert[:, 2] * m2km*km2R,
                             triangles=shape.order_face-1,
                             color=color_asteroid,
                             zorder=0, alpha=.7)
        ax[idx].set_xlim([-40*km2R, 40*km2R])
        ax[idx].set_ylim([-40*km2R, 40*km2R])
        ax[idx].set_zlim([-40*km2R, 40*km2R])
        ax[idx].set_title(title, fontsize=font, pad=0)
        ax[idx].set_xlabel('$x/R$ [-]', fontsize=font_label, labelpad=10)
        ax[idx].set_ylabel('$y/R$ [-]', fontsize=font_label, labelpad=10)
        ax[idx].set_zlabel('$z/R$ [-]', fontsize=font_label, labelpad=10)
        ax[idx].tick_params(axis='both', labelsize=font)
        ax[idx].view_init(elev=35, azim=-110)
        ax[idx].grid(color='white')

    # Prepare for 3D plotting with 2x1 plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                           subplot_kw={'projection': '3d'})

    # # Plot the first subplot
    # idx = 0
    # for i in range(len(transfers1.pos_list)):
    #     if transfers1.pos_list[i] is not None:
    #         pos = transfers1.pos_list[i]
    #         posf = transfers1.posf[i]
    #         if i % 2:
    #             ax[0].plot(pos[0, 0] * m2km, pos[0, 1] * m2km, pos[0, 2] * m2km,
    #                        marker='.', color=colors[0])
    #             ax[0].plot(pos[:, 0] * m2km, pos[:, 1] * m2km, pos[:, 2] * m2km,
    #                        color=colors[0], linewidth=1.5)
    #             ax[0].plot(posf[0] * m2km, posf[1] * m2km, posf[2] * m2km,
    #                        marker='s', color='k', markersize=3, zorder=10)
    #         idx += 1

    # Plot the corresponding transfers
    plot_transfer(transfers1, 0, title='Transfers to Psyche crater')
    plot_transfer(transfers2, 1, title='Transfers to North Pole')
    if model == 'poly':
        fig.suptitle('Constant density polyhedron', fontsize=font)
    elif model == 'polyheterogeneous':
        fig.suptitle('Heterogeneous polyhedron', fontsize=font)
    plt.tight_layout()
    plt.savefig('Plots/transfers_' + model + '.png',
                format='png', dpi=dpi)


# This function plot transfer errors
def plot_errors(transfers1, transfers2):
    def plot_plane(transfer1, transfer2, idx, title='', map='xy'):
        # Get number of transfers
        n = len(transfer1.posf)
        posf_err1 = np.zeros((n, 3))
        posf_err2 = np.zeros((n, 3))
        err1 = np.ones(n) * np.nan
        err2 = np.ones(n) * np.nan
        line_list = []

        # Plot the first subplot
        for i in range(n):
            if transfer2.sol_list[i] is not None:
                # Errors for the first set of data
                posf_truth = transfer1.pos_list[i][-1, 0:3]
                posf = transfer1.posf[i, 0:3]
                posf_err1[i, 0:3] = posf_truth - posf
                err1[i] = np.linalg.norm(posf_err1[i, 0:3])
                if map == 'xy':
                    ax[idx].plot(posf_err1[i, 0], posf_err1[i, 1],
                                 marker='.', markersize=10,
                                 linestyle='', color=colors[0],
                                 alpha=0.8)
                elif map == 'xz':
                    ax[idx].plot(posf_err1[i, 0], posf_err1[i, 2],
                                 marker='.', markersize=10, linestyle='',
                                 color=colors[0],
                                 alpha=0.8)
                if i == 0:
                    ax[idx].plot(np.nan, np.nan, marker='.', color=colors[0],
                                 markersize=10, linestyle='', label='Mascon',
                                 alpha=0.8)

                # Errors for the second set of data
                posf_truth = transfer2.pos_list[i][-1, 0:3]
                posf = transfer2.posf[i, 0:3]
                posf_err2[i, 0:3] = posf_truth - posf
                err2[i] = np.linalg.norm(posf_err2[i, 0:3])
                if map == 'xy':
                    ax[idx].plot(posf_err2[i, 0], posf_err2[i, 1],
                                 marker='.', markersize=10, color=colors[1],
                                 alpha=0.8)
                elif map == 'xz':
                    ax[idx].plot(posf_err2[i, 0], posf_err2[i, 2],
                                 marker='.',  markersize=10, color=colors[1],
                                 alpha=0.8)
                if i == 0:
                    ax[idx].plot(np.nan, np.nan, marker='.', color=colors[1],
                                 markersize=10, linestyle='', label='Mascon-PINN',
                                 alpha=0.8)

        # Create accuracy circles for the first subplot
        rad = [0.5, 1, 1.5, 2, 2.5, 3]
        nu = np.linspace(0, 2*np.pi, 100)
        for i in range(len(rad)):
            r = rad[i]
            ax[idx].plot(r*np.cos(nu), r*np.sin(nu), color='k', linewidth=0.25)

        # Fanciness for the first subplot
        ax[idx].set_xlim([-2.5, 2.5])
        ax[idx].set_ylim([-2.5, 2.5])
        if idx == 0:
            ax[idx].set_ylabel('$\delta y\'$ [m]', fontsize=font_label)
            ax[idx].legend(fontsize=font_legend, loc='upper right')
        #ax[idx].set_xlabel(xlabel, fontsize=font)
        ax[idx].tick_params(axis='both', labelsize=font)
        ax[idx].set_title(title, fontsize=font, pad=1.5)
        ax[idx].set_xticks([-2, -1, 0, 1, 2])
        ax[idx].grid()

    # Prepare for 3D plotting with 1x2 plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 6),
                           gridspec_kw={'width_ratios': [1, 1.035]})
    plot_plane(transfers1[0], transfers2[0], 0,
               title='Transfers to Psyche crater', map='xz')
    plot_plane(transfers1[1], transfers2[1], 1,
               title='Transfers to North Pole', map='xy')
    ax[1].set_yticklabels([])
    fig.supxlabel('$\delta x\'$ [m]', fontsize=font, x=0.53, y=0.05)
    if model == 'poly':
        fig.suptitle('Constant density polyhedron', fontsize=font)
    elif model == 'polyheterogeneous':
        fig.suptitle('Heterogeneous polyhedron', fontsize=font)
    plt.tight_layout()
    plt.savefig('Plots/errortransfers_' + model + '.png',
                format='png', dpi=dpi, bbox_inches='tight')


# This function plot transfer errors
def plot_2errors(transfers1, transfers2):
    def plot_plane(transfer1, transfer2, idx, title='', map='xy'):
        # Min and max
        err_min = 0
        err_max = 2.5

        # Get number of transfers
        n = len(transfer1.posf)
        posf_err1 = np.zeros((n, 3))
        posf_err2 = np.zeros((n, 3))
        err1 = np.ones(n) * np.nan
        err2 = np.ones(n) * np.nan
        line_list = []

        # Plot the first subplot
        for i in range(n):
            if transfer2.sol_list[i] is not None:
                # Errors for the first set of data
                posf_truth = transfer1.pos_list[i][-1, 0:3]
                posf = transfer1.posf[i, 0:3]
                posf_err1[i, 0:3] = posf_truth - posf
                err1[i] = np.linalg.norm(posf_err1[i, 0:3])
                if map == 'xy':
                    sc = ax[idx].scatter(posf_err1[i, 0], posf_err1[i, 1], c=abs(posf_err1[i, 2]),
                                         cmap='viridis', marker='^', zorder=10,
                                         vmin=err_min, vmax=err_max, s=45)
                elif map == 'xz':
                    sc = ax[idx].scatter(posf_err1[i, 0], posf_err1[i, 2], c=abs(posf_err1[i, 1]),
                                         cmap='viridis', marker='^', zorder=10,
                                         vmin=err_min, vmax=err_max, s=45)
                if i == 0:
                    ax[idx].plot(np.nan, np.nan, marker='^', color='#440154FF', markersize=7.5,
                                 linestyle='', label='Mascon')

                # Errors for the second set of data
                posf_truth = transfer2.pos_list[i][-1, 0:3]
                posf = transfer2.posf[i, 0:3]
                posf_err2[i, 0:3] = posf_truth - posf
                err2[i] = np.linalg.norm(posf_err2[i, 0:3])
                if map == 'xy':
                    ax[idx].scatter(posf_err2[i, 0], posf_err2[i, 1], c=abs(posf_err2[i, 2]),
                                    cmap='viridis', marker='.', zorder=10,
                                    vmin=0, vmax=2, s=100)
                elif map == 'xz':
                    ax[idx].scatter(posf_err2[i, 0], posf_err2[i, 2], c=abs(posf_err2[i, 1]),
                                    cmap='viridis', marker='.', zorder=10,
                                    vmin=0, vmax=2, s=100)
                if i == 0:
                    ax[idx].plot(np.nan, np.nan, marker='.', color='#440154FF', markersize=10,
                                 linestyle='', label='Mascon-PINN')

        if map == 'xz':
            cbar = plt.colorbar(sc, ax=ax[1])
            cbar.set_label('Longitudinal error [m]', fontsize=font_map)
            cbar.ax.tick_params(labelsize=font_map)

        # Create accuracy circles for the first subplot
        rad = [0.5, 1, 1.5, 2, 2.5, 3]
        nu = np.linspace(0, 2*np.pi, 100)
        for i in range(len(rad)):
            r = rad[i]
            ax[idx].plot(r*np.cos(nu), r*np.sin(nu), color='k', linewidth=0.25)

        # Fanciness for the first subplot
        ax[idx].set_xlim([-2.5, 2.5])
        ax[idx].set_ylim([-2.5, 2.5])
        if map == 'xy':
            xlabel = '$\delta x$ [m]'
            ylabel = '$\delta y$ [m]'
        elif map == 'xz':
            xlabel = '$\delta x$ [m]'
            ylabel = '$\delta z$ [m]'
        if idx == 0:
            ax[idx].legend(fontsize=font_legend)
        ax[idx].set_xlabel(xlabel, fontsize=font)
        ax[idx].set_ylabel(ylabel, fontsize=font)
        ax[idx].tick_params(axis='both', labelsize=font)
        ax[idx].set_title(title, fontsize=font-3, pad=1.5)
        ax[idx].grid()

    # Prepare for 3D plotting with 1x2 plots
    fig, ax = plt.subplots(1, 2, figsize=(13, 6),
                           gridspec_kw={'width_ratios': [1, 1.2]})
    plot_plane(transfers1[0], transfers2[0], 0,
               title='Transfers to Psyche crater', map='xz')
    plot_plane(transfers1[1], transfers2[1], 1,
               title='Transfers to North pole', map='xy')
    if model == 'poly':
        fig.suptitle('Constant density polyhedron', fontsize=font-3)
    elif model == 'polyheterogeneous':
        fig.suptitle('Heterogeneous polyhedron', fontsize=font)
    plt.tight_layout()
    plt.savefig('Plots/errortransfers_' + model + '.png',
                format='png', dpi=dpi)


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
plot_transfers(pinn_list[0], pinn_list[1])
plot_errors(mascon_list, pinn_list)
plt.show()
