import numpy as np
import pickle as pck
import os
import matplotlib.pyplot as plt

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


def plot_orbits():
    # Get polyhedron
    #oe_list = orbits_pinn.groundtruth.oe_0
    xyz_vert = orbits_ref.asteroid.shape.xyz_vert
    order_face = orbits_ref.asteroid.shape.order_face

    # Position list
    n_a = orbits_ref.n_a
    n_inc = orbits_ref.n_inc

    # Create figure, plot asteroid and orbits
    fig, axs = plt.subplots(2, 2, figsize=(10, 8),
                            subplot_kw={'projection': '3d'})

    # Flatten the axs array for easier iteration
    axs = axs.flatten()

    # Plot data in each subplot
    for j, ax in enumerate(axs):
        ax.plot_trisurf(xyz_vert[:, 0] * m2km,
                        xyz_vert[:, 1] * m2km,
                        xyz_vert[:, 2] * m2km,
                        triangles=order_face-1,
                        color=color_asteroid, zorder=0)
        for i in range(n_a):
            pos_ref = orbits_ref.orbits[i][j].pos_BP_P
            #pos_pinn = poslist_pinn[i][j]
            # ax.plot(pos_groundtruth[0, 0] * m2km, pos_groundtruth[0, 1] * m2km,
            #         pos_groundtruth[0, 2] * m2km, 'b', marker='.')
            ax.plot(pos_ref[:, 0] * m2km, pos_ref[:, 1] * m2km,
                    pos_ref[:, 2] * m2km, color=colors[i],
                    zorder=20, linewidth=1)
            # ax.plot(pos_pinn[:, 0] * m2km, pos_pinn[:, 1] * m2km,
            #         pos_pinn[:, 2] * m2km, 'r', zorder=20, linewidth=0.5)
            # ax.plot(pos[-1, 0] * m2km, pos[-1, 1] * m2km, pos[-1, 2] * m2km,
            #         'b', marker='s')

        ax.set_xlim([-50, 50])
        ax.set_ylim([-50, 50])
        ax.set_zlim([-50, 50])
        ax.set_xlabel('$x$ [km]', labelpad=15)
        ax.set_ylabel('$y$ [km]', labelpad=15)
        ax.set_zlabel('$z$ [km]', labelpad=15)
        #ax.set_title('$i_0=$' + str(oe_list[0][i][2] * rad2deg) + ' deg.')

    # Set labels
    ax.tick_params(axis='both', labelsize=font)
    ax.set_facecolor('white')


def plot_RMSE():
    # Position list
    n_a = orbits_ref.n_a
    n_inc = orbits_ref.n_inc
    RMSEf_pinn = np.zeros((n_a, n_inc))
    RMSEf_mascon = np.zeros((n_a, n_inc))

    plt.gcf()
    fig, ax = plt.subplots(int(n_inc/2), 2, figsize=(12, 8))
    a = np.zeros(n_a)
    for i in range(n_a):
        a[i] = orbits_pinn.orbits[i][0].spacecraft.a
        for j in range(n_inc):
            n = len(orbits_pinn.orbits[i][j].pos_err)
            RMSEf_mascon[i, j] = \
                np.sqrt(np.nansum(orbits_mascon.orbits[i][j].pos_err**2) / n)
            RMSEf_pinn[i, j] = \
                np.sqrt(np.nansum(orbits_pinn.orbits[i][j].pos_err**2) / n)

    for i in range(int(n_inc/2)):
        # Plots
        ax[i, 0].plot(a * m2km, RMSEf_pinn[:, 2*i],
                      marker='s', clip_on=False,
                      label='100 mascon (Adam) + 8x40 PINN SIREN')
        ax[i, 0].plot(a * m2km, RMSEf_mascon[:, 2*i],
                      marker='s', clip_on=False,
                      label='100 mascon (Adam)')
        ax[i, 1].plot(a * m2km, RMSEf_pinn[:, 2*i+1],
                      marker='s', clip_on=False,
                      label='100 mascon (Adam) + 8x40 PINN SIREN')
        ax[i, 1].plot(a * m2km, RMSEf_mascon[:, 2*i+1],
                      marker='s', clip_on=False,
                      label='100 mascon (Adam)')

        # Aesthetic
        ax[i, 0].set_yscale('log')
        ax[i, 1].set_yscale('log')
        ax[i, 0].set_xlim(np.min(a) * m2km, np.max(a) * m2km)
        ax[i, 1].set_xlim(np.min(a) * m2km, np.max(a) * m2km)
        ax[i, 0].set_ylim(1e-1, 500)
        ax[i, 1].set_ylim(1e-1, 500)
        # ax[i, 0].set_title('$i_0=$' + str(oe_list[i][2*i][2] * rad2deg) + ' deg.',
        #                    fontsize=font)
        # ax[i, 1].set_title('$i_0=$' + str(oe_list[i][2*i+1][2] * rad2deg) + ' deg.',
        #                    fontsize=font)
        ax[i, 0].set_ylabel('RMSE [m]', fontsize=font)
        ax[i, 0].tick_params(axis='both', labelsize=font)
        ax[i, 1].tick_params(axis='both', labelsize=font)
        ax[i, 0].grid()
        ax[i, 1].grid()
    ax[-1, 0].set_xlabel('$a_0$ [km]', fontsize=font)
    ax[-1, 1].set_xlabel('$a_0$ [km]', fontsize=font)
    ax[0, 0].legend(loc='upper right', fontsize=font_legend-5)


# Load files
file_path = os.path.dirname(os.getcwd())
file_ref = file_path + \
            '/Results/eros/groundtruth/polyheterogeneous/propagation.pck'
file_pinn = file_path + \
            '/Results/eros/results/polyheterogeneous/ideal/dense_alt50km_10000samples/' +\
            'pinn8x40SIREN_masconMLE_propagation.pck'
file_mascon = file_path +\
              '/Results/eros/results/polyheterogeneous/ideal/dense_alt50km_10000samples/' +\
              'mascon100_muxyzMLE_octant_rand0_propagation.pck'

# Extract orbits class
orbits_ref = pck.load(open(file_ref, "rb"))
orbits_pinn = pck.load(open(file_pinn, "rb"))
orbits_mascon = pck.load(open(file_mascon, "rb"))

# Plot orbits and RMSE
plot_orbits()
plot_RMSE()
plt.show()
