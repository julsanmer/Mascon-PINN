import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib import rc

from plots.utils import set_axes_equal

# ------ GLOBAL VARIABLES ------ #
rad2deg = 180 / np.pi
sec2day = 1.0 / (3600*24)
km2R = []
m2km = 1e-3
m2mm = 1e3
m2mu = 1e6
err2perc = 1e2

# ------ PLOTS VARIABLES ------ #
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
font = 20
font_legend = 15
font_map = 15
color_asteroid = [105/255, 105/255, 105/255]


# This function plots the dataset
def plot_dataset(pos_data, xyz_vert, order_face):
    plt.gcf()
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(xyz_vert[:, 0]*m2km, xyz_vert[:, 1]*m2km, xyz_vert[:, 2]*m2km,
                    triangles=order_face-1, color=color_asteroid, zorder=0, alpha=0.5)
    if type(pos_data) is np.ndarray:
        ax.plot(pos_data[:, 0]*m2km, pos_data[:, 1]*m2km, pos_data[:, 2]*m2km,
                'b', zorder=20, linewidth=0.5, linestyle='', marker='.', markersize=1.5)

    # Commands for fanciness
    ax.set_xlabel('$x$ [km]', fontsize=font, labelpad=15)
    ax.set_ylabel('$y$ [km]', fontsize=font, labelpad=15)
    ax.set_zlabel('$z$ [km]', fontsize=font, labelpad=15)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-40, 40)
    ax.set_xticks([-30, -15, 0, 15, 30])
    ax.set_yticks([-30, -15, 0, 15, 30])
    ax.set_zticks([-30, 0, 30])
    ax.tick_params(axis='both', labelsize=font)
    ax.set_facecolor('white')
    set_axes_equal(ax)
    ax.view_init(elev=59., azim=360 - 144)


# This function plots the gravity potential
def plot_acc2D(gt):
    def plot_map(X, Y, normacc, map):
        if map == 'xy':
            col = 0
            x_vert = xyz_vert[:, 0]
            y_vert = xyz_vert[:, 1]
        elif map == 'xz':
            col = 1
            x_vert = xyz_vert[:, 0]
            y_vert = xyz_vert[:, 2]
        elif map == 'yz':
            col = 2
            x_vert = xyz_vert[:, 1]
            y_vert = xyz_vert[:, 2]

        # Plot contour
        cs = ax[col].contourf(X * m2km*km2R, Y * m2km*km2R, normacc,
                              cmap=mpl.colormaps['viridis'])
        clines = ax[col].contour(Xy * m2km*km2R, Yx * m2km*km2R, normacc,
                                 colors='white',
                                 linewidths=0.25)
        ax[col].clabel(clines,
                       levels=clines.levels,
                       inline=True,
                       fontsize=8)
        ax[col].plot(x_vert * m2km*km2R, y_vert * m2km*km2R,
                     color=color_asteroid, linestyle='', marker='.')
        # if map == 1:
        #     ax[col].plot(-10*1e3 * m2km*km2R, 0,
        #                  marker='.', markersize=10, color='b')
        #     ax[col].plot(10*1e3 * m2km*km2R, 0,
        #                  marker='.', markersize=10, color='r')
        ax[col].set_xlim([x_min * m2km*km2R, x_max * m2km*km2R])
        ax[col].set_ylim([x_min * m2km*km2R, x_max * m2km*km2R])
        #ax[col].set_title(title, fontsize=font_map)

        if col == 0:
            ax[col].set_ylabel('$y/R$ [-]', fontsize=font)
        ax[col].tick_params(axis='both', labelsize=font)

        if col == 2:
            cbar = fig.colorbar(cs, ax=ax[col], shrink=0.95,
                                ticks=[0, 200, 400, 600, 800])
            cbar.ax.tick_params(labelsize=font)
            cbar.set_label('Gravity [mGal]', rotation=90, fontsize=font)

    # Plot the global gravity results
    plt.gcf()
    fig, ax = plt.subplots(1, 3,
                           sharex=True,
                           figsize=(7.5, 3.6),
                           gridspec_kw={'width_ratios': [1, 1, 1.25]})

    # Conversion factor
    global km2R
    km2R = 1 / (gt.asteroid.shape.axes[0] * m2km)

    # Retrieve polyhedron shape
    xyz_vert = gt.asteroid.shape.xyz_vert

    # Obtain trajectory and mesh bounds
    x_min = -1.5 * 16 * 1e3
    x_max = 1.5 * 16 * 1e3

    # Plot XY
    Xy = gt.gravmap.map_2D.Xy
    Yx = gt.gravmap.map_2D.Yx
    acc_XY = gt.gravmap.map_2D.acc_XY
    normacc_XY = np.linalg.norm(acc_XY, axis=2)
    plot_map(Xy, Yx, normacc_XY, 'xy')

    # Plot XZ
    Xz = gt.gravmap.map_2D.Xz
    Zx = gt.gravmap.map_2D.Zx
    acc_XZ = gt.gravmap.map_2D.acc_XZ
    normacc_XZ = np.linalg.norm(acc_XZ, axis=2)
    plot_map(Xz, Zx, normacc_XZ, 'xz')

    # Plot YZ
    Yz = gt.gravmap.map_2D.Yz
    Zy = gt.gravmap.map_2D.Zy
    acc_YZ = gt.gravmap.map_2D.acc_YZ
    normacc_YZ = np.linalg.norm(acc_YZ, axis=2)
    plot_map(Yz, Zy, normacc_YZ, 'yz')

    # Plot maps
    fig.supxlabel('$x/R$ [-]', fontsize=font, x=0.5, y=0.1)


def all_groundtruth_plots(groundtruth):
    # Abbreviate
    gt = groundtruth

    # Plot normal dataset
    plot_dataset(gt.spacecraft.data.pos_BP_P,
                 gt.asteroid.shape.xyz_vert,
                 gt.asteroid.shape.order_face)

    # Plot low altitude dataset
    plot_dataset(gt.ejecta.data.pos_BP_P,
                 gt.asteroid.shape.xyz_vert,
                 gt.asteroid.shape.order_face)

    # Plot acceleration contour
    plot_acc2D(gt)

    plt.show()
