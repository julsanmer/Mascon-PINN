import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
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
            xlabel = '$x/R [-]$'
            ylabel = '$y/R [-]$'
        elif map == 'xz':
            col = 1
            x_vert = xyz_vert[:, 0]
            y_vert = xyz_vert[:, 2]
            xlabel = '$x/R [-]$'
            ylabel = '$z/R [-]$'
        elif map == 'yz':
            col = 2
            x_vert = xyz_vert[:, 1]
            y_vert = xyz_vert[:, 2]
            xlabel = '$y/R [-]$'
            ylabel = '$z/R [-]$'

        # Plot contour
        cs = ax[col].contourf(X * m2km*km2R, Y * m2km*km2R, normacc * m2mm,
                              cmap=mpl.colormaps['viridis'])
        clines = ax[col].contour(Xy * m2km*km2R, Yx * m2km*km2R, normacc * m2mm,
                                 colors='white',
                                 linewidths=0.25)
        ax[col].clabel(clines,
                       levels=clines.levels,
                       inline=True,
                       fontsize=8)
        ax[col].plot(x_vert * m2km*km2R, y_vert * m2km*km2R,
                     color=color_asteroid, linestyle='', marker='.')
        ax[col].set_xlim([x_min * m2km*km2R, x_max * m2km*km2R])
        ax[col].set_ylim([x_min * m2km*km2R, x_max * m2km*km2R])
        ax[col].set_xlabel(xlabel, fontsize=font)
        ax[col].set_ylabel(ylabel, fontsize=font)
        ax[col].tick_params(axis='both', labelsize=font)
        if map == 'yz':
            cbar = fig.colorbar(cs, ax=ax[col], shrink=0.95)
            cbar.ax.tick_params(labelsize=font)
            cbar.set_label('Gravity [mm/s$^2$]', rotation=90, fontsize=font)

    # Plot the global gravity results
    plt.gcf()
    fig, ax = plt.subplots(1, 3,
                           sharex=True,
                           figsize=(12.0, 3.6),
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


def plot_acc3D(gt):
    # Retrieve variables
    r = np.ravel(gt.gravmap.map_3D.r)
    acc = gt.gravmap.map_3D.acc_XYZ
    normacc = np.ravel(np.linalg.norm(acc, axis=3))

    # Make figure
    plt.gcf()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Plot
    ax.plot(r * m2km * km2R, normacc * m2mm,
            marker='o',
            linestyle='',
            markersize=2.5,
            alpha=0.05,
            zorder=5)

    # Set logarithmic scale
    ax.set_yscale('log')

    # Set labels
    ax.set_xlabel('$r/R$ [-]', fontsize=font)
    ax.set_ylabel('Gravity [mm/s$^2$]', fontsize=font)

    # Set ticks, grid and legend
    ax.tick_params(axis='both', labelsize=font)
    ax.grid()
    ax.legend(loc='upper right', fontsize=font_legend)


# This function plots the gravity potential
def plot_U2D(gt):
    def plot_map(X, Y, U, map):
        if map == 'xy':
            col = 0
            x_vert = xyz_vert[:, 0]
            y_vert = xyz_vert[:, 1]
            xlabel = '$x/R [-]$'
            ylabel = '$y/R [-]$'
        elif map == 'xz':
            col = 1
            x_vert = xyz_vert[:, 0]
            y_vert = xyz_vert[:, 2]
            xlabel = '$x/R [-]$'
            ylabel = '$z/R [-]$'
        elif map == 'yz':
            col = 2
            x_vert = xyz_vert[:, 1]
            y_vert = xyz_vert[:, 2]
            xlabel = '$y/R [-]$'
            ylabel = '$z/R [-]$'

        # Plot contour
        cs = ax[col].contourf(X * m2km*km2R, Y * m2km*km2R, U,
                              cmap=mpl.colormaps['viridis'])
        clines = ax[col].contour(Xy * m2km*km2R, Yx * m2km*km2R, U,
                                 colors='white',
                                 linewidths=0.25)
        ax[col].clabel(clines,
                       levels=clines.levels,
                       inline=True,
                       fontsize=8)
        ax[col].plot(x_vert * m2km*km2R, y_vert * m2km*km2R,
                     color=color_asteroid, linestyle='', marker='.')
        ax[col].set_xlim([x_min * m2km*km2R, x_max * m2km*km2R])
        ax[col].set_ylim([x_min * m2km*km2R, x_max * m2km*km2R])
        ax[col].set_xlabel(xlabel, fontsize=font)
        ax[col].set_ylabel(ylabel, fontsize=font)
        ax[col].tick_params(axis='both', labelsize=font)
        if map == 'yz':
            cbar = fig.colorbar(cs, ax=ax[col], shrink=0.95)
            cbar.ax.tick_params(labelsize=font)
            cbar.set_label('Potential [m$^2$/s$^2$]', rotation=90, fontsize=font)

    # Plot the global gravity results
    plt.gcf()
    fig, ax = plt.subplots(1, 3,
                           sharex=True,
                           figsize=(12, 3.6),
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
    U_XY = gt.gravmap.map_2D.U_XY
    plot_map(Xy, Yx, U_XY, 'xy')

    # Plot XZ
    Xz = gt.gravmap.map_2D.Xz
    Zx = gt.gravmap.map_2D.Zx
    U_XZ = gt.gravmap.map_2D.U_XZ
    plot_map(Xz, Zx, U_XZ, 'xz')

    # Plot YZ
    Yz = gt.gravmap.map_2D.Yz
    Zy = gt.gravmap.map_2D.Zy
    U_YZ = gt.gravmap.map_2D.U_YZ
    plot_map(Yz, Zy, U_YZ, 'yz')


def plot_U3D(gt):
    # Retrieve variables
    r = np.ravel(gt.gravmap.map_3D.r)
    U = np.ravel(gt.gravmap.map_3D.U_XYZ)

    # Make figure
    plt.gcf()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Plot
    ax.plot(r * m2km * km2R, U,
            marker='o',
            linestyle='',
            markersize=2.5,
            alpha=0.05,
            zorder=5)

    # Set logarithmic scale
    ax.set_yscale('log')

    # Set labels
    ax.set_xlabel('$r/R$ [-]', fontsize=font)
    ax.set_ylabel('Gravity potential [m/s$^2$]', fontsize=font)

    # Set ticks, grid and legend
    ax.tick_params(axis='both', labelsize=font)
    ax.grid()
    ax.legend(loc='upper right', fontsize=font_legend)


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

    # Plot acc-U contours
    plot_acc2D(gt)
    plot_U2D(gt)

    # Plot acc-U w.r.t. r
    plot_acc3D(gt)
    plot_U3D(gt)

    plt.show()
