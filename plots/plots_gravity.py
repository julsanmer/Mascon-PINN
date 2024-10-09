import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib import rc

# ------ GLOBAL VARIABLES ------ #
rad2deg = 180 / np.pi
sec2day = 1.0 / (3600*24)
km2R = []
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


# ------------------------------------- MAIN PLOT HANDLING ------------------------------------------------------ #
# This function plots mascon distribution
def plot_mascon(xyz_M, mu_M, xyz_vert, order_face):
    # Make figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Filter small masses for visualization purposes
    mu_lim = 1
    mu_M[mu_M < mu_lim] = mu_lim

    # Plot mascon distribution
    cmap = plt.get_cmap('viridis')
    cmap.set_under(cmap(0))
    ax.plot_trisurf(xyz_vert[:, 0] * m2km, xyz_vert[:, 1] * m2km, xyz_vert[:, 2] * m2km,
                    triangles=order_face-1,
                    color=color_asteroid,
                    zorder=0,
                    alpha=.2)
    p = ax.scatter3D(xyz_M[:, 0] * m2km, xyz_M[:, 1] * m2km, xyz_M[:, 2] * m2km, c=mu_M,
                     cmap=cmap,
                     norm=mpl.colors.LogNorm())
    cbar = fig.colorbar(p, ax=ax, shrink=0.7, extend='min')

    # Set ticks
    cbar.ax.tick_params(labelsize=font)
    cbar.ax.yaxis.get_offset_text().set_fontsize(font)
    cbar.ax.yaxis.set_offset_position('left')
    ax.tick_params(axis='both', labelsize=font)

    # Set labels
    cbar.set_label('$\mu_M$ [m$^3$/s$^2$]', rotation=90, fontsize=font)
    ax.set_xlabel('$x$ [km]', fontsize=font, labelpad=15)
    ax.set_ylabel('$y$ [km]', fontsize=font, labelpad=15)
    ax.set_zlabel('$z$ [km]', fontsize=font, labelpad=15)

    # Set grid and view
    ax.set_facecolor('white')
    set_axes_equal(ax)
    ax.view_init(elev=59., azim=360-144)
    # plt.savefig('Plots/mascon.pdf', format='pdf', bbox_inches='tight')


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
    # if parameters.grav_est.data_type == 'orbit' and parameters.grav_est.flag_ejecta:
    #     n_ejecta = parameters.grav_est.n_ejecta
    #     pos_ejecta = outputs.groundtruth.states.pos_EP_P[0:n_ejecta, 0:3]
    #     ax.plot(pos_ejecta[:, 0]*m2km, pos_ejecta[:, 1]*m2km, pos_ejecta[:, 2]*m2km,
    #             'r', zorder=20, linewidth=0.5, linestyle='', marker='.', markersize=1.5)

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

# This function plots the loss
def plot_loss(loss):
    if loss.ndim == 2:
        n_iter, n_segments = loss.shape
    elif loss.ndim == 1:
        n_iter = loss.shape[0]
        n_segments = 1

    # Make figure
    plt.gcf()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    if loss.ndim == 2:
        for i in range(n_segments):
            iterCount = np.linspace(i, i+1, n_iter)
            plt.axvline(i+1, color='k', linewidth=1, linestyle='--')
            ax.plot(iterCount, loss[:, i], linewidth=2, color='b')
    elif loss.ndim == 1:
        iterCount = np.linspace(0, 1, n_iter)
        ax.plot(iterCount, loss, linewidth=2, color='b')

    ax.set_yscale('log')

    plt.xlabel('Segments [-]', fontsize=font)
    plt.ylabel('Loss function [-]', fontsize=font)

    # Set limits on axis
    ax.set_xlim([0, n_segments])

    # Set ticks, grid and legend
    ax.tick_params(axis='both', labelsize=font)
    ax.grid()

def plot_Uproxy(r_data, Uproxy, Uproxy_data):
    # Make figure
    plt.gcf()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(r_data, Uproxy_data, marker='x', linestyle='', alpha=.5)
    ax.plot(r_data, Uproxy, marker='o', linestyle='', alpha=.1)

    plt.xlabel('$r$ [km]', fontsize=font)
    plt.ylabel('$U$ [-]', fontsize=font)

    # Set ticks, grid and legend
    ax.tick_params(axis='both', labelsize=font)
    ax.grid()
    ax.legend(fontsize=font_legend, loc='upper right')


def plot_tcpu(tcpu, tcpu_ref):
    tcpu = tcpu[~np.isnan(tcpu)]
    tcpu_ref = tcpu_ref[~np.isnan(tcpu_ref)]

    # Make figure
    plt.gcf()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    eval = np.linspace(1, len(tcpu), len(tcpu))

    ax.plot(eval, tcpu*1e3,  marker='.', linestyle='', alpha=1, label='Model')
    ax.plot(eval, tcpu_ref*1e3, marker='.', linestyle='', alpha=1, label='Groundtruth')

    plt.xlabel('Evaluation [-]', fontsize=font)
    plt.ylabel('Computation time [ms]', fontsize=font)

    # Set ticks, grid and legend
    ax.set_yscale('log')
    ax.set_xlim(eval[0], eval[-1])
    ax.tick_params(axis='both', labelsize=font)
    ax.grid()
    ax.legend(fontsize=font_legend, loc='upper right')


# This function sets equal axes for 3D plots
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# TBD
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


# This function plots all gravity
def all_gravityplots(gt, grav_optimizer):
    # Extract data and asteroid
    pos_data = gt.spacecraft.data.pos_BP_P
    xyz_vert = \
        gt.asteroid.shape.xyz_vert
    order_face = \
        gt.asteroid.shape.order_face

    # Extract gravity error maps
    gravmap = grav_optimizer.gravmap

    # Conversion factor
    global km2R
    km2R = 1 / (gt.asteroid.shape.axes[0] * m2km)

    # Plot dataset
    plot_dataset(pos_data, xyz_vert, order_face)

    # Plot gravity errors
    plot_gravity2D(gravmap.map_2D, gt.asteroid.shape)
    plot_gravity3D(gravmap.map_3D, gravmap.intervals, [])
    plot_gravitysurf2D(gravmap.map_surf2D)
    plot_gravitysurf3D(gravmap.map_surf3D, xyz_vert, order_face)

    if grav_optimizer.config['grav_model'] == 'mascon':
        mascon = grav_optimizer.asteroid.gravity[0]
        plot_mascon(mascon.xyz_M, mascon.mu_M,
                    xyz_vert, order_face)

    # if scenario.config['regression']['grav_model'] == 'pinn':
    #     pinn = scenario.regression.asteroid.gravity[0]
    #     plot_loss(pinn.loss)
    #     # plot_Uproxy(pinn.r_data, pinn.Uproxy,
    #     #             pinn.Uproxy_data)

    # Plot execution times
    plot_tcpu(np.ravel(gravmap.map_3D.t_cpu),
              np.ravel(gt.gravmap.map_3D.t_cpu))

    plt.show()
