import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
from matplotlib.cm import get_cmap
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


# This function plots the 2D gravity map
def plot_gravity2D(map_2D, shape):
    def plot_map(plane, XY, err, colorbar=False):
        # Unfold
        if plane == 'xy':
            col = 0
            x_vert, y_vert = \
                shape.xyz_vert[:, 0], shape.xyz_vert[:, 1]
            xlabel, ylabel = '$x/R$ [-]', '$y/R$ [-]'
        elif plane == 'xz':
            col = 1
            x_vert, y_vert = \
                shape.xyz_vert[:, 0], shape.xyz_vert[:, 2]
            xlabel, ylabel = '$x/R$ [-]', '$z/R$ [-]'
        elif plane == 'yz':
            col = 2
            x_vert, y_vert = \
                shape.xyz_vert[:, 1], shape.xyz_vert[:, 2]
            xlabel, ylabel = '$y/R$ [-]', '$z/R$ [-]'

        # Retrieve arrays
        X, Y = XY[0], XY[1]

        # Plot mascon xy map error
        cs = ax[col].contourf(X * m2km*km2R, Y * m2km*km2R, err * err2perc,
                              levels=100,
                              cmap=mpl.colormaps['viridis'],
                              norm=lognorm)
        clines = ax[col].contour(X * m2km*km2R, Y * m2km*km2R, err * err2perc,
                                 levels=[1e-4, 1e-3, 1e-2, 1e-1, 1],
                                 colors='white',
                                 linewidths=0.25)
        ax[col].clabel(clines,
                       levels=clines.levels,
                       fmt=lines_labels,
                       inline=True,
                       fontsize=8)
        ax[col].plot(x_vert * m2km*km2R, y_vert * m2km*km2R,
                     color=color_asteroid)
        ax[col].set_xlim([x_min * m2km*km2R, x_max * m2km*km2R])
        ax[col].set_ylim([x_min * m2km*km2R, x_max * m2km*km2R])
        ax[col].set_xlabel(xlabel, fontsize=font_map)
        ax[col].set_ylabel(ylabel, fontsize=font_map)
        ax[col].tick_params(axis='both', labelsize=font_map)

        if colorbar:
            cbar = fig.colorbar(cs, ax=ax[col], shrink=0.95)
            cbar.ax.tick_params(labelsize=font_map)
            cbar.set_label('Gravity error [\%]', rotation=90, fontsize=font_map)

    # Obtain XYZ and gravity errors meshes
    Xy, Yx = map_2D.Xy, map_2D.Yx
    Xz, Zx = map_2D.Xz, map_2D.Zx
    Yz, Zy = map_2D.Yz, map_2D.Zy
    aErrXY, aErrXZ, aErrYZ = \
        map_2D.aErrXY, map_2D.aErrXZ, map_2D.aErrYZ

    # Prune upper errors for visualization purposes
    aErr_up = 0.099999999
    aErr_low = 0.01000000001*1e-4

    # Do clippings
    np.clip(aErrXY, aErr_low, aErr_up, out=aErrXY)
    np.clip(aErrXZ, aErr_low, aErr_up, out=aErrXZ)
    np.clip(aErrYZ, aErr_low, aErr_up, out=aErrYZ)

    # Min and max
    aErr_min = 1e-6
    aErr_max = 1e-1

    # Obtain trajectory and mesh bounds
    x_min = -map_2D.rmax
    x_max = map_2D.rmax

    # Plot the global gravity results
    plt.gcf()
    fig, ax = plt.subplots(1, 3,
                           sharex=True,
                           figsize=(12, 3),
                           gridspec_kw={'width_ratios': [1, 1, 1.2]})
    lines_labels = {1e-4: '0.0001\%',
                    1e-3: '0.001\%',
                    1e-2: '0.01\%',
                    1e-1: '0.1\%',
                    1: '1\%'}

    # Logarithmic norm
    lognorm = mpl.colors.LogNorm(vmin=aErr_min*err2perc,
                                 vmax=aErr_max*err2perc,
                                 clip=True)

    # Plot first row
    plot_map('xy', [Xy, Yx], aErrXY)
    plot_map('xz', [Xz, Zx], aErrXZ)
    plot_map('yz', [Yz, Zy], aErrYZ, colorbar=True)

    plt.tight_layout()


# This function plots the 3D gravity map
def plot_gravity3D(map_3D, intervals, h_data):
    def plot_map3D(h, aErr, h_bins, aErr_bins, h_data, marker='', label=''):
        # Plot 3D gravity error and their average
        ax.plot(h * m2km*km2R, aErr * err2perc,
                marker='o',
                linestyle='',
                markersize=2.5,
                alpha=0.05,
                zorder=5)
        ax.plot(h_bins * m2km*km2R, aErr_bins * err2perc,
                marker=marker,
                markersize=5,
                zorder=10,
                linestyle='--',
                label=label)

        if isinstance(h_data, np.ndarray):
            ax_twin = ax.twinx()

            # Define the intervals
            hdata_bins = np.linspace(np.min(h_data),
                                     np.max(h_data),
                                     10)
            counts, _ = np.histogram(h_data, bins=hdata_bins)
            ax_twin.bar(hdata_bins[:-1]*m2km*km2R, counts,
                        width=np.diff(hdata_bins*m2km*km2R),
                        align='edge',
                        alpha=0.1,
                        color='gray')
            ax_twin.set_ylabel('Data samples [-]', fontsize=font)
            plt.axvline(x=np.max(h_data)*m2km*km2R,
                        color='gray',
                        linewidth=2)
            ax_twin.set_ylim(100, 2*1e5)
            ax_twin.set_yscale('log')
            ax_twin.tick_params(axis='y', labelsize=font)

        # Set limits on axis
        ax.set_xlim([np.nanmin(h) * m2km*km2R,
                     np.nanmax(h) * m2km*km2R])

    # Retrieve variables
    h = np.ravel(map_3D.h)
    aErr = np.ravel(map_3D.aErrXYZ)
    hbins = intervals.h_bins
    aErrbins = intervals.aErrAlt_bins

    # Make figure
    plt.gcf()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Plot
    plot_map3D(h, aErr,
               hbins, aErrbins,
               h_data)

    # Set logarithmic scale
    ax.set_yscale('log')

    # Set labels
    ax.set_xlabel('Altitude/$R$ [-]', fontsize=font)
    ax.set_ylabel('Gravity error [\%]', fontsize=font)

    # Set ticks, grid and legend
    ax.tick_params(axis='both', labelsize=font)
    ax.grid()
    ax.legend(loc='upper right', fontsize=font_legend)
    # title = f'{type} polyhedron'
    # ax.set_title(title, fontsize=font_legend)


# This function plots gravity error at surface
def plot_gravitysurf2D(map_surf):
    lon_grid = map_surf.lon
    lat_grid = map_surf.lat
    aErr_surf = map_surf.aErr_surf

    # Min and max
    aErr_min = 1.00000001*1e-4
    aErr_max = 0.99999999*1e0

    # Clip to min and max
    np.clip(aErr_surf, aErr_min, aErr_max, out=aErr_surf)

    # Make min and max reflect to appear on corners
    aErr_surf[0, 0] = aErr_min
    aErr_surf[-1, 0] = aErr_max

    # Logarithmic norm
    lognorm = mpl.colors.LogNorm(vmin=aErr_min*err2perc,
                                 vmax=aErr_max*err2perc,
                                 clip=True)

    lines_labels = {1e-2: '0.01\%',
                    1e-1: '0.1\%',
                    1e0: '1\%',
                    1e1: '10\%',
                    1e2: '100\%'}

    # Plot the data
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(111)
    surf = ax.contourf(lon_grid * rad2deg,
                       lat_grid * rad2deg,
                       aErr_surf * err2perc,
                       cmap=mpl.colormaps['viridis'],
                       norm=lognorm)
    cbar = fig.colorbar(surf, ax=ax, shrink=1.)
    clines = ax.contour(lon_grid * rad2deg,
                        lat_grid * rad2deg,
                        aErr_surf * err2perc,
                        levels=[1e-2, 1e-1, 1e0, 1e1, 1e2],
                        colors='white',
                        linewidths=0.25)
    ax.clabel(clines,
              fmt=lines_labels,
              levels=clines.levels,
              inline=True,
              fontsize=8)
    cbar.ax.tick_params(labelsize=font_map)
    cbar.set_label('Gravity error [\%]', rotation=90, fontsize=font_map)
    ax.tick_params(axis='both', labelsize=font_map)

    # Set labels and title
    ax.set_xlabel('Longitude [$^{\circ}$]', fontsize=font)
    ax.set_ylabel('Latitude [$^{\circ}$]', fontsize=font)


# This function plots gravity error at surface
def plot_gravitysurf3D(map_surf, xyz_vert, order_face):
    # Switch to error percentage
    aErr_low = 1e-2
    aErr_up = 10
    aErr_surf = map_surf.aErr_surf * err2perc

    # Create figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(xyz_vert[:, 0] * m2km, xyz_vert[:, 1] * m2km, xyz_vert[:, 2] * m2km,
                           triangles=order_face-1,
                           cmap='viridis',
                           edgecolor='k',
                           linewidth=0.05,
                           norm=mpl.colors.LogNorm(vmin=aErr_low, vmax=aErr_up, clip=True))
    surf.set_array(aErr_surf)

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=.6)
    cbar.ax.tick_params(labelsize=font_map)
    cbar.set_label('Gravity error [\%]', rotation=90, fontsize=font_map)

    # Set labels and title
    ax.set_xlabel('x [km]', fontsize=font)
    ax.set_ylabel('y [km]', fontsize=font)
    ax.set_zlabel('z [km]', fontsize=font)

    # Add accuracy textbox
    aErr_max = np.max(aErr_surf)
    aErr_mean = np.sum(aErr_surf) / len(aErr_surf)
    text = f'Mean: {"{:.3f}".format(aErr_mean)}\%\n' \
           f' Max.: {"{:.3f}".format(aErr_max)}\%'
    ax.text(2, 2, 20, text,
            fontsize=14,
            bbox=dict(facecolor='white', alpha=0.5))

    # Set axes equal
    ax.tick_params(axis='both', labelsize=font)
    ax.set_facecolor('white')
    set_axes_equal(ax)


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
def all_gravityplots(scenario):
    # Extract data and asteroid
    groundtruth = scenario.groundtruth
    pos_data = groundtruth.spacecraft.data.pos_BP_P
    xyz_vert = \
        groundtruth.asteroid.shape.xyz_vert
    order_face = \
        groundtruth.asteroid.shape.order_face

    # Extract gravity error maps
    gravmap = scenario.estimation.gravmap

    # Conversion factor
    global km2R
    km2R = 1 / (groundtruth.asteroid.shape.axes[0] * m2km)

    # Plot dataset
    plot_dataset(pos_data, xyz_vert, order_face)

    # Plot gravity errors
    plot_gravity2D(gravmap.map_2D, groundtruth.asteroid.shape)
    plot_gravity3D(gravmap.map_3D, gravmap.intervals, [])
    plot_gravitysurf2D(gravmap.map_surf2D)
    plot_gravitysurf3D(gravmap.map_surf3D, xyz_vert, order_face)

    if scenario.config['estimation']['grav_model'] == 'mascon':
        mascon = scenario.estimation.asteroid.gravity[0]
        plot_mascon(mascon.xyz_M, mascon.mu_M,
                    xyz_vert, order_face)

    if scenario.config['estimation']['grav_model'] == 'pinn':
        pinn = scenario.estimation.asteroid.gravity[0]
        plot_loss(pinn.loss)
        # plot_Uproxy(pinn.r_data, pinn.Uproxy,
        #             pinn.Uproxy_data)

    # Plot execution times
    plot_tcpu(np.ravel(gravmap.map_3D.t_cpu),
              np.ravel(groundtruth.gravmap.map_3D.t_cpu))

    plt.show()
