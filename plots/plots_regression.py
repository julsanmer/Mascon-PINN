import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib import rc

from plots.plots_groundtruth import plot_dataset
from plots.plots_mascon import plot_mascon
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


# This function plots the 2D gravity map
def plot_accerr2D(map_2D, shape):
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
    accerr_XY, accerr_XZ, accerr_YZ = \
        map_2D.accerr_XY, map_2D.accerr_XZ, map_2D.accerr_YZ

    # Prune upper errors for visualization purposes
    accerr_up = 0.099999999
    accerr_low = 0.01000000001*1e-4

    # Do clippings
    np.clip(accerr_XY, accerr_low, accerr_up, out=accerr_XY)
    np.clip(accerr_XZ, accerr_low, accerr_up, out=accerr_XZ)
    np.clip(accerr_YZ, accerr_low, accerr_up, out=accerr_YZ)

    # Min and max
    accerr_min = 1e-6
    accerr_max = 1e-1

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
    lognorm = mpl.colors.LogNorm(vmin=accerr_min*err2perc,
                                 vmax=accerr_max*err2perc,
                                 clip=True)

    # Plot first row
    plot_map('xy', [Xy, Yx], accerr_XY)
    plot_map('xz', [Xz, Zx], accerr_XZ)
    plot_map('yz', [Yz, Zy], accerr_YZ, colorbar=True)

    plt.tight_layout()


# This function plots the gravity error w.r.t. radius
def plot_accerr3D(map_3D, intervals):
    def plot_map3D(r, accerr, r_bins, accerr_bins, marker='', label=''):
        # Plot 3D gravity error and their average
        ax.plot(r * m2km*km2R, accerr * err2perc,
                marker='o',
                linestyle='',
                markersize=2.5,
                alpha=0.05,
                zorder=5)
        ax.plot(r_bins * m2km*km2R, accerr_bins * err2perc,
                marker=marker,
                markersize=5,
                zorder=10,
                linestyle='--',
                label=label)

    # Retrieve variables
    r = np.ravel(map_3D.r)
    accerr = np.ravel(map_3D.accerr_XYZ)
    rbins = intervals.r_bins
    accerr_bins = intervals.accerr_rad_bins

    # Make figure
    plt.gcf()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Plot
    plot_map3D(r, accerr, rbins, accerr_bins)

    # Set logarithmic scale
    ax.set_yscale('log')

    # Set labels
    ax.set_xlabel('r/$R$ [-]', fontsize=font)
    ax.set_ylabel('Gravity error [\%]', fontsize=font)

    # Set ticks, grid and legend
    ax.tick_params(axis='both', labelsize=font)
    ax.grid()
    ax.legend(loc='upper right', fontsize=font_legend)


# This function plots the potential error w.r.t. radius
def plot_Uerr3D(map_3D, intervals):
    def plot_map3D(r, Uerr, r_bins, Uerr_bins, marker='', label=''):
        # Plot 3D gravity error and their average
        ax.plot(r * m2km*km2R, Uerr * err2perc,
                marker='o',
                linestyle='',
                markersize=2.5,
                alpha=0.05,
                zorder=5)
        # ax.plot(r_bins * m2km*km2R, Uerr_bins * err2perc,
        #         marker=marker,
        #         markersize=5,
        #         zorder=10,
        #         linestyle='--',
        #         label=label)

    # Retrieve variables
    r = np.ravel(map_3D.r)
    Uerr = np.ravel(map_3D.Uerr_XYZ)
    rbins = intervals.r_bins
    #Uerr_bins = intervals.Uerr_rad_bins

    # Make figure
    plt.gcf()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Plot
    plot_map3D(r, Uerr, rbins, [])

    # Set logarithmic scale
    ax.set_yscale('log')

    # Set labels
    ax.set_xlabel('r/$R$ [-]', fontsize=font)
    ax.set_ylabel('Potential error [\%]', fontsize=font)

    # Set ticks, grid and legend
    ax.tick_params(axis='both', labelsize=font)
    ax.grid()
    ax.legend(loc='upper right', fontsize=font_legend)


# This function plots gravity error at surface
def plot_accerrsurf2D(map_surf):
    lon_grid = map_surf.lon
    lat_grid = map_surf.lat
    accerr_surf = map_surf.accerr_surf

    # Min and max
    accerr_min = 1.00000001*1e-4
    accerr_max = 0.99999999*1e0

    # Clip to min and max
    np.clip(accerr_surf, accerr_min, accerr_max, out=accerr_surf)

    # Make min and max reflect to appear on corners
    accerr_surf[0, 0] = accerr_min
    accerr_surf[-1, 0] = accerr_max

    # Logarithmic norm
    lognorm = mpl.colors.LogNorm(vmin=accerr_min*err2perc,
                                 vmax=accerr_max*err2perc,
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
                       accerr_surf * err2perc,
                       cmap=mpl.colormaps['viridis'],
                       norm=lognorm)
    cbar = fig.colorbar(surf, ax=ax, shrink=1.)
    clines = ax.contour(lon_grid * rad2deg,
                        lat_grid * rad2deg,
                        accerr_surf * err2perc,
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
def plot_accerrsurf3D(map_surf, xyz_vert, order_face):
    # Switch to error percentage
    accerr_low = 1e-2
    accerr_up = 10
    accerr_surf = map_surf.accerr_surf * err2perc

    # Create figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(xyz_vert[:, 0] * m2km, xyz_vert[:, 1] * m2km, xyz_vert[:, 2] * m2km,
                           triangles=order_face-1,
                           cmap='viridis',
                           edgecolor='k',
                           linewidth=0.05,
                           norm=mpl.colors.LogNorm(vmin=accerr_low, vmax=accerr_up, clip=True))
    surf.set_array(accerr_surf)

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=.6)
    cbar.ax.tick_params(labelsize=font_map)
    cbar.set_label('Gravity error [\%]', rotation=90, fontsize=font_map)

    # Set labels and title
    ax.set_xlabel('x [km]', fontsize=font)
    ax.set_ylabel('y [km]', fontsize=font)
    ax.set_zlabel('z [km]', fontsize=font)

    # Add accuracy textbox
    accerr_max = np.max(accerr_surf)
    accerr_mean = np.sum(accerr_surf) / len(accerr_surf)
    text = f'Mean: {"{:.3f}".format(accerr_mean)}\%\n' \
           f' Max.: {"{:.3f}".format(accerr_max)}\%'
    ax.text(2, 2, 20, text,
            fontsize=14,
            bbox=dict(facecolor='white', alpha=0.5))

    # Set axes equal
    ax.tick_params(axis='both', labelsize=font)
    ax.set_facecolor('white')
    set_axes_equal(ax)


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


# This function plots all regression plots
def all_regression_plots(gt, grav_optimizer):
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
    plot_accerr2D(gravmap.map_2D, gt.asteroid.shape)
    plot_accerr3D(gravmap.map_3D, gravmap.intervals)
    plot_accerrsurf2D(gravmap.map_surf2D)
    plot_accerrsurf3D(gravmap.map_surf3D, xyz_vert, order_face)
    plot_Uerr3D(gravmap.map_3D, gravmap.intervals)

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
