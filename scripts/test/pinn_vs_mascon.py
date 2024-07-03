import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pck

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


# This function plots the 2D gravity map
def plot_gravity2D(map_2D, shape):
    def plot_map(row, plane, XY, err, colorbar=False):
        # Unfold
        if plane == 'xy':
            i, j = row, 0
            x_vert, y_vert = \
                shape.xyz_vert[:, 0], shape.xyz_vert[:, 1]
            xlabel, ylabel = '$x/R$ [-]', '$y/R$ [-]'
        elif plane == 'xz':
            i, j = row, 1
            x_vert, y_vert = \
                shape.xyz_vert[:, 0], shape.xyz_vert[:, 2]
            xlabel, ylabel = '$x/R$ [-]', '$z/R$ [-]'
        elif plane == 'yz':
            i, j = row, 2
            x_vert, y_vert = \
                shape.xyz_vert[:, 1], shape.xyz_vert[:, 2]
            xlabel, ylabel = '$y/R$ [-]', '$z/R$ [-]'

        # Retrieve arrays
        X, Y = XY[0], XY[1]

        # Plot mascon xy map error
        cs = ax[i, j].contourf(X * m2km * km2R, Y * m2km * km2R, err * err2perc,
                               levels=100,
                               cmap=mpl.colormaps['viridis'],
                               norm=lognorm)
        clines = ax[i, j].contour(X * m2km * km2R, Y * m2km * km2R, err * err2perc,
                                  levels=[1e-4, 1e-3, 1e-2, 1e-1, 1],
                                  colors='white',
                                  linewidths=0.25)
        ax[i, j].clabel(clines,
                        levels=clines.levels,
                        fmt=lines_labels,
                        inline=True,
                        fontsize=8)
        ax[i, j].plot(x_vert * m2km * km2R, y_vert * m2km * km2R,
                      color=color_asteroid)
        # if type(pos_data) is np.ndarray:
        #     ax[1, 0].plot(pos_data[:, 0]*m2km, pos_data[:, 1]*m2km, color='white',
        #                   linestyle='', marker='.', markersize=0.2)
        ax[i, j].set_xlim([x_min * m2km * km2R, x_max * m2km * km2R])
        ax[i, j].set_ylim([x_min * m2km * km2R, x_max * m2km * km2R])
        ax[i, j].set_xlabel(xlabel, fontsize=font_map)
        ax[i, j].set_ylabel(ylabel, fontsize=font_map)
        ax[i, j].tick_params(axis='both', labelsize=font_map)

        if colorbar:
            cbar = fig.colorbar(cs, ax=ax[i, j], shrink=0.95)
            cbar.ax.tick_params(labelsize=font_map)
            cbar.set_label('Gravity error [\%]', rotation=90, fontsize=font_map)

    # Conversion factor
    km2R = 1 / (shape.axes[0] * m2km)

    # Obtain XYZ and gravity errors meshes
    Xy, Yx = map_2D[0].Xy, map_2D[0].Yx
    Xz, Zx = map_2D[0].Xz, map_2D[0].Zx
    Yz, Zy = map_2D[0].Yz, map_2D[0].Zy
    aErrXY_0, aErrXY_1 = map_2D[0].aErrXY, map_2D[1].aErrXY
    aErrXZ_0, aErrXZ_1 = map_2D[0].aErrXZ, map_2D[1].aErrXZ
    aErrYZ_0, aErrYZ_1 = map_2D[0].aErrYZ, map_2D[1].aErrYZ

    # Prune upper errors for visualization purposes
    aErr_up = 0.099999999
    aErr_low = 0.01000000001*1e-4

    # Do clippings
    np.clip(aErrXY_0, aErr_low, aErr_up, out=aErrXY_0)
    np.clip(aErrXZ_0, aErr_low, aErr_up, out=aErrXZ_0)
    np.clip(aErrYZ_0, aErr_low, aErr_up, out=aErrYZ_0)

    # Do clippings
    np.clip(aErrXY_1, aErr_low, aErr_up, out=aErrXY_1)
    np.clip(aErrXZ_1, aErr_low, aErr_up, out=aErrXZ_1)
    np.clip(aErrYZ_1, aErr_low, aErr_up, out=aErrYZ_1)

    # Min and max
    aErr_min = 1e-6
    aErr_max = 1e-1

    # Obtain trajectory and mesh bounds
    x_min = -map_2D[0].rmax
    x_max = map_2D[0].rmax

    # Plot the global gravity results
    plt.gcf()
    fig, ax = plt.subplots(2, 3,
                           sharex=True,
                           figsize=(12, 6),
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
    plot_map(0, 'xy', [Xy, Yx], aErrXY_0)
    plot_map(0, 'xz', [Xz, Zx], aErrXZ_0)
    plot_map(0, 'yz', [Yz, Zy], aErrYZ_0, colorbar=True)

    # Plot second row
    plot_map(1, 'xy', [Xy, Yx], aErrXY_1)
    plot_map(1, 'xz', [Xz, Zx], aErrXZ_1)
    plot_map(1, 'yz', [Yz, Zy], aErrYZ_1, colorbar=True)

    title_mascon = f'Mascon $n=100$' #({type} polyhedron)'
    title_pinn = f'Mascon $n=100$ + NN 6x40 SIREN' #({type} polyhedron)'
    ax[0, 1].set_title(title_mascon,
                       fontsize=font_map)
    ax[1, 1].set_title(title_pinn,
                       fontsize=font_map)

    plt.tight_layout()

    plt.savefig('map2D.pdf', format='pdf')


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
    h_0, h_1 = np.ravel(map_3D[0].h), np.ravel(map_3D[1].h)
    aErr_0, aErr_1 = np.ravel(map_3D[0].aErrXYZ), np.ravel(map_3D[1].aErrXYZ)
    hbins_0, hbins_1 = intervals[0].h_bins, intervals[1].h_bins
    aErrbins_0, aErrbins_1 = intervals[0].aErrAlt_bins, intervals[1].aErrAlt_bins

    # Make figure
    plt.gcf()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Plot
    plot_map3D(h_0, aErr_0,
               hbins_0, aErrbins_0,
               h_data,
               marker='s',
               label=r'Mascon $n=100$')
    plot_map3D(h_1, aErr_1,
               hbins_1, aErrbins_1,
               h_data,
               marker='^',
               label=r'Mascon $n=100$ + PINN 6x40 SIREN')

    # Set logarithmic scale
    ax.set_yscale('log')

    # Set labels
    ax.set_xlabel('Altitude/$R$ [-]', fontsize=font)
    ax.set_ylabel('Gravity error [\%]', fontsize=font)

    # Set ticks, grid and legend
    ax.tick_params(axis='both', labelsize=font)
    ax.grid()
    ax.legend(loc='upper right', fontsize=font_legend)
    title = f'{type} polyhedron'
    ax.set_title(title, fontsize=font_legend)


# This function plots gravity error at surface
def plot_gravitysurf(map_surf, xyz_vert, order_face):
    def plot_surf(idx, aErr_surf, title=''):
        ax = fig.add_subplot(idx, projection='3d')
        surf = ax.plot_trisurf(xyz_vert[:, 0] * m2km, xyz_vert[:, 1] * m2km, xyz_vert[:, 2] * m2km,
                               triangles=order_face - 1,
                               cmap='viridis',
                               edgecolor='k',
                               linewidth=0.05,
                               norm=mpl.colors.LogNorm(vmin=aErr_low, vmax=aErr_up, clip=True))
        surf.set_array(aErr_surf)

        # Add accuracy textbox
        aErr_max = np.max(aErr_surf)
        aErr_mean = np.sum(aErr_surf) / len(aErr_surf)
        text = f'Mean: {"{:.3f}".format(aErr_mean)}\%\n' \
               f' Max.: {"{:.3f}".format(aErr_max)}\%'
        ax.text(2, 2, 10, text,
                fontsize=14,
                bbox=dict(facecolor='white', alpha=0.5))
        ax.set_title(title,
                     fontsize=font_map)
        ax.set_aspect('equal')
        ax.axis('off')

        return surf

    # Switch to error percentage
    aErr_low = 1e-2
    aErr_up = 10
    aErrsurf_0 = map_surf[0].aErr_surf
    aErrsurf_1 = map_surf[1].aErr_surf

    title_mascon = r'Mascon $n=100$\\ (' + type + ' polyhedron)'
    title_pinn = r'Mascon $n=100$ + PINN 6x40 SIREN\\ (' + type + ' polyhedron)'

    # Plot 1
    fig = plt.figure(figsize=(10, 5))
    surf = plot_surf(121,
                     aErrsurf_0 * err2perc,
                     title=title_mascon)
    surf = plot_surf(122,
                     aErrsurf_1 * err2perc,
                     title=title_pinn)

    # Create a separate axis for the colorbar
    cbar_ax = fig.add_axes([0.47, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
    cbar = plt.colorbar(surf, cax=cbar_ax, pad=0.1)
    cbar.set_label('Gravity error [\%]',
                   rotation=90,
                   fontsize=font_map)
    cbar.ax.tick_params(labelsize=font_map)
    #plt.tight_layout()


def plot_orbits(orbits_list):
    def plot_RMSE(orbits, label='', idx='', marker='o'):
        RMSEf = np.zeros((n_a, n_inc))
        a = np.zeros(n_a)
        inc = np.zeros(n_inc)
        for i in range(n_a):
            for j in range(n_inc):
                a[i] = orbits.orbits[i][j].oe[0]
                inc[j] = orbits.orbits[i][j].oe[2]
                pos_err = orbits.orbits[i][j].data.pos_err
                n = len(pos_err)
                RMSEf[i, j] = np.sqrt(np.nansum(pos_err**2) / n)

        for i in range(int(n_inc / 2)):
            if i == 0:
                if idx == 0:
                    ax[i, 0].plot(a * m2km, RMSEf[:, 2*i],
                                  marker=marker,
                                  clip_on=False,
                                  label=label)
                    ax[i, 1].plot(a * m2km, RMSEf[:, 2*i+1],
                                  marker=marker,
                                  clip_on=False)
                    ax[i, 0].legend()
                if idx == 1:
                    ax[i, 0].plot(a * m2km, RMSEf[:, 2*i],
                                  marker=marker,
                                  clip_on=False)
                    ax[i, 1].plot(a * m2km, RMSEf[:, 2*i+1],
                                  marker=marker,
                                  clip_on=False,
                                  label=label)
                    ax[i, 1].legend()
            else:
                ax[i, 0].plot(a * m2km, RMSEf[:, 2*i],
                              marker=marker,
                              clip_on=False)
                ax[i, 1].plot(a * m2km, RMSEf[:, 2*i+1],
                              marker=marker,
                              clip_on=False)

            # Aesthetic
            ax[i, 0].set_yscale('log')
            ax[i, 1].set_yscale('log')
            ax[i, 0].set_xlim(np.min(a) * m2km,
                              np.max(a) * m2km)
            ax[i, 1].set_xlim(np.min(a) * m2km,
                              np.max(a) * m2km)
            ax[i, 0].set_ylim(1e-2, 500)
            ax[i, 1].set_ylim(1e-2, 500)
            ax[i, 0].set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2])
            ax[i, 1].set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2])
            ax[i, 0].grid(visible=True)
            ax[i, 1].grid(visible=True)
            ax[i, 0].set_title('$i_0=$' + str(inc[2*i] * rad2deg) + ' deg.',
                               fontsize=font)
            ax[i, 1].set_title('$i_0=$' + str(inc[2*i+1] * rad2deg) + ' deg.',
                               fontsize=font)
            ax[i, 0].set_ylabel('RMSE [m]', fontsize=font)
            ax[i, 0].tick_params(axis='both', labelsize=font)
            ax[i, 1].tick_params(axis='both', labelsize=font)

            #
            if i == int(n_inc/2) - 1:
                ax[i, 0].set_xlabel('$a_0$ [km]',
                                    fontsize=font)
                ax[i, 1].set_xlabel('$a_0$ [km]',
                                    fontsize=font)

    # Get number of a, inc
    n_a = orbits_list[0].n_a
    n_inc = orbits_list[0].n_inc

    # Create figure
    plt.gcf()
    fig, ax = plt.subplots(int(n_inc/2), 2, figsize=(10, 7.5))

    # Plot
    plot_RMSE(orbits_list[0],
              label=r'Mascon $n=100$',
              idx=0,
              marker='^')
    plot_RMSE(orbits_list[1],
              label=r'Mascon $n=100$ + \\ PINN 6x40 SIREN',
              idx=1,
              marker='s')


if __name__ == "__main__":
    # Load files
    file_path = os.path.dirname(os.getcwd())

    homogeneous = True
    heterogeneous = False
    if homogeneous:
        # classic
        name_mascon = '/Results/eros/results/poly/ideal/dense_alt50km_100000samples/' + \
                       'mascon100_muxyzMSE_octant_rand0'
        # name_pinn = '/Results/eros/results/poly/ideal/dense_alt50km_100000samples/' + \
        #             'pinn6x40SIREN_masconMLE'

        # 6x40
        # name_mascon = '/Results/eros/results/poly/ideal/dense_alt50km_100000samples/' + \
        #               'pinn6x40SIREN_masconMLE_new'
        # name_pinn = '/Results/eros/results/poly/ideal/dense_alt50km_100000samples/' + \
        #             'pinn6x40SIREN_masconMLE_new2'

        # 6x160
        # name_mascon = '/Results/eros/results/poly/ideal/dense_alt50km_100000samples/' + \
        #               'pinn6x160SIREN_masconMLE_old'
        name_pinn = '/Results/eros/results/poly/ideal/dense_alt50km_100000samples/' + \
                    'pinn6x40SIREN_mascon100'
        type = 'homogeneous'
    if heterogeneous:
        name_mascon = '/Results/eros/results/polyheterogeneous/ideal/dense_alt50km_10000samples/' + \
                      'mascon100_muxyzMSE_octant_rand0'
        name_pinn = '/Results/eros/results/polyheterogeneous/ideal/dense_alt50km_100000samples/' + \
                    'pinn6x40SIREN_masconMLE'
        type = 'heterogeneous'

    # Import mascon
    file_mascon = file_path + \
                  name_mascon + '.pck'
    scen_mascon = pck.load(open(file_mascon, "rb"))

    # Import pinn
    file_pinn = file_path + \
                name_pinn + '.pck'
    scen_pinn = pck.load(open(file_pinn, "rb"))

    # Map 2D list
    map2D_list = [scen_mascon.estimation.gravmap.map_2D,
                  scen_pinn.estimation.gravmap.map_2D]
    map3D_list = [scen_mascon.estimation.gravmap.map_3D,
                  scen_pinn.estimation.gravmap.map_3D]
    mapsurf_list = [scen_mascon.estimation.gravmap.map_surf,
                    scen_pinn.estimation.gravmap.map_surf]
    intervals_list = [scen_mascon.estimation.gravmap.intervals,
                      scen_pinn.estimation.gravmap.intervals]

    # Shape
    shape = scen_pinn.groundtruth.asteroid.shape
    km2R = 1 / (shape.axes[0] * m2km)

    # Call script
    plot_gravity2D(map2D_list, shape)
    plot_gravity3D(map3D_list,
                   intervals_list,
                   scen_pinn.groundtruth.spacecraft.data.h_BP)
    plot_gravitysurf(mapsurf_list,
                     shape.xyz_vert,
                     shape.order_face)

    # # Make figure
    # plt.gcf()
    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(scen_mascon.estimation.asteroid.gravity[0].loss)
    # ax.plot(scen_pinn.estimation.asteroid.gravity[0].loss)


    if homogeneous:
        # Import mascon
        file_mascon = file_path + name_mascon + '_propagation.pck'
        orbits_mascon = pck.load(open(file_mascon, "rb"))

        # Import pinn
        file_pinn = file_path + name_pinn + '_propagation.pck'
        orbits_pinn = pck.load(open(file_pinn, "rb"))

        orbits_list = [orbits_mascon,
                       orbits_pinn]

        # Plot orbits
        plot_orbits(orbits_list)

    plt.show()
