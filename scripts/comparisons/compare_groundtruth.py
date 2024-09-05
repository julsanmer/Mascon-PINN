import numpy as np
import pickle as pck
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc


# ------ GLOBAL VARIABLES ------ #
err2perc = 1e2
m2km = 1e-3
km2R = 1/16
acc2gal = 1e2
gal2mgal = 1e3

# ------ PLOT VARIABLES ------ #
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
font = 20
font_legend = 15
font_map = 15
color_asteroid = [105/255, 105/255, 105/255]
dpi = 600


def plot2_grav2D():
    def plot_map(normacc, col, title='', colorbar=False):
        if col == 0:
            # Plot mascon xy map error
            cs = ax[col].contourf(Xy * m2km*km2R, Yx * m2km*km2R, normacc * acc2gal*gal2mgal,
                                  cmap=mpl.colormaps['viridis'],
                                  levels=100,
                                  vmin=0, vmax=600)
            clines = ax[col].contour(Xy * m2km*km2R, Yx * m2km*km2R, normacc * acc2gal*gal2mgal,
                                     levels=[100, 200, 300, 400, 500],
                                     colors='white',
                                     linewidths=0.25)
        elif col == 1:
            # Plot mascon xy map error
            cs = ax[col].contourf(Xy * m2km*km2R, Yx * m2km*km2R, normacc,
                                  cmap=mpl.colormaps['viridis'],
                                  levels=100,
                                  vmin=-60, vmax=60)
            clines = ax[col].contour(Xy * m2km*km2R, Yx * m2km*km2R, normacc,
                                     levels=[-40, -30, -20, -10, 0, 10, 20, 30, 40],
                                     colors='white',
                                     linewidths=0.25)
        if col == 0:
            ax[col].clabel(clines,
                           levels=clines.levels,
                           inline=True,
                           fontsize=8)
        elif col == 1:
            def percent_formatter(x):
                return f'{int(x)}\%' if x.is_integer() else f'{x:.0f}\%'

            # Create a dictionary for line labels formatted as percentages
            lines_labels = {level: percent_formatter(level) for level in clines.levels}
            #fmt = {'$-40\%$', '$-30\%$', '$-20\%$', '$-10\%$', '$0\%$',
            #       '$10\%$', '$20\%$', '$30\%$', '$40\%$'}
            fmt = clines.labels
            ax[col].clabel(clines,
                           fmt=lines_labels,
                           levels=clines.levels,
                           inline=True,
                           fontsize=8)

        ax[col].plot(xyz_vert[:, 0] * m2km*km2R, xyz_vert[:, 1] * m2km*km2R,
                     color=color_asteroid, linestyle='', marker='.')
        if col == 1:
            ax[col].plot(-10*1e3 * m2km*km2R, 0,
                         marker='.', markersize=10, color='b')
            ax[col].plot(10*1e3 * m2km*km2R, 0,
                         marker='.', markersize=10, color='r')
            ax[col].text(0.16, 0.8, f'0\%', fontsize=8, ha='center',
                         va='center', color='white')
            ax[col].text(-0.30, -0.8, f'0\%', fontsize=8, color='white')
        ax[col].set_xlim([x_min * m2km*km2R, x_max * m2km*km2R])
        ax[col].set_ylim([x_min * m2km*km2R, x_max * m2km*km2R])
        ax[col].set_title(title, fontsize=font_map)
        if col == 0:
            ax[col].set_ylabel('$y/R$ [-]', fontsize=font_map)
        ax[col].tick_params(axis='both', labelsize=font_map)

        ax[col].set_xticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        ax[col].set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])

        if col == 0:
            cbar = fig.colorbar(cs, ax=ax[col], shrink=0.95,
                                ticks=[0, 100, 200, 300, 400, 500])
            cbar.ax.tick_params(labelsize=font_map)
            cbar.set_label('Gravity [mGal]', rotation=90, fontsize=font_map)
        elif col == 1:
            cbar = fig.colorbar(cs, ax=ax[col], shrink=0.95,
                                ticks=[-50, -25, 0, 25, 50])
            cbar.ax.tick_params(labelsize=font_map)
            cbar.set_label('Relative difference [\%]', rotation=90, fontsize=font_map)

    # Plot the global gravity results
    plt.gcf()
    fig, ax = plt.subplots(1, 2,
                           sharex=True,
                           figsize=(9, 3.6),
                           gridspec_kw={'width_ratios': [1.0, 1.0]})
    acc_max = np.max(np.array([np.nanmax(normaccXY_poly),
                               np.nanmax(normaccXY_polyheterogeneous)]))

    plot_map(normaccXY_poly, 0, title='Constant density polyhedron')
    plot_map(erraccXY, 1, title='Heterogeneous polyhedron', colorbar=True)
    fig.supxlabel('$x/R$ [-]', fontsize=font_map, x=0.5, y=0.1)

    # Save figure
    plt.savefig('Plots/gravity_groundtruth.png', format='png',
                dpi=dpi)


def plot_grav2D():
    def plot_map(normacc, col, title='', colorbar=False):
        # Plot mascon xy map error
        cs = ax[col].contourf(Xy * m2km*km2R, Yx * m2km*km2R, normacc * acc2gal*gal2mgal,
                              cmap=mpl.colormaps['viridis'],
                              levels=100,
                              vmin=0, vmax=600)
        clines = ax[col].contour(Xy * m2km*km2R, Yx * m2km*km2R, normacc * acc2gal*gal2mgal,
                                 levels=[100, 200, 300, 400, 500, 600],
                                 colors='white',
                                 linewidths=0.25)
        ax[col].clabel(clines,
                       levels=clines.levels,
                       inline=True,
                       fontsize=8)
        ax[col].plot(xyz_vert[:, 0] * m2km*km2R, xyz_vert[:, 1] * m2km*km2R,
                     color=color_asteroid, linestyle='', marker='.')
        if col == 1:
            ax[col].plot(-10*1e3 * m2km*km2R, 0,
                         marker='.', markersize=10, color='b')
            ax[col].plot(10*1e3 * m2km*km2R, 0,
                         marker='.', markersize=10, color='r')
        ax[col].set_xlim([x_min * m2km*km2R, x_max * m2km*km2R])
        ax[col].set_ylim([x_min * m2km*km2R, x_max * m2km*km2R])
        ax[col].set_title(title, fontsize=font_map)
        #ax[col].set_xlabel('$x/R$ [-]', fontsize=font_map)
        if col == 0:
            ax[col].set_ylabel('$y/R$ [-]', fontsize=font)
        ax[col].tick_params(axis='both', labelsize=font)

        if colorbar:
            cbar = fig.colorbar(cs, ax=ax[col], shrink=0.95,
                                ticks=[0, 200, 400, 600, 800])
            cbar.ax.tick_params(labelsize=font)
            cbar.set_label('Gravity [mGal]', rotation=90, fontsize=font)

    # Plot the global gravity results
    plt.gcf()
    fig, ax = plt.subplots(1, 2,
                           sharex=True,
                           figsize=(7.5, 3.6),
                           gridspec_kw={'width_ratios': [1, 1.25]})

    acc_max = np.max(np.array([np.nanmax(normaccXY_poly),
                               np.nanmax(normaccXY_polyheterogeneous)]))

    plot_map(normaccXY_poly, 0, title='Constant density polyhedron')
    plot_map(normaccXY_polyheterogeneous, 1, title='Heterogeneous polyhedron', colorbar=True)
    fig.supxlabel('$x/R$ [-]', fontsize=font, x=0.5, y=0.1)

    # Save figure
    plt.savefig('Plots/gravity_groundtruth.pdf', format='pdf')

# Change working path
os.chdir('/Users/julio/Desktop/python_scripts/THOR/scripts')

#
n_faces = '200700faces'
file_poly = 'Results/eros/groundtruth/poly' + n_faces + '/dense_alt50km_100000samples.pck'
file_polyheterogeneous = 'Results/eros/groundtruth/polyheterogeneous' + n_faces + '/dense_alt50km_100000samples.pck'

# Import groundtruth data from file
scen_poly = pck.load(open(file_poly, "rb"))
scen_polyheterogeneous = pck.load(open(file_polyheterogeneous, "rb"))

# Retrieve polyhedron shape
xyz_vert = scen_poly.groundtruth.asteroid.shape.xyz_vert

# Obtain trajectory and mesh bounds
x_min = -1.5*16*1e3
x_max = 1.5*16*1e3

# Retrieve cartesian coordinates
Xy = scen_poly.groundtruth.gravmap.map_2D.Xy
Yx = scen_poly.groundtruth.gravmap.map_2D.Yx

# Retrieve acceleration field
accXY_poly = scen_poly.groundtruth.gravmap.map_2D.acc_XY
accXY_polyheterogeneous = scen_polyheterogeneous.groundtruth.gravmap.map_2D.acc_XY

# Compute norm
normaccXY_poly = np.linalg.norm(accXY_poly, axis=2)
normaccXY_polyheterogeneous = np.linalg.norm(accXY_polyheterogeneous, axis=2)
erraccXY = (normaccXY_polyheterogeneous - normaccXY_poly) / normaccXY_poly * 100
#plot_grav2D()
plot2_grav2D()
plt.show()
