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
