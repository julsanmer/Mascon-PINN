import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
from matplotlib.cm import get_cmap
import matplotlib.colors as colors
from matplotlib import rc

# --------------------------------- COMPONENTS & SUBPLOT HANDLING ----------------------------------------------- #
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


# ------------------------------------- MAIN PLOT HANDLING ------------------------------------------------------ #
def plot_orb(parameters, outputs, frame='inertial'):
    # Choose frame to represent
    if frame == 'inertial':
        pos = outputs.groundtruth.pos_CA_N
    elif frame == 'asteroid':
        pos = outputs.groundtruth.pos_CA_A

    # Get polyhedron and landmarks
    xyz_vert = parameters.asteroid.xyz_vert
    order_face = parameters.asteroid.order_face
    xyz_lmk = parameters.sensors.xyz_lmk

    # Create figure, plot asteroid, landmarks and orbit
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(xyz_vert[:, 0]*m2km, xyz_vert[:, 1]*m2km, xyz_vert[:, 2]*m2km,
                    triangles=order_face-1, color=color_asteroid, zorder=0)
    ax.plot(pos[:, 0]*m2km, pos[:, 1]*m2km, pos[:, 2]*m2km, 'b', zorder=20, linewidth=0.5)
    ax.plot(xyz_lmk[:, 0]*m2km, xyz_lmk[:, 1]*m2km, xyz_lmk[:, 2]*m2km,
            'k', linestyle='', marker='s', markersize=2.25, zorder=10)

    # Set labels according to frame
    if frame == 'inertial':
        ax.set_xlabel('$x^N$ [km]', fontsize=font, labelpad=15)
        ax.set_ylabel('$y^N$ [km]', fontsize=font, labelpad=15)
        ax.set_zlabel('$z^N$ [km]', fontsize=font, labelpad=15)
        ax.set_title('Inertial frame', fontsize=font)
    elif frame == 'asteroid':
        ax.set_xlabel('$x$ [km]', fontsize=font, labelpad=15)
        ax.set_ylabel('$y$ [km]', fontsize=font, labelpad=15)
        ax.set_zlabel('$z$ [km]', fontsize=font, labelpad=15)
    ax.tick_params(axis='both', labelsize=font)
    ax.set_facecolor('white')
    plt.savefig('Plots/position.pdf', format='pdf')


def plot_mascon(pos_M, mu_M, xyz_vert, order_face):
    # Make figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Filter small masses for visualization purposes
    mu_lim = 1
    mu_M[mu_M < mu_lim] = mu_lim

    # Plot mascon distribution
    cmap = plt.get_cmap('viridis')
    cmap.set_under(cmap(0))
    ax.plot_trisurf(xyz_vert[:, 0]*m2km, xyz_vert[:, 1]*m2km, xyz_vert[:,2]*m2km,
                    triangles=order_face-1, color=color_asteroid, zorder=0, alpha=.2)
    p = ax.scatter3D(pos_M[:, 0]*m2km, pos_M[:, 1]*m2km, pos_M[:, 2]*m2km, c=mu_M,
                     cmap=cmap, norm=mpl.colors.LogNorm())
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
    plt.savefig('Plots/mascon.pdf', format='pdf', bbox_inches='tight')


def plot_pos(t, r_truth, r_dmcukf, flag_nav):
    # Make figure
    fig, ax = plt.subplots(3, sharex=True, figsize=(12, 6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    # Plot ground truth position
    ax[0].plot(t*sec2day, r_truth[:, 0]*m2km, 'b', label='truth')
    ax[1].plot(t*sec2day, r_truth[:, 1]*m2km, 'b')
    ax[2].plot(t*sec2day, r_truth[:, 2]*m2km, 'b')

    # Plot dmc-ukf position
    ax[0].plot(t*sec2day, r_dmcukf[:, 0]*m2km * flag_nav, color='orange', label='estimate')
    ax[1].plot(t*sec2day, r_dmcukf[:, 1]*m2km * flag_nav, color='orange')
    ax[2].plot(t*sec2day, r_dmcukf[:, 2]*m2km * flag_nav, color='orange')

    # Set labels
    plt.xlabel('Time [days]', fontsize=font)
    ax[0].set_ylabel('$x$ [km]', fontsize=font)
    ax[1].set_ylabel('$y$ [km]', fontsize=font)
    ax[2].set_ylabel('$z$ [km]', fontsize=font)
    plt.title('Spacecraft position', fontsize=font)

    # Set limits on axis
    ax[0].set_xlim([t[0]*sec2day, t[-1]*sec2day])
    ax[1].set_xlim([t[0]*sec2day, t[-1]*sec2day])
    ax[2].set_xlim([t[0]*sec2day, t[-1]*sec2day])

    # Set ticks, grid and legend
    ax[0].tick_params(axis='both', labelsize=font)
    ax[1].tick_params(axis='both', labelsize=font)
    ax[2].tick_params(axis='both', labelsize=font)
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].legend(fontsize=font_legend)


def plot_vel(t, v_truth, v_dmcukf, flag_nav):
    # Make figure
    plt.gcf()
    fig, ax = plt.subplots(3, sharex=True, figsize=(12, 6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    # Plot ground truth velocity
    ax[0].plot(t*sec2day, v_truth[:, 0], 'b', label='truth')
    ax[1].plot(t*sec2day, v_truth[:, 1], 'b')
    ax[2].plot(t*sec2day, v_truth[:, 2], 'b')

    # Plot dmc-ukf velocity
    ax[0].plot(t*sec2day, v_dmcukf[:, 0] * flag_nav, color='orange', label='estimate')
    ax[1].plot(t*sec2day, v_dmcukf[:, 1] * flag_nav, color='orange')
    ax[2].plot(t*sec2day, v_dmcukf[:, 2] * flag_nav, color='orange')

    # Set labels
    plt.xlabel('Time [days]', fontsize=font)
    ax[0].set_ylabel('$v_{x}$ [m/s]', fontsize=font)
    ax[1].set_ylabel('$v_{y}$ [m/s]', fontsize=font)
    ax[2].set_ylabel('$v_{z}$ [m/s]', fontsize=font)
    plt.title('Spacecraft velocity', fontsize=font)

    # Set limits on axis
    ax[0].set_xlim([t[0]*sec2day, t[-1]*sec2day])
    ax[1].set_xlim([t[0]*sec2day, t[-1]*sec2day])
    ax[2].set_xlim([t[0]*sec2day, t[-1]*sec2day])

    # Set ticks, grid and legend
    ax[0].tick_params(axis='both', labelsize=font)
    ax[1].tick_params(axis='both', labelsize=font)
    ax[2].tick_params(axis='both', labelsize=font)
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].legend(fontsize=font_legend)


def plot_acc(t, a_truth, a_dmcukf, flag_nav):
    # Preallocate measurement outages variables
    sol_nav = np.ones(len(flag_nav))
    sol_nav[flag_nav == 0] = np.nan
    t_outage = []
    ysup_outage = []
    ylow_outage = []
    switch = 1
    for i in range(len(flag_nav)):
        if switch == 1 and abs(flag_nav[i]) < 1e-6:
            switch = 0
            t0 = t[i]
        if switch == 0 and abs(flag_nav[i]-1) < 1e-6:
            tf = t[i]
            switch = 1
            t_outage.append([t0, tf])
            ysup_outage.append([100, 100])
            ylow_outage.append([-100, -100])
    t_outage = np.array(t_outage)
    ysup_outage = np.array(ysup_outage)
    ylow_outage = np.array(ylow_outage)

    # Make figure
    plt.gcf()
    fig, ax = plt.subplots(3, sharex=True, figsize=(12, 8))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    # Plot ground truth inhomogeneous gravity acceleration
    ax[0].plot(t*sec2day, a_truth[:, 0]*m2mu * sol_nav, color='blue')
    ax[1].plot(t*sec2day, a_truth[:, 1]*m2mu * sol_nav, color='blue', label='Truth')
    ax[2].plot(t*sec2day, a_truth[:, 2]*m2mu * sol_nav, color='blue')

    # Plot dmc-ukf inhomogeneous gravity acceleration
    ax[0].plot(t*sec2day, a_dmcukf[:, 0]*m2mu * sol_nav, color='orange', marker='.', markersize=2,
               linestyle='')
    ax[1].plot(t*sec2day, a_dmcukf[:, 1]*m2mu * sol_nav, color='orange', marker='.', markersize=2,
               linestyle='', label='DMC-UKF')
    ax[2].plot(t*sec2day, a_dmcukf[:, 2]*m2mu * sol_nav, color='orange', marker='.', markersize=2,
               linestyle='')

    # Plot measurement outages
    for i in range(len(t_outage)):
        if i == 0:
            ax[1].fill_between(t_outage[i, 0:2]*sec2day, ylow_outage[i, 0:2], ysup_outage[i, 0:2],
                               color='red', alpha=.1, label='Nav. gaps')
        else:
            ax[1].fill_between(t_outage[i, 0:2]*sec2day, ylow_outage[i, 0:2], ysup_outage[i, 0:2],
                               color='red', alpha=.1)
        ax[0].fill_between(t_outage[i, 0:2]*sec2day, ylow_outage[i, 0:2], ysup_outage[i, 0:2],
                           color='red', alpha=.1)
        ax[2].fill_between(t_outage[i, 0:2]*sec2day, ylow_outage[i, 0:2], ysup_outage[i, 0:2],
                           color='red', alpha=.1)

    # Set labels
    plt.xlabel('Time [days]', fontsize=font, labelpad=10)
    ax[0].set_ylabel('$a_{\mathrm{grav},x}$ [$\mu$m/s$^2$]', fontsize=font)
    ax[1].set_ylabel('$a_{\mathrm{grav},y}$ [$\mu$m/s$^2$]', fontsize=font)
    ax[2].set_ylabel('$a_{\mathrm{grav},z}$ [$\mu$m/s$^2$]', fontsize=font)
    ax[0].set_title('Mascon $n$=400', fontsize=font)

    # Set limits on axis
    ax[0].set_xlim([t[0]*sec2day, t[-1]*sec2day])
    ax[1].set_xlim([t[0]*sec2day, t[-1]*sec2day])
    ax[2].set_xlim([t[0]*sec2day, t[-1]*sec2day])
    ax[0].set_ylim([-100, 100])
    ax[1].set_ylim([-100, 100])
    ax[2].set_ylim([-100, 100])

    # Set ticks, grid and legend
    ax[0].tick_params(axis='both', labelsize=font)
    ax[1].tick_params(axis='both', labelsize=font)
    ax[2].tick_params(axis='both', labelsize=font)
    ax[0].set_yticks([-100, -50, 0, 50, 100])
    ax[1].set_yticks([-100, -50, 0, 50, 100])
    ax[2].set_yticks([-100, -50, 0, 50, 100])
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[1].legend(fontsize=font_legend, bbox_to_anchor=(1.32, 0.15), borderaxespad=0.)

    plt.tight_layout()
    #plt.savefig('Plots/TAES/accError.pdf', format='pdf')


def plot_pos_error(t, dr, P, flag_nav):
    # Preallocate measurement outages variables
    sol_nav = np.ones(len(flag_nav))
    sol_nav[sol_nav == 0] = np.nan
    t_outage = []
    ysup_outage = []
    ylow_outage = []
    switch = 1
    for i in range(len(flag_nav)):
        if switch == 1 and abs(flag_nav[i]) < 1e-6:
            switch = 0
            t0 = t[i]
        if switch == 0 and abs(flag_nav[i]-1) < 1e-6:
            tf = t[i]
            switch = 1
            t_outage.append([t0, tf])
            ysup_outage.append([100, 100])
            ylow_outage.append([-100, -100])
    t_outage = np.array(t_outage)
    ysup_outage = np.array(ysup_outage)
    ylow_outage = np.array(ylow_outage)

    # Make figure
    plt.gcf()
    fig, ax = plt.subplots(3, sharex=True, figsize=(12, 8))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    # Plot x position error and uncertainty
    ax[0].plot(t*sec2day, dr[:, 0] * sol_nav, 'b', label='Error', marker='.',
               markersize=2, linestyle='')
    ax[0].plot(t*sec2day, 3*np.sqrt(P[:, 0, 0]) * sol_nav, 'k--')
    ax[0].plot(t*sec2day, -3*np.sqrt(P[:, 0, 0]) * sol_nav, 'k--')

    # Plot y position error and uncertainty
    ax[1].plot(t*sec2day, dr[:, 1] * sol_nav, 'b', marker='.', markersize=2, linestyle='', label='Error')
    ax[1].plot(t*sec2day, 3*np.sqrt(P[:, 1, 1]) * sol_nav, 'k--', label=r'3-$\sigma$ bounds')
    ax[1].plot(t*sec2day, -3*np.sqrt(P[:, 1, 1]) * sol_nav, 'k--')

    # Plot z position error and uncertainty
    ax[2].plot(t*sec2day, dr[:, 2] * sol_nav, 'b', marker='.', markersize=2, linestyle='')
    ax[2].plot(t*sec2day, 3*np.sqrt(P[:, 2, 2]) * sol_nav, 'k--')
    ax[2].plot(t*sec2day, -3*np.sqrt(P[:, 2, 2]) * sol_nav, 'k--')

    # Plot measurement outages
    for i in range(len(t_outage)):
        if i == 0:
            ax[1].fill_between(t_outage[i, 0:2]*sec2day, ylow_outage[i, 0:2], ysup_outage[i, 0:2],
                               color='red', alpha=.1, label='Nav. gaps')
        else:
            ax[1].fill_between(t_outage[i, 0:2]*sec2day, ylow_outage[i, 0:2], ysup_outage[i, 0:2],
                               color='red', alpha=.1)
        ax[0].fill_between(t_outage[i, 0:2]*sec2day, ylow_outage[i, 0:2], ysup_outage[i, 0:2],
                           color='red', alpha=.1)
        ax[2].fill_between(t_outage[i, 0:2]*sec2day, ylow_outage[i, 0:2], ysup_outage[i, 0:2],
                           color='red', alpha=.1)

    # Set labels
    plt.xlabel('Time [days]', fontsize=font, labelpad=10)
    ax[0].set_ylabel('$x$ [m]', fontsize=font)
    ax[1].set_ylabel('$y$ [m]', fontsize=font)
    ax[2].set_ylabel('$z$ [m]', fontsize=font)
    ax[0].set_title('Mascon $n$=400', fontsize=font)

    # Set limits on axis
    ax[0].set_xlim([t[0]*sec2day, t[-1]*sec2day])
    ax[1].set_xlim([t[0]*sec2day, t[-1]*sec2day])
    ax[2].set_xlim([t[0]*sec2day, t[-1]*sec2day])
    ax[0].set_ylim([-30, 30])
    ax[1].set_ylim([-30, 30])
    ax[2].set_ylim([-30, 30])

    # Set ticks, grid and legend
    ax[0].tick_params(axis='both', labelsize=font)
    ax[1].tick_params(axis='both', labelsize=font)
    ax[2].tick_params(axis='both', labelsize=font)
    ax[0].set_yticks([-30, -15, 0, 15, 30])
    ax[1].set_yticks([-30, -15, 0, 15, 30])
    ax[2].set_yticks([-30, -15, 0, 15, 30])
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[1].legend(fontsize=font_legend, bbox_to_anchor=(1.33, 0.15), borderaxespad=0.)
    plt.tight_layout()


def plot_vel_error(t, dv, P, flag_nav):
    # Make figure
    plt.gcf()
    fig, ax = plt.subplots(3, sharex=True, sharey=True, figsize=(12, 6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    # Plot vx velocity error and uncertainty
    ax[0].plot(t*sec2day, dv[:, 0]*m2mm * flag_nav, 'b', label='error')
    ax[0].plot(t*sec2day, 3*np.sqrt(P[:, 3, 3])*m2mm * flag_nav, 'k--', label=r'$3\sigma$')
    ax[0].plot(t*sec2day, -3*np.sqrt(P[:, 3, 3])*m2mm * flag_nav, 'k--')

    # Plot vy velocity error and uncertainty
    ax[1].plot(t*sec2day, dv[:, 1]*m2mm * flag_nav, 'b')
    ax[1].plot(t*sec2day, 3*np.sqrt(P[:, 4, 4])*m2mm * flag_nav, 'k--')
    ax[1].plot(t*sec2day, -3*np.sqrt(P[:, 4, 4])*m2mm * flag_nav, 'k--')

    # Plot vz velocity error and uncertainty
    ax[2].plot(t*sec2day, dv[:, 2]*m2mm * flag_nav, 'b')
    ax[2].plot(t*sec2day, 3*np.sqrt(P[:, 5, 5])*m2mm * flag_nav, 'k--')
    ax[2].plot(t*sec2day, -3*np.sqrt(P[:, 5, 5])*m2mm * flag_nav, 'k--')

    # Set labels
    plt.xlabel('Time [days]', fontsize=font)
    ax[0].set_ylabel('$v_{x}$ [mm/s]', fontsize=font)
    ax[1].set_ylabel('$v_{y}$ [mm/s]', fontsize=font)
    ax[2].set_ylabel('$v_{z}$ [mm/s]', fontsize=font)
    plt.title('Velocity error and covariance', fontsize=font)

    # Set limits on axis
    ax[0].set_xlim([t[0]*sec2day, t[-1]*sec2day])
    ax[1].set_xlim([t[0]*sec2day, t[-1]*sec2day])
    ax[2].set_xlim([t[0]*sec2day, t[-1]*sec2day])

    # Set ticks, grid and legend
    ax[0].tick_params(axis='both', labelsize=font)
    ax[1].tick_params(axis='both', labelsize=font)
    ax[2].tick_params(axis='both', labelsize=font)
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].legend(fontsize=font_legend)


def plot_gravity3D(h_3D, aErr_3D, a0Err_3D, h):
    # Preallocate number of data for radius range
    n_truth = 10
    h_min = np.min(h)
    h_max = np.max(h)
    N_truth = np.zeros(n_truth)

    # Obtain number of data in each radius range
    for i in range(n_truth):
        hmin_i = h_min + i * (h_max-h_min) / n_truth
        hmax_i = h_min + (i+1) * (h_max-h_min) / n_truth
        idx = np.where(np.logical_and(h >= hmin_i, h <= hmax_i))[0]
        N_truth[i] = len(idx)
    N_truth /= np.sum(N_truth)

    # Transpose variables to 1D vectors
    h_1D = np.ravel(h_3D)
    aErr_1D = np.ravel(aErr_3D)
    a0Err_1D = np.ravel(a0Err_3D)

    # Prepare bins for gravity errors
    h0 = np.nanmin(h_1D)
    hf = np.nanmax(h_1D)
    n_bins = 80
    h_lim = np.linspace(h0, hf + 1e-6*hf, n_bins+1)
    h_bins = h_lim[0:-1] + (h_lim[1] - h_lim[0]) / 2
    aErr_bins = np.zeros(n_bins)
    a0Err_bins = np.zeros(n_bins)
    N_bins = np.zeros(n_bins)

    # Compute average gravity error in each radius range
    for i in range(len(h_1D)):
        if np.isnan(h_1D[i]):
            continue
        dh = h_1D[i] - h0
        idx = int(np.floor(dh / (h_lim[1]-h_lim[0])))
        aErr_bins[idx] += aErr_1D[i]
        a0Err_bins[idx] += a0Err_1D[i]
        N_bins[idx] += 1
    aErr_bins /= N_bins
    a0Err_bins /= N_bins

    # Make figure
    plt.gcf()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Plot 3D gravity error and their average
    ax.plot(h_1D*m2km, aErr_1D*err2perc, marker='o', linestyle='', markersize=1)
    ax.plot(h_1D*m2km, a0Err_1D*err2perc, marker='o', linestyle='', markersize=1)
    ax.plot(h_bins*m2km, aErr_bins*err2perc)
    ax.plot(h_bins*m2km, a0Err_bins*err2perc)

    # Plot data lines
    plt.axvline(x=np.min(h)*m2km, color='b', label='Data bounds')
    plt.axvline(x=np.max(h)*m2km, color='b')
    plt.axvline(x=17.68, color='r', label='Brillouin sphere')
    for i in range(n_truth):
        ax.bar(h_min*m2km + (i+0.5)*(h_max-h_min)/n_truth*m2km, 1e3*N_truth[i],
               (h_max-h_min)/n_truth*m2km, 0, alpha=0.2, color='red')

    # Set logarithmic scale
    ax.set_yscale('log')

    # Set labels
    plt.xlabel('Radius [km]', fontsize=font)
    plt.ylabel('Gravity acceleration error [\%]', fontsize=font)

    # Set limits on axis
    ax.set_xlim([np.nanmin(h_1D)*m2km, np.nanmax(h_1D)*m2km])

    # Set ticks, grid and legend
    ax.tick_params(axis='both', labelsize=font)
    ax.grid()
    ax.legend(loc='upper right', fontsize=font_legend)


def plot_gravity2D(outputs, parameters):
    # Obtain XYZ and gravity errors meshes
    Xy = outputs.groundtruth.Xy_2D
    Yx = outputs.groundtruth.Yx_2D
    Xz = outputs.groundtruth.Xz_2D
    Zx = outputs.groundtruth.Zx_2D
    Yz = outputs.groundtruth.Yz_2D
    Zy = outputs.groundtruth.Zy_2D
    aErrXY_2D = outputs.results.aErrXY_2D
    aErrXZ_2D = outputs.results.aErrXZ_2D
    aErrYZ_2D = outputs.results.aErrYZ_2D
    a0ErrXY_2D = outputs.groundtruth.a0ErrXY_2D
    a0ErrXZ_2D = outputs.groundtruth.a0ErrXZ_2D
    a0ErrYZ_2D = outputs.groundtruth.a0ErrYZ_2D

    # Prune upper errors for visualization purposes
    aErr_lim = 0.1
    aErrXY_2D[aErrXY_2D > aErr_lim] = aErr_lim
    aErrXZ_2D[aErrXZ_2D > aErr_lim] = aErr_lim
    aErrYZ_2D[aErrYZ_2D > aErr_lim] = aErr_lim
    a0ErrXY_2D[a0ErrXY_2D > aErr_lim] = aErr_lim
    a0ErrXZ_2D[a0ErrXZ_2D > aErr_lim] = aErr_lim
    a0ErrYZ_2D[a0ErrYZ_2D > aErr_lim] = aErr_lim

    # Obtain trajectory and mesh bounds
    pos = outputs.groundtruth.pos_CA_A
    x_min = -outputs.groundtruth.rmax
    x_max = outputs.groundtruth.rmax
    linestyle_orbit = 'solid'

    # Retrieve asteroid
    xyz_vert = parameters.asteroid.xyz_vert

    # Plot the global gravity results
    plt.gcf()
    fig, ax = plt.subplots(2, 3, sharex=True, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1, 1.2]})

    ticks = np.linspace(0, aErr_lim, 11) * 100

    # Plot Keplerian xy map error
    cs = ax[0, 0].contourf(Xy*m2km, Yx*m2km, a0ErrXY_2D*err2perc, levels=100, cmap=get_cmap("viridis"))
    ax[0, 0].plot(xyz_vert[:, 0]*m2km, xyz_vert[:, 1]*m2km, color=color_asteroid)
    ax[0, 0].set_xlim([x_min*m2km, x_max*m2km])
    ax[0, 0].set_ylim([x_min*m2km, x_max*m2km])
    ax[0, 0].set_ylabel('$y$ [km]', fontsize=font_map)
    ax[0, 0].tick_params(axis='both', labelsize=font_map)

    # Plot Keplerian xz map error
    cs = ax[0, 1].contourf(Xz*m2km, Zx*m2km, a0ErrXZ_2D*err2perc, levels=100, cmap=get_cmap("viridis"))
    ax[0, 1].plot(xyz_vert[:, 0]*m2km, xyz_vert[:, 2]*m2km, color=color_asteroid)
    ax[0, 1].set_xlim([x_min*m2km, x_max*m2km])
    ax[0, 1].set_ylim([x_min*m2km, x_max*m2km])
    ax[0, 1].set_ylabel('$z$ [km]', fontsize=font_map)
    ax[0, 1].tick_params(axis='both', labelsize=font_map)
    ax[0, 1].set_title('Initial', fontsize=font_map, fontweight='bold')

    # Plot Keplerian yz map error
    cs = ax[0, 2].contourf(Yz*m2km, Zy*m2km, a0ErrYZ_2D*err2perc, levels=100, cmap=get_cmap("viridis"))
    cbar = fig.colorbar(cs, ax=ax[0, 2], shrink=0.95, ticks=ticks)
    cbar.ax.set_yticklabels(['$0$', '', '$2$', '', '$4$', '', '$6$', '', '$8$', '', '$\geq$$10$'])
    cbar.ax.tick_params(labelsize=font_map)
    cbar.set_label('Gravity error [\%]', rotation=90, fontsize=font_map)
    ax[0, 2].plot(xyz_vert[:, 1]*m2km, xyz_vert[:, 2]*m2km, color=color_asteroid)
    ax[0, 2].set_xlim([x_min*m2km, x_max*m2km])
    ax[0, 2].set_ylim([x_min*m2km, x_max*m2km])
    ax[0, 2].set_ylabel('$z$ [km]', fontsize=font_map)
    ax[0, 2].tick_params(axis='both', labelsize=font_map)

    # Plot mascon xy map error
    cs = ax[1, 0].contourf(Xy*m2km, Yx*m2km, aErrXY_2D*err2perc, levels=100, cmap=get_cmap("viridis"))
    ax[1, 0].plot(xyz_vert[:, 0]*m2km, xyz_vert[:, 1]*m2km, color=color_asteroid)
    #ax[1, 0].plot(r[:, 0]*m2km, r[:, 1]*m2km, 'k', linestyle=linestyle_orbit, linewidth=0.5)
    ax[1, 0].set_xlim([x_min*m2km, x_max*m2km])
    ax[1, 0].set_ylim([x_min*m2km, x_max*m2km])
    ax[1, 0].set_xlabel('$x$ [km]', fontsize=font_map)
    ax[1, 0].set_ylabel('$y$ [km]', fontsize=font_map)
    ax[1, 0].tick_params(axis='both', labelsize=font_map)

    # Plot mascon xz map error
    cs = ax[1, 1].contourf(Xy*m2km, Zx*m2km, aErrXZ_2D*err2perc, levels=100, cmap=get_cmap("viridis"))
    ax[1, 1].plot(xyz_vert[:, 0]*m2km, xyz_vert[:, 2]*m2km, color=color_asteroid)
    #ax[1, 1].plot(r[:, 0]*m2km, r[:, 2]*m2km, 'k', linestyle=linestyle_orbit, linewidth=0.5)
    ax[1, 1].set_xlim([x_min*m2km, x_max*m2km])
    ax[1, 1].set_ylim([x_min*m2km, x_max*m2km])
    ax[1, 1].set_xlabel('$x$ [km]', fontsize=font_map)
    ax[1, 1].set_ylabel('$z$ [km]', fontsize=font_map)
    ax[1, 1].tick_params(axis='both', labelsize=font_map)
    ax[1, 1].set_title('Final', fontsize=font_map, fontweight='bold')

    # Plot mascon yz map error
    cs = ax[1, 2].contourf(Yz*m2km, Zy*m2km, aErrYZ_2D*err2perc, levels=100, cmap=get_cmap("viridis"))
    cbar = fig.colorbar(cs, ax=ax[1, 2], shrink=0.95, ticks=ticks)
    cbar.ax.set_yticklabels(['$0$', '', '$2$', '', '$4$', '', '$6$', '', '$8$', '', '$\geq$$10$'])
    cbar.ax.tick_params(labelsize=font_map)
    cbar.set_label('Gravity error [\%]', rotation=90, fontsize=font_map)
    ax[1, 2].plot(xyz_vert[:,1]*m2km, xyz_vert[:, 2]*m2km, color=color_asteroid)
    #ax[1, 2].plot(r[:, 1]*m2km, r[:, 2]*m2km, 'k', linestyle=linestyle_orbit, linewidth=0.5)
    ax[1, 2].set_xlim([x_min*m2km, x_max*m2km])
    ax[1, 2].set_ylim([x_min*m2km, x_max*m2km])
    ax[1, 2].set_xlabel('$y$ [km]', fontsize=font_map)
    ax[1, 2].set_ylabel('$z$ [km]', fontsize=font_map)
    ax[1, 2].tick_params(axis='both', labelsize=font_map)

    plt.tight_layout()
    #plt.savefig('Plots/Eros2Dgrav.eps', format='eps')


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


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
