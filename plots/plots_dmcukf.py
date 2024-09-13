import matplotlib.pyplot as plt
import numpy as np

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


# ------------------------------------- MAIN PLOT HANDLING ------------------------------------------------------ #
def plot_orb(asteroid, pos, frame='inertial'):
    # Get polyhedron and landmarks
    xyz_vert = asteroid.shape.xyz_vert
    order_face = asteroid.shape.order_face
    #xyz_lmk = parameters.sensors.xyz_lmk

    # Create figure, plot asteroid, landmarks and orbit
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(xyz_vert[:, 0]*m2km, xyz_vert[:, 1]*m2km, xyz_vert[:, 2]*m2km,
                    triangles=order_face-1, color=color_asteroid, zorder=0)
    ax.plot(pos[:, 0]*m2km, pos[:, 1]*m2km, pos[:, 2]*m2km, 'b', zorder=20, linewidth=2)
    #ax.plot(xyz_lmk[:, 0]*m2km, xyz_lmk[:, 1]*m2km, xyz_lmk[:, 2]*m2km,
    #        'k', linestyle='', marker='s', markersize=2.25, zorder=10)

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
    set_axes_equal(ax)
    # plt.savefig('Plots/position.pdf', format='pdf')


def plot_pos(t, pos_truth, pos_dmcukf, flag_meas):
    # Make figure
    fig, ax = plt.subplots(3, sharex=True, figsize=(12, 6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    # Plot ground truth position
    ax[0].plot(t*sec2day, pos_truth[:, 0]*m2km, 'b', label='truth')
    ax[1].plot(t*sec2day, pos_truth[:, 1]*m2km, 'b')
    ax[2].plot(t*sec2day, pos_truth[:, 2]*m2km, 'b')

    # Plot dmc-ukf position
    ax[0].plot(t[flag_meas]*sec2day, pos_dmcukf[flag_meas, 0]*m2km, color='orange', linestyle='',
               marker='.', markersize=5, label='estimate')
    ax[1].plot(t[flag_meas]*sec2day, pos_dmcukf[flag_meas, 1]*m2km, color='orange', linestyle='',
               marker='.', markersize=5)
    ax[2].plot(t[flag_meas]*sec2day, pos_dmcukf[flag_meas, 2]*m2km, color='orange', linestyle='',
               marker='.', markersize=5)

    # Plot measurement outages
    plot_outage(ax, t, flag_meas)

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
    ax[0].set_ylim([-40, 40])
    ax[1].set_ylim([-40, 40])
    ax[2].set_ylim([-40, 40])

    # Set ticks, grid and legend
    ax[0].tick_params(axis='both', labelsize=font)
    ax[1].tick_params(axis='both', labelsize=font)
    ax[2].tick_params(axis='both', labelsize=font)
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].legend(fontsize=font_legend)


def plot_vel(t, vel_truth, vel_dmcukf, flag_meas):
    # Make figure
    plt.gcf()
    fig, ax = plt.subplots(3, sharex=True, figsize=(12, 6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    # Plot ground truth velocity
    ax[0].plot(t*sec2day, vel_truth[:, 0], 'b', label='truth')
    ax[1].plot(t*sec2day, vel_truth[:, 1], 'b')
    ax[2].plot(t*sec2day, vel_truth[:, 2], 'b')

    # Plot dmc-ukf velocity
    ax[0].plot(t[flag_meas]*sec2day, vel_dmcukf[flag_meas, 0], color='orange', linestyle='',
               marker='.', markersize=5, label='estimate')
    ax[1].plot(t[flag_meas]*sec2day, vel_dmcukf[flag_meas, 1], color='orange', linestyle='',
               marker='.', markersize=5)
    ax[2].plot(t[flag_meas]*sec2day, vel_dmcukf[flag_meas, 2], color='orange', linestyle='',
               marker='.', markersize=5)

    # Plot measurement outages
    plot_outage(ax, t, flag_meas)

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
    ax[0].set_ylim([-5, 5])
    ax[1].set_ylim([-5, 5])
    ax[2].set_ylim([-5, 5])

    # Set ticks, grid and legend
    ax[0].tick_params(axis='both', labelsize=font)
    ax[1].tick_params(axis='both', labelsize=font)
    ax[2].tick_params(axis='both', labelsize=font)
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].legend(fontsize=font_legend)


def plot_acc(t, a_truth, a_dmcukf, flag_meas):
    # Make figure
    plt.gcf()
    fig, ax = plt.subplots(3, sharex=True, figsize=(12, 8))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    # Plot ground truth inhomogeneous gravity acceleration
    ax[0].plot(t*sec2day, a_truth[:, 0]*m2mu, color='blue')
    ax[1].plot(t*sec2day, a_truth[:, 1]*m2mu, color='blue', label='Truth')
    ax[2].plot(t*sec2day, a_truth[:, 2]*m2mu, color='blue')

    # Plot dmc-ukf inhomogeneous gravity acceleration
    ax[0].plot(t[flag_meas]*sec2day, a_dmcukf[flag_meas, 0]*m2mu, color='orange',
               marker='.', markersize=2, linestyle='')
    ax[1].plot(t[flag_meas]*sec2day, a_dmcukf[flag_meas, 1]*m2mu, color='orange',
               marker='.', markersize=2, linestyle='', label='DMC-UKF')
    ax[2].plot(t[flag_meas]*sec2day, a_dmcukf[flag_meas, 2]*m2mu, color='orange',
               marker='.', markersize=2, linestyle='')

    # Plot measurement outages
    plot_outage(ax, t, flag_meas)

    # Set labels
    plt.xlabel('Time [days]', fontsize=font, labelpad=10)
    ax[0].set_ylabel('$a_{\mathrm{grav},x}$ [$\mu$m/s$^2$]', fontsize=font)
    ax[1].set_ylabel('$a_{\mathrm{grav},y}$ [$\mu$m/s$^2$]', fontsize=font)
    ax[2].set_ylabel('$a_{\mathrm{grav},z}$ [$\mu$m/s$^2$]', fontsize=font)

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


def plot_pos_error(t, dr, P, flag_meas):
    # Make figure
    plt.gcf()
    fig, ax = plt.subplots(3, sharex=True, figsize=(12, 8))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    # Plot x position error and uncertainty
    ax[0].plot(t[flag_meas]*sec2day, dr[flag_meas, 0], 'b', label='Error',
               marker='.', markersize=2, linestyle='')
    ax[0].plot(t[flag_meas]*sec2day, 3*np.sqrt(P[flag_meas, 0, 0]), 'k--')
    ax[0].plot(t[flag_meas]*sec2day, -3*np.sqrt(P[flag_meas, 0, 0]), 'k--')

    # Plot y position error and uncertainty
    ax[1].plot(t[flag_meas]*sec2day, dr[flag_meas, 1], 'b', label='Error',
               marker='.', markersize=2, linestyle='')
    ax[1].plot(t[flag_meas]*sec2day, 3*np.sqrt(P[flag_meas, 1, 1]), 'k--',
               label=r'3-$\sigma$ bounds')
    ax[1].plot(t[flag_meas]*sec2day, -3*np.sqrt(P[flag_meas, 1, 1]), 'k--')

    # Plot z position error and uncertainty
    ax[2].plot(t[flag_meas]*sec2day, dr[flag_meas, 2], 'b',
               marker='.', markersize=2, linestyle='')
    ax[2].plot(t[flag_meas]*sec2day, 3*np.sqrt(P[flag_meas, 2, 2]), 'k--')
    ax[2].plot(t[flag_meas]*sec2day, -3*np.sqrt(P[flag_meas, 2, 2]), 'k--')

    plot_outage(ax, t, flag_meas)

    # Set labels
    plt.xlabel('Time [days]', fontsize=font, labelpad=10)
    ax[0].set_ylabel('$x$ [m]', fontsize=font)
    ax[1].set_ylabel('$y$ [m]', fontsize=font)
    ax[2].set_ylabel('$z$ [m]', fontsize=font)

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


def plot_vel_error(t, dv, P, flag_meas):
    # Make figure
    plt.gcf()
    fig, ax = plt.subplots(3, sharex=True, sharey=True, figsize=(12, 6))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    # Plot vx velocity error and uncertainty
    ax[0].plot(t[flag_meas]*sec2day, dv[flag_meas, 0]*m2mm, 'b',
               marker='.', markersize=2, linestyle='')
    ax[0].plot(t[flag_meas]*sec2day, 3*np.sqrt(P[flag_meas, 3, 3])*m2mm, 'k--',
               label=r'$3\sigma$')
    ax[0].plot(t[flag_meas]*sec2day, -3*np.sqrt(P[flag_meas, 3, 3])*m2mm, 'k--')

    # Plot vy velocity error and uncertainty
    ax[1].plot(t[flag_meas]*sec2day, dv[flag_meas, 1]*m2mm, 'b',
               marker='.', markersize=2, linestyle='')
    ax[1].plot(t[flag_meas]*sec2day, 3*np.sqrt(P[flag_meas, 4, 4])*m2mm, 'k--')
    ax[1].plot(t[flag_meas]*sec2day, -3*np.sqrt(P[flag_meas, 4, 4])*m2mm, 'k--')

    # Plot vz velocity error and uncertainty
    ax[2].plot(t[flag_meas]*sec2day, dv[flag_meas, 2]*m2mm, 'b',
               marker='.', markersize=2, linestyle='')
    ax[2].plot(t[flag_meas]*sec2day, 3*np.sqrt(P[flag_meas, 5, 5])*m2mm, 'k--')
    ax[2].plot(t[flag_meas]*sec2day, -3*np.sqrt(P[flag_meas, 5, 5])*m2mm, 'k--')

    # Plot measurement outages
    plot_outage(ax, t, flag_meas)

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
    ax[0].set_ylim([-50, 50])
    ax[1].set_ylim([-50, 50])
    ax[2].set_ylim([-50, 50])

    # Set ticks, grid and legend
    ax[0].tick_params(axis='both', labelsize=font)
    ax[1].tick_params(axis='both', labelsize=font)
    ax[2].tick_params(axis='both', labelsize=font)
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].legend(fontsize=font_legend)


def plot_RMSE(a0, inc0, RMSE):
    # Make figure
    plt.gcf()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    for j in range(len(inc0)):
        ax.plot(a0*m2km, RMSE[:, j], linewidth=2, linestyle='--',
                marker='s', markersize=10, clip_on=False,
                label='$i_0$=' + str(int(inc0[j]*rad2deg)) + '$^\circ$')

    ax.set_yscale('log')

    plt.xlabel('$a_0$ [km]', fontsize=font)
    plt.ylabel('Final RMSE [m]', fontsize=font)

    # Set limits on axis
    ax.set_xlim([a0[0]*m2km, a0[-1]*m2km])

    # Set ticks, grid and legend
    ax.tick_params(axis='both', labelsize=font)
    ax.grid()
    ax.legend(fontsize=font_legend, loc='upper right')


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


def plot_outage(ax, t, flag_meas):
    def outage_intervals():
        # Preallocate measurement outages variables
        sol_nav = np.ones(len(flag_meas))
        sol_nav[sol_nav == 0] = np.nan
        t_outage = []
        ysup_outage = []
        ylow_outage = []
        switch = 1
        for i in range(len(flag_meas)):
            if switch == 1 and abs(flag_meas[i]) < 1e-6:
                switch = 0
                t0 = t[i]
            if switch == 0 and abs(flag_meas[i] - 1) < 1e-6:
                tf = t[i]
                switch = 1
                t_outage.append([t0, tf])
                ysup_outage.append([1e3, 1e3])
                ylow_outage.append([-1e3, -1e3])
        t_outage = np.array(t_outage)
        ysup_outage = np.array(ysup_outage)
        ylow_outage = np.array(ylow_outage)

        return t_outage, ylow_outage, ysup_outage

    t_outage, ylow_outage, ysup_outage = outage_intervals()

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


def all_dmcukfplots(scenario):
    asteroid = scenario.groundtruth.asteroid
    sc_data = scenario.groundtruth.spacecraft.data
    dmcukf_data = scenario.dmcukf.data
    camera_data = scenario.measurements.camera.data

    plot_orb(asteroid, sc_data.pos_BP_P)
    plot_pos(sc_data.t,
             sc_data.pos_BP_N1,
             dmcukf_data.pos_BP_N1,
             camera_data.flag_meas)
    plot_vel(sc_data.t,
             sc_data.vel_BP_N1,
             dmcukf_data.vel_BP_N1,
             camera_data.flag_meas)
    plot_pos_error(sc_data.t,
                   dmcukf_data.pos_BP_N1-sc_data.pos_BP_N1,
                   dmcukf_data.Pxx,
                   camera_data.flag_meas)
    plot_vel_error(sc_data.t,
                   dmcukf_data.vel_BP_N1-sc_data.vel_BP_N1,
                   dmcukf_data.Pxx,
                   camera_data.flag_meas)
    plt.show()
