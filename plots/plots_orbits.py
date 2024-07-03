import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

from src.orbits.check_collision import check_collision

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
colors = ['#ffd700',
          '#ffb14e',
          '#fa8775',
          '#ea5f94',
          '#cd34b5',
          '#9d02d7',
          '#0000ff']
colorsGray = ['#FFFFFF',
              '#B2BEB5',
              '#676767']


# ------------------------------------- MAIN PLOT HANDLING ------------------------------------------------------ #
def plot_orb(orbits):
    # Get polyhedron
    shape = orbits.asteroid.shape
    xyz_vert = shape.xyz_vert
    order_face = shape.order_face

    # Create figure, plot asteroid and orbits
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(xyz_vert[:, 0] * m2km, xyz_vert[:, 1] * m2km, xyz_vert[:, 2] * m2km,
                    triangles=order_face-1, color=color_asteroid, zorder=0)
    for i in range(orbits.n_a):
        for j in range(orbits.n_inc):
            pos = orbits.orbits[i][j].data.pos_BP_P
            # ax.plot(pos[0, 0] * m2km, pos[0, 1] * m2km, pos[0, 2] * m2km,
            #         color=colors[2*j], marker='.')
            ax.plot(pos[:, 0] * m2km, pos[:, 1] * m2km, pos[:, 2] * m2km,
                    color=colors[2*j], zorder=20, linewidth=0.5)
            # ax.plot(pos[-1, 0] * m2km, pos[-1, 1] * m2km, pos[-1, 2] * m2km,
            #         'b', marker='s')

    # Set limits
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 50)

    # Set labels
    ax.set_xlabel('$x$ [km]', fontsize=font, labelpad=15)
    ax.set_ylabel('$y$ [km]', fontsize=font, labelpad=15)
    ax.set_zlabel('$z$ [km]', fontsize=font, labelpad=15)
    ax.tick_params(axis='both', labelsize=font)
    ax.set_facecolor('white')


# Plot orbits
def plot_RMSE(orbits, oe_dict):
    n_a = len(oe_dict['a'])
    n_inc = len(oe_dict['inc'])
    RMSEf = np.zeros((n_a, n_inc))

    plt.gcf()
    fig, ax = plt.subplots(int(n_inc/2), 2, figsize=(12, 8))
    a = np.zeros(n_a)
    for i in range(n_a):
        a[i] = oe_dict['a'][i]
        for j in range(n_inc):
            pos_err = orbits.orbits[i][j].data.pos_err
            n = len(pos_err)
            RMSEf[i, j] = np.sqrt(np.nansum(pos_err**2)/n)

    for i in range(int(n_inc/2)):
        ax[i, 0].plot(a/1e3, RMSEf[:, 2*i], clip_on=False)
        ax[i, 1].plot(a/1e3, RMSEf[:, 2*i+1], clip_on=False)

        # Aesthetic
        ax[i, 0].set_yscale('log')
        ax[i, 1].set_yscale('log')
        ax[i, 0].set_xlim(np.min(a) * m2km, np.max(a) * m2km)
        ax[i, 1].set_xlim(np.min(a) * m2km, np.max(a) * m2km)
        ax[i, 0].set_ylim(1e-2, 50)
        ax[i, 1].set_ylim(1e-2, 50)
        ax[i, 0].set_title('$i_0=$' + str(oe_dict['inc'][2*i] * rad2deg) + ' deg.',
                           fontsize=font)
        ax[i, 1].set_title('$i_0=$' + str(oe_dict['inc'][2*i+1] * rad2deg) + ' deg.',
                           fontsize=font)
        ax[i, 0].set_ylabel('RMSE [m]', fontsize=font)
        ax[i, 0].tick_params(axis='both', labelsize=font)
        ax[i, 1].tick_params(axis='both', labelsize=font)
        ax[i, 0].grid()
        ax[i, 1].grid()


def plot_tcpu(tcpu_list, oe_list):
    n_a = len(oe_list)
    n_inc = len(oe_list[0])
    tcpu = np.zeros((n_a, n_inc))

    plt.gcf()
    fig, ax = plt.subplots(int(n_inc/2),2, figsize=(10,7.5))
    a = np.zeros(n_a)
    for i in range(n_a):
        a[i] = oe_list[i][0][0]
        for j in range(n_inc):
            tcpu[i, j] = tcpu_list[i][j]

    for i in range(int(n_inc/2)):
        ax[i, 0].plot(a/1e3, tcpu[:, 2*i], clip_on=False)
        ax[i, 1].plot(a/1e3, tcpu[:, 2*i+1], clip_on=False)

        # Aesthetic
        ax[i, 0].set_xlim(np.min(a) * m2km, np.max(a) * m2km)
        ax[i, 1].set_xlim(np.min(a) * m2km, np.max(a) * m2km)
        ax[i, 0].set_title('$i_0=$' + str(oe_list[i][2*i][2] * rad2deg) + ' deg.',
                           fontsize=font)
        ax[i, 1].set_title('$i_0=$' + str(oe_list[i][2*i+1][2] * rad2deg) + ' deg.',
                           fontsize=font)
        ax[i, 0].set_ylabel('CPU time [s]', fontsize=font)
        ax[i, 0].tick_params(axis='both', labelsize=font)
        ax[i, 1].tick_params(axis='both', labelsize=font)
        ax[i, 0].grid()
        ax[i, 1].grid()


def all_orbitplots(orbits, config):
    # Plot orbits
    plot_orb(orbits)
    if 'estimation' in config:
        plot_RMSE(orbits, config['oe'])

    plt.show()
