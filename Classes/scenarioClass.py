import numpy as np

from Basilisk.simulation.gravityEffector import loadPolyFromFileToList
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk import __path__

bsk_path = __path__[0]


class Parameters:
    def __init__(self, configuration):
        # Set sampling times
        self.dyn_rate = 1
        self.save_rate = 1
        self.dmcukf_rate = configuration.dmcukf_rate
        self.grav_rate = configuration.gravest_rate

        # Choose if Sun perturbation is to be considered and fsw
        self.flag_sun = True

        # Set subclasses
        self.asteroid = self.Asteroid(configuration)
        self.spacecraft = self.Spacecraft(configuration)
        self.sensors = self.Sensors(configuration)
        self.dmcukf = self.DMCUKF()
        self.grav_est = self.GravityEst(configuration, self.asteroid)

        # Set ground truth and DMC-UKF time intervals
        orb_period = 2*np.pi / np.sqrt(self.asteroid.mu / self.spacecraft.a**3)
        self.simCount = 0
        self.times_groundtruth = orb_period * configuration.orbits_groundtruth
        self.times_dmcukf = orb_period * configuration.orbits_dmcukf
        self.n_segments = len(self.times_groundtruth) - 1

    class Asteroid:
        def __init__(self, configuration):
            # Asteroid name and gravity model
            self.name = configuration.asteroid_name
            self.grav = configuration.grav_groundtruth

            # Set standard gravity parameter
            self.mu = 4.46275472004*1e5

            # Set polyhedron
            self.polyFile = bsk_path + '/supportData/LocalGravData/eros007790.tab'
            vert_list, face_list, n_vert, n_face = loadPolyFromFileToList(self.polyFile)
            self.xyz_vert = np.array(vert_list)
            self.order_face = np.array(face_list)
            self.n_vert = n_vert
            self.n_face = n_face

            # Set small body orbital elements w.r.t. Sun
            self.a = 1.4583 * 149597870.7*1e3
            self.ecc = 0.2227
            self.inc = 10.829 * np.pi/180
            self.RAAN = 304.3 * np.pi/180
            self.omega = 178.9 * np.pi/180
            self.f = 246.9 * np.pi/180

            # Set polyhedron parameters
            self.ra = 11.369 * np.pi/180
            self.dec = 17.227 * np.pi/180
            self.lst0 = 0 * np.pi/180
            self.rot_period = 5.27 * 3600
            self.dcm_N1N0 = rbk.euler3232C([self.ra, np.pi/2 - self.dec, self.lst0])

            # Set ejecta properties
            self.n_ejecta = 100
            self.rmax_ejecta = 17.648*1e3
            self.rmaxiter_ejecta = 18*1e3
            self.maxiter_ejecta = 100

    class Spacecraft:
        def __init__(self, configuration):
            # Set mass and srp properties
            self.mass = 750.0
            self.srpArea = 1.1
            self.CR = 1.2

            # Set initial orbital elements
            self.a = configuration.a0
            self.ecc = 0.001
            self.inc = configuration.i0
            self.RAAN = 48.2 * np.pi/180
            self.omega = 347.8 * np.pi/180
            self.f = 85.3 * np.pi/180

    class Sensors:
        def __init__(self, configuration):
            def landmark_distribution(type_lmk):
                # Place random seed
                np.random.seed(0)
                if type_lmk == 'poly':
                    # Set
                    poly_file = bsk_path + '/supportData/LocalGravData/EROS856Vert1708Fac.txt'
                    vert_list, face_list, n_vert, n_face = loadPolyFromFileToList(poly_file)
                    xyz_vert = np.array(vert_list)
                    order_face = np.array(face_list) - 1

                    # Randomly choose landmarks
                    idx_lmk = np.random.choice(n_vert, n_lmk, replace=False)
                    idx_lmk.sort()
                    for i in range(n_lmk):
                        idx = idx_lmk[i]
                        v1 = order_face[idx, 0]
                        v2 = order_face[idx, 1]
                        v3 = order_face[idx, 2]
                        xyz1 = xyz_vert[v1, 0:3]
                        xyz2 = xyz_vert[v2, 0:3]
                        xyz3 = xyz_vert[v3, 0:3]
                        xyz_lmk[i, 0:3] = (xyz1 + xyz2 + xyz3) / 3
                        xyz21 = xyz2 - xyz1
                        xyz31 = xyz3 - xyz1
                        normal_lmk[i, 0:3] = np.cross(xyz21, xyz31) \
                                             / np.linalg.norm(np.cross(xyz21, xyz31))
                elif type_lmk == 'ellipsoid':
                    # Set ellipsoid axes
                    axes = np.array([17.5781, 8.43072, 6.01272])*1e3

                    # Set random longitude and latitude for landmarks
                    lon = np.random.uniform(0, 2*np.pi, n_lmk)
                    lat = np.random.uniform(-np.pi/2, np.pi/2, n_lmk)
                    for i in range(n_lmk):
                        xyz_lmk[i, 0:3] = np.array([axes[0] * np.cos(lon[i]) * np.cos(lat[i]),
                                                    axes[1] * np.sin(lon[i]) * np.cos(lat[i]),
                                                    axes[2] * np.sin(lat[i])])
                        normal_lmk[i, 0:3] = (2*xyz_lmk[i, 0:3] / axes**2)\
                                             / np.linalg.norm((2*xyz_lmk[i, 0:3] / axes**2))

            # Set camera parameters
            self.nx_pixel = 2048
            self.ny_pixel = 1536
            self.w_pixel = (17.3*1e-3) / 2048
            self.f = configuration.f
            self.dcm_CB = np.eye(3)

            # Set artificial parameters to filter non-visible pixels
            self.maskangle_cam = 20 * np.pi/180
            self.maskangle_sun = configuration.maskangle_sun

            # Set number of landmarks and associated shape
            n_lmk = configuration.n_lmk
            xyz_lmk = np.zeros((n_lmk, 3))
            normal_lmk = np.zeros((n_lmk, 3))
            landmark_distribution('poly')
            self.n_lmk = n_lmk
            self.xyz_lmk = xyz_lmk
            self.normal_lmk = normal_lmk

            # Add uncertainty to landmarks
            devR_lmk = np.sqrt(configuration.dev_lmk**2 / 3)
            R_lmk = np.zeros((3, 3))
            np.fill_diagonal(R_lmk, devR_lmk**2)
            np.random.seed(0)
            self.dxyz_lmk = np.random.multivariate_normal(np.zeros(3), R_lmk, self.n_lmk)

    class DMCUKF:
        def __init__(self):
            # Set initial state uncertainty
            devPk_pos = 10
            devPk_vel = 1e-2
            devPk_acc = 1e-6

            # Set process uncertainty
            devProc_pos = 0.1
            devProc_vel = 1e-3
            devProc_acc = 2*1e-6

            # Fill initial covariance
            P_k = np.zeros((9, 9))
            np.fill_diagonal(P_k[0:3, 0:3], devPk_pos**2)
            np.fill_diagonal(P_k[3:6, 3:6], devPk_vel**2)
            np.fill_diagonal(P_k[6:9, 6:9], devPk_acc**2)
            self.P_k = P_k

            # Fill process covariance
            P_proc = np.zeros((9, 9))
            np.fill_diagonal(P_proc[0:3, 0:3], devProc_pos**2)
            np.fill_diagonal(P_proc[3:6, 3:6], devProc_vel**2)
            np.fill_diagonal(P_proc[6:9, 6:9], devProc_acc**2)
            self.P_proc = P_proc

            # Preallocate initial state
            self.dx_k = np.zeros(9)
            self.x_k = np.zeros(9)

            # Set measurements covariance
            self.P_pixel = np.array([[1**2, 0],
                                     [0, 1**2]])

            # Set hyperparameters
            self.alpha = 0
            self.beta = 2
            self.kappa = 1e-3

            # Set propagation steps
            self.n_prop = 10

    class GravityEst:
        def __init__(self, configuration, asteroid):
            # Set dataset
            self.data_type = configuration.data_type

            # Set loss type, learning rate and maximum iterations
            self.loss_type = configuration.loss_type
            self.lr = configuration.lr
            self.maxiter = configuration.maxiter

            # Set standard gravity parameter
            self.mu = asteroid.mu

            # Mascon distribution properties
            self.mascon_type = configuration.mascon_type
            self.mascon_init = configuration.mascon_init

            # Number of masses, seed and non-dimensional parameters
            self.n_M = configuration.n_M
            self.seed_M = configuration.seed_M
            self.muM_ad = self.mu / (self.n_M+1)
            self.posM_ad = np.array([16342.6, 8410.61, 5973.615]) / 10

            # Set polyhedron properties for interior constraint
            self.polyFile = bsk_path + '/supportData/LocalGravData/EROS856Vert1708Fac.txt'
            vert_list, face_list, n_vert, n_face = loadPolyFromFileToList(self.polyFile)
            self.xyz_vert = np.array(vert_list)
            self.order_face = np.array(face_list)
            self.n_vert = n_vert
            self.n_face = n_face
            self.xyz_face = np.zeros((n_face, 3))
            for i in range(n_face):
                idx = self.order_face[i, 0:3]
                self.xyz_face[i, 0:3] = (self.xyz_vert[idx[0]-1, 0:3] + self.xyz_vert[idx[1]-1, 0:3]
                                         + self.xyz_vert[idx[2]-1, 0:3])/3

            # Preallocate initial and final distributions
            self.pos0_M = []
            self.mu0_M = []
            self.posM = []
            self.muM = []

            # Set ejecta configuration
            self.flag_ejecta = configuration.flag_ejecta
            self.n_ejecta = configuration.n_ejecta
            self.dev_ejecta = configuration.dev_ejecta


class Outputs:
    def __init__(self):
        # Output classes
        self.groundtruth = self.GroundTruth()
        self.camera = self.Camera()
        self.results = self.Results()

    class GroundTruth:
        def __init__(self):
            # Preallocate simulation time
            self.t = []

            # Preallocate spacecraft truth position and velocity
            self.pos_BP_N0 = []
            self.vel_BP_N0 = []
            self.pos_BP_N1 = []
            self.vel_BP_N1 = []
            self.pos_BP_P = []
            self.vel_BP_P = []

            # Preallocate spacecraft orientation in asteroid frame
            self.mrp_BP = []

            # Preallocate spacecraft truth radius and altitude
            self.r_BP = []
            self.h_BP = []

            # Preallocate truth gravity acceleration
            self.acc_BP_N0 = []
            self.acc_BP_P = []
            self.accHigh_BP_P = []

            # Preallocate ejecta truth data
            self.pos_EP_P = []
            self.acc_EP_P = []
            self.r_EP = []
            self.h_EP = []

            # Preallocate asteroid position, orientation and Sun's direction
            self.pos_PS_N1 = []
            self.e_SP_P = []
            self.mrp_PN0 = []

            # Define maximum radius to be evaluated
            self.rmax = 50*1e3

            # Preallocate 2D gravity map parameters
            self.n_2D = 100
            self.Xy_2D = []
            self.Yx_2D = []
            self.Xz_2D = []
            self.Zx_2D = []
            self.Yz_2D = []
            self.Zy_2D = []
            self.aXY_2D = []
            self.aXZ_2D = []
            self.aYZ_2D = []
            self.extXY_2D = []
            self.extXZ_2D = []
            self.extYZ_2D = []
            self.a0XY_2D = []
            self.a0XZ_2D = []
            self.a0YZ_2D = []
            self.a0ErrXY_2D = []
            self.a0ErrXZ_2D = []
            self.a0ErrYZ_2D = []

            # Preallocate 3D gravity map parameters
            self.nr_3D = 40
            self.nlat_3D = 40
            self.nlon_3D = 40
            self.X_3D = []
            self.Y_3D = []
            self.Z_3D = []
            self.aXYZ_3D = []
            self.extXYZ_3D = []
            self.rXYZ_3D = []
            self.hXYZ_3D = []
            self.a0XYZ_3D = []
            self.a0ErrXYZ_3D = []
            self.a0ErrTotal_3D = []

            # Gravity accuracy results w.r.t. radius and altitude
            self.n_bins = 40
            self.r_bins = []
            self.a0Err_binsRad = []
            self.N_rad = []
            self.h_bins = []
            self.a0Err_binsAlt = []
            self.N_alt = []

    class Camera:
        def __init__(self):
            # Time
            self.t = []

            # Number of visible landmarks, status and pixel
            self.nvisible_lmk = []
            self.isvisible_lmk = []
            self.pixel_lmk = []

            # Flag telling if there are available measurements
            self.flag_meas = []

            # Camera mrp
            self.mrp_CP = []

    class Results:
        def __init__(self):
            # DMC-UKF position, velocity and acceleration
            self.pos_BP_P = []
            self.pos_BP_N1 = []
            self.vel_BP_P = []
            self.acc_BP_P = []

            # DMC-UKF uncertainty covariances
            self.Pxx = []
            self.Ppos_P = []
            self.Pacc_P = []

            # DMC-UKF measurement residual
            self.resZ = []

            # Trained mascon distributions
            self.pos_M = []
            self.mu_M = []

            # Preallocate loss function
            self.loss = []

            # Preallocate gravity estimation data
            self.t_data = []
            self.pos_data = []
            self.acc_data = []

            # Preallocate polyhedron acceleration
            # evaluated at data
            self.accNK_data = []
            self.accNK_poly = []

            # Preallocate execution times
            self.tcpu_dmcukf = []
            self.tcpu_gravest = []

            # Preallocate 2D gravity map result
            self.aXY_2D = []
            self.aXZ_2D = []
            self.aYZ_2D = []
            self.aErrXY_2D = []
            self.aErrXZ_2D = []
            self.aErrYZ_2D = []

            # Preallocate 3D gravity map result
            self.aXYZ_3D = []
            self.aErrXYZ_3D = []
            self.aErrTotal_3D = []

            # Gravity accuracy results w.r.t. radius and altitude
            self.aErr_binsRad = []
            self.aErr_binsAlt = []
