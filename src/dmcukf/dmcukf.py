import numpy as np

from Basilisk.ExternalModules import dmcukf
from Basilisk.utilities import RigidBodyKinematics as rbk


# DMC-UKF attributes
class DMCUKF:
    def __init__(self):
        # Set sampling rate
        self.dmcukf_rate = []

        # Set initial state uncertainty
        devPk_pos = 10
        devPk_vel = 1e-2
        devPk_acc = 1e-6

        # Set process uncertainty
        devProc_pos = 0.1
        devProc_vel = 1e-3
        devProc_acc = 2 * 1e-6

        # Fill initial covariance
        P_k = np.zeros((9, 9))
        np.fill_diagonal(P_k[0:3, 0:3], devPk_pos ** 2)
        np.fill_diagonal(P_k[3:6, 3:6], devPk_vel ** 2)
        np.fill_diagonal(P_k[6:9, 6:9], devPk_acc ** 2)
        self.P_k = P_k

        # Fill process covariance
        P_proc = np.zeros((9, 9))
        np.fill_diagonal(P_proc[0:3, 0:3], devProc_pos ** 2)
        np.fill_diagonal(P_proc[3:6, 3:6], devProc_vel ** 2)
        np.fill_diagonal(P_proc[6:9, 6:9], devProc_acc ** 2)
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

        # Create data instance
        self.data = self.Data()

    class Data:
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
            self.xyz_M = []
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

    # This function creates dmc-ukf object
    def create_dmcukf(self, asteroid, spacecraft, measurements):
        # Create instance of BSK dmcukf
        self.dmcukf_bsk = dmcukf.DMCUKF()

        # Set SRP and Sun's 3rd body if required
        # Set spacecraft physical parameters
        self.dmcukf_bsk.useSRP = True
        self.dmcukf_bsk.useSun = True
        self.dmcukf_bsk.m = spacecraft.mass
        self.dmcukf_bsk.ASRP = spacecraft.srp_area
        self.dmcukf_bsk.CR = spacecraft.CR

        # Set dmc-ukf state, uncertainty, process and measurement covariances
        self.x_k = np.concatenate((spacecraft.data.pos_BP_N1[0, 0:3],
                                  spacecraft.data.vel_BP_N1[0, 0:3],
                                  np.zeros(3)))
        self.dmcukf_bsk.x = self.x_k + self.dx_k
        self.dmcukf_bsk.Pxx = self.P_k.tolist()
        self.dmcukf_bsk.Pproc = self.P_proc.tolist()
        self.dmcukf_bsk.Ppixel = self.P_pixel.tolist()

        # Set integration steps
        self.dmcukf_bsk.Nint = self.n_prop

        # Set landmark and camera information
        camera = measurements.camera
        self.dmcukf_bsk.nLmk = camera.n_lmk
        self.dmcukf_bsk.rLmk = camera.xyz_lmk + camera.dxyz_lmk
        self.dmcukf_bsk.f = camera.f
        self.dmcukf_bsk.wPixel = camera.w_pixel
        self.dmcukf_bsk.dcm_CB = camera.dcm_CB

        # Set dcm of planet equatorial w.r.t. inertial and
        # camera w.r.t. body frame
        self.dmcukf_bsk.dcm_N1N0 = asteroid.dcm_N1N0

        # Set the use of a mascon gravity model
        gravityModel = self.dmcukf_bsk
        gravityModel.useMasconGravityModel()
        gravityModel.mascon.muMascon = [0,0,asteroid.mu]
        gravityModel.mascon.xyzMascon = [[1,0,0],[4,0,0], [0,0,0]]

        # Reset
        self.dmcukf_bsk.Reset(0)

    # This method adds mascon model
    def add_mascon(self, mu_M, xyz_M):
        # Set mascon model
        self.dmcukf_bsk.setMascon(mu_M.tolist(),
                                  xyz_M.tolist())

    # This method loads ephemeris
    def load_ephemeris(self, pos_PS_N1, mrp_PN0, mrp_BP):
        # Set asteroid position and orientation
        self.dmcukf_bsk.batch.r_PS_N1 = pos_PS_N1
        self.dmcukf_bsk.batch.mrp_PN0 = mrp_PN0

        # Set spacecraft orientation w.r.t. asteroid
        self.dmcukf_bsk.batch.mrp_BP = mrp_BP

    # This function loads measurements into bsk dmc-ukf module
    def load_camera_meas(self, t_meas, isvisible_lmk, pixel_lmk):
        # Load measurements into module
        self.dmcukf_bsk.batch.t = t_meas
        self.dmcukf_bsk.batch.isvisibleLmk = isvisible_lmk
        self.dmcukf_bsk.batch.pLmk = pixel_lmk

    # This method runs filter forward
    def run_forward(self):
        self.dmcukf_bsk.processBatch()
        self.save_dmcukf()

    # This function saves dmc-ukf results
    def save_dmcukf(self):
        # Obtain filter state, covariance,
        # training data and residuals
        X = np.array(self.dmcukf_bsk.batch.x)
        Pxx_array = np.array(self.dmcukf_bsk.batch.Pxx)

        # Get small body orientation variables
        dcm_N1N0 = np.array(self.dmcukf_bsk.dcm_N1N0)
        mrp_PN0 = np.array(self.dmcukf_bsk.batch.mrp_PN0)

        # Set estimated variables
        n = len(X)
        pos_BP_N1 = X[:, 0:3]
        vel_BP_N1 = X[:, 3:6]
        acc_BP_N1 = X[:, 6:9]
        Pxx = Pxx_array.reshape((n, 9, 9))

        # Initialize variables in small body frame
        pos_data = np.zeros((n, 3))
        acc_data = np.zeros((n, 3))
        Ppos_P = np.zeros((n, 3, 3))
        Pacc_P = np.zeros((n, 3, 3))

        # Fill variables in small body frame
        for k in range(n):
            # Obtain spacecraft to planet dcm
            dcm_PN0 = rbk.MRP2C(mrp_PN0[k, 0:3])
            dcm_PN1 = np.matmul(dcm_PN0, dcm_N1N0.T)

            # Compute data position and acceleration
            pos_data[k, 0:3] = dcm_PN1.dot(pos_BP_N1[k, 0:3])
            acc_data[k, 0:3] = dcm_PN1.dot(acc_BP_N1[k, 0:3]) \
                               + np.array(self.dmcukf_bsk.mascon.computeField(pos_data[k, 0:3])).T

            # Compute uncertainty in small body rotating frame
            Ppos_P[k, 0:3, 0:3] = np.matmul(dcm_PN1, np.matmul(Pxx[k, 0:3, 0:3], dcm_PN1.T))
            Pacc_P[k, 0:3, 0:3] = np.matmul(dcm_PN1, np.matmul(Pxx[k, 6:9, 6:9], dcm_PN1.T))

        try:
            # Fill position, velocity and unmodeled acceleration
            self.data.pos_BP_N1 = np.concatenate((self.data.pos_BP_N1,
                                                  pos_BP_N1[1:n, :]))
            self.data.vel_BP_N1 = np.concatenate((self.data.vel_BP_N1,
                                                  vel_BP_N1[1:n, :]))
            self.data.acc_BP_N1 = np.concatenate((self.data.acc_BP_N1,
                                                  acc_BP_N1[1:n, :]))

            # Fill covariances
            self.data.Pxx = np.concatenate((self.data.Pxx, Pxx[1:n, :, :]))
            self.data.Ppos_P = np.concatenate((self.data.Ppos_P, Ppos_P[1:n, :, :]))
            self.data.Pacc_P = np.concatenate((self.data.Pacc_P, Pacc_P[1:n, :, :]))
        # Save variables in outputs
        except:
            # Initialize position, velocity and unmodeled acceleration
            self.data.pos_BP_N1 = pos_BP_N1
            self.data.vel_BP_N1 = vel_BP_N1
            self.data.acc_BP_N1 = acc_BP_N1

            # Initialize covariances
            self.data.Pxx = Pxx
            self.data.Ppos_P = Ppos_P
            self.data.Pacc_P = Pacc_P
