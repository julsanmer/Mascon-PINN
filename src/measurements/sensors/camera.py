import numpy as np

from Basilisk.simulation import pinholeCamera
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk import __path__

bsk_path = __path__[0]

# Conversion constants
deg2rad = np.pi/180
km2m = 1e3


# Measurements class
class Camera:
    def __init__(self):
        # Set sampling rate
        self.meas_rate = 1

        # Height and width number of pixels,
        # pixel length, focal length and
        # camera dcm w.r.t. body
        self.nx_pixel, self.ny_pixel = [], []
        self.w_pixel = []
        self.f = []
        self.dcm_CB = np.eye(3)

        # Camera and Sun's mask angles
        self.maskangle_cam = []
        self.maskangle_sun = []

        # Landmark distribution: number, normals,
        # positions and error
        self.n_lmk = []
        self.normal_lmk, self.xyz_lmk = [], []
        self.dev_lmk = []
        self.dxyz_lmk = []

        # Create instance of data subclass
        self.data = self.Data()

        # Preallocate BSK camera object
        self.camera_bsk = None

    # This subclass stores camera data
    class Data:
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

    # This method initializes BSK camera
    def initialize(self):
        # Initialize BSK camera module
        self.camera_bsk = pinholeCamera.PinholeCamera()

        # Set camera properties
        self.camera_bsk.nxPixel = self.nx_pixel
        self.camera_bsk.nyPixel = self.ny_pixel
        self.camera_bsk.wPixel = self.w_pixel
        self.camera_bsk.f = self.f

        # Add landmark information
        for j in range(self.n_lmk):
            self.camera_bsk.addLandmark(self.xyz_lmk[j, 0:3],
                                        self.normal_lmk[j, 0:3])

        # Set Sun mask angle and camera dcm
        #self.camera_bsk.maskCam = self.maskangle_cam
        self.camera_bsk.maskSun = self.maskangle_sun
        self.camera_bsk.dcm_CB = self.dcm_CB.tolist()
        self.camera_bsk.Reset(0)

    # This method generates camera measurements
    def generate_meas(self, asteroid, spacecraft):
        # Get spacecraft position and orientation
        # Get small body to Sun unit-vector
        pos_BP_P = spacecraft.data.pos_BP_P
        mrp_BP = spacecraft.data.mrp_BP
        e_SP_P = asteroid.data.e_SP_P

        # Generate camera measurements
        self.camera_bsk.processBatch(pos_BP_P.tolist(),
                                     mrp_BP.tolist(),
                                     e_SP_P.tolist(),
                                     show_progress=True)

        # Save camera times and get number of samples
        t = spacecraft.data.t
        n = len(t)

        # Process outputs
        isvisibleBatchLmk = np.array(self.camera_bsk.isvisibleBatchLmk)
        nvisibleBatchLmk = np.sum(isvisibleBatchLmk, axis=1)
        pixelBatchLmk = np.array(self.camera_bsk.pixelBatchLmk)
        mrp_CP = np.zeros((n, 3))
        for i in range(n):
            dcm_BP = rbk.MRP2C(mrp_BP[i, 0:3])
            dcm_CP = self.dcm_CB @ dcm_BP
            mrp_CP[i, 0:3] = rbk.C2MRP(dcm_CP)

        # Save outputs
        self.data.t = t
        self.data.pixel_lmk = np.array(pixelBatchLmk).astype(int)
        self.data.nvisible_lmk = np.array(nvisibleBatchLmk).astype(int)
        self.data.isvisible_lmk = np.array(isvisibleBatchLmk).astype(int)
        self.data.flag_meas = self.data.nvisible_lmk > 0
        self.data.mrp_CP = np.array(mrp_CP)

    # This method deletes swigpy objects
    def delete_swigpy(self):
        self.camera_bsk = None

    # This method creates landmark distribution using asteroid shape
    def create_landmark_distribution(self, asteroid, n_lmk=100, dev_lmk=5, seed=0):
        # Place random seed
        np.random.seed(seed)

        # Set number of landmarks and error
        self.n_lmk = n_lmk
        self.dev_lmk = dev_lmk

        # Retrieve asteroid shape
        shape = asteroid.shape

        # Randomly choose landmarks
        idx_lmk = np.random.choice(shape.n_face, self.n_lmk, replace=False)
        idx_lmk.sort()
        self.normal_lmk = shape.normal_face[idx_lmk, 0:3]
        self.xyz_lmk = shape.xyz_face[idx_lmk, 0:3]

        # Apply errors to landmark positions
        devR_lmk = np.sqrt(self.dev_lmk**2 / 3)
        R_lmk = np.zeros((3, 3))
        np.fill_diagonal(R_lmk, devR_lmk**2)
        self.dxyz_lmk = np.random.multivariate_normal(np.zeros(3), R_lmk, self.n_lmk)

        # elif type_lmk == 'ellipsoid':
        #     # Set ellipsoid axes
        #     axes = np.array([17.5781, 8.43072, 6.01272]) * km2m
        #
        #     # Set random longitude and latitude for landmarks
        #     lon = np.random.uniform(0, 2 * np.pi, n_lmk)
        #     lat = np.random.uniform(-np.pi / 2, np.pi / 2, n_lmk)
        #     for i in range(n_lmk):
        #         xyz_lmk[i, 0:3] = np.array([axes[0] * np.cos(lon[i]) * np.cos(lat[i]),
        #                                     axes[1] * np.sin(lon[i]) * np.cos(lat[i]),
        #                                     axes[2] * np.sin(lat[i])])
        #         normal_lmk[i, 0:3] = (2 * xyz_lmk[i, 0:3] / axes ** 2) \
        #                              / np.linalg.norm((2 * xyz_lmk[i, 0:3] / axes ** 2))

