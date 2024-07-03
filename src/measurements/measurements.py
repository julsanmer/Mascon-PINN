import numpy as np
import pickle as pck

from src.measurements.sensors.camera import Camera

from Basilisk.simulation import pinholeCamera
from Basilisk.simulation.gravityEffector import loadPolyFromFileToList
from Basilisk import __path__

bsk_path = __path__[0]

# Conversion constants
deg2rad = np.pi/180
km2m = 1e3


# Measurements class
class Measurements:
    def __init__(self):
        # Set file
        self.file = []

        # Set sampling rate
        self.meas_rate = 1

        # Preallocate camera
        self.camera = None

    # This method initializes camera parameters
    def add_camera(self, nx_pixel=2048, ny_pixel=1536,
                   f=25*1e-3, w_pixel=(17.3*1e-3)/2048,
                   maskangle_cam=0., maskangle_sun=0.,
                   n_lmk=100):
        # Create camera instance
        self.camera = Camera()

        # Set camera attributes
        self.camera.nx_pixel = nx_pixel
        self.camera.ny_pixel = ny_pixel
        self.camera.f = f
        self.camera.w_pixel = w_pixel
        self.camera.maskangle_cam = maskangle_cam
        self.camera.maskangle_sun = maskangle_sun
        self.camera.n_lmk = n_lmk

    # This method creates data
    def generate_data(self, asteroid, spacecraft):
        self.camera.generate_meas(asteroid,
                                  spacecraft)

    # This method imports data
    def import_data(self, dt):
        # Import groundtruth data from file
        inputs = pck.load(open(self.file, "rb"))

        # Load camera
        self.camera = inputs.measurements.camera

        # Define camera copy
        data_in = inputs.measurements

        # Determine indexes to prune data
        t0 = self.camera.data.t[0]
        tf = self.camera.data.t[-1]
        idx = np.linspace(0, np.floor((tf - t0) / dt) * dt,
                          int(np.floor((tf - t0) / dt)) + 1).astype(int)

        # Copy camera variables
        self.camera.data.t = data_in.camera.data.t[idx]
        self.camera.data.pixel_lmk = data_in.camera.data.pixel_lmk[idx, :]
        self.camera.data.nvisible_lmk = data_in.camera.data.nvisible_lmk[idx]
        self.camera.data.isvisible_lmk = data_in.camera.data.isvisible_lmk[idx, :]
        self.camera.data.flag_meas = data_in.camera.data.flag_meas[idx]
        self.camera.data.mrp_CP = data_in.camera.data.mrp_CP[idx, :]

