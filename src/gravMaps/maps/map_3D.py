import numpy as np

from timeit import default_timer as timer


# This class is a 3D map for gravity
class Map3D:
    def __init__(self, rmax, nr, nlat, nlon):
        # Preallocate 3D gravity
        # map parameters
        self.rmax = rmax
        self.nr, self.nlat, self.nlon = nr, nlat, nlon
        self.X, self.Y, self.Z = [], [], []
        self.r, self.h = [], []
        self.ext_XYZ = []
        self.acc_XYZ = []
        self.aErrXYZ = []
        self.t_cpu = []

    # This creates 3D grid
    def create_grid(self, shape):
        # Retrieve maximum radius and
        # number of spherical points
        rmax = self.rmax
        nr, nlat, nlon = self.nr, self.nlat, self.nlon

        # Create 3D exterior meshgrids
        r = np.linspace(rmax / nr, rmax, nr)
        lat = np.linspace(-np.pi / 2 * (1 - 2 / nlat),
                          np.pi / 2 * (1 - 2 / nlat),
                          nlat)
        lon = np.linspace(0,
                          2 * np.pi * (1 - 1 / nlon),
                          nlon)
        self.X = np.zeros((nr, nlat, nlon))
        self.Y = np.zeros((nr, nlat, nlon))
        self.Z = np.zeros((nr, nlat, nlon))
        self.r = np.zeros((nr, nlat, nlon))
        self.h = np.zeros((nr, nlat, nlon))

        # Initialize seed for rng
        np.random.seed(0)

        # Initialize first radius range
        r0, rf = 0, r[0]

        # Loop through spherical grid points
        for i in range(nr):
            for j in range(nlat):
                for k in range(nlon):
                    # Sample random radius within range
                    r_jk = np.random.uniform(r0, rf)

                    # Fill xyz spherical grid
                    self.X[i, j, k] = \
                        r_jk * np.cos(lat[j]) * np.cos(lon[k])
                    self.Y[i, j, k] = \
                        r_jk * np.cos(lat[j]) * np.sin(lon[k])
                    self.Z[i, j, k] = \
                        r_jk * np.sin(lat[j])

                    # Fill radius and altitude
                    self.r[i, j, k] = r_jk
                    self.h[i, j, k] = shape.compute_altitude(
                        [self.X[i, j, k],
                         self.Y[i, j, k],
                         self.Z[i, j, k]])

            # Switch to next radial interval
            if i < nr - 1:
                r0 = r[i]
                rf = r[i + 1]

        # Check interior shape points
        self.ext_XYZ = np.zeros((nr, nlat, nlon), dtype='bool')
        for i in range(nr):
            for j in range(nlat):
                for k in range(nlon):
                    self.ext_XYZ[i, j, k] = shape.check_exterior(
                        [self.X[i, j, k],
                         self.Y[i, j, k],
                         self.Z[i, j, k]])

        # Set nan
        self.r[np.invert(self.ext_XYZ)] = np.nan
        self.h[np.invert(self.ext_XYZ)] = np.nan

    # This generates 3D map
    def generate_acc(self, grav_model):
        # Preallocate gravity 3D map
        nr = self.nr
        nlat = self.nlat
        nlon = self.nlon
        self.acc_XYZ = np.zeros((nr, nlat, nlon, 3))
        self.t_cpu = np.zeros((nr, nlat, nlon))

        # Fill gravity 3D map
        for i in range(nr):
            for j in range(nlat):
                for k in range(nlon):
                    # Compute gravity
                    if self.ext_XYZ[i, j, k]:
                        pos = [self.X[i, j, k],
                               self.Y[i, j, k],
                               self.Z[i, j, k]]

                        # Start measuring cpu time
                        t_start = timer()

                        # Evaluate gravity
                        self.acc_XYZ[i, j, k, :] = \
                            grav_model.compute_gravity(pos)

                        # End measuring cpu time
                        t_end = timer()
                        self.t_cpu[i, j, k] = t_end - t_start
                    else:
                        self.acc_XYZ[i, j, k, :] = np.nan * np.ones(3)
                        self.t_cpu[i, j, k] = np.nan

    # This computes 3D gravity potential
    def generate_U(self, grav_model):
        # Preallocate gravity 3D map
        nr = self.nr
        nlat = self.nlat
        nlon = self.nlon
        self.U_XYZ = np.zeros((nr, nlat, nlon))

        # Fill gravity 3D map
        for i in range(nr):
            for j in range(nlat):
                for k in range(nlon):
                    # Compute gravity
                    if self.ext_XYZ[i, j, k]:
                        pos = [self.X[i, j, k],
                               self.Y[i, j, k],
                               self.Z[i, j, k]]

                        # Evaluate gravity
                        self.U_XYZ[i, j, k] = \
                            grav_model.compute_potential(pos)
                    else:
                        self.U_XYZ[i, j, k] = np.nan

    # This computes 3D error map
    def compute_errors(self, refmap_3D):
        # Compute 3D gravity map error
        self.aErrXYZ = \
            np.linalg.norm(self.acc_XYZ - refmap_3D.acc_XYZ, axis=3) \
            / np.linalg.norm(refmap_3D.acc_XYZ, axis=3)

    # This function imports 3D grid
    def import_grid(self, refmap_3D):
        # Copy 3D map
        self.rmax = refmap_3D.rmax
        self.nr, self.nlat, self.nlon = \
            refmap_3D.nr, refmap_3D.nlat, refmap_3D.nlon
        self.X, self.Y, self.Z = \
            refmap_3D.X, refmap_3D.Y, refmap_3D.Z
        self.ext_XYZ = refmap_3D.ext_XYZ
        self.r, self.h = refmap_3D.r, refmap_3D.h
