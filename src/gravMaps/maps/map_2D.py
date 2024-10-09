import numpy as np


# This class is a 2D map for gravity
class Map2D:
    def __init__(self, rmax, n):
        # Maximum coordinate and number
        # of samples per dimension
        self.rmax = rmax
        self.n = n

        # 2D map grid
        self.Xy, self.Yx = [], []
        self.Xz, self.Zx = [], []
        self.Yz, self.Zy = [], []

        # Exterior boolean
        self.ext_XY, self.ext_XZ, self.ext_YZ = \
            [], [], []

        # Acceleration and error maps
        self.acc_XY, self.acc_XZ, self.acc_YZ = \
            [], [], []
        self.accerr_XY, self.accerr_XZ, self.accerr_YZ = \
            [], [], []

        # Potential and error maps
        self.U_XY, self.U_XZ, self.U_YZ = \
            [], [], []
        self.Uerr_XY, self.Uerr_XZ, self.Uerr_YZ = \
            [], [], []

    # This method creates 2D grid
    def create_grid(self, shape):
        # Define maximum dimension
        # and number of points
        rmax = self.rmax
        n = self.n

        # Create 2D mesh grids
        x = np.linspace(-rmax, rmax, n)
        y = np.linspace(-rmax, rmax, n)
        z = np.linspace(-rmax, rmax, n)
        self.Xy, self.Yx = np.meshgrid(x, y)
        self.Xz, self.Zx = np.meshgrid(x, z)
        self.Yz, self.Zy = np.meshgrid(y, z)

        # Preallocate exterior 2D mesh grids
        self.ext_XY = np.zeros((n, n), dtype='bool')
        self.ext_XZ = np.zeros((n, n), dtype='bool')
        self.ext_YZ = np.zeros((n, n), dtype='bool')

        # Loop all points
        for i in range(n):
            for j in range(n):
                # Check exterior condition for xy plane
                self.ext_XY[i, j] = shape.check_exterior(
                    [self.Xy[i, j], self.Yx[i, j], 0])

                # Check exterior condition for xz plane
                self.ext_XZ[i, j] = shape.check_exterior(
                    [self.Xz[i, j], 0, self.Zx[i, j]])

                # Check exterior condition for yz plane
                self.ext_YZ[i, j] = shape.check_exterior(
                    [0, self.Yz[i, j], self.Zy[i, j]])

    # This method computes acceleration
    def generate_acc(self, grav_model):
        # Preallocate gravity 2D map
        n = self.n
        self.acc_XY = np.zeros((n, n, 3))
        self.acc_XZ = np.zeros((n, n, 3))
        self.acc_YZ = np.zeros((n, n, 3))

        # Loop through 2D map points
        for i in range(n):
            for j in range(n):
                # Retrieve 2D map exterior flags
                ext_xy = self.ext_XY[i, j]
                ext_xz = self.ext_XZ[i, j]
                ext_yz = self.ext_YZ[i, j]

                # Compute gravity for xy plane
                if ext_xy:
                    xy = [self.Xy[i, j],
                          self.Yx[i, j],
                          0]
                    acc_xy = grav_model.compute_gravity(xy)
                else:
                    acc_xy = np.nan * np.ones(3)

                # Compute gravity for xz plane
                if ext_xz:
                    xz = [self.Xz[i, j],
                          0,
                          self.Zx[i, j]]
                    acc_xz = grav_model.compute_gravity(xz)
                else:
                    acc_xz = np.nan * np.ones(3)

                # Compute gravity for yz plane
                if ext_yz:
                    yz = [0,
                          self.Yz[i, j],
                          self.Zy[i, j]]
                    acc_yz = grav_model.compute_gravity(yz)
                else:
                    acc_yz = np.nan * np.ones(3)

                # Save gravity
                self.acc_XY[i, j, :] = \
                    np.array(acc_xy).reshape(3)
                self.acc_XZ[i, j, :] = \
                    np.array(acc_xz).reshape(3)
                self.acc_YZ[i, j, :] = \
                    np.array(acc_yz).reshape(3)

    # This method computes potential
    def generate_U(self, grav_model):
        # Preallocate gravity 2D map
        n = self.n
        self.U_XY = np.zeros((n, n))
        self.U_XZ = np.zeros((n, n))
        self.U_YZ = np.zeros((n, n))

        # Loop through 2D map points
        for i in range(n):
            for j in range(n):
                # Retrieve 2D map exterior flags
                ext_xy = self.ext_XY[i, j]
                ext_xz = self.ext_XZ[i, j]
                ext_yz = self.ext_YZ[i, j]

                # Compute gravity for xy plane
                if ext_xy:
                    xy = [self.Xy[i, j],
                          self.Yx[i, j],
                          0]
                    U_xy = grav_model.compute_potential(xy)
                else:
                    U_xy = np.nan

                # Compute gravity for xz plane
                if ext_xz:
                    xz = [self.Xz[i, j],
                          0,
                          self.Zx[i, j]]
                    U_xz = grav_model.compute_potential(xz)
                else:
                    U_xz = np.nan

                # Compute gravity for yz plane
                if ext_yz:
                    yz = [0,
                          self.Yz[i, j],
                          self.Zy[i, j]]
                    U_yz = grav_model.compute_potential(yz)
                else:
                    U_yz = np.nan

                # Save gravity
                self.U_XY[i, j] = U_xy
                self.U_XZ[i, j] = U_xz
                self.U_YZ[i, j] = U_yz

    # This method computes 2D acceleration error map
    def compute_accerr(self, refmap_2D):
        # Compute xy plane errors
        accerr_XY = self.acc_XY - refmap_2D.acc_XY
        self.accerr_XY = np.linalg.norm(accerr_XY, axis=2) \
                         / np.linalg.norm(refmap_2D.acc_XY, axis=2)

        # Compute xz plane errors
        accerr_XZ = self.acc_XZ - refmap_2D.acc_XZ
        self.accerr_XZ = np.linalg.norm(accerr_XZ, axis=2) \
                      / np.linalg.norm(refmap_2D.acc_XZ, axis=2)

        # Compute yz plane errors
        accerr_YZ = self.acc_YZ - refmap_2D.acc_YZ
        self.accerr_YZ = np.linalg.norm(accerr_YZ, axis=2) \
                         / np.linalg.norm(refmap_2D.acc_YZ, axis=2)

    # This method computes 2D potential error map
    def compute_Uerr(self, refmap_2D):
        # Compute xy plane errors
        Uerr_XY = self.U_XY - refmap_2D.U_XY
        self.Uerr_XY = np.linalg.norm(Uerr_XY, axis=2) \
                       / np.linalg.norm(refmap_2D.U_XY, axis=2)

        # Compute xz plane errors
        Uerr_XZ = self.U_XZ - refmap_2D.U_XZ
        self.Uerr_XZ = np.linalg.norm(Uerr_XZ, axis=2) \
                       / np.linalg.norm(refmap_2D.U_XZ, axis=2)

        # Compute yz plane errors
        Uerr_YZ = self.U_YZ - refmap_2D.U_YZ
        self.Uerr_YZ = np.linalg.norm(Uerr_YZ, axis=2) \
                      / np.linalg.norm(refmap_2D.U_YZ, axis=2)

    # This method imports 2D map
    def import_grid(self, refmap_2D):
        # Copy 2D map
        self.rmax = refmap_2D.rmax
        self.n = refmap_2D.n
        self.Xy, self.Yx = refmap_2D.Xy, refmap_2D.Yx
        self.Xz, self.Zx = refmap_2D.Xz, refmap_2D.Zx
        self.Yz, self.Zy = refmap_2D.Yz, refmap_2D.Zy
        self.ext_XY, self.ext_XZ, self.ext_YZ =\
            refmap_2D.ext_XY, refmap_2D.ext_XZ, refmap_2D.ext_YZ
