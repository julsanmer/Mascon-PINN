import numpy as np


# This class defines a surface map
class MapSurface2D:
    def __init__(self, nlat, nlon, mode='ellipsoid'):
        # Set grid mode
        self.mode = mode

        # Preallocate latitude-longitude
        lat = np.linspace(-np.pi/2, np.pi/2, nlat)
        lon = np.linspace(-np.pi, np.pi, nlon)
        self.nlat, self.nlon = nlat, nlon
        self.lat, self.lon = np.meshgrid(lat, lon)

        # b/a and b/a factors
        self.ba, self.ca = 1, 1

        # Preallocate evaluation and acceleration
        self.h_surf = []
        self.xyz_surf = []
        self.acc_surf = []
        self.aErr_surf = []

    # This method creates surface grid
    def create_grid(self, shape):
        def _find_surface(x0, dx0, tol=1e-12):
            # Initialize exterior flags
            is_exterior0 = shape.check_exterior(x0 * line)
            is_exterior = not is_exterior0

            # Initialize unknown variable
            x = x0
            dx = dx0

            # While tolerance is not meet and point is interior
            while dx > tol or not is_exterior:
                # Check current point exterior status
                is_exterior = shape.check_exterior(x * line)

                # Subtract (exterior) or add (interior)
                if is_exterior:
                    x -= dx
                else:
                    x += dx

                # If an interior-exterior cross is detected,
                # decrease step
                if is_exterior0 != is_exterior:
                    dx *= 0.5

                # Reset exterior flag
                is_exterior0 = is_exterior

            return x

        # If ellipsoid mode is chosen
        if self.mode == 'ellipsoid':
            # Ellipsoid axes
            a = shape.axes[0]
            b = shape.axes[1]
            c = shape.axes[2]

            # This is b/a and c/a
            self.ba = b / a
            self.ca = c / a

        # Preallocate variables to fill
        self.h_surf = np.zeros((self.nlon, self.nlat))
        self.xyz_surf = np.zeros((self.nlon, self.nlat, 3))

        # Set initial guess and step
        r0 = np.max(np.linalg.norm(shape.xyz_vert))
        dr0 = 0.05 * r0

        # Loop through longitude
        for i in range(self.nlon):
            # Loop through latitude
            for j in range(self.nlat):
                # Retrieve longitude and latitude
                lon_ij = self.lon[i, j]
                lat_ij = self.lat[i, j]

                # Compute direction line
                line = np.array([np.cos(lon_ij) * np.cos(lat_ij),
                                 self.ba * np.sin(lon_ij) * np.cos(lat_ij),
                                 self.ca * np.sin(lat_ij)])

                # Find radius
                r = _find_surface(r0, dr0, tol=1e-9)

                # Save surface point and altitude
                xyz_ij = r * line
                self.xyz_surf[i, j, 0:3] = xyz_ij
                self.h_surf[i, j] = shape.compute_altitude(xyz_ij)

    # This method computes surface acceleration
    def generate_acc(self, grav_model):
        # Preallocate acceleration
        self.acc_surf = np.zeros((self.nlon, self.nlat, 3))

        # Loop through surface
        for i in range(self.nlon):
            for j in range(self.nlat):
                self.acc_surf[i, j, 0:3] =\
                    grav_model.compute_gravity(self.xyz_surf[i, j, 0:3])

    # This computes surface error map
    def compute_errors(self, refmap_surf):
        # Preallocate errors
        self.aErr_surf = np.zeros((self.nlon, self.nlat))

        # Loop through surface
        for i in range(self.nlon):
            for j in range(self.nlat):
                self.aErr_surf[i, j] = \
                    np.linalg.norm(self.acc_surf[i, j, 0:3] - refmap_surf.acc_surf[i, j, 0:3]) \
                    / np.linalg.norm(refmap_surf.acc_surf[i, j, 0:3])

    # This method imports surface map
    def import_grid(self, refmap_surf):
        # Copy surface map evaluation
        self.h_surf = refmap_surf.h_surf
        self.xyz_surf = refmap_surf.xyz_surf
