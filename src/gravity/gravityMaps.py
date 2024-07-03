from src.gravity.maps.intervals import Intervals
from src.gravity.maps.map_2D import Map2D
from src.gravity.maps.map_3D import Map3D
from src.gravity.maps.map_surf2D import MapSurface2D
from src.gravity.maps.map_surf3D import MapSurface3D


# Gravity map class
class GravityMap:
    def __init__(self, rmax_2D=None, rmax_3D=None,
                 n_2D=160, nr_3D=40, nlat_3D=40, nlon_3D=40):
        # Initializes 2D and 3D maps
        self.map_2D = Map2D(rmax_2D, n_2D)
        self.map_3D = Map3D(rmax_3D, nr_3D, nlat_3D, nlon_3D)
        self.map_surf2D = MapSurface2D(100, 200)
        self.map_surf3D = MapSurface3D()

        # Initializes alt-rad error intervals
        self.intervals = Intervals()

    # This method creates 2D and 3D grids
    def create_grids(self, shape):
        # Create 2D and 3D grids
        self.map_2D.create_grid(shape)
        self.map_3D.create_grid(shape)
        self.map_surf2D.create_grid(shape)
        self.map_surf3D.create_grid(shape)

    # This method generates 2D and 3D maps
    def generate_maps(self, grav_model):
        # Create 2D and 3D maps
        self.map_2D.generate_acc(grav_model)
        self.map_3D.generate_acc(grav_model)
        self.map_surf2D.generate_acc(grav_model)
        self.map_surf3D.generate_acc(grav_model)

    # This method imports 2D and 3D grids
    def import_grids(self, refmaps):
        # Import grids
        self.map_2D.import_grid(refmaps.map_2D)
        self.map_3D.import_grid(refmaps.map_3D)
        self.map_surf2D.import_grid(refmaps.map_surf2D)
        self.map_surf3D.import_grid(refmaps.map_surf3D)

    # This method computes errors maps w.r.t. reference
    def error_maps(self, refmaps):
        # Compute 2D and 3D error maps
        self.map_2D.compute_acc_error(refmaps.map_2D)
        self.map_3D.compute_errors(refmaps.map_3D)
        self.map_surf2D.compute_errors(refmaps.map_surf2D)
        self.map_surf3D.compute_errors(refmaps.map_surf3D)

        # Compute rad-alt error intervals
        self.intervals.compute_errors(self.map_3D)
