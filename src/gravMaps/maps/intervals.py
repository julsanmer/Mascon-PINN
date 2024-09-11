import numpy as np


# Error intervals
class Intervals:
    def __init__(self):
        # Gravity accuracy results
        # w.r.t. radius and altitude
        self.n_bins = 100
        self.r_bins, self.h_bins = [], []
        self.N_rad, self.N_alt = [], []
        self.aErrRad_bins = []
        self.aErrAlt_bins = []
        self.aErrTotal = []

    # This method computes rad-alt error intervals
    def compute_errors(self, map_3D):
        # Get bins, altitude and radius
        n_bins = self.n_bins
        h = np.ravel(map_3D.h)
        r = np.ravel(map_3D.r)

        # Obtain min-max altitude and radius
        h_min, h_max = np.nanmin(h), np.nanmax(h)
        r_min, r_max = np.nanmin(r), np.nanmax(r)

        # Set altitude bins
        h_lim = np.linspace(h_min,
                            h_max + 1e-6*h_max,
                            n_bins+1)
        dh = (h_lim[1] - h_lim[0]) / 2
        self.h_bins = h_lim[0:-1] + dh

        # Set radius bins
        r_lim = np.linspace(r_min,
                            r_max + 1e-6*r_max,
                            n_bins+1)
        dr = (r_lim[1] - r_lim[0]) / 2
        self.r_bins = r_lim[0:-1] + dr

        # Number of intervals
        self.N_rad = np.zeros(n_bins)
        self.N_alt = np.zeros(n_bins)

        # Preallocate mean error
        self.aErrRad_bins = np.zeros(n_bins)
        self.aErrAlt_bins = np.zeros(n_bins)

        # Reshape gravity errors
        aErr = np.ravel(map_3D.aErrXYZ)

        # Loop through bins
        for i in range(n_bins):
            # Compute mean gravity error per radius range
            rmin_i = self.r_bins[i] - dr
            rmax_i = self.r_bins[i] + dr
            idx = np.where(np.logical_and(r >= rmin_i,
                                          r < rmax_i))[0]
            self.N_rad[i] = len(idx)
            self.aErrRad_bins[i] =\
                np.sum(aErr[idx]) / len(idx)

            # Compute mean gravity error per altitude range
            hmin_i = self.h_bins[i] - dh
            hmax_i = self.h_bins[i] + dh
            idx = np.where(np.logical_and(h >= hmin_i,
                                          h < hmax_i))[0]
            self.N_alt[i] = len(idx)
            self.aErrAlt_bins[i] =\
                np.sum(aErr[idx]) / len(idx)

        # Compute total error
        n_samples = np.count_nonzero(~np.isnan(aErr))
        self.aErrTotal = np.nansum(aErr) / n_samples
