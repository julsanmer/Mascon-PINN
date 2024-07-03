import numpy as np


# This draws a spherical sample
def rad_sample(rmax):
    # Create a random spherical sample
    lat = np.random.uniform(-np.pi/2, np.pi/2)
    lon = np.random.uniform(0, 2*np.pi)
    r = np.random.uniform(0, rmax)
    pos_sample = r * np.array([np.cos(lat) * np.cos(lon),
                               np.cos(lat) * np.sin(lon),
                               np.sin(lat)])

    return pos_sample


# This draws an ellipsoid sample
def ell_sample(rmax, axes):
    # Do a random position sample
    a, b, c = axes
    lat = np.random.uniform(-np.pi/2, np.pi/2)
    lon = np.random.uniform(0, 2*np.pi)
    s = np.random.uniform(0, rmax / np.min([a, b, c]))
    pos_sample = s * np.array([a * np.cos(lat) * np.cos(lon),
                               b * np.cos(lat) * np.sin(lon),
                               c * np.sin(lat)])

    return pos_sample


# This draws a shape-based sample
def alt_sample(hmax, xyz_face):
    # Get faces
    n_face = len(xyz_face)

    # Do a random position sample
    idx = np.random.choice(n_face)
    h = np.random.uniform(0, hmax)
    pos_surface = xyz_face[idx, 0:3]
    r_surface = np.linalg.norm(pos_surface)
    pos_sample = pos_surface * (1 + h/r_surface)

    return pos_sample
