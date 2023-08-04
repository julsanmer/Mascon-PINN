import numpy as np

from Basilisk.utilities import simIncludeGravBody
from Basilisk.ExternalModules import masconFit


# This function computes a 2D gravity map
def gravity_map2D(parameters, outputs, map_type):
    # Choose type of gravity map
    if map_type == 'groundtruth':
        # Set shape object
        masconfit_bsk = masconFit.MasconFit()
        shape = masconfit_bsk.shape
        shape.initPolyhedron(parameters.asteroid.xyz_vert.tolist(),
                             parameters.asteroid.order_face.tolist())

        # Extract parameters for grid
        rmax = outputs.groundtruth.rmax
        n = outputs.groundtruth.n_2D

        # Make 2D grid
        x = np.linspace(-rmax, rmax, n)
        y = np.linspace(-rmax, rmax, n)
        z = np.linspace(-rmax, rmax, n)
        Xy_2D, Yx_2D = np.meshgrid(x, y)
        Xz_2D, Zx_2D = np.meshgrid(x, z)
        Yz_2D, Zy_2D = np.meshgrid(y, z)

        # Reshape to 1D
        Xy_1D = Xy_2D.reshape(n*n)
        Yx_1D = Yx_2D.reshape(n*n)
        Xz_1D = Xz_2D.reshape(n*n)
        Zx_1D = Zx_2D.reshape(n*n)
        Yz_1D = Yz_2D.reshape(n*n)
        Zy_1D = Zy_2D.reshape(n*n)

        # Fill positions
        XY = np.vstack((Xy_1D, Yx_1D, np.zeros(n*n))).transpose()
        XZ = np.vstack((Xz_1D, np.zeros(n*n), Zx_1D)).transpose()
        YZ = np.vstack((np.zeros(n*n), Yz_1D, Zy_1D)).transpose()

        # Compute laplacian and exterior points
        lapXY_2D = np.reshape(np.array(shape.computeLaplacianBatch(XY.tolist())),
                              (n, n))
        lapXZ_2D = np.reshape(np.array(shape.computeLaplacianBatch(XZ.tolist())),
                              (n, n))
        lapYZ_2D = np.reshape(np.array(shape.computeLaplacianBatch(YZ.tolist())),
                              (n, n))
        extXY_2D = abs(lapXY_2D) < 2*np.pi
        extXZ_2D = abs(lapXZ_2D) < 2*np.pi
        extYZ_2D = abs(lapYZ_2D) < 2*np.pi

        # Save to outputs class
        outputs.groundtruth.Xy_2D = Xy_2D
        outputs.groundtruth.Yx_2D = Yx_2D
        outputs.groundtruth.Xz_2D = Xz_2D
        outputs.groundtruth.Zx_2D = Zx_2D
        outputs.groundtruth.Yz_2D = Yz_2D
        outputs.groundtruth.Zy_2D = Zy_2D
        outputs.groundtruth.extXY_2D = extXY_2D
        outputs.groundtruth.extXZ_2D = extXZ_2D
        outputs.groundtruth.extYZ_2D = extYZ_2D

        # Create gravity object
        mu = parameters.asteroid.mu
        gravFactory = simIncludeGravBody.gravBodyFactory()
        gravity = gravFactory.createCustomGravObject("eros", mu=mu)

        # Set polyhedron model
        gravity.poly.muBody = mu
        gravity.poly.nVertex = parameters.asteroid.n_vert
        gravity.poly.nFacet = parameters.asteroid.n_face
        gravity.poly.xyzVertex = parameters.asteroid.xyz_vert.tolist()
        gravity.poly.orderFacet = parameters.asteroid.order_face.tolist()
        grav = gravity.poly
        grav.initializeParameters()
    elif map_type == 'results':
        # Retry grid
        Xy_2D = outputs.groundtruth.Xy_2D
        Yx_2D = outputs.groundtruth.Yx_2D
        Xz_2D = outputs.groundtruth.Xz_2D
        Zx_2D = outputs.groundtruth.Zx_2D
        Yz_2D = outputs.groundtruth.Yz_2D
        Zy_2D = outputs.groundtruth.Zy_2D
        extXY_2D = outputs.groundtruth.extXY_2D
        extXZ_2D = outputs.groundtruth.extXZ_2D
        extYZ_2D = outputs.groundtruth.extYZ_2D

        # Create gravity object
        gravFactory = simIncludeGravBody.gravBodyFactory()
        mu = parameters.grav_est.mu
        gravity = gravFactory.createCustomGravObject("eros", mu=mu)

        # Set mascon model
        gravity.mascon.muM = parameters.grav_est.mu_M.tolist()
        gravity.mascon.xyzM = parameters.grav_est.pos_M.tolist()
        grav = gravity.mascon
        grav.initializeParameters()

    # Preallocate gravity 2D map
    n = outputs.groundtruth.n_2D
    a_XY = np.zeros((n, n, 3))
    a_XZ = np.zeros((n, n, 3))
    a_YZ = np.zeros((n, n, 3))
    a0_XY = np.zeros((n, n, 3))
    a0_XZ = np.zeros((n, n, 3))
    a0_YZ = np.zeros((n, n, 3))

    # Fill gravity 2D map
    for i in range(n):
        for j in range(n):
            # Save coordinates
            xy = [Xy_2D[i, j], Yx_2D[i, j], 0]
            xz = [Xz_2D[i, j], 0, Zx_2D[i, j]]
            yz = [0, Yz_2D[i, j], Zy_2D[i, j]]

            # Choose type of gravity map
            if map_type == 'groundtruth':
                # Map on the xy plane
                if extXY_2D[i, j]:
                    a_xy = np.array(grav.computeField(xy)).reshape(3)
                    a0_xy = -mu * np.array(xy) / (np.linalg.norm(xy))**3
                else:
                    a_xy = np.nan * np.ones(3)
                    a0_xy = np.nan * np.ones(3)

                # Map on the xz plane
                if extXZ_2D[i, j]:
                    a_xz = np.array(grav.computeField(xz)).reshape(3)
                    a0_xz = -mu * np.array(xz) / (np.linalg.norm(xz)) ** 3
                else:
                    a_xz = np.nan * np.ones(3)
                    a0_xz = np.nan * np.ones(3)

                # Map on the yz plane
                if extYZ_2D[i, j]:
                    a_yz = np.array(grav.computeField(yz)).reshape(3)
                    a0_yz = -mu * np.array(yz) / (np.linalg.norm(yz))**3
                else:
                    a_yz = np.nan * np.ones(3)
                    a0_yz = np.nan * np.ones(3)

                # Save Keplerian gravity
                a0_XY[i, j, :] = np.array(a0_xy).reshape(3)
                a0_XZ[i, j, :] = np.array(a0_xz).reshape(3)
                a0_YZ[i, j, :] = np.array(a0_yz).reshape(3)
            elif map_type == 'results':
                # Map on the xy plane
                if extXY_2D[i, j]:
                    a_xy = np.array(grav.computeField(xy)).reshape(3)
                else:
                    a_xy = np.nan * np.ones(3)

                # Map on the xz plane
                if extXZ_2D[i, j]:
                    a_xz = np.array(grav.computeField(xz)).reshape(3)
                else:
                    a_xz = np.nan * np.ones(3)

                # Map on the yz plane
                if extYZ_2D[i, j]:
                    a_yz = np.array(grav.computeField(yz)).reshape(3)
                else:
                    a_yz = np.nan * np.ones(3)

            # Save gravity
            a_XY[i, j, :] = np.array(a_xy).reshape(3)
            a_XZ[i, j, :] = np.array(a_xz).reshape(3)
            a_YZ[i, j, :] = np.array(a_yz).reshape(3)

    # Save outputs
    if map_type == 'groundtruth':
        # Save gravity map
        outputs.groundtruth.aXY_2D = a_XY
        outputs.groundtruth.aXZ_2D = a_XZ
        outputs.groundtruth.aYZ_2D = a_YZ
        outputs.groundtruth.a0XY_2D = a0_XY
        outputs.groundtruth.a0XZ_2D = a0_XZ
        outputs.groundtruth.a0YZ_2D = a0_YZ

        # Compute gravity map errors
        outputs.groundtruth.a0ErrXY_2D = np.linalg.norm(a0_XY - a_XY, axis=2) / np.linalg.norm(a_XY, axis=2)
        outputs.groundtruth.a0ErrXZ_2D = np.linalg.norm(a0_XZ - a_XZ, axis=2) / np.linalg.norm(a_XZ, axis=2)
        outputs.groundtruth.a0ErrYZ_2D = np.linalg.norm(a0_YZ - a_YZ, axis=2) / np.linalg.norm(a_YZ, axis=2)
    elif map_type == 'results':
        # Save gravity map
        outputs.results.aXY_2D = a_XY
        outputs.results.aXZ_2D = a_XZ
        outputs.results.aYZ_2D = a_YZ

        # Compute gravity map errors
        outputs.results.aErrXY_2D = np.linalg.norm(a_XY - outputs.groundtruth.aXY_2D, axis=2) \
                                    / np.linalg.norm(outputs.groundtruth.aXY_2D, axis=2)
        outputs.results.aErrXZ_2D = np.linalg.norm(a_XZ - outputs.groundtruth.aXZ_2D, axis=2) \
                                    / np.linalg.norm(outputs.groundtruth.aXZ_2D, axis=2)
        outputs.results.aErrYZ_2D = np.linalg.norm(a_YZ - outputs.groundtruth.aYZ_2D, axis=2) \
                                    / np.linalg.norm(outputs.groundtruth.aYZ_2D, axis=2)

# This function computes a 3D gravity map
def gravity_map3D(parameters, outputs, map_type):
    # Choose type of gravity map
    if map_type == 'groundtruth':
        # Set shape object
        masconfit_bsk = masconFit.MasconFit()
        shape = masconfit_bsk.shape
        shape.initPolyhedron(parameters.asteroid.xyz_vert.tolist(),
                             parameters.asteroid.order_face.tolist())

        # Parameters for 3D grid
        rmax = outputs.groundtruth.rmax
        nr = outputs.groundtruth.nr_3D
        nlat = outputs.groundtruth.nlat_3D
        nlon = outputs.groundtruth.nlon_3D

        # Preallocate spatial grid
        r = np.linspace(rmax / nr, rmax, nr)
        lat = np.linspace(-np.pi/2 * (1-2/nlat), np.pi/2 * (1-2/nlat), nlat)
        lon = np.linspace(0, 2*np.pi * (1-1/nlon), nlon)
        X_3D = np.zeros((nr, nlat, nlon))
        Y_3D = np.zeros((nr, nlat, nlon))
        Z_3D = np.zeros((nr, nlat, nlon))
        r_3D = np.zeros((nr, nlat, nlon))
        h_3D = np.zeros((nr, nlat, nlon))

        # Create 3D grid with radial randomness
        np.random.seed(0)
        r0 = 0
        rf = r[0]
        for i in range(nr):
            for j in range(nlat):
                for k in range(nlon):
                    # Assign random radius within range
                    r_jk = np.random.uniform(r0, rf)

                    # Fill spatial grid
                    X_3D[i, j, k] = r_jk * np.cos(lat[j]) * np.cos(lon[k])
                    Y_3D[i, j, k] = r_jk * np.cos(lat[j]) * np.sin(lon[k])
                    Z_3D[i, j, k] = r_jk * np.sin(lat[j])
                    r_3D[i, j, k] = r_jk
                    h_3D[i, j, k] = shape.computeAltitude([X_3D[i, j, k],
                                                           Y_3D[i, j, k],
                                                           Z_3D[i, j, k]])

            # Switch to next radial range
            if i < nr-1:
                r0 = r[i]
                rf = r[i+1]

        # Reshape to 1D arrays
        X_1D = X_3D.reshape(nr * nlat * nlon)
        Y_1D = Y_3D.reshape(nr * nlat * nlon)
        Z_1D = Z_3D.reshape(nr * nlat * nlon)
        XYZ = np.vstack((X_1D, Y_1D, Z_1D)).transpose()

        # Check interior shape points
        lapXYZ_3D = np.reshape(np.array(shape.computeLaplacianBatch(XYZ.tolist())), (nr, nlat, nlon))
        extXYZ_3D = abs(lapXYZ_3D) < 2*np.pi
        r_3D[np.invert(extXYZ_3D)] = np.nan
        h_3D[np.invert(extXYZ_3D)] = np.nan

        # Save outputs
        outputs.groundtruth.X_3D = X_3D
        outputs.groundtruth.Y_3D = Y_3D
        outputs.groundtruth.Z_3D = Z_3D
        outputs.groundtruth.rXYZ_3D = r_3D
        outputs.groundtruth.hXYZ_3D = h_3D
        outputs.groundtruth.extXYZ_3D = extXYZ_3D

        # Create gravity object
        mu = parameters.asteroid.mu
        gravFactory = simIncludeGravBody.gravBodyFactory()
        gravity = gravFactory.createCustomGravObject("eros", mu=mu)

        # Set polyhedron model
        gravity.poly.muBody = mu
        gravity.poly.nVertex = parameters.asteroid.n_vert
        gravity.poly.nFacet = parameters.asteroid.n_face
        gravity.poly.xyzVertex = parameters.asteroid.xyz_vert.tolist()
        gravity.poly.orderFacet = parameters.asteroid.order_face.tolist()
        grav = gravity.poly
        grav.initializeParameters()
    elif map_type == 'results':
        # Retry grid
        nr = outputs.groundtruth.nr_3D
        nlat = outputs.groundtruth.nlat_3D
        nlon = outputs.groundtruth.nlon_3D
        X_3D = outputs.groundtruth.X_3D
        Y_3D = outputs.groundtruth.Y_3D
        Z_3D = outputs.groundtruth.Z_3D
        extXYZ_3D = outputs.groundtruth.extXYZ_3D

        # Create gravity object
        gravFactory = simIncludeGravBody.gravBodyFactory()
        mu = parameters.grav_est.mu
        gravity = gravFactory.createCustomGravObject("eros", mu=mu)

        # Set mascon model
        gravity.mascon.muM = parameters.grav_est.mu_M.tolist()
        gravity.mascon.xyzM = parameters.grav_est.pos_M.tolist()
        grav = gravity.mascon
        grav.initializeParameters()

    # Preallocate acceleration
    a_XYZ = np.zeros((nr, nlat, nlon, 3))
    a0_XYZ = np.zeros((nr, nlat, nlon, 3))
    for i in range(nr):
        for j in range(nlat):
            for k in range(nlon):
                xyz = [X_3D[i, j, k], Y_3D[i, j, k], Z_3D[i, j, k]]

                if map_type == 'groundtruth':
                    if extXYZ_3D[i, j, k]:
                        a_xyz = np.array(grav.computeField(xyz)).reshape(3)
                        a0_xyz = -mu * np.array(xyz) / (np.linalg.norm(xyz))**3
                    else:
                        a_xyz = np.nan * np.ones(3)
                        a0_xyz = np.nan * np.ones(3)

                    a_XYZ[i, j, k, :] = a_xyz
                    a0_XYZ[i, j, k, :] = a0_xyz
                elif map_type == 'results':
                    if extXYZ_3D[i, j, k]:
                        a_xyz = np.array(grav.computeField(xyz)).reshape(3)
                    else:
                        a_xyz = np.nan * np.ones(3)
                    a_XYZ[i, j, k, :] = a_xyz

    # Save outputs
    if map_type == 'groundtruth':
        # Save gravity map
        outputs.groundtruth.aXYZ_3D = a_XYZ
        outputs.groundtruth.a0XYZ_3D = a0_XYZ

        # Compute gravity map errors
        outputs.groundtruth.a0ErrXYZ_3D = np.linalg.norm(a0_XYZ - a_XYZ, axis=3) \
                                          / np.linalg.norm(a_XYZ, axis=3)

        # Prepare radius, altitude and Keplerian gravity bins
        n_bins = outputs.groundtruth.n_bins
        r0 = np.nanmin(r_3D)
        rf = np.nanmax(r_3D)
        r_bins = np.zeros((n_bins-1, 2))
        r_bins[:, 0] = np.linspace(r0, rf-(rf-r0) / (n_bins-1), n_bins-1)
        r_bins[:, 1] = np.linspace(r0+(rf-r0) / (n_bins-1), rf, n_bins-1)
        outputs.groundtruth.r_bins = r_bins
        outputs.groundtruth.a0Err_binsRad = np.zeros(n_bins-1)
        outputs.groundtruth.N_rad = np.zeros(n_bins-1)

        h0 = np.nanmin(h_3D)
        hf = np.nanmax(h_3D)
        h_bins = np.zeros((n_bins-1, 2))
        h_bins[:, 0] = np.linspace(h0, hf-(hf-h0) / (n_bins-1), n_bins-1)
        h_bins[:, 1] = np.linspace(h0+(hf-h0) / (n_bins-1), hf, n_bins-1)
        outputs.groundtruth.h_bins = h_bins
        outputs.groundtruth.a0Err_binsAlt = np.zeros(n_bins-1)
        outputs.groundtruth.N_alt = np.zeros(n_bins-1)

        # Reshape radius, altitude and gravity errors as 1D
        r_1D = np.ravel(r_3D)
        h_1D = np.ravel(h_3D)
        a0ErrXYZ_1D = np.ravel(outputs.groundtruth.a0ErrXYZ_3D)

        # Loop through bins
        for i in range(len(r_bins)):
            # Compute mean gravity error per radius range
            idx = np.where(np.logical_and(r_1D >= r_bins[i, 0], r_1D < r_bins[i, 1]))[0]
            outputs.groundtruth.a0Err_binsRad[i] = np.sum(a0ErrXYZ_1D[idx]) / len(idx)
            outputs.groundtruth.N_rad[i] = len(idx)

            # Compute mean gravity error per altitude range
            idx = np.where(np.logical_and(h_1D >= h_bins[i, 0], h_1D < h_bins[i, 1]))[0]
            outputs.groundtruth.a0Err_binsAlt[i] = np.sum(a0ErrXYZ_1D[idx]) / len(idx)
            outputs.groundtruth.N_alt[i] = len(idx)

        # Compute total error
        outputs.groundtruth.a0ErrTotal_3D = np.sum(outputs.groundtruth.a0Err_binsAlt * outputs.groundtruth.N_alt) \
                                            / np.sum(outputs.groundtruth.N_alt)

    elif map_type == 'results':
        # Save gravity map
        outputs.results.aXYZ_3D = a_XYZ

        # Compute gravity map errors
        outputs.results.aErrXYZ_3D = np.linalg.norm(a_XYZ - outputs.groundtruth.aXYZ_3D, axis=3) \
                                     / np.linalg.norm(outputs.groundtruth.aXYZ_3D, axis=3)

        # Prepare radius, altitude and mascon gravity bins
        n_bins = outputs.groundtruth.n_bins
        r_3D = outputs.groundtruth.rXYZ_3D
        r_bins = outputs.groundtruth.r_bins
        h_3D = outputs.groundtruth.hXYZ_3D
        h_bins = outputs.groundtruth.h_bins
        outputs.results.aErr_binsRad = np.zeros(n_bins-1)
        outputs.results.aErr_binsAlt = np.zeros(n_bins-1)

        # Reshape radius, altitude and gravity errors as 1D
        r_1D = np.ravel(r_3D)
        h_1D = np.ravel(h_3D)
        aErrXYZ_1D = np.ravel(outputs.results.aErrXYZ_3D)

        # Loop through bins
        for i in range(len(r_bins)):
            # Compute mean gravity error per radius range
            idx = np.where(np.logical_and(r_1D >= r_bins[i, 0], r_1D < r_bins[i, 1]))[0]
            outputs.results.aErr_binsRad[i] = np.sum(aErrXYZ_1D[idx]) / len(idx)

            # Compute mean gravity error per altitude range
            idx = np.where(np.logical_and(h_1D >= h_bins[i, 0], h_1D < h_bins[i, 1]))[0]
            outputs.results.aErr_binsAlt[i] = np.sum(aErrXYZ_1D[idx]) / len(idx)

        # Compute total error
        outputs.results.aErrTotal_3D = np.sum(outputs.results.aErr_binsAlt * outputs.groundtruth.N_alt) \
                                       / np.sum(outputs.groundtruth.N_alt)


# This function computes the gravity acceleration
def gravity_series(parameters, outputs):
    # Extract estimated position
    pos = outputs.results.pos_data
    n = len(pos)

    # Preallocate polyhedron and mascon ground truth
    acc_poly = np.zeros((n, 3))

    # Create gravity objects
    mu = parameters.asteroid.mu
    gravFactory = simIncludeGravBody.gravBodyFactory()

    # Set polyhedron model
    gravity = gravFactory.createCustomGravObject("eros1", mu=mu)
    gravity.poly.muBody = mu
    gravity.poly.nVertex = parameters.asteroid.n_vert
    gravity.poly.nFacet = parameters.asteroid.n_face
    gravity.poly.xyzVertex = parameters.asteroid.xyz_vert.tolist()
    gravity.poly.orderFacet = parameters.asteroid.order_face.tolist()
    grav = gravity.poly
    grav.initializeParameters()

    # Loop through samples
    for i in range(n):
        # Compute Keplerian term
        acc0 = -mu * pos[i, 0:3]/np.linalg.norm(pos[i, 0:3])**3

        # Compute polyhedron and mascon gravity
        acc_poly[i, 0:3] = np.array(grav.computeField(pos[i, 0:3])).reshape(3) \
                           - acc0

    outputs.results.accNK_poly = acc_poly