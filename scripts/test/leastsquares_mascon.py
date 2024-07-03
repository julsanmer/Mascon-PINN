import numpy as np
import pickle as pck
import os

from Basilisk.fswAlgorithms import masconFit

from src.gravitymodels.mascon.initializers import initialize_mascon


def modify_data(pos, acc, xyz0_M, mu):
    accMod = acc
    for i in range(len(acc)):
        dpos0 = pos[i, 0:3] - xyz0_M
        dpos0_norm = np.linalg.norm(dpos0)

        accMod[i, 0:3] += mu*dpos0/dpos0_norm**3

    return accMod


def mascon_jacobian(pos, xyz_M):
    # Get number of masses and
    # Jacobian
    n_M = len(xyz_M)
    n = len(pos)

    # Preallocate Jacobian
    jac = np.zeros((3*n, n_M-1))

    # Loop through data and mascon
    for i in range(n):
        # Compute relative position
        # and norm of 0th mass
        dpos0 = pos[i, 0:3] - xyz_M[0, 0:3]
        dpos0_norm = np.linalg.norm(dpos0)

        for j in range(n_M-1):
            # Obtain relative position and norm
            dpos = pos[i, 0:3] - xyz_M[j+1, 0:3]
            dpos_norm = np.linalg.norm(dpos)

            # Store in Jacobian
            jac[3*i:3*(i+1), j] = - dpos/dpos_norm**3\
                                  + dpos0/dpos0_norm**3

    return jac


# Load data file
file_path = os.path.dirname(os.getcwd())
file_data = file_path + '/Results/eros/groundtruth/poly' +\
            '/dense_alt50km_100000samples.pck'
parameters, inputs = pck.load(open(file_data, "rb"))
pos = inputs.groundtruth.states.pos_BP_P[0:10000, 0:3]
acc = inputs.groundtruth.states.acc_BP_P[0:10000, 0:3]

# Set polyhedron
masconfit_bsk = masconFit.MasconFit()
shape = masconfit_bsk.shapeModel
shape.xyzVertex = parameters.asteroid.xyz_vert.tolist()
shape.orderFacet = parameters.asteroid.order_face.tolist()
shape.initializeParameters()

# Initialize mascon distribution
xyz_M0, mu_M0 = initialize_mascon(shape, 100, parameters.asteroid.mu)
jac = mascon_jacobian(pos, xyz_M0)
accMod = modify_data(pos, acc, xyz_M0[0, 0:3], parameters.asteroid.mu)
y = np.linalg.pinv(jac).dot(accMod.reshape(-1))
mu_M = np.concatenate(([parameters.asteroid.mu - np.sum(y)], y))
print(np.sum(mu_M))
