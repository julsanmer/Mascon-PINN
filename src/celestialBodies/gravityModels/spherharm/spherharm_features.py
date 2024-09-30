import math as mt
import numpy as np
import os

from Basilisk.simulation.gravityEffector import loadPolyFromFileToList
from Basilisk import __path__

bsk_path = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])


# This computes spherical harmonics potential
def computePotentialSpherharm(pos, C, S, deg, rE, mu):
    # K parameter
    def getK(input):
        # Conditional
        K = 1 if input == 0 else 2

        return K

    # This computes spherical harmonics potential
    def initializePines():
        # Preallocate variables
        nq1 = np.zeros((deg+2, deg+2))
        nq2 = np.zeros((deg+2, deg+2))

        # Loop through degree
        for n in range(deg+2):
            # Fill diagonal terms
            if n == 0:
                aBar[n, n] = 1
            else:
                aBar[n, n] = np.sqrt((2*n+1) * getK(n) / (2*n*getK(n-1)))\
                             * aBar[n-1, n-1]

            # Loop through order
            for m in range(n+1):
                if n >= m + 2:
                    n1[n, m] = np.sqrt((2*n+1) * (2*n-1)
                                       / ((n-m) * (n+m)))
                    n2[n, m] = np.sqrt((n+m-1) * (2*n+1) * (n-m-1)
                                       / ((n+m) * (n-m) * (2*n-3)))

        for n in range(deg+1):
            for m in range(n+1):
                if m < n:
                    nq1[n, m] = np.sqrt((n-m) * getK(m) * (n+m+1)
                                        / getK(m+1))
                nq2[n, m] = np.sqrt((n+m+2) * (n+m+1) * (2*n+1) * getK(m)
                                    / ((2*n+3) * getK(m+1)))

        return aBar, n1, n2, nq1, nq2

    # Compute Pines coordinates
    r = np.linalg.norm(pos)
    s, t, u = pos / r

    # Initialize Pines params
    aBar = np.zeros((deg+2, deg+2))
    n1 = np.zeros((deg+2, deg+2))
    n2 = np.zeros((deg+2, deg+2))
    initializePines()

    for n in range(1, deg+2):
        # Compute low diagonal terms
        aBar[n, n-1] = np.sqrt((2*n) * getK(n-1) / getK(n))\
                       * aBar[n, n] * u

    # Compute lower terms of A_bar
    rEVec = np.zeros(deg+2)
    iM = np.zeros(deg+2)
    for m in range(deg+2):
        for n in range(m+2, deg+2):
            aBar[n, m] = u * n1[n, m] * aBar[n-1, m] \
                         - n2[n, m] * aBar[n-2, m]
        if m == 0:
            rEVec[m] = 1.0
            iM[m] = 0.0
        else:
            rEVec[m] = s*rEVec[m-1] - t*iM[m-1]
            iM[m] = s*iM[m-1] + t*rEVec[m-1]

    # Define variables
    rho = rE / r
    rhoVec = np.zeros(deg+2)
    rhoVec[0] = mu / r
    rhoVec[1] = rhoVec[0] * rho

    # Initialize potential
    U = 0.0

    # Loop through degree
    for n in range(1, deg+1):
        # Add rhoVec term
        rhoVec[n+1] = rho * rhoVec[n]

        # Initialize sum(A_nm*D_nm)
        sum_AD = 0.0

        # loop through order
        for m in range(n+1):
            # Compute D and add sum(A_nm*D_nm)
            D = C[n, m] * rEVec[m] + S[n, m] * iM[m]
            sum_AD += D * aBar[n, m]

        # Add degree contribution
        U += rhoVec[n+1] * sum_AD

    U += mu / r

    return U
