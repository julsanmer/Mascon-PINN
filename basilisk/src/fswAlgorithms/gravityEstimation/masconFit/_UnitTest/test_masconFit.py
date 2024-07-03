# 
#  ISC License
# 
#  Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado Boulder
# 
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
# 
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


import pytest

from Basilisk.simulation.gravityEffector import loadPolyFromFileToList
from Basilisk.utilities import unitTestSupport
from Basilisk.fswAlgorithms import masconFit
import numpy as np
from matplotlib import pyplot as plt

from Basilisk import __path__

bsk_path = __path__[0]

def test_masconFit(show_plots):
    r"""
    **Validation Test Description**

    This unit test checks that the filter converges to a constant position and null velocity estimates under the presence of static measurements.
    Then, the non-Keplerian gravity estimation should match the Keplerian gravity with opposite sign.

    **Test Parameters**

    Args:
        :param show_plots: flag if plots should be shown.
    """
    [testResults, testMessage] = masconFitTestFunction(show_plots)
    assert testResults < 1, testMessage


def masconFitTestFunction(show_plots):
    """Test method"""
    testFailCount = 0
    testMessages = []

    # Define a simple truth mascon gravity
    # 90% of mass at origin, 5% of mass at two lobes
    mu = 4.46275472004 * 1e5
    nM = 3
    true_mu_M = np.array([0.9, 0.05, 0.05]) * mu
    true_xyz_M = np.array([[0, 0, 0],
                           [5*1e3, 0, 0],
                           [-5*1e3, 0, 0]])

    # Generate 400 data samples with true model
    n_samples = 400
    pos_data = np.zeros((n_samples, 3))
    acc_data = np.zeros((n_samples, 3))
    r_min = 18 * 1e3
    r_max = 30 * 1e3
    np.random.seed(0)
    for j in range(n_samples):
        # Create pseudorandom position for sample
        r = np.random.uniform(r_min, r_max)
        lat = np.random.uniform(-np.pi/2, np.pi/2)
        lon = np.random.uniform(0, 2*np.pi)
        pos = r * np.array([np.cos(lon)*np.cos(lat),
                            np.sin(lon)*np.cos(lat),
                            np.sin(lat)])

        # Compute mascon gravity
        acc = np.zeros(3)
        for k in range(nM):
            dr = pos - true_xyz_M[k, 0:3]
            acc += -true_mu_M[k] * dr/np.linalg.norm(dr)**3

        # Store in data batches
        pos_data[j, 0:3] = pos
        acc_data[j, 0:3] = acc

    # Setup module to be tested
    module = masconFit.MasconFit()

    # Set adimensional variables
    module.mu = mu
    module.muMad = mu / nM
    module.xyzMad = np.array([16342.6, 8410.61, 5973.615])/10

    # Choose algorithm and loss function
    module.useMSE = True
    module.useMLE = False

    # Set training variables flag
    xyz_vert, order_face, n_vert, n_face = \
        loadPolyFromFileToList(bsk_path + '/supportData/LocalGravData/EROS856Vert1708Fac.txt')
    module.shape.initPolyhedron(xyz_vert, order_face)
    module.trainXYZ = True

    # Set Adam parameters
    module.setMaxIter(4000)
    module.setLR(1e-3)

    # Initialize mascon distribution
    mu_M0 = np.array([0.8, 0.1, 0.1]) * mu
    xyz_M0 = np.array([[0, 0, 0],
                       [5*1e3, 0, 0],
                       [-5*1e3, 0, 0]])
    module.muM = mu_M0.tolist()
    module.xyzM = xyz_M0.tolist()

    # Train mascon distribution with data
    module.train(pos_data.tolist(), acc_data.tolist(), True)

    muM = np.array(module.muM).squeeze()
    xyzM = np.array(module.xyzM).squeeze()
    loss = module.getLoss()

    testFailCount, testMessages = unitTestSupport.compareArrayRelative(
        [true_mu_M], [muM], 1e-3, "mu_M",
        testFailCount, testMessages)
    testFailCount, testMessages = unitTestSupport.compareVectorRelative(
        true_xyz_M[0, 0:3], xyzM[0, 0:3], 1e-3, "xyz_M0",
        testFailCount, testMessages)
    testFailCount, testMessages = unitTestSupport.compareVectorRelative(
        true_xyz_M[1, 0:3], xyzM[1, 0:3], 1e-3, "xyz_M1",
        testFailCount, testMessages)
    testFailCount, testMessages = unitTestSupport.compareVectorRelative(
        true_xyz_M[2, 0:3], xyzM[2, 0:3], 1e-3, "xyz_M2",
        testFailCount, testMessages)

    plt.figure(1)
    plt.clf()
    plt.figure(1, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k')
    plt.ticklabel_format(useOffset=False)
    plt.plot(loss)
    plt.xlabel('Iterations [-]')
    plt.ylabel('Loss [-]')
    plt.title('Loss history')

    if show_plots:
        plt.show()

    if testFailCount == 0:
        print("PASSED: " + "masconFit")
    else:
        print(testMessages)

    return [testFailCount, "".join(testMessages)]


if __name__ == "__main__":
    test_masconFit(True)
