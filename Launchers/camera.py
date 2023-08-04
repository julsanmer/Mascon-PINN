import numpy as np

from Basilisk.simulation import pinholeCamera
from Basilisk.utilities import RigidBodyKinematics as rbk


# This function executes the pinhole camera module
# to generate landmark-based measurements
def run_camera(parameters, outputs):
    # This function initializes pinhole camera module parameters
    def initialize_camera():
        # Set camera properties
        camera_bsk.nxPixel = parameters.sensors.nx_pixel
        camera_bsk.nyPixel = parameters.sensors.ny_pixel
        camera_bsk.wPixel = parameters.sensors.w_pixel
        camera_bsk.f = parameters.sensors.f

        # Add landmark information
        for j in range(n_lmk):
            camera_bsk.addLandmark(parameters.sensors.xyz_lmk[j, 0:3],
                                   parameters.sensors.normal_lmk[j, 0:3])

        # Set Sun mask angle and camera dcm
        camera_bsk.maskCam = parameters.sensors.maskangle_cam
        camera_bsk.maskSun = parameters.sensors.maskangle_sun
        camera_bsk.dcm_CB = parameters.sensors.dcm_CB.tolist()

        # Reset module
        camera_bsk.Reset(0)

    # Retrieve ground truth times
    t = outputs.groundtruth.t

    # Get number of landmarks and
    # camera orientation w.r.t. body
    n_lmk = parameters.sensors.n_lmk
    dcm_CB = parameters.sensors.dcm_CB

    # Initialize BSK camera module
    camera_bsk = pinholeCamera.PinholeCamera()
    initialize_camera()

    # Get spacecraft position and orientation
    # Get small body to Sun unit-vector
    pos_BP_P = outputs.groundtruth.pos_BP_P
    mrp_BP = outputs.groundtruth.mrp_BP
    e_SP_P = outputs.groundtruth.e_SP_P

    # Run module
    camera_bsk.processBatch(pos_BP_P.tolist(), mrp_BP.tolist(),
                            e_SP_P.tolist(), True)

    # Save camera times and get number of samples
    outputs.camera.t = t
    n = len(t)

    # Process outputs
    isvisibleBatchLmk = np.array(camera_bsk.isvisibleBatchLmk)
    nvisibleBatchLmk = np.sum(isvisibleBatchLmk, axis=1)
    pixelBatchLmk = np.array(camera_bsk.pixelBatchLmk)
    mrp_CP = np.zeros((n, 3))
    for i in range(n):
        dcm_BP = rbk.MRP2C(mrp_BP[i, 0:3])
        dcm_CP = np.matmul(dcm_CB, dcm_BP)
        mrp_CP[i, 0:3] = rbk.C2MRP(dcm_CP)

    # Save outputs
    outputs.camera.pixel_lmk = np.array(pixelBatchLmk).astype(int)
    outputs.camera.nvisible_lmk = np.array(nvisibleBatchLmk).astype(int)
    outputs.camera.isvisible_lmk = np.array(isvisibleBatchLmk).astype(int)
    outputs.camera.flag_meas = outputs.camera.nvisible_lmk > 0
    outputs.camera.mrp_CP = np.array(mrp_CP)
