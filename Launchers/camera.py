import numpy as np
from Basilisk.ExternalModules import cameraNav4


def run_camera(parameters, outputs):
    def initialize_camera():
        # Set camera properties
        camera_bsk.nxPixel = parameters.sensors.nx_pixel
        camera_bsk.nyPixel = parameters.sensors.ny_pixel
        camera_bsk.wPixel = parameters.sensors.w_pixel
        camera_bsk.f = parameters.sensors.f

        # Set mask angles
        camera_bsk.maskangleCam = parameters.sensors.maskangle_cam
        camera_bsk.maskangleSun = parameters.sensors.maskangle_sun

        # Set landmarks uncertainty
        camera_bsk.dxyzLandmark = parameters.sensors.dxyz_lmk

        # Set polyhedron shape properties
        camera_bsk.body.xyzVertex = parameters.sensors.xyz_vert
        camera_bsk.body.orderFacet = parameters.sensors.order_face
        camera_bsk.body.idxLandmark = parameters.sensors.idx_lmk

        # Reset module
        camera_bsk.Reset(0)

    # Retrieve times
    t = outputs.groundtruth.t

    # Initialize BSK camera class
    camera_bsk = cameraNav4.CameraNav4()
    initialize_camera()

    # Run module
    e_SA_A = outputs.groundtruth.e_SA_A
    pos_CA_A = outputs.groundtruth.pos_CA_A
    camera_bsk.computePixelBatch(pos_CA_A.tolist(), e_SA_A)

    # Save outputs
    outputs.camera.t = t
    outputs.camera.pixel = np.array(camera_bsk.pixelBatch)
    outputs.camera.n_visible = np.array(camera_bsk.nVisibleBatch).squeeze()
    outputs.camera.flag_nav = np.array(camera_bsk.visibleBatch).squeeze()
    outputs.camera.latlon = np.array(camera_bsk.latlonBatch)
