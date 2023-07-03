class Configuration:
    """
    Class that sets up variable parameters of the simulation.
    """
    def __init__(self):
        # Asteroid name
        self.asteroid_name = []

        # Simulation and data type
        self.data_type = []
        self.results_type = []

        # Ground truth variables
        self.grav_groundtruth = []
        self.orbits_groundtruth = []

        # Training segments
        self.orbits_dmcukf = []
        self.dmcukf_rate = []
        self.gravest_rate = []

        # Mascon training, initialization, number of masses and random realizations
        self.mascon_type = []
        self.mascon_init = []
        self.nM_array = []
        self.rand_M = []

        # Gradient descent maximum iterations, learning rate and loss type
        self.maxiter = []
        self.lr = []
        self.loss_type = []

        # Camera focal length, number of landmarks, error and lighting condition
        self.f = []
        self.n_lmk = []
        self.dev_lmk = []
        self.maskangle_sun = []

        # Spacecraft initial semi-major axis and inclination
        self.a0 = []
        self.i0 = []

        # Ejecta
        self.flag_ejecta = []
        self.n_ejecta = []
        self.dev_ejecta = []

        # Files and paths
        self.file_groundtruth = []
        self.file_camera = []
        self.filepath_results = []

        # Simulation options flags
        self.flag_groundtruth = []
        self.flag_camera = []
        self.flag_results = []
        self.flag_plot = []
        self.flag_map2D = []
        self.flag_map3D = []
