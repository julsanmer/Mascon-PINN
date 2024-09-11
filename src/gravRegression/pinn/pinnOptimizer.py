import numpy as np
import os
import torch

from timeit import default_timer as timer

from src.gravRegression.gravityOptimizer import GravityOptimizer
from src.gravRegression.pinn.trainer import Optimizer
from src.gravRegression.pinn.pinn_eval import PINNeval
from src.gravRegression.pinn.pinn_train import PINNtrain

# Conversion constants
deg2rad = np.pi/180
km2m = 1e3


# This is the PINN class
class PINNOptimizer(GravityOptimizer):
    def __init__(self, file_torch=None):
        super().__init__()

        # Set model name
        self.name = 'pinn'

        # Define layers and activation function
        self.layers = []
        self.activation = []

        # Adimensional radius and proxy potential
        self.r_ad = []
        self.Uprx_ad = []

        # Set switch function parameters
        self.r_bc, self.k_bc, self.l_bc = \
            [], [], []

        # Loss function and training time
        self.t_train = []
        self.loss = []

        # Define boundary model
        self.model_bc = self.ModelBC

        # Define optimizer and network
        self.optimizer = None
        self.network = None

        # Set saving files
        self.file_onnx = []
        self.file_torch = file_torch

    # This is the boundary model class
    class ModelBC:
        def __init__(self):
            # Preallocate type
            self.type = []

            # Preallocate mascon variables
            self.n_M, self.mu_M, self.xyz_M = \
                [], [], []

            # Preallocate spherical harmonics variables
            self.mu, self.rE, self.deg = [], [], []
            self.C, self.S = [], []

    # This method prepares pinn optimizer
    def prepare_optimizer(self, maxiter=1000, lr=1e-3, batch_size=1,
                          loss_type='linear'):
        # Initialize mascon fit module
        self.optimizer = Optimizer(self.network)

        # Set some variables
        self.optimizer.lr = lr
        self.optimizer.maxiter = maxiter
        self.optimizer.batch_size = batch_size
        self.optimizer.loss_type = loss_type

        # Initialise optimizer
        self.optimizer.initialize()

    # This method trains PINN
    def optimize(self, pos_data, acc_data, U_data):
        # Normalise potential
        self.normalize_data(pos_data, acc_data, U_data)

        # Start measuring cpu time
        t_start = timer()

        # Call optimizer
        self.optimizer.train(pos_data, acc_data)

        # End measuring cpu time
        t_end = timer()
        self.t_train.append(t_end - t_start)

        # Save loss
        self.loss = self.optimizer.loss

    # This method adds a mascon model
    def add_mascon(self, mu_M, xyz_M):
        # Set masses and positions
        self.model_bc.type = 'mascon'
        self.model_bc.mu_M, self.model_bc.xyz_M = mu_M, xyz_M
        self.model_bc.n_M = len(self.model_bc.mu_M)

    # # This method adds a spherical harmonics model
    # def add_spherharm(self):
    #     # Define polyhedron file
    #     poly_file = bsk_path + '/supportData/LocalGravData/eros007790.tab'
    #
    #     # Compute spherical harmonics coefficients
    #     self.mu, self.rE, self.deg = mu, 16 * km2m, 2
    #
    #     # Compute spherical harmonics from shape
    #     from src.Gravity.spherharm.spherharm_features import poly2spherharm
    #     self.C, self.S = \
    #         poly2spherharm(self.deg, poly_file, self.rE)

    # This method computes proxy potential
    def compute_proxy(self, pos_data):
        # Compute data radius
        r_data = np.linalg.norm(pos_data, axis=1)

        # Transform position and radius to tensors
        pos_data = torch.from_numpy(pos_data).to(dtype=torch.float32)
        r_data = torch.from_numpy(r_data).to(dtype=torch.float32)

        # Do a forward call of the potential
        U_prx = self.network.forward(pos_data, r_data)

        # Set proxy potential
        self.Uprx = U_prx.detach().numpy()

    # This method initializes network for training
    def init_network(self, n_layers=8, n_neurons=40,
                     activation='SIREN', eval_mode='BSK'):
        # Set layers, neurons and activation
        layers = []
        for i in range(n_layers + 2):
            if i == 0:
                layers.append(5)
            elif i == n_layers + 1:
                layers.append(1)
            else:
                layers.append(n_neurons)
        self.layers = np.array(layers)
        self.activation = activation

        # Set network
        self.network = PINNtrain(self, device='cpu')

    # This method sets adimensional parameters
    def set_extra_params(self, r_ad=1., r_bc=1.,
                         k_bc=1., l_bc=1.):
        # Adimensionalization for inputs
        self.r_ad = r_ad

        # Switch function variables
        self.r_bc = r_bc
        self.k_bc = k_bc
        self.l_bc = l_bc

    # This method computes boundary model potential
    def bc_potential(self, pos_data):
        U_bc = self.network.model_bc.compute_potential(pos_data)
        U_bc = U_bc.cpu().detach().numpy()

        return U_bc

    # This method normalizes potential
    def normalize_data(self, pos_data, acc_data, U_data):
        # Compute adimensional potential for analytic model
        U_bc = self.bc_potential(pos_data)

        # Compute data radius
        r_data = np.linalg.norm(pos_data, axis=1)

        # Apply proxy potential formula
        U_prx = (U_data - U_bc) \
                * (r_data / self.r_ad) ** self.l_bc

        # Set adimensional proxy potential
        self.Uprx_ad = np.max(abs(U_prx))
        self.r_data = r_data
        self.Uprx_data = U_prx / self.Uprx_ad

        # Save boundary model potential
        self.U_bc = U_bc

        # Set adimensional proxy in training network
        self.network.acc_ad = np.max(abs(acc_data))
        self.network.Uprx_ad = self.Uprx_ad

    # This method saves model
    def save_model(self, path):
        # Set path to save
        parts = path.split('/')
        path_pinn = '/'.join(parts[:-1])
        path_pinnmodel = path_pinn + '/pinn_models'

        # Create path if it does not exist
        exist = os.path.exists(path_pinnmodel)
        if not exist:
            os.makedirs(path_pinnmodel)
        model_name = parts[-1].split('.')[0]

        # Set torch and onnx file names
        self.file_torch = path_pinnmodel + '/' + model_name + '.pt'
        self.file_onnx = path_pinnmodel + '/' + model_name + '.onnx'

        # Create only the pinn potential model
        self.network_eval = PINNeval(self.network, 'cpu')

        # Export the full model to Torch
        torch.save(self.network_eval, self.file_torch)

        # Export the model to ONNX format
        dummy_input = torch.tensor([[30.*1e3, 15*1e3, -15*1e3]])
        torch.onnx.export(self.network_eval, dummy_input,
                          self.file_onnx,
                          verbose=True,
                          input_names=["input"],
                          output_names=["output"])
