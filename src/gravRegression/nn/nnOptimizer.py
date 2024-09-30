import numpy as np
import os
import torch

from timeit import default_timer as timer

from src.gravRegression.gravityOptimizer import GravityOptimizer
from src.gravRegression.nn.trainer import Optimizer
from src.gravRegression.nn import Mascon
from src.celestialBodies.gravityModels.nn.nn_eval import NNeval
from src.celestialBodies.gravityModels.nn.pinn_eval import PINNeval
from src.celestialBodies.gravityModels.nn.weight import Weight

# Conversion constants
deg2rad = np.pi/180
km2m = 1e3


# This is the PINN class
class NNOptimizer(GravityOptimizer):
    def __init__(self, file_torch=None):
        super().__init__()

        # Loss function and training time
        self.t_train = []
        self.loss = []

        # Define optimizer and network
        self.optimizer = None
        self.grav_nn = None

        # Set saving files
        self.file_onnx = []
        self.file_torch = file_torch

    # This method prepares pinn optimizer
    def prepare_optimizer(self, maxiter=1000, lr=1e-3, batch_size=1,
                          loss_type='linear'):
        # Initialize mascon fit module
        self.optimizer = Optimizer(self.grav_nn)

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
        self.compute_adparams(pos_data, acc_data, U_data)

        # Start measuring cpu time
        t_start = timer()

        # Call optimizer
        self.optimizer.acc_bc = self.grav_nn.acc_bc
        self.optimizer.train(pos_data, acc_data)

        # End measuring cpu time
        t_end = timer()
        self.t_train.append(t_end - t_start)

        # Save loss
        self.loss = self.optimizer.loss

    # This method computes proxy potential
    def compute_proxy(self, pos_data):
        # Compute data radius
        r_data = np.linalg.norm(pos_data, axis=1)

        # Transform position and radius to tensors
        pos_data = torch.from_numpy(pos_data).to(dtype=torch.float32)
        r_data = torch.from_numpy(r_data).to(dtype=torch.float32)

        # Do a forward call of the potential
        U_prx = self.grav_nn.forward(pos_data, r_data)

        # Set proxy potential
        self.Uprx = U_prx.detach().numpy()

    # This method initializes network for training
    def init_network(self, n_layers=8, n_neurons=40,
                     activation='SIREN', model='PINN',
                     eval_mode='BSK'):
        # Set network
        if model == 'PINN':
            self.grav_nn = PINNeval(n_layers,
                                    n_neurons,
                                    activation,
                                    device='cpu')
        elif model == 'NN':
            self.grav_nn = NNeval(n_layers,
                                  n_neurons,
                                  activation,
                                  device='cpu')
        else:
            raise ValueError(f"Unsupported gravNN")

    # This method sets adimensional parameters
    def set_extra_params(self, R=1., r_bc=1.,
                         k_bc=1., l_bc=1.):
        # Adimensionalization for inputs
        self.grav_nn.R = R

        # Switch function variables
        self.grav_nn.r_bc = r_bc
        self.grav_nn.k_bc = k_bc
        self.grav_nn.l_bc = l_bc

    # This method adds a boundary model
    def add_mascon(self, mu_M, xyz_M):
        # Add mascon model as bc
        self.grav_nn.model_bc = Mascon(mu_M,
                                       xyz_M)

    # This method adds a weight to nn model
    def add_wnn(self, k_bc, r_bc, R):
        #
        self.grav_nn.w_nn = Weight(k_bc, r_bc, R)

    # This method adds a weight to lf model
    def add_wlf(self, k_lf, r_lf, R):
        #
        self.grav_nn.w_lf = Weight(k_lf, r_lf, R)

    # This method normalizes dataset
    def compute_adparams(self, pos_data,
                         acc_data,
                         U_data):
        # Compute potential and acceleration
        # for boundary model
        U_bc = self.grav_nn.model_bc.compute_U(
            torch.from_numpy(pos_data)).numpy()
        acc_bc = self.grav_nn.model_bc.compute_acc(
            torch.from_numpy(pos_data)).numpy()

        # Compute data radius
        r_data = np.linalg.norm(pos_data, axis=1)
        r_data_ad = r_data / self.grav_nn.R

        # Compute discrepancy between bc model and
        # ground truth
        U_err = U_data - U_bc
        acc_err = acc_data - acc_bc

        # Apply the condition: U *= r where r > 1
        U_prx = np.where(r_data_ad > 1,
                         U_err * r_data_ad ** self.grav_nn.l_bc,
                         U_err)
        # import matplotlib.pyplot as plt
        # plt.plot(r_data_ad, U_prx, '.')
        # plt.show()
        acc_prx = np.where(r_data_ad[:, np.newaxis] > 1,
                           acc_err * r_data_ad[:, np.newaxis] ** self.grav_nn.l_bc,
                           acc_err)

        # Set adimensional proxy potential
        self.grav_nn.Uprx_ad = np.max(abs(U_prx))
        self.grav_nn.r_data = r_data
        self.grav_nn.Uprx_data = U_prx / self.grav_nn.Uprx_ad

        # Save boundary model potential
        self.grav_nn.U_bc = U_bc
        self.grav_nn.acc_bc = acc_bc

        # Set adimensional proxy in training network
        self.grav_nn.acc_ad = np.max(abs(acc_data))
        self.grav_nn.accprx_ad = np.max(np.linalg.norm(acc_prx,
                                                       axis=1))

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
        #self.network_eval = PINNeval(self.network, 'cpu')

        # Export the full model to Torch
        self.grav_nn.graph = False
        torch.save(self.grav_nn, self.file_torch)

        # Export the model to ONNX format
        # dummy_input = torch.tensor([[30.*1e3, 15*1e3]])
        # torch.onnx.export(self.network, dummy_input,
        #                   self.file_onnx,
        #                   verbose=True,
        #                   input_names=["input"],
        #                   output_names=["output"])
