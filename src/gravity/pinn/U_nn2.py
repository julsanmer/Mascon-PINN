import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

from src.gravity.pinn.model_bc import Mascon_BC


# This defines the PINN class for training
class PINNtrain(nn.Module):
    def __init__(self, params, device='cpu'):
        super().__init__()

        # Set device
        self.device = device

        # Set number of layers and activation function
        self.layers = params.layers
        self.is_SIREN = False
        if params.activation == 'GELU':
            self.activation = nn.GELU()
        elif params.activation == 'SiLU':
            self.activation = nn.SiLU()
        elif params.activation == 'SIREN':
            self.is_SIREN = True

        # Initialise neural network as a list
        self.linears = nn.ModuleList(
            [nn.Linear(self.layers[i], self.layers[i+1])
             for i in range(len(self.layers)-1)])

        # Initialise weights and biases
        self.initialise_weights_biases()

        # Set adimensional parameters
        self.r_ad = params.r_ad
        self.Uprx_ad = params.Uprx_ad

        # Set boundary condition transition
        self.rad_bc = params.rad_bc
        self.l_bc = params.l_bc
        self.r_bc = params.r_bc
        self.k_bc = params.k_bc

        # Set boundary model
        self.bc_type = params.model_bc.type
        self.model_bc = Mascon_BC(mu_M=np.array([np.sum(params.model_bc.mu_M)]),
                                  xyz_M=np.array([[0, 0, 0]]),
                                  device='cpu')

        # Initialize data
        self.U_bc = []

    # This method initialises weights and biases
    def initialise_weights_biases(self):
        # Set seed
        torch.manual_seed(0)

        # Loop through layers
        for i in range(len(self.layers)-1):
            # Initialise weights and biases
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

        # Set last layer weights to zero
        self.linears[-1].weight.data.zero_()

    # This method forwards the pinn
    def forward(self, pos, r):
        # Preallocate pinn inputs
        input = torch.zeros(pos.shape[0], 5).to(
            self.device, dtype=torch.float32)

        # Normalise inputs
        x_ad = pos[:, 0] / r
        y_ad = pos[:, 1] / r
        z_ad = pos[:, 2] / r

        # Set interior and exterior features
        ri = torch.clamp(r / self.rad_bc, min=None, max=1.)
        re = torch.clamp(self.rad_bc / r, min=None, max=1.)

        # Set input
        input[:, 0] = x_ad
        input[:, 1] = y_ad
        input[:, 2] = z_ad
        input[:, 3] = ri
        input[:, 4] = re

        # Forward layers
        for i in range(len(self.layers)-2):
            # Apply weight and biases
            z = self.linears[i](input)

            # Apply activation function
            if self.is_SIREN:
                input = torch.sin(z)
            else:
                input = self.activation(z)

        # Get pinn proxy potential
        U = self.linears[-1](input)

        return U

    # This method computes the potential
    def potential(self, pos):
        # This is the switch function
        def compute_switch(r_ad, k_bc=2., r_bc=1.):
            # Compute switch function
            h = k_bc * (r_ad-r_bc)
            H = (1 + torch.tanh(h))/2

            return H

        # This is the proxy transform
        def rescale_proxy(U_prx, r_ad, l_bc=1.):
            # Create a mask
            mask = r_ad > 1
            U_prx[mask] *= r_ad[mask]**l_bc
            U_pinn = U_prx

            return U_pinn

        # Compute radius
        r = torch.norm(pos, dim=1)

        # Compute PINN potential
        U_prx = self.forward(pos, r)

        # Compute boundary potential
        U_bc = self.model_bc.compute_potential(pos)

        # For further computations we need
        # to unsqueeze these variables
        r = torch.unsqueeze(r, dim=1)
        U_bc = torch.unsqueeze(U_bc, dim=1)

        # Rescale proxy potential
        U_prx *= self.Uprx_ad
        U_pinn = rescale_proxy(U_prx, r/self.rad_bc, l_bc=self.l_bc)

        # Assign weights
        H_nn = compute_switch(r/self.rad_bc,
                              k_bc=self.k_bc,
                              r_bc=self.r_bc/self.rad_bc)
        w_nn = 1 - H_nn
        H_bc = compute_switch(r/self.rad_bc,
                              k_bc=0.5,
                              r_bc=1)
        w_bc = H_bc

        # Do total potential
        U = w_nn*U_pinn + w_bc*U_bc

        return U

    # This computes the gradient of the potential
    def gradient(self, pos):
        # If it is not a tensor
        if not torch.is_tensor(pos):
            pos = torch.from_numpy(pos)

        # Track gradient w.r.t. position
        pos.requires_grad = True

        # Compute potentials
        U = self.potential(pos)

        # Compute gradient
        dU = autograd.grad(U, pos,
                           torch.ones([pos.shape[0], 1]).to(self.device),
                           create_graph=True)[0]

        return dU


# This defines the PINN class
# for evaluation
class PINNeval(nn.Module):
    def __init__(self, input_pinn, device):
        super().__init__()

        # Set device
        self.device = device

        # Set number of layers and activation function
        self.layers = input_pinn.layers
        self.is_SIREN = False
        if input_pinn.is_SIREN:
            self.is_SIREN = True
        else:
            self.activation = input_pinn.activation

        # Initialise neural network as a list
        self.linears = nn.ModuleList(
            [nn.Linear(self.layers[i], self.layers[i+1])
             for i in range(len(self.layers)-1)])

        # Copy weights and biases
        self.load_state_dict(input_pinn.state_dict(),
                             strict=True)

        # Set adimensional parameters
        self.r_ad = input_pinn.r_ad
        self.Uprx_ad = input_pinn.Uprx_ad

        # Set boundary condition transition
        self.rad_bc = input_pinn.rad_bc
        self.l_bc = input_pinn.l_bc
        self.r_bc = input_pinn.r_bc
        self.k_bc = input_pinn.k_bc

    # This forwards the pinn
    def forward(self, pos):
        # This is the switch function
        def compute_switch(r_ad, k_bc=2., r_bc=1.):
            # Compute switch function
            h = k_bc * (r_ad-r_bc)
            H = (1 + torch.tanh(h))/2

            return H

        # This is the proxy transform
        def rescale_proxy(U_prx, r_ad, l_bc=1.):
            # Rescale proxy potential
            U_pinn = U_prx / (r_ad**l_bc)

            return U_pinn

        # Preallocate pinn inputs
        input = torch.zeros(pos.shape[0], 4).to(
            self.device, dtype=torch.float32)

        # Normalise inputs
        r = torch.norm(pos)
        x_ad = pos[:, 0] / r
        y_ad = pos[:, 1] / r
        z_ad = pos[:, 2] / r

        # Set r_ad/r
        input[:, 3] = self.r_ad / r

        # Set input
        input[:, 0] = x_ad
        input[:, 1] = y_ad
        input[:, 2] = z_ad

        # Forward layers
        for i in range(len(self.layers)-2):
            # Apply weight and biases
            z = self.linears[i](input)

            # Apply activation function
            if self.is_SIREN:
                input = torch.sin(z)
            else:
                input = self.activation(z)

        # Get pinn proxy potential
        U_prx = self.linears[-1](input)

        # Rescale proxy potential
        U_prx *= self.Uprx_ad
        U_pinn = rescale_proxy(U_prx,
                               r/self.rad_bc,
                               l_bc=self.l_bc)

        # Assign weights
        H_nn = compute_switch(r/self.rad_bc,
                              k_bc=self.k_bc,
                              r_bc=self.r_bc/self.rad_bc)
        w_nn = 1 - H_nn

        # Weight pinn potential
        U = w_nn * U_pinn

        return U
