import torch
import torch.nn as nn

from src.gravRegression.nn.bcModels.mascon import Mascon


# This defines the PINN class for training
class NNeval(nn.Module):
    def __init__(self, n_layers,
                 n_neurons,
                 activation,
                 device='cpu'):
        super().__init__()

        # Set device
        self.device = device

        # Define layers
        layers = []
        for i in range(n_layers + 2):
            # Input layer
            if i == 0:
                layers.append(5)
            # Output layer
            elif i == n_layers + 1:
                layers.append(3)
            # Hidden layers
            else:
                layers.append(n_neurons)
        self.layers = layers

        # Set activation function
        self.is_SIREN = False
        if activation == 'GELU':
            self.activation = nn.GELU()
        elif activation == 'SiLU':
            self.activation = nn.SiLU()
        elif activation == 'SIREN':
            self.is_SIREN = True

        # Initialise neural network as a list
        self.linears = nn.ModuleList(
            [nn.Linear(self.layers[i], self.layers[i+1])
             for i in range(len(self.layers)-1)])

        # Initialise weights and biases
        self.initialise_weights_biases()

        # Set adimensional parameters
        self.R = []
        self.Uprx_ad = []

        # Set boundary condition transition
        self.l_bc = []
        self.r_bc = []
        self.k_bc = []

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
        input = torch.zeros(pos.shape[0], self.layers[0]).to(
            self.device, dtype=torch.float32)

        # Normalise inputs
        x_ad = pos[:, 0] / r
        y_ad = pos[:, 1] / r
        z_ad = pos[:, 2] / r

        # Set r_e and r_i
        input[:, 3] = torch.clamp(self.R/r, max=1.0)
        input[:, 4] = torch.clamp(r/self.R, max=1.0)

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
        acc = self.linears[-1](input)

        # #
        # acc_norm = output[:, 0]
        # ex = torch.clamp(output[:, 1], min=-1.0, max=1.0)
        # ey = torch.clamp(output[:, 2], min=-1.0, max=1.0)
        # ez = torch.clamp(output[:, 3], min=-1.0, max=1.0)

        # acc = torch.zeros_like(pos)
        # acc[:, 0] = acc_norm*ex
        # acc[:, 1] = acc_norm*ey
        # acc[:, 2] = acc_norm*ez

        return acc

    # This method computes the potential
    def compute_U(self, pos):
        pass

    # This computes the gradient of the potential
    def compute_acc(self, pos):
        # This is the proxy transform
        def rescale_accprx(acc_prx, r_ad, l_bc=1.):
            # Rescale proxy potential
            acc_pinn = acc_prx / (r_ad**l_bc)

            return acc_pinn

        # Compute radius
        r = torch.norm(pos, dim=1)

        # Compute PINN acceleration
        acc_prx = self.forward(pos, r)

        # For further computations we need
        # to unsqueeze these variables
        r = torch.unsqueeze(r, dim=1)

        # Rescale proxy potential
        acc_prx *= self.accprx_ad
        acc_nn = rescale_accprx(acc_prx,
                                r/self.R,
                                l_bc=self.l_bc)

        ## Compute nn and bc weights
        #w_nn, w_bc = self.w_nn.compute_w(r)
        #_, w_lf = self.w_lf.compute_w(r)

        acc_bc = self.model_bc.compute_acc(pos)

        #acc = w_nn*(acc_nn + acc_bc) + w_bc*acc_bc

        acc = torch.where(r < self.R, acc_nn, acc_bc)

        return acc
