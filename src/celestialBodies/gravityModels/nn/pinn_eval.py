import torch
import torch.autograd as autograd
import torch.nn as nn


# This defines the PINN class for training
class PINNeval(nn.Module):
    def __init__(self, n_layers,
                 n_neurons,
                 activation,
                 device='cpu'):
        super().__init__()

        # Set device
        self.device = device

        # Set layers, neurons and activation
        layers = []
        for i in range(n_layers + 2):
            if i == 0:
                layers.append(5)
            elif i == n_layers + 1:
                layers.append(1)
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

        # Switch models
        self.w_nn = None
        self.w_lf = None

        # Set if create graph is necessary
        self.graph = True

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
        U_prx = self.linears[-1](input)

        return U_prx

    # This method computes the potential
    def compute_U(self, pos):
        # This is the proxy transform
        def rescale_Uprx(U_prx, r_ad, l_bc=1.):
            # Rescale proxy potential
            U_nn = torch.where(r_ad > 1,
                               U_prx / (r_ad**l_bc),
                               U_prx)

            return U_nn

        # Compute radius
        r = torch.norm(pos, dim=1)

        # Compute PINN potential
        U_prx = self.forward(pos, r)

        # For further computations we need
        # to unsqueeze these variables
        r = torch.unsqueeze(r, dim=1)

        # Rescale proxy potential
        U_prx *= self.Uprx_ad
        U_nn = rescale_Uprx(U_prx,
                            r/self.R,
                            l_bc=self.l_bc)

        # Compute nn and bc weights
        w_bc, w_nn = self.w_nn.compute_w(r)
        w_lf, _ = self.w_lf.compute_w(r)

        #
        U_bc = self.model_bc.compute_U(pos).unsqueeze(1)

        # Do total potential
        U = w_nn*(U_nn+U_bc) + w_bc*U_bc

        return U

    # This computes the gradient of the potential
    def compute_acc(self, pos):
        # Track gradient w.r.t. position
        pos.requires_grad = True

        # Compute potentials
        U = self.compute_U(pos)

        # Compute gradient
        # acc_nn = autograd.grad(U, pos,
        #                        torch.ones([pos.shape[0], 1]).to(self.device),
        #                        create_graph=True,
        #                        retain_graph=True)[0]
        acc = autograd.grad(U, pos,
                            torch.ones([pos.shape[0], 1]).to(self.device),
                            create_graph=self.graph,
                            retain_graph=self.graph)[0]
        #acc_bc = self.model_bc.compute_acc(pos)

        return acc
