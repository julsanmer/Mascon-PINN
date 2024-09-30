import numpy as np
import torch
import torch.optim as optim
from src.gravRegression.nn.dataset import TrainDataset
from torch.utils.data import DataLoader, random_split


# This class manages the optimization
# of a physics-informed neural network
class Optimizer:
    def __init__(self, grav_nn):
        # Declare model to be trained
        self.grav_nn = grav_nn

        # Declare empty trainer and
        # its parameters
        self.trainer = None
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.lr = 1e-3
        self.eps = 1e-6
        self.maxiter = 1000
        self.loss = 'linear'

        # Internal training data
        self.n = []
        self.pos, self.acc = [], []
        self.acc_norm = []
        self.acc_ad = []
        self.acc_bc = []

        # Mini-batch
        self.batch_size = []
        self.train_loader = None
        self.val_loader = None
        self.train_dataset = None

        # Set loss type
        self.loss = []

        # Set device
        self.device = 'cpu'

    # This method initializes trainer
    def initialize(self):
        # Declare Adam gradient descent
        self.trainer = optim.Adam([{'params': self.grav_nn.parameters()}],
                                  lr=self.lr,
                                  betas=(self.beta1, self.beta2),
                                  eps=self.eps,
                                  weight_decay=0)
        total_params = sum(p.numel() for p in self.grav_nn.parameters() if p.requires_grad)
        print("Total parameters:", total_params)

    # This method computes loss
    def compute_loss(self, pos_data, acc_data, acc_bc):
        # Get data
        n = self.batch_size
        accdata_norm = torch.norm(acc_data, dim=1).unsqueeze(1)

        # Compute adimensional acceleration
        accdata_perc = acc_data / accdata_norm
        accdata_abs = acc_data / self.grav_nn.acc_ad

        # Extract components and track derivatives
        pos_data = pos_data.clone().to(self.device,
                                       dtype=torch.float32)

        # # Compute gradient of the potential
        # acc_nn = self.grav_nn.compute_acc(pos_data)
        # acc_perc = (acc_nn + acc_bc) / accdata_norm
        # acc_abs = (acc_nn + acc_bc) / self.grav_nn.acc_ad

        # Compute gradient of the potential
        acc = self.grav_nn.compute_acc(pos_data)
        acc_perc = acc / accdata_norm
        acc_abs = acc / self.grav_nn.acc_ad

        # Compute loss function
        if self.loss_type == 'linear':
            loss_rel = torch.sum(torch.norm(acc_perc - accdata_perc, dim=1))
            loss_abs = torch.sum(torch.norm(acc_abs - accdata_abs, dim=1))
        elif self.loss_type == 'quadratic':
            loss_rel = torch.sum(torch.norm(acc_perc - accdata_perc, dim=1)**2)
            loss_abs = torch.sum(torch.norm(acc_abs - accdata_abs, dim=1)**2)

        loss = (loss_abs + loss_rel) / n

        return loss

    # This method initialises data
    def prepare_data(self, pos_data, acc_data):
        # Set data to tensors
        self.n = pos_data.shape[0]
        self.pos = torch.from_numpy(pos_data).float().to(
            self.device, dtype=torch.float32)
        self.acc = torch.from_numpy(acc_data).float().to(
            self.device, dtype=torch.float32)

        # Custom dataset
        self.dataset = TrainDataset(self.pos,
                                    self.acc,
                                    self.acc_bc)

        # Define the split ratio (80/20 in this case)
        train_ratio = 0.8
        train_size = int(train_ratio * len(self.dataset))
        val_size = len(self.dataset) - train_size

        # Split the dataset into training and validation sets
        train_dataset, val_dataset = \
            random_split(self.dataset, [train_size, val_size])

        # Create DataLoader for the training dataset
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True)

        # Create DataLoader for the validation dataset
        self.val_loader = DataLoader(val_dataset,
                                     batch_size=val_size,
                                     shuffle=False)

    # This method trains a PINN based on a
    # position-gravity dataset
    def train(self, pos_data, acc_data):
        # Preprocess data
        self.prepare_data(pos_data, acc_data)

        # Create lr scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.trainer,
                                                         factor=0.5,
                                                         patience=50,
                                                         mode='min')

        # Preallocate loss to save
        loss_np = np.zeros(self.maxiter)

        # Loop through training
        for k in range(self.maxiter):
            # Initialize epoch loss
            loss_sum = 0

            # Loop through data batches
            for pos, acc, acc_bc in self.train_loader:
            #for pos, acc in self.train_loader:
                # Compute loss and reset optimizer gradient
                loss = self.compute_loss(pos, acc, acc_bc)
                self.trainer.zero_grad()

                # Backpropagate and do gradient step
                loss.backward()
                self.trainer.step()

            ## Set model to evaluation mode
            ##self.trainer.eval()

            # Loop through validation data batches
            for pos_val, acc_val, acc_bc_val in self.val_loader:
                # Compute validation loss
                loss_val = self.compute_loss(pos_val, acc_val, acc_bc_val)

            # Apply lr scheduler
            scheduler.step(loss_val)

            # Save loss
            loss_np[k] = loss_val
            print(k, loss_val*1e2)

        # Save loss
        self.loss = loss_np
