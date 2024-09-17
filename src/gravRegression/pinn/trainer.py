import numpy as np
import torch
import torch.optim as optim
from src.gravRegression.pinn.dataset import TrainDataset
from torch.utils.data import DataLoader


# This class manages the optimization
# of a physics-informed neural network
class Optimizer:
    def __init__(self, model):
        # Declare model to be trained
        self.model = model

        # Declare empty trainer and
        # its parameters
        self.trainer = []
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
        self.train_loader = []
        self.train_dataset = []

        # Set loss type
        self.loss = []

        # Set device
        self.device = 'cpu'

    # This method initializes trainer
    def initialize(self):
        # Declare Adam gradient descent
        self.trainer = optim.Adam([{'params': self.model.parameters()}],
                                  lr=self.lr,
                                  betas=(self.beta1, self.beta2),
                                  eps=self.eps,
                                  weight_decay=0)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total parameters:", total_params)

    # This method computes loss
    def compute_loss(self, pos, acc, acc_bc):
        # Get data
        n = self.batch_size
        acc_norm = torch.norm(acc, dim=1).unsqueeze(1)

        # Compute adimensional acceleration
        acc_perc = acc / acc_norm
        acc_ad = acc / self.model.acc_ad

        # Extract components and track derivatives
        pos = pos.clone().to(self.device,
                             dtype=torch.float32)

        # Compute gradient of the potential
        dU = self.model.gradient(pos)
        #pos_nograd = pos.detach().clone()
        #acc_M = self.model.model_bc.compute_acc(pos_nograd)
        dU_perc = (dU + acc_bc) / acc_norm
        dU_ad = (dU + acc_bc) / self.model.acc_ad

        # Compute loss function
        if self.loss_type == 'linear':
            loss_MAE = torch.sum(torch.norm(dU_perc - acc_perc, dim=1))
            loss_MSE = torch.sum(torch.norm(dU_ad - acc_ad, dim=1))
        elif self.loss_type == 'quadratic':
            loss_MAE = torch.sum(torch.norm(dU_perc - acc_perc, dim=1)**2)
            loss_MSE = torch.sum(torch.norm(dU_ad - acc_ad, dim=1)**2)

        loss = (loss_MAE + loss_MSE) / n

        return loss

    # This method initialises data
    def prepare_data(self, pos_data, acc_data):
        # Set data to tensors
        self.n = pos_data.shape[0]
        self.pos = torch.from_numpy(pos_data).float().to(
            self.device, dtype=torch.float32)
        self.acc = torch.from_numpy(acc_data).float().to(
            self.device, dtype=torch.float32)

        # Set training dataset and loader
        # self.train_dataset = TensorDataset(self.pos,
        #                                    self.acc)
        self.train_dataset = TrainDataset(self.pos,
                                          self.acc,
                                          self.acc_bc)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True)

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

                # Add data batch loss
                loss_sum += loss / len(self.train_loader)

            # Apply lr scheduler
            scheduler.step(loss_sum)

            # Save loss
            loss_np[k] = loss_sum
            print(k, loss_sum*1e2)

        # Save loss
        self.loss = loss_np
