from torch.utils.data import Dataset


# Custom Dataset that returns data and indices
class TrainDataset(Dataset):
    def __init__(self, pos, acc, acc_bc):
        self.pos = pos
        self.acc = acc
        self.acc_bc = acc_bc

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        # Return data, label, and the index
        return self.pos[idx], self.acc[idx], self.acc_bc[idx]
