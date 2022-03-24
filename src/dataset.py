import torch
from torch.utils.data import Dataset, DataLoader

class AlsDataset(Dataset):
    def __init__(self, X, y):
      #data loading
      self.x = torch.tensor(X.values, dtype=torch.float)
      self.y = torch.tensor(y.values, dtype=torch.float)
      self.n_samples = X.shape[0]

    def __len__(self):
      # len(dataset)
      return self.n_samples

    def __getitem__(self, index):
      # dataset[0]
      return self.x[index], self.y[index]

    def get_label_count(self):
      return self.y.unique()
