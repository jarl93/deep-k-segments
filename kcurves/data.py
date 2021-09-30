# libraries
from torch.utils.data import Dataset, DataLoader
import torch
from constants import DEVICE


class SyntheticDataset(Dataset):
    """
    Class to handle the synthetic data set.
    """
    def __init__(self, data, labels):
        """
        Initialization for the synthetic data set.
        Arguments:
            data: numpy array with the data.
            labels: labels of the data.
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        """
        Gets the length of the data set.
        Outputs:
           _ :length of the data set.
        """
        return (len(self.labels))

    def __getitem__(self, idx):
        """
        Gets the pair (x,y) given an index.
        Arguments:
            idx: index to retrieve the data and label.
        Outputs:
            x: data
            y: label
        """
        x = torch.from_numpy(self.data[idx]).type(torch.FloatTensor).to(DEVICE)
        y = self.labels[idx]

        return x, y