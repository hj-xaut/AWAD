import numpy as np
import torch
import torch.utils.data as data
from os import listdir
from os.path import join
import scipy.io as scio
import warnings
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])

class TrainsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir):
        super(TrainsetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir , x) for x in listdir(dataset_dir ) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        label = torch.from_numpy(mat['label'].astype(np.float32))
        lr = torch.from_numpy(mat['lr'].astype(np.float32))

        return lr, label

    def __len__(self):
        return len(self.image_filenames)

class ValsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir):
        super(ValsetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir , x) for x in listdir(dataset_dir ) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index])
        label = torch.from_numpy(mat['label'].astype(np.float32))
        lr = torch.from_numpy(mat['lr'].astype(np.float32))

        return lr, label

    def __len__(self):
        return len(self.image_filenames)