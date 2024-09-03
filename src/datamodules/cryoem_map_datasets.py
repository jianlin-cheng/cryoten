import gzip
import os
import mrcfile
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from monai.transforms import Randomizable, apply_transform
from monai.utils import MAX_SEED, get_seed
from monai.data.meta_tensor import MetaTensor
import glob

class CryoemDensityMapBlockTestDataset(Dataset):
    def __init__(self, dataset_dir, emdb_id):
        blocks_dir = os.path.join(dataset_dir, "test", "input")
        self.files = glob.glob(os.path.join(blocks_dir, emdb_id+"*.mrc"))
        self.block_indices = torch.load(os.path.join(dataset_dir, "block_indices", "emd_"+emdb_id+".pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        with mrcfile.open(file) as mrc:
            X = torch.from_numpy(np.copy(mrc.data))

        # Images generally have 3 channels (R, G, B). But, in cryoem maps, its just 1 channel 3D image of shape (h, w, d)
        # so reshape it as (c=1, h, w, d)
        X = X.unsqueeze(0)

        return X, file

class CryoemDensityMapBlockPredictDataset(Dataset):
    def __init__(self, blocks_dir):
        self.files = glob.glob(os.path.join(blocks_dir, "*.mrc"))
        self.block_indices = torch.load(os.path.join(blocks_dir, "block_indices.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        with mrcfile.open(file) as mrc:
            X = torch.from_numpy(np.copy(mrc.data))

        # Images generally have 3 channels (R, G, B). But, in cryoem maps, its just 1 channel 3D image of shape (h, w, d)
        # so reshape it as (c=1, h, w, d)
        X = X.unsqueeze(0)

        return X, file

class CryoemDensityMapBlockDataset(Dataset):
    def __init__(self, filename_list, input_maps_data_dir, target_maps_data_dir):
        self.filename_list = filename_list
        self.input_maps_data_dir = input_maps_data_dir
        self.target_maps_data_dir = target_maps_data_dir

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        filename = self.filename_list[idx]

        with mrcfile.open(os.path.join(self.input_maps_data_dir, filename)) as mrc:
            X = torch.from_numpy(np.copy(mrc.data))

        with mrcfile.open(os.path.join(self.target_maps_data_dir, filename)) as mrc:
            y = torch.from_numpy(np.copy(mrc.data))

        # Images generally have 3 channels (R, G, B). But, in cryoem maps, its just 1 channel 3D image of shape (h, w, d)
        # so reshape it as (c=1, h, w, d)
        X = X.unsqueeze(0)
        y = y.unsqueeze(0)

        return X, y, filename

class CryoemDensityMapBlockAugmentedDataset(Dataset, Randomizable):
    def __init__(
        self,
        filename_list,
        input_maps_data_dir,
        target_maps_data_dir,
        i_transform,
        t_transform,
    ):
        self.filename_list = filename_list
        self.input_maps_data_dir = input_maps_data_dir
        self.target_maps_data_dir = target_maps_data_dir
        self.set_random_state(seed=get_seed())
        self._seed = 0  # transform synchronization seed
        self.i_transform = i_transform
        self.t_transform = t_transform

    def __len__(self):
        return len(self.filename_list)

    def randomize(self):
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, idx):
        self.randomize()
        filename = self.filename_list[idx]
        
        with mrcfile.open(os.path.join(self.input_maps_data_dir, filename)) as mrc:
            X = torch.from_numpy(np.copy(mrc.data))

        with mrcfile.open(os.path.join(self.target_maps_data_dir, filename)) as mrc:
            y = torch.from_numpy(np.copy(mrc.data))

        # Images generally have 3 channels (R, G, B). But, in cryoem maps, its just 1 channel 3D image of shape (h, w, d)
        # so reshape it as (c=1, h, w, d)
        X = X.unsqueeze(0)
        y = y.unsqueeze(0)

        # apply the transforms
        if self.i_transform is not None:
            if isinstance(self.i_transform, Randomizable):
                self.i_transform.set_random_state(seed=self._seed)
            X = apply_transform(self.i_transform, X, map_items=False)

        if self.t_transform is not None:
            if isinstance(self.t_transform, Randomizable):
                self.t_transform.set_random_state(seed=self._seed)
            y = apply_transform(self.t_transform, y, map_items=False)

        return X, y, filename
