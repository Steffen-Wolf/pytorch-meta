import os
import json
import glob
import h5py
from PIL import Image, ImageOps
import numpy as np

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import list_dir, download_url
from torchmeta.datasets.utils import get_asset


class FastCombinationMetaDataset(CombinationMetaDataset):
    def __init__(self, folder, num_classes_per_task, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, class_augmentations=None, target_transform=None,
                 dataset_transform=None):
        dataset = FastClassDataset(folder, meta_train=meta_train,
                                   meta_val=meta_val, meta_test=meta_test,
                                   transform=transform,
                                   meta_split=meta_split, class_augmentations=class_augmentations)
        super().__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class FastClassDataset(ClassDataset):

    def __init__(self, folder, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None,
                 class_augmentations=None):
        super(FastClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split)

        self.folder = folder
        self._num_classes = 20

    def __getitem__(self, index):
        """
        Each item from a `ClassDataset` is 
            a dataset containing examples from the same class.
        """
        return FastDataset(len(self.folder), index)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(self.split_filename, 'r')
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

class FastDataset(Dataset):
    def __init__(self, offset, index):
        self.offset = offset
        super().__init__(index)

    def __getitem__(self, index):
        print("loading index=",index, self.offset)
        return (index * np.ones((1, 28, 28)), self.offset)
        
    def __len__(self):
        return 10
