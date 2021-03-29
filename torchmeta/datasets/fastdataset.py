import os
import json
import glob
import h5py
from PIL import Image, ImageOps
import numpy as np
import zarr

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

    def __init__(self, folders, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None,
                 class_augmentations=None):
        super(FastClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split)

        self.folders = folders
        self.transform = transform
        self._data = None
        # trigger the data loading immediately
        print(len(self.data))


    def __getitem__(self, index):
        """
        Each item from a `ClassDataset` is 
            a dataset containing examples from the same class.
        """
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return FastDataset(self.data[index], index,
                           transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return len(self.data)

    @property
    def data(self):
        if self._data is None:
            gt_zarr = zarr.open(
                self.folders["gt_segmentation"][0], "r")
            gt_key = self.folders["gt_segmentation"][1]
            gt_segmentation = gt_zarr[gt_key][:]

            emb_zarr = zarr.open(
                self.folders["embedding"][0], "r")
            emb_key = self.folders["embedding"][1]
            emb_segmentation = emb_zarr[emb_key][0]

            x = np.arange(gt_segmentation.shape[-1], dtype=np.float32)
            y = np.arange(gt_segmentation.shape[-2], dtype=np.float32)

            coords = np.meshgrid(x, y, copy=True)
            emb_segmentation = np.concatenate([coords[0:1],
                                               coords[1:2],
                                               emb_segmentation], axis=0)

            self._data = []
            for idx in np.unique(gt_segmentation):
                mask = gt_segmentation == idx
                if mask.sum() > self.folders["min_samples"]:
                    instance_embedding = np.transpose(emb_segmentation[:, mask])
                    instance_embedding = instance_embedding.astype(np.float32)
                    self._data.append(instance_embedding)
                else:
                    print("skipping", idx, mask.sum())
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels


class FastDataset(Dataset):
    def __init__(self, data, target,
                 transform=None, target_transform=None):
        super(FastDataset, self).__init__(target, transform=transform,
                                              target_transform=target_transform)
        self.target = target
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target = self.target
        embedding = self.data[index]

        if self.transform is not None:
            embedding = self.transform(embedding)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return embedding, target
