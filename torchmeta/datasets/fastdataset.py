import os
import json
import glob
import h5py
from PIL import Image, ImageOps
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
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
        self._semantic_class = None
        # trigger the data loading immediately
        self.data


    def __getitem__(self, index):
        """
        Each item from a `ClassDataset` is 
            a dataset containing examples from the same class.
        """
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return FastDataset(self.data[index], index, self.semantic_class[index],
                           transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return len(self.data)

    def duplicate_and_interpolate(self, embeddings):
        mixed = embeddings[np.random.permutation(len(embeddings))]
        return np.concatenate((embeddings, (embeddings + mixed) / 2), axis=0)

    def extract_data(self):
        gt_zarr = zarr.open(
            self.folders["gt_segmentation"][0], "r")
        gt_key = self.folders["gt_segmentation"][1]
        gt_segmentation = gt_zarr[gt_key][:]

        emb_segmentation = []
        for ef in self.folders["embedding"]:
            emb_zarr = zarr.open(ef[0], "r")
            emb_key = ef[1]
            emb_segmentation.append(emb_zarr[emb_key][0])
        emb_segmentation = np.concatenate(emb_segmentation, axis=0)

        x = np.arange(gt_segmentation.shape[-1], dtype=np.float32)
        y = np.arange(gt_segmentation.shape[-2], dtype=np.float32)

        coords = np.meshgrid(x, y, copy=True)
        emb_segmentation = np.concatenate([coords[0:1],
                                            coords[1:2],
                                            emb_segmentation], axis=0)

        self._data = []
        self._semantic_class = []

        bg_mask = gt_segmentation == 0
        for idx in np.unique(gt_segmentation):
            mask = gt_segmentation == idx
            instance_embedding = np.transpose(emb_segmentation[:, mask])
            instance_embedding = instance_embedding.astype(np.float32)
            
            if len(instance_embedding) <= 1:
                continue

            while len(instance_embedding) < self.folders["min_samples"]:
                print("extending", len(instance_embedding))
                instance_embedding = self.duplicate_and_interpolate(instance_embedding)
                print("to ", len(instance_embedding))

            self._data.append(instance_embedding)
            if idx == 0:
                # we assume that the background instance is
                # always at index zero
                self._semantic_class.append(0)
            else:
                self._semantic_class.append(1)

                # add a background instance in close proximity to the object
                background_distance = distance_transform_edt(
                    gt_segmentation != idx)
                bg_close_to_instance_mask = np.logical_and(background_distance < self.folders["bg_distance"],
                                                        bg_mask)
                bg_instance_embedding = np.transpose(
                    emb_segmentation[:, bg_close_to_instance_mask])
                bg_instance_embedding = bg_instance_embedding.astype(np.float32)

                if bg_close_to_instance_mask.sum() > self.folders["min_samples"]:
                    self._data.append(bg_instance_embedding)
                    self._semantic_class.append(0)

    @property
    def data(self):
        if self._data is None:
            self.extract_data()
        return self._data

    @property
    def semantic_class(self):
        if self._semantic_class is None:
            self.extract_data()
        return self._semantic_class

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels


class FastDataset(Dataset):
    def __init__(self, data, target, semantic_class,
                 transform=None, target_transform=None):
        super(FastDataset, self).__init__(target, transform=transform,
                                              target_transform=target_transform)
        self.target = target
        self.semantic_class = semantic_class
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

        return embedding, target, self.semantic_class
