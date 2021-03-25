import random
import warnings
from itertools import combinations
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data import ConcatDataset

from torchmeta.utils.data.dataset import CombinationMetaDataset

__all__ = ['CombinationSequentialSampler', 'CombinationRandomSampler']


class CombinationSequentialSampler(SequentialSampler):
    def __init__(self, data_source):
        if not isinstance(data_source, CombinationMetaDataset):
            raise TypeError('Expected `data_source` to be an instance of '
                            '`CombinationMetaDataset`, but found '
                            '{0}'.format(type(data_source)))
        super(CombinationSequentialSampler, self).__init__(data_source)

    def __iter__(self):
        num_classes = len(self.data_source.dataset)
        num_classes_per_task = self.data_source.num_classes_per_task
        return combinations(range(num_classes), num_classes_per_task)


class CombinationRandomSampler(RandomSampler):
    def __init__(self, data_source):
        if not isinstance(data_source, CombinationMetaDataset):
            raise TypeError('Expected `data_source` to be an instance of '
                            '`CombinationMetaDataset`, but found '
                            '{0}'.format(type(data_source)))
        # Temporarily disable the warning if the length of the length of the 
        # dataset exceeds the machine precision. This avoids getting this
        # warning shown with MetaDataLoader, even though MetaDataLoader itself
        # does not use the length of the dataset.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            super(CombinationRandomSampler, self).__init__(data_source,
                                                           replacement=True)

    def __iter__(self):
        num_classes = len(self.data_source.dataset)
        num_classes_per_task = self.data_source.num_classes_per_task
        for _ in combinations(range(num_classes), num_classes_per_task):
            print(tuple(random.sample(range(num_classes), num_classes_per_task)))
            yield tuple(random.sample(range(num_classes), num_classes_per_task))


class MultiCombinationRandomSampler(RandomSampler):
    def __init__(self, list_of_data_sources):
        # Temporarily disable the warning if the length of the length of the
        # dataset exceeds the machine precision. This avoids getting this
        # warning shown with MetaDataLoader, even though MetaDataLoader itself
        # does not use the length of the dataset.
        self.list_of_data_sources = list_of_data_sources
        self.samplers = [CombinationRandomSampler(ds) for ds in list_of_data_sources]
        
        # determine the index ranges that belong to the same dataset
        self.sampeling_info = []
        start_index = 0
        self.total_combinations = 0
        for ds in list_of_data_sources:
            dataset_sampel_info = {}
            dataset_sampel_info["num_classes"] = len(ds.dataset)
            dataset_sampel_info["num_classes_per_task"] = ds.num_classes_per_task
            dataset_sampel_info["start_index"] = start_index
            start_index += dataset_sampel_info["num_classes"]

            self.total_combinations += sum(1 for _ in combinations(
                                           range(dataset_sampel_info["num_classes"]),
                                           dataset_sampel_info["num_classes_per_task"]))

            self.sampeling_info.append(dataset_sampel_info)


        super().__init__(ConcatDataset(list_of_data_sources),
                         replacement=True)

    def __iter__(self):

        for _ in range(self.total_combinations):
            # sample a random dataset
            sinfo = random.choice(self.sampeling_info)
            # sample a random combination of indexes
            # starting at the dataset index
            start_index = sinfo["start_index"]
            end_index = sinfo["start_index"] + sinfo["num_classes"]
            num_classes_per_task = sinfo["num_classes_per_task"]

            index_range = range(start_index, end_index)
            index_combination = tuple(random.sample(index_range, num_classes_per_task))
            print(index_combination)
            yield index_combination
