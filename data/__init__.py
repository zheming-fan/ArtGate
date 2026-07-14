import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from .datasets import dataset_folder
from .datasets import preprocess_artgate_image, read_data_artgate

def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler




    



def create_dataloader_artgate(opt):
    shuffle = True if opt.isTrain else False
    dataset = read_data_artgate(opt)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              num_workers=int(0))
    return data_loader


def create_dataloader_test_artgate(opt, max_real_size=None, max_fake_size=None):
    shuffle = True if opt.isTrain else False

    dataset = read_data_artgate(opt, max_real_size=max_real_size, max_fake_size=max_fake_size)


    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              num_workers=int(0))
    return data_loader
