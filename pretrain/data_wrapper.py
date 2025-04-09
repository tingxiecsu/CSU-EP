# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 17:29:07 2024

@author: ZNDX002
"""
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from matchms.importing import load_from_mgf
from torch.utils.data import Dataset
from toolz.sandbox import unzip

import logging

logging.getLogger("matchms").setLevel(logging.ERROR)

def collate_func(input_list):
    input_ids,attention_masks,intens,num_peaks = map(list, unzip(input_list))
    num_peaks = torch.LongTensor(num_peaks)
    intens = [torch.from_numpy(spec_intens).float() for spec_intens in intens]
    input_ids = [torch.from_numpy(input_id).long() for input_id in input_ids]
    attention_masks = [torch.from_numpy(attention_mask).long() for attention_mask in attention_masks]
    input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=1002
        )
    intens_tensors = torch.nn.utils.rnn.pad_sequence(
            intens, batch_first=True, padding_value=0
        )
    attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
    return input_ids,attention_masks,intens_tensors,num_peaks

class ClrDataset(Dataset):
    """Contrastive Learning Representations Dataset."""

    def __init__(self, 
                spectra):

        self.spectra = spectra
        self.word_list = list(np.linspace(0,1001,1001,endpoint=False))
        self.word_list = [str(i) for i in self.word_list]
        #self.word2idx = {'[CLS]' : 1502, '[PAD]':1503,'[MASK]':1504}
        self.word2idx = { '[PAD]':1002,'[MASK]':1003}
        for i, w in enumerate(self.word_list):
            self.word2idx[w] = i + 1
            
    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spec = self.spectra[idx]
        spec_mz = spec.mz
        spec_intens = spec.intensities
        input_ids=np.array([self.word2idx[str(s)] for s in spec_mz])
        #input_ids=np.insert(input_ids, 0, 1502)
        #spec_intens=np.insert(spec_intens,0,2)
        #input_ids=np.append(input_ids, 1503)
        num_peak = len(input_ids)
        attention_mask = np.ones_like(input_ids)
        return input_ids,attention_mask,spec_intens,num_peak

class DataSetWrapper(object):
    def __init__(self,
                world_size,
                rank,
                batch_size, 
                num_workers, 
                valid_size, 
                ms_file):
        self.world_size = world_size
        self.valid_size = valid_size
        self.rank = rank
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ms_file = ms_file

    def get_data_loaders(self):
        
        # obtain training indices that will be used for validation
        self.ms = list(load_from_mgf(self.ms_file))
        num_train = len(self.ms)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_ms = [self.ms[i] for i in train_idx]
        self.valid_ms = [self.ms[i] for i in valid_idx]
        train_dataset = ClrDataset(self.train_ms)
        valid_dataset = ClrDataset(self.valid_ms)

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset,valid_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset,valid_dataset):
        train_sampler = DistributedSampler(train_dataset, num_replicas = self.world_size, rank=self.rank, shuffle = True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, 
                                                   sampler=train_sampler,collate_fn = collate_func)
        valid_sampler = DistributedSampler(valid_dataset, num_replicas = self.world_size, rank=self.rank, shuffle = False)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, 
                                                   sampler=valid_sampler,collate_fn = collate_func)

        return train_loader, valid_loader
