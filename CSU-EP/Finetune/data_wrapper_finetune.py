
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
import random
from matchms.Fragments import Fragments
import matchms.filtering as msfilters


def collate_func(input_list):
    input_ids,attention_masks,intens,num_peaks,input_ids_pre,attention_masks_pre,intens_pre,num_peaks_pre = map(list, unzip(input_list))
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
    
    num_peaks_pre = torch.LongTensor(num_peaks_pre)
    intens_pre = [torch.from_numpy(spec_intens).float() for spec_intens in intens_pre]
    input_ids_pre = [torch.from_numpy(input_id).long() for input_id in input_ids_pre]
    attention_masks_pre = [torch.from_numpy(attention_mask).long() for attention_mask in attention_masks_pre]
    input_ids_pre = torch.nn.utils.rnn.pad_sequence(
            input_ids_pre, batch_first=True, padding_value=1002
        )
    intens_tensors_pre = torch.nn.utils.rnn.pad_sequence(
            intens_pre, batch_first=True, padding_value=0
        )
    attention_masks_pre = torch.nn.utils.rnn.pad_sequence(
            attention_masks_pre, batch_first=True, padding_value=0
        )
    return input_ids,attention_masks,intens_tensors,num_peaks,input_ids_pre,attention_masks_pre,intens_tensors_pre,num_peaks_pre

class ClrDataset(Dataset):
    """Contrastive Learning Representations Dataset."""

    def __init__(self, 
                spectra_real,
                spectra_pre):

        self.spectra_real = spectra_real
        self.spectra_pre = spectra_pre

        self.word_list = list(np.linspace(0,1001,1001,endpoint=False))
        self.word_list = [str(i) for i in self.word_list]
        self.word2idx = {'[PAD]':1002,'[MASK]':1003}
        for i, w in enumerate(self.word_list):
            self.word2idx[w] = i + 1
            
    def __len__(self):
        return len(self.spectra_real)

    def __getitem__(self, idx):
        spec_real = self.spectra_real[idx]
        spec_pre = self.spectra_pre[idx]
        
        spec_mz = spec_real.mz
        spec_intens = spec_real.intensities
        input_ids=[self.word2idx[str(s)] for s in spec_mz]
        input_ids=np.array(input_ids)
        num_peak = len(input_ids)
        attention_mask = np.ones_like(input_ids)
        
        spec_pre_mz = spec_pre.mz
        spec_pre_intens = spec_pre.intensities
        input_ids_pre=[self.word2idx[str(s)] for s in spec_pre_mz]
        input_ids_pre=np.array(input_ids_pre)
        num_peak_pre = len(input_ids_pre)
        attention_mask_pre = np.ones_like(input_ids_pre)
        return input_ids,attention_mask,spec_intens,num_peak,input_ids_pre,attention_mask_pre,spec_pre_intens,num_peak_pre

class DataSetWrapper(object):
    def __init__(self,
                world_size,
                rank,
                batch_size, 
                num_workers, 
                valid_size, 
                train_ms_file,
                valid_ms_file,
                train_ms_pre_file,
                valid_ms_pre_file):
        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.train_ms_file = train_ms_file
        self.valid_ms_file = valid_ms_file
        self.train_ms_pre_file = train_ms_pre_file
        self.valid_ms_pre_file = valid_ms_pre_file

    def get_data_loaders(self):
        
        self.train_ms = list(load_from_mgf(self.train_ms_file))
        self.valid_ms = list(load_from_mgf(self.valid_ms_file))
        self.train_ms_pre = list(load_from_mgf(self.train_ms_pre_file))
        self.valid_ms_pre = list(load_from_mgf(self.valid_ms_pre_file))
        
        train_dataset = ClrDataset(self.train_ms,self.train_ms_pre)
        valid_dataset = ClrDataset(self.valid_ms,self.valid_ms_pre)

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset,valid_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset,valid_dataset):
        train_sampler = DistributedSampler(train_dataset, num_replicas = self.world_size, rank=self.rank, shuffle = True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, 
                                                   sampler=train_sampler,shuffle=False,collate_fn = collate_func)
        valid_sampler = DistributedSampler(valid_dataset, num_replicas = self.world_size, rank=self.rank, shuffle = False)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, 
                                                   sampler=valid_sampler,shuffle=False,collate_fn = collate_func)

        return train_loader, valid_loader

