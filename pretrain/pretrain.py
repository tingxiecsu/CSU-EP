# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 16:37:56 2024

@author: ZNDX002
"""


import torch
from model_pretrain import EIMSBERT
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm
from transformers import AdamW
from optimizer import StableAdamW
from scheduler import  get_wsd_schedule
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import sys


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

class ModalTrain(object):
    def __init__(self, dataset, config,device,world_size,rank):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.writer = SummaryWriter()
        self.rank = rank
        self.dataset = dataset

    def _get_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Running on:", device)
        return device

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs/', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), weights_only=True)
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
        return model

    def _validate(self,model, valid_loader):
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            counter = 0
            if dist.get_rank() == 0:
              print(f'Validation step')
            for step,(input_ids,attention_masks,intensity_,num_peaks) in enumerate(tqdm(valid_loader)):
                input_ids = input_ids.to(self.device)
                intensity_ = intensity_.to(self.device)
                attention_masks = attention_masks.to(self.device)
                num_peaks = num_peaks.to(self.device)
                loss = model(input_ids,attention_masks,intensity_,num_peaks)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss

    def train(self):
        #Dataloaders
        model = EIMSBERT()
        model = model.to(self.device)
        model = self._load_pre_trained_weights(model)

        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.rank) 
            model = DistributedDataParallel(model, device_ids=[self.rank], find_unused_parameters = True)
            
        train_loader, valid_loader = self.dataset.get_data_loaders()

        optimizer = StableAdamW(model.parameters(), lr=eval(self.config['learning_rate']), betas=list(eval(self.config['betas'])), eps=eval(self.config['eps']), weight_decay=self.config['weight_decay'])
        #optimizer = torch.optim.AdamW(model.parameters(), 
        #                                eval(self.config['learning_rate']), 
        #                                weight_decay=self.config['weight_decay'])
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
        #                                                        T_max=len(train_loader), 
        #                                                        eta_min=0, 
        #                                                        last_epoch=-1)
       
        scheduler = get_wsd_schedule(optimizer,self.config['num_warmup_steps'],self.config['num_stable_steps'],self.config['num_decay_steps'], float(self.config['min_lr_ratio']),
                                     self.config['num_cycles'])

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        if dist.get_rank() == 0:
            _save_config_file(model_checkpoints_folder)

        valid_n_iter = 0
        n_iter = 0
        best_valid_loss = 10000
        if dist.get_rank() == 0:
            print(f'Training...')
        for epoch_counter in range(self.config['epochs']):
            train_loader.sampler.set_epoch(epoch_counter)
            if dist.get_rank() == 0:
                print(f'Epoch {epoch_counter}')
            train_loss = 0
            train_n_iter = 0
            for step,(input_ids,attention_masks,intensity_,num_peaks) in enumerate(tqdm(train_loader)):
                input_ids = input_ids.to(self.device)
                intensity_ = intensity_.to(self.device)
                attention_masks = attention_masks.to(self.device)
                num_peaks = num_peaks.to(self.device)
                loss = model(input_ids,attention_masks,intensity_,num_peaks)
                if dist.get_rank() == 0:
                    if n_iter % self.config['log_every_n_steps'] == 0:
                        self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                loss.backward()

                optimizer.step()
                train_loss += loss.item()
                train_n_iter += 1
                n_iter += 1
            if dist.get_rank() == 0:
                print('Training at Epoch ' + str(epoch_counter + 1) + 'with loss ' + str(train_loss/train_n_iter))
                self.writer.add_scalar('train_divid_loss', train_loss/train_n_iter, global_step=epoch_counter + 1)

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if dist.get_rank() == 0:
                    print('Validation at Epoch ' + str(epoch_counter + 1) + 'with loss ' + str(valid_loss))
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    if dist.get_rank() == 0:
                        torch.save(model.module.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                if dist.get_rank() == 0:
                    self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            scheduler.step()
            if dist.get_rank() == 0:
                self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)
        if self.world_size > 1:
            dist.destroy_process_group()
