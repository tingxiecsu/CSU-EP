
import torch
from model_finetune import CSUEP_finetune
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
from torch.amp import GradScaler, autocast
from contextlib import nullcontext

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

class ModelTrain(object):
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
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('text_encoder.model.'):
                    new_key = key.replace('text_encoder.model.', 'text_encoder.')
                    new_state_dict[new_key] = value
                else:
                    continue
            model.load_state_dict(new_state_dict, strict=False)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
        return model

    def _load_finetune_trained_weights(self, model):
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
            for step,(input_ids,attention_masks,intensity_,num_peaks,input_ids_pre,attention_masks_pre,intensity_pre_,num_peaks_pre) in enumerate(tqdm(valid_loader)):
                input_ids_pre = input_ids_pre.to(self.device)              
                intensity_pre_ = intensity_pre_.to(self.device)
                attention_masks_pre = attention_masks_pre.to(self.device)
                num_peaks_pre = num_peaks_pre.to(self.device)
                input_ids = input_ids.to(self.device)
                intensity_ = intensity_.to(self.device)
                attention_masks = attention_masks.to(self.device)
                num_peaks = num_peaks.to(self.device)
                embeds,embeds_pre,loss = model(input_ids,attention_masks,intensity_,num_peaks,input_ids_pre,attention_masks_pre,intensity_pre_,num_peaks_pre)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss

    def train(self):
        #Dataloaders
        model = CSUEP_finetune()
        model = model.to(self.device)
        model = self._load_pre_trained_weights(model)

        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.rank) 
            model = DistributedDataParallel(model, device_ids=[self.rank], find_unused_parameters = True)
            
        train_loader, valid_loader = self.dataset.get_data_loaders()

        optimizer = StableAdamW(model.parameters(), lr=eval(self.config['learning_rate']), betas=list(eval(self.config['betas'])), eps=eval(self.config['eps']), weight_decay=self.config['weight_decay'])
       
        
        accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        updates_per_epoch = len(train_loader) // accumulation_steps
        total_steps = updates_per_epoch * self.config["epochs"]

        warmup_steps  = int(total_steps * self.config["warmup_ratio"])
        stable_steps  = int(total_steps * self.config["stable_ratio"])
        decay_steps   = total_steps - warmup_steps - stable_steps

        scheduler = get_wsd_schedule(
            optimizer,
            warmup_steps,
            stable_steps,
            decay_steps,
            float(self.config['min_lr_ratio']),
            self.config['num_cycles'],
        )
                                     
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        if dist.get_rank() == 0:
            _save_config_file(model_checkpoints_folder)

        valid_n_iter = 0
        n_iter = 0
        best_valid_loss = float("inf")
        
        scaler = GradScaler()
        
        if dist.get_rank() == 0:
            print(f'Training...')
            
        for epoch_counter in range(self.config['epochs']):
            train_loader.sampler.set_epoch(epoch_counter)
            if dist.get_rank() == 0:
                print(f'Epoch {epoch_counter}')
            train_loss = 0
            train_n_iter = 0
            
            optimizer.zero_grad()
            
            for step,(input_ids,attention_masks,intensity_,num_peaks,input_ids_pre,attention_masks_pre,intensity_pre_,num_peaks_pre) in enumerate(tqdm(train_loader)):
                is_update_step = (step + 1) % accumulation_steps == 0
                sync_context = model.no_sync() if self.world_size > 1 and not is_update_step else nullcontext()
                
                with sync_context:
                    input_ids_pre = input_ids_pre.to(self.device, non_blocking=True)              
                    intensity_pre_ = intensity_pre_.to(self.device, non_blocking=True)
                    attention_masks_pre = attention_masks_pre.to(self.device, non_blocking=True)
                    num_peaks_pre = num_peaks_pre.to(self.device, non_blocking=True)
                    input_ids = input_ids.to(self.device, non_blocking=True)
                    intensity_ = intensity_.to(self.device, non_blocking=True)
                    attention_masks = attention_masks.to(self.device, non_blocking=True)
                    num_peaks = num_peaks.to(self.device, non_blocking=True)
                
                    with autocast(device_type="cuda"):
                        embeds,embeds_pre,loss = model(input_ids,attention_masks,intensity_,num_peaks,input_ids_pre,attention_masks_pre,intensity_pre_,num_peaks_pre)
                        loss = loss / accumulation_steps
                
                    scaler.scale(loss).backward()
                    
                train_loss += loss.item()
                train_n_iter += 1
                
                if is_update_step:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    n_iter += 1
                
                if dist.get_rank() == 0 and n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item() * accumulation_steps, global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                
                
            if (step + 1) % accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
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

        if self.world_size > 1:
            dist.destroy_process_group()