
from modular_csuep import  CsuepForMaskedLM, CsuepConfig

import torch
from torch import nn


class CSUEP_Pretrain(nn.Module):
    def __init__(self,                 
                 config = None,    
                 ):
        super().__init__()
        
        self.config= CsuepConfig()
        self.text_encoder = CsuepForMaskedLM(config=self.config)

    def forward(self,input_ids,attention_masks,intens_tensors,num_peaks):

        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.config.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.config.vocab_size, input_ids.device, targets=labels,
                                      probability_matrix = probability_matrix) 
        
 
        mlm_output = self.text_encoder(input_ids=input_ids, 
                                       intensities=intens_tensors,
                                       attention_mask = attention_masks,   
                                       return_dict = True,
                                       labels = labels
                                      )                           
        loss_mlm = mlm_output.loss        

        return loss_mlm

  
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.config.pad_token_id] = False
        #masked_indices[input_ids == self.config.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.config.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

