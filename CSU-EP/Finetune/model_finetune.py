
from modular_csuep import CsuepModel, CsuepConfig

import torch
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

class MeanPooling(nn.Module):
    def __init__(self, feature_dim, out_dim):
        super().__init__()
        self.ln_f = nn.LayerNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, out_dim)
        
    def forward(self, hidden_states, attention_mask):
        # hidden_states: (B, N, C_in)
        logits = self.ln_f(hidden_states) 
        features = self.linear(logits) # (B, N, C_out)

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(features.size()).to(features.dtype)
        sum_embeddings = torch.sum(features * input_mask_expanded, dim=1) 
        sum_mask = input_mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings
        
class MaxPooling(nn.Module):
    def __init__(self, feature_dim, out_dim):
        super().__init__()
        self.ln_f = nn.LayerNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, out_dim)
        
    def forward(self, hidden_states, attention_mask):
        logits = self.ln_f(hidden_states) 
        features = self.linear(logits) # (B, N, C_out)
        mask = attention_mask.unsqueeze(-1) # (B, N, 1)
        min_value = torch.finfo(features.dtype).min
        features = features.masked_fill(mask == 0, min_value) 
        aggr_feature = torch.max(features, dim=1)[0] # (B, C_out)
        return aggr_feature
        
class ESA(nn.Module):
    def __init__(self, feature_dim, out_dim):
        super().__init__()
        self.ln_f = nn.LayerNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, out_dim)
        self.linear1 = nn.Linear(out_dim, out_dim)
        
    def forward(self, hidden_states,attention_mask):
        logits = self.ln_f(hidden_states) # (B, N, C)
        cap_embes = self.linear(logits) # Q
        features_in = self.linear1(cap_embes) # M
        mask = attention_mask.unsqueeze(-1) # (B, N, 1)
        features_in = features_in.masked_fill(mask == 0, -1e4) # (B, N, C)
        features_k_softmax = nn.Softmax(dim=1)(features_in)
        attn = features_k_softmax.masked_fill(mask == 0, 0)
        aggr_feature = torch.sum(attn * cap_embes, dim=1) # (B, C)
        return aggr_feature
    
    
class CSUEP_finetune(nn.Module):
    def __init__(self,                 
                 config = None,  
                 embed_dim = 768,
                 temperature = 0.07,
                 ):
        super().__init__()
        
        self.config= CsuepConfig()                  
        self.text_encoder = CsuepModel(config=self.config)  
        #self.esa = ESA(self.config.hidden_size, embed_dim)
        self.pooler = MeanPooling(self.config.hidden_size, embed_dim)
        self.proj = nn.Linear(embed_dim,embed_dim)
        self.temperature = temperature
        
    def info_nce_loss(self, features1, features2):
        batch_size = features1.shape[0]
        device = features1.device
    
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
    
        similarity_matrix = torch.matmul(features1, features2.T)
        labels = torch.arange(batch_size).to(device)
    
        logits1 = similarity_matrix / self.temperature
        loss1 = F.cross_entropy(logits1, labels)
    
        logits2 = similarity_matrix.T / self.temperature
        loss2 = F.cross_entropy(logits2, labels)
    
        return (loss1 + loss2) / 2
    
    def forward(self,input_ids,attention_masks,intens_tensors,num_peaks,input_ids_pre,attention_masks_pre,intens_tensors_pre,num_peaks_pre):

 
        output = self.text_encoder(input_ids=input_ids, 
                                       intensities=intens_tensors,
                                       attention_mask = attention_masks,   
                                       return_dict = True,
                                      )
        output_pre = self.text_encoder(input_ids=input_ids_pre, 
                                       intensities=intens_tensors_pre,
                                       attention_mask = attention_masks_pre,   
                                       return_dict = True,
                                      )
        output_feats = output.last_hidden_state
        output_pre_feats = output_pre.last_hidden_state
        
        output_aggr_feats = self.pooler(output_feats,attention_masks)
        output_pre_aggr_feats = self.pooler(output_pre_feats,attention_masks_pre)
        
        #output_aggr_feats = self.esa(output_feats,attention_masks)
        #output_pre_aggr_feats = self.esa(output_pre_feats,attention_masks_pre)
        output_aggr_feats = self.proj(output_aggr_feats)
        output_pre_aggr_feats = self.proj(output_pre_aggr_feats)
        loss = self.info_nce_loss(output_aggr_feats, output_pre_aggr_feats)
                                  
        return output_aggr_feats,output_pre_aggr_feats,loss

