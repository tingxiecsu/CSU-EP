import torch
from model_finetune import CSUEP_finetune
from matchms.exporting import save_as_mgf
import numpy as np
import os
from tqdm import tqdm
from matchms.importing import load_from_mgf
import matchms.filtering as msfilters
import torch.nn.functional as F
from scipy.sparse import csr_matrix, save_npz,load_npz

def count_elements_less_than(lst, value):
    count = sum(1 for element in lst if element <= value)
    return count

def spectrum_processing(s):
    """This is how one would typically design a desired pre- and post-
    processing pipeline."""
    s = msfilters.normalize_intensities(s)
    s = msfilters.select_by_mz(s, mz_from=0, mz_to=1000)
    s = msfilters.require_minimum_number_of_peaks(s, n_required=2)
    return s

test=list(load_from_mgf("/test.mgf"))
test=[spectrum_processing(s) for s in test]
test=[s for s in test if s is not None]


model_path ="/checkpoints/model.pth"
device = torch.device('cpu')
state_dict = torch.load(model_path,map_location=device)
model = CSUEP_finetune()
model.load_state_dict(state_dict)
model.to(device) 
model.eval()
print(model.config._attn_implementation)

word_list = list(np.linspace(0,1001,1001,endpoint=False))
word_list = [str(i) for i in word_list]
word2idx = {'[PAD]':1002,'[MASK]':1003}
for i, w in enumerate(word_list):
    word2idx[w] = i + 1
    

batch_size=200
test_vectors=[]

for i in tqdm(range(0, len(test), batch_size)):
    batch = test[i:i + batch_size]
    
    input_ids_list = []
    intensities_list = []
    attention_masks_list = []
    num_peaks_list = []

    for spec in batch:
        spec_mz = spec.mz
        spec_intens = spec.intensities
        input_ids = [word2idx[str(float(int(s)))] for s in spec_mz]
        input_ids = np.array(input_ids)
        attention_mask = np.ones_like(input_ids)

        input_ids_list.append(torch.from_numpy(input_ids).long())
        intensities_list.append(torch.from_numpy(spec_intens).float())
        attention_masks_list.append(torch.from_numpy(attention_mask).long())
        num_peaks_list.append(len(input_ids))

    # Padding to the same length
    max_len = max(num_peaks_list)
    input_ids_batch = torch.nn.utils.rnn.pad_sequence( input_ids_list, batch_first=True, padding_value=1002)
    intensities_batch = torch.nn.utils.rnn.pad_sequence( intensities_list, batch_first=True, padding_value=0)
    attention_mask_batch = torch.nn.utils.rnn.pad_sequence( attention_masks_list, batch_first=True, padding_value=0)
    num_peaks_tensor = torch.LongTensor(num_peaks_list)
    
    input_ids_batch = input_ids_batch.to(device)
    intensities_batch = intensities_batch.to(device)
    attention_mask_batch = attention_mask_batch.to(device)
    num_peaks_tensor = num_peaks_tensor.to(device)

    # Forward pass
    with torch.no_grad():
        output = model.text_encoder(
            input_ids=input_ids_batch,
            intensities=intensities_batch,
            attention_mask=attention_mask_batch,
            return_dict=True
        )
        output_feats = output.last_hidden_state  # shape: (B, L, D)
        output_aggr_feats = model.pooler(output_feats,attention_mask_batch)
        output_aggr_feats = model.proj(output_aggr_feats)
        spectrum_embeddings = F.normalize(output_aggr_feats, dim=1)
        test_vectors.extend(spectrum_embeddings.detach().cpu().numpy())
test_matrix = np.vstack(test_vectors)
test_embeddings=csr_matrix(test_matrix)
np.save("/measured_spectra_embeddings.npz",test_vectors)
