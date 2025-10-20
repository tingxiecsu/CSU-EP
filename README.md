# CUREI
<p align="center">
  <img src="https://github.com/tingxiecsu/CUREI/blob/main/img/logo.png" width="300">
</p>

This is the official code repository for the paper **"Contrastive Alignment of Simulated and Experimental Electron Ionization Mass Spectra for High-Fidelity Library Matching."**

We developed a method named **CUREI** to bridge simulated and experimental EI-MS spectra through **self-supervised pretraining** and **contrastive fine-tuning**, enabling robust **cross-domain spectral alignment** and **accurate compound identification**.

### ?? ### ?? Overview

CUREI introduces a unified framework that jointly learns representations from both simulated and experimental spectra, capturing domain-invariant features to enhance library matching accuracy. The method achieves state-of-the-art performance on public benchmarks and supports large-scale database retrieval.

Key highlights:
- ?? **FlashAttention** acceleration for large-batch training
- ?? **Transformer-based spectral encoder** adapted from *ModernBERT*.  
- ?? **Self-supervised pretraining** for robust embedding learning and Contrastive fine-tuning for accurate compound identification.  
- ? **Fast spectral retrieval** over a database of 2 million spectra using an optimized **HNSW-based index**.
- ?? Integrated **web server** for interactive spectrum search and visualization.

---

## Setup

We have fully documented the environment used to train CUREI, which can be installed on a GPU-equipped machine with the following commands:

```bash
conda env create -f environment.yaml
conda activate bert24
# install flash attention 2 (model uses just FA2 as FA3 isn't supported)
pip install "flash_attn==2.6.3" --no-build-isolation
# or download a precompiled wheel from https://github.com/Dao-AILab/flash-attention/releases/tag/v2.6.3
# or limit the number of parallel compilation jobs
# MAX_JOBS=8 pip install "flash_attn==2.6.3" --no-build-isolation
```

## Training
Pretrain and fine-tune the model based on your own spectrum datasets with [run_ddp.py](https://github.com/tingxiecsu/CUREI/blob/main/CUREI/run_ddp.py) function and [run_finetune_ddp.py](https://github.com/tingxiecsu/CUREI/blob/main/CUREI/run_finetune_ddp.py). Multi-gpu or multi-node parallel training can be performed using Distributed Data Parallel (DDP) provided in the code.

    main(rank, world_size, num_gpus, rank_is_set, ds_args)

## ?? Usage Example: Compute new spectrum embeddings & library search

The **CUREI** framework supports computing spectral embeddings for newly collected EI-MS data and performing large-scale retrieval using a pre-built **HNSW index** (˜2 million spectra).
There are two main scripts involved:

1. **[`calculate_embeddings.py`](https://github.com/tingxiecsu/CUREI/blob/main/calculate_embeddings.py)**  
   ? Generates embeddings for new mass spectra using the trained CUREI model.

2. **[`hnswlib_index_searching.py`](https://github.com/tingxiecsu/CUREI/blob/main/hnswlib_index_searching.py)**  
   ? Performs similarity-based search in the HNSW index to retrieve top candidate molecules.

### 1?? Generate Spectrum Embeddings
You can compute embeddings for a list of new spectra (e.g., in `.mgf` format) as follows:

    with torch.no_grad():
        output = model.text_encoder(
            input_ids=input_ids_batch,
            intensities=intensities_batch,
            attention_mask=attention_mask_batch,
            return_dict=True
        )
        output_feats = output.last_hidden_state  # shape: (B, L, D)
        output_aggr_feats = model.esa(output_feats,attention_mask_batch)
        output_aggr_feats = model.proj(output_aggr_feats)
        spectrum_embeddings = F.normalize(output_aggr_feats, dim=1)
	    

### 2?? Search with HNSW Index
Once embeddings are generated, you can search the HNSW-based CUREI database:

    xq= load_npz("/meassured_spectra_embeddings.npz").todense().astype('float32')
    xq_len = np.linalg.norm(xq, axis=1, keepdims=True)
    xq = xq/xq_len
    dim = 256
    start_time=time.time()*1000
    p = hnswlib.Index(space='l2', dim=dim) 
    p.load_index("references_index.bin")
    end_time=time.time()*1000
    print('loadindex_time %.4f'%((end_time-start_time)/100))
    start_time=time.time()*1000
    # Controlling the recall by setting ef:
    p.set_ef(300) # ef should always be > k   ##
    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    k=100
    I, D = p.knn_query(xq, k)
    end_time=time.time()*1000
    print('search_time %.4f'%((end_time-start_time)/100))

    np.save("/meassured_index_results.npy",I)
    np.save("/meassured_score_results.npy",D)
	    

### ?? Web Server

We provide an online **CUREI Web Server** that allows users to:
- Upload EI-MS spectra in `.msp` format or manually input peak data.  
- Retrieve top candidate molecules from the CUREI spectral library.  
- Visualize experimental and predicted spectra interactively.  

The CSU-MS2 web server and CUREIDB are hosted on Hugging Face, and can be visited through the following links:

- ?? **CUREI web server**: The application interface allows users to upload unknow spectra and accsess results in real time. Visit the app here: [CUREI web server](https://huggingface.co/spaces/Tingxie/CUREI).

- ?? **CUREIDB**: Explore the dataset here: [CUREIDB](https://huggingface.co/datasets/Tingxie/CUREIDB).


---

### ?? Contact

For questions or collaboration inquiries, please contact:  
?? **212307003@csu.edu.cn**

---