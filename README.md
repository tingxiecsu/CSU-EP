# CUREI
This is the official code repository for the paper **"Contrastive Alignment of Simulated and Experimental Electron Ionization Mass Spectra for High-Fidelity Library Matching."**

We developed a method named **CUREI** to bridge simulated and experimental EI-MS spectra through **self-supervised pretraining** and **contrastive fine-tuning**, enabling robust **cross-domain spectral alignment** and **accurate compound identification**.

<p align="center">
  <img src="https://github.com/tingxiecsu/CUREI/blob/main/img/logo.png" width="200">
</p>

---

### 🔍 Overview

CUREI introduces a unified framework that jointly learns representations from both simulated and experimental spectra, capturing domain-invariant features to enhance library matching accuracy. The method achieves state-of-the-art performance on public benchmarks and supports large-scale database retrieval.

Key highlights:
- 🚀 **FlashAttention** acceleration for large-batch training
- 🧠 **Transformer-based spectral encoder** adapted from *ModernBERT*.  
- 🧩 **Self-supervised pretraining** for robust embedding learning and Contrastive fine-tuning for accurate compound identification.  
- ⚡ **Fast spectral retrieval** over a database of 2 million spectra using an optimized **HNSW-based index**.
- 🌐 Integrated **web server** for interactive spectrum search and visualization.

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

### 🧪 Web Server

We provide an online **CUREI Web Server** that allows users to:
- Upload EI-MS spectra in `.msp` format or manually input peak data.  
- Retrieve top candidate molecules from the CUREI spectral library.  
- Visualize experimental and predicted spectra interactively.  

The CSU-MS2 web server and CUREIDB are hosted on Hugging Face, and can be visited through the following links:

- ðŸŒ **CUREI web server**: The application interface allows users to upload unknow spectra and accsess results in real time. Visit the app here: [CUREI web server](https://huggingface.co/spaces/Tingxie/CUREI).

- ðŸ“‚ **CUREIDB**: Explore the dataset here: [CUREIDB](https://huggingface.co/datasets/Tingxie/CUREIDB).


---

### 📫 Contact

For questions or collaboration inquiries, please contact:  
📧 **212307003@csu.edu.cn**

---