### DHGCDA

## Introduction

DHGCDA is a dual-channel hypergraph learning framework for circRNA-disease association prediction. It integrates multiple biological views and jointly models node-level and hyperedge-level information to capture higher-order relationships between circRNAs and diseases.


## File Contents

- `main_GPU.py`  
  The main script for 5-fold training and testing, responsible for data loading, model training, validation, testing, and result statistics.
- `model_GPU.py`  
  The model definition file, including biological feature encoding, dual-channel hypergraph encoding, multi-view fusion, and pairwise decoding modules.
- `build_hyg.py`  
  The multi-view hypergraph construction script, used to generate the hypergraph `pickle` files required for training.
- `data/`  
  The data directory currently required by the prepared version of the scripts.

## Data Directory Requirements

By default, the scripts depend on the following contents under the `data/` directory:

- circRNA and disease ID files
- embedding files
- training/validation/test split files in `CD_data/`
- hypergraph `pickle` files in `hyg_new/`
- circRNA similarity files
- miRNA-disease association files


## Usage

### 1. Build Hypergraphs

If you need to regenerate the hypergraph input files, run:

```bash
python build_hyg.py
```

After running, `hyg_dict0.pickle` to `hyg_dict4.pickle` will be generated under `hyg_new/`.

### 2. Training and Testing

Run the main script:

```bash
python main_GPU.py
```

The main optional arguments supported by `main_GPU.py` include:

- `--ablation`
- `--ablation_mode`
- `--dual_ablation`
- `--view_fusion`
- `--view_drop`
- `--use_view_gate`
- `--hgnn_dim`
- `--lr`
- `--exp_tag`

The current default core settings in the script are:

- `epochs = 100`
- `lr = 0.004`
- `bio_out_dim = 32`
- `hgnn_dim = 512`
- CUDA is used by default if available; otherwise, the script falls back to CPU

## Environment Dependencies

The scripts require at least the following Python packages:

- `torch`
- `torch_geometric`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

## Notes and Important Considerations

- `main_GPU.py` imports `model_GPU.py` from the same directory using `from model_GPU import *`, so these two files should be kept in the same folder
- `build_hyg.py` still contains some absolute paths from the original environment. If you want to run it on another machine, it is recommended to manually convert these paths into portable relative paths
