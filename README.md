# iRMS-Net

This repository contains the code for the paper model iRMS-Net.

## Environment

Recommended versions:

- Python 3.8.20
- PyTorch 2.1.1 + CUDA 12.1
- torch-geometric 2.6.1
- torch-scatter 2.1.2+pt21cu121
- torch-sparse 0.6.18+pt21cu121
- torch-cluster 1.6.3+pt21cu121
- torch-spline-conv 1.2.2+pt21cu121
- pyg-lib 0.4.0+pt21cu121
- RDKit 2024.3.2
- numpy 1.24.3
- pandas 2.0.3
- scipy 1.10.1
- scikit-learn 1.3.2

## Folder layout

- `data/comb-2.csv`: interaction table
- `data/smiles/smile.csv`: drug SMILES table
- `data/strain/strainfeature.csv`: strain/cell feature table
- `data/processed/`: generated `.pt` files
- `data/iRMS-Net/`: training logs and `.model` files

## Pipeline

1. `creat_data_DC.py`
   - reads `comb-2.csv`
   - converts SMILES to molecular graphs with RDKit
   - loads strain features
   - saves PyG data to `data/processed/`

2. `utils_test.py`
   - defines `TestbedDataset`
   - provides metric and utility helpers

3. `model/iRMSNet.py`
   - defines the `iRMSNet` network
   - fuses drug graph features and strain features
   - outputs binary logits and an intermediate feature vector

4. `newtrain.py`
   - runs 5-fold cross validation
   - trains and evaluates each fold
   - records AUC, PR-AUC, ACC, BACC, PREC, TPR, KAPPA, RECALL
   - saves the best model and log files to `data/iRMS-Net/`

## Run steps

```bash
python creat_data_DC.py
python newtrain.py
```

## Notes

- If CUDA is unavailable, the training script falls back to CPU.
- Generated `.pt` files are stored in `data/processed/`.
- Training outputs are stored in `data/iRMS-Net/`.
