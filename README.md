# GQE (Generalized Quantum Eigensolver) Project

## Overview

This project combines quantum chemistry calculations using CUDA-Q with molecular generation using ChemVAE (Chemical Variational Autoencoder). The GQE framework performs quantum eigenvalue calculations for molecular systems, while ChemVAE enables molecular design and optimization.

## Project Structure

```
/home/leo07010/GQE/
├── GQE_tool/              # Main quantum chemistry calculation framework
│   ├── cudaqlib/          # CUDA-Q library integration
│   └── gqe/               # GQE algorithm implementation
├── chemvae_20250313/      # ChemVAE molecular generation framework
│   ├── chemvae/           # Core VAE modules
│   ├── *.smi              # SMILES training data (77MB)
│   └── config.ini         # Configuration file
├── data/                  # Molecular calculation data
│   └── molecules/
│       ├── H/             # Hydrogen molecule data
│       ├── Li/            # Lithium molecule data
│       └── N/             # Nitrogen molecule data
├── notebooks/             # Jupyter notebooks
│   ├── GQE_test.ipynb     # Main testing notebook
│   └── GQE_test.py        # Python export
├── scripts/               # Utility scripts
│   ├── fix_plots.py       # Plot generation and fixing
│   └── run_mpi.sh         # MPI execution script
├── experiments_logs/      # Parameter sweep experiment results
│   └── gqe_sweep/
│       ├── lr_*/          # Learning rate experiments
│       └── temperature_*/ # Temperature experiments
├── plots/                 # Generated comparison plots
│   ├── learning_rate_comparison.png
│   └── temperature_comparison.png
├── docs/                  # Documentation
│   └── PLOT_FIX_SUMMARY.md
└── chemvae_20250313.tar.gz  # Original archive (backup)
```

## Components

### 1. GQE Tool (Quantum Chemistry)

**Purpose**: Quantum eigenvalue calculations for molecular systems using CUDA-Q

**Key Features**:
- Hamiltonian construction from molecular data
- Variational quantum eigensolver (VQE) implementation
- UCCSD operator pool generation
- Parameter optimization using gradient-based methods
- MPI support for distributed computing

**Main Modules**:
- `GQE_tool/gqe/` - Core GQE algorithm
- `GQE_tool/cudaqlib/` - CUDA-Q integration

### 2. ChemVAE (Molecular Generation)

**Purpose**: SMILES-based molecular generation and optimization using variational autoencoders

**Key Features**:
- SMILES string encoding/decoding
- Latent space molecular representation
- Bayesian optimization for molecular property optimization
- LSTM-based sequence modeling

**Main Scripts**:
- `smiles_lstm_main.py` - LSTM training for SMILES
- `smiles_vae_main.py` - VAE training
- `smiles_vae_bo_main.py` - Bayesian optimization

**Core Modules** (in `chemvae/`):
- `smiles_vae.py` - VAE architecture (9.7 KB)
- `smiles_lstm.py` - LSTM model (4.9 KB)
- `smiles_vocab.py` - SMILES vocabulary handling
- `selfies.py` - SELFIES representation support
- `metrics.py` - Evaluation metrics

**Training Data**:
- `train.smi` - Training set (61.8 MB, ~620K molecules)
- `val.smi` - Validation set (3.9 MB, ~39K molecules)
- `test.smi` - Test set (11.6 MB, ~116K molecules)

**Configuration**: `config.ini`
- VAE latent dimension: 64
- Encoder/Decoder hidden size: 512
- Learning rate: 0.0001
- Batch size: 256
- Max sequence length: 150

## Molecular Data

### Calculated Molecules

#### Hydrogen (H)
- Checkpoint: `data/molecules/H/H 0-pyscf.chk` (105 KB)
- Log: `data/molecules/H/H 0-pyscf.log` (7.5 KB)
- Metadata: `data/molecules/H/H 0_metadata.json` (4.7 KB)

#### Lithium (Li)
- Checkpoint: `data/molecules/Li/Li 0-pyscf.chk` (122 KB)
- Log: `data/molecules/Li/Li 0-pyscf.log` (8.2 KB)
- Metadata: `data/molecules/Li/Li 0_metadata.json` (338 KB)

#### Nitrogen (N)
- Checkpoint: `data/molecules/N/N 0-pyscf.chk` (61 KB)
- Log: `data/molecules/N/N 0-pyscf.log` (7.8 KB)
- Metadata: `data/molecules/N/N 0_metadata.json` (2.6 MB)

## Usage

### GQE Quantum Calculations

```bash
# Run parameter sweep experiments
cd /home/leo07010/GQE
jupyter notebook notebooks/GQE_test.ipynb

# Generate comparison plots
python scripts/fix_plots.py
```

### ChemVAE Molecular Generation

```bash
cd /home/leo07010/GQE/chemvae_20250313

# Train LSTM model
python smiles_lstm_main.py

# Train VAE
python smiles_vae_main.py

# Run Bayesian optimization
python smiles_vae_bo_main.py
```

## Dependencies

### GQE Requirements
- CUDA-Q
- cudaq-solvers
- PySCF
- OpenFermion
- PyTorch
- Lightning Fabric

### ChemVAE Requirements
- Python 3.x
- CUDA (for GPU acceleration)
- RDKit
- PyTorch
- TorchDrug
- BoTorch
- Matplotlib
- Pandas
- tqdm

## Experiment Results

### Parameter Sweep Studies

**Learning Rate Comparison**:
- Tested values: 0.001, 0.0001, 1e-05
- Results: `plots/learning_rate_comparison.png`
- Logs: `experiments_logs/gqe_sweep/lr_*/`

**Temperature Comparison**:
- Tested values: 0.1, 1.0, 5.0
- Results: `plots/temperature_comparison.png`
- Logs: `experiments_logs/gqe_sweep/temperature_*/`

## Integration Potential

The GQE and ChemVAE components can be integrated for:

1. **Molecular Design Loop**:
   - ChemVAE generates candidate molecules
   - GQE calculates quantum properties
   - Feedback loop optimizes molecular properties

2. **Property Prediction**:
   - Use VAE latent space for molecular representation
   - Train quantum property predictors
   - Guide molecular generation with quantum constraints

3. **Inverse Design**:
   - Define target quantum properties
   - Use Bayesian optimization to search VAE latent space
   - Generate molecules with desired quantum characteristics

## References

- ChemVAE: Based on [機械学習による分子最適化](https://github.com/kanojikajino/ml4chem)
- SMILES data: [Figshare dataset](https://figshare.com/articles/dataset/SMILES_data/7034426)

## Notes

- All molecule calculations use PySCF with STO-3G basis
- GQE uses UCCSD operator pool for ansatz construction
- ChemVAE supports both SMILES and SELFIES representations
- Original ChemVAE archive kept as backup: `chemvae_20250313.tar.gz`
