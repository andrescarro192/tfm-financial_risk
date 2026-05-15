# TFM Financial Risk

# TFM — Financial Risk Modeling with Hybrid ML Pipeline

Master's thesis project combining TabPFN, Temporal GNN, and anomaly detection
for financial risk modeling on US banking data (FDIC RIS dataset, 2016–2025).

## Pipeline
1. Tabular encoding with TabPFN V2.5 (temporal feature engineering)
2. Temporal GNN (T-GCN / EvolveGCN) on dynamic financial graph
3. Anomaly detection (LOF (Local Outlier Factor) + LSTM Autoencoder)
4. Baseline comparison (LSTM per entity)

## Stack

- PyTorch
- PyTorch Geometric
- TabPFN
- scikit-learn

## Setup
```bash
conda env create -f environment.yml
conda activate tfm-financial-risk
```

## Repository structure
```
tfm-financial_risk/
├── data/          # not tracked — place raw RIS files in data/dataraw/
├── src/           # source code (loader, models, anomaly, evaluation)
├── experiments/   # training scripts and YAML configs
├── notebooks/     # EDA, feature engineering, results
└── tests/         # unit tests
```
