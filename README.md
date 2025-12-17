# Dual-Graph GConvGRU for Accelerating Abaqus Simulations

This repository provides an end-to-end pipeline for accelerating Abaqus finite-element
simulations using a dual-graph temporal Graph Neural Network (GConvGRU).

The workflow consists of:
1. Exporting Abaqus `.odb` files into compressed `.npz` datasets.
2. Training a dual-graph recurrent GNN to predict nodal displacement, element stress,
   and global reaction forces in an autoregressive rollout setting.

The project is designed for research and engineering applications where repeated
finite-element simulations are computationally expensive.

---

## Repository Structure

```text
.
├── src/                    # Training code (PyTorch + PyG Temporal)
│   └── train.py
├── tools/
│   ├── export_odb_to_npz.py # Abaqus ODB → NPZ data exporter
│   └── README.md            # Detailed documentation for the exporter
├── configs/
│   └── config.yaml          # Training configuration
└── README.md
