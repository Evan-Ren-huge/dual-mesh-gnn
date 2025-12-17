# Dual-Graph GConvGRU for Abaqus Finite Element Simulations

A PyTorch implementation of a dual-graph recurrent Graph Neural Network (GNN) using `GConvGRU` from PyTorch Geometric Temporal. The model processes Abaqus simulation data to predict:

- Node displacements `U(t)`
- Element-averaged stress `s_elem(t)`
- Global reaction force `RF2(t)`

Key features:
- Dual graph structure: separate node graph and element graph
- True batched training across multiple simulation cases
- Global or per-case normalization
- Teacher forcing with decay schedule
- Validation split and best model checkpointing
- Configurable via YAML

## Installation

```bash
pip install torch torch-geometric-temporal numpy pyyaml
