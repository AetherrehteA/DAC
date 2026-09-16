# DAC for Multi-Objective Combinatorial Optimization

This repository contains the implementation of a dynamic algorithm configuration framework for multi-objective combinatorial optimization.

The method uses reinforcement learning to dynamically configure evolutionary parameters during the optimization process and currently supports:

* Multi-objective Flexible Job Shop Scheduling Problem (FJSP)
* Multi-objective Capacitated Vehicle Routing Problem (CVRP)

## Requirements

Install the required packages with:

```bash
pip install -r requirements.txt
```
## Repository Structure

```text
DAC/
├── rl_environments/    # RL environments and configuration files
├── routing/            # CVRP-related code and data
├── scheduling/         # FJSP-related code and data
├── train.py            # Training
├── settings.py         
└── requirements.txt
```

## Training

run:

```bash
python train.py
```
## Citation

Citation information will be provided after publication.

