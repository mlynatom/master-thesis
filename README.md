# Master Thesis Repository - Tomáš Mlynář

This repository contains all code, experiments, and documentation related to the master thesis of Tomáš Mlynář. The project focuses on the adaptation large language models (LLMs), their training, evaluation, and benchmarking, with a particular emphasis on Czech language resources and evaluation frameworks.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Acknowledgments](#acknowledgments)

## Project Structure
```python
datasets_creation/ # Scripts and notebooks for dataset creation and preprocessing 
evaluation/ # Evaluation scripts, benchmarks, and analysis - notebooks 
scripts/ # Shell scripts for running experiments and evaluations 
training/ # Training scripts and notebooks (pretraining, finetuning, NLI, etc.)
```

## Installation

1. Clone the repository:
    ```sh
    git clone https://gitlab.fel.cvut.cz/factchecking/master-thesis-repository-tomas-mlynar.git
    cd master-thesis-repository-tomas-mlynar
    ```

2. (Recommended) Create and activate a Python virtual environments (there are 3 requirements files available for different components):
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies (from the desired requirements file):
    ```sh
    pip install -r master_venv_requirements.txt # main venv for the project
    ```
    ```sh
    pip install -r unsloth_venv_requirements.txt # for training with Unsloth
    ```
    ```sh
    pip install -r wildbench_venv_requirements.txt # for evaluation with WildBench
    ```

## Acknowledgments

- [WildBench](https://github.com/allenai/WildBench) for evaluation scripts and benchmarks.
- Supervisors, collaborators, and the Czech Technical University in Prague for support and guidance.