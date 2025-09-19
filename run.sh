#!/bin/bash

#SBATCH -p hgx
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --job-name=cluster_test
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out

uv run lightning_model.py