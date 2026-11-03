#!/bin/bash
# Generic job wrapper:  sbatch scripts/slurm/job.sbatch <psychad args...>
#   sbatch --gres=gpu:1 scripts/slurm/job.sbatch benchmark --models cell_mlp cell_mlp_large mil_attn
#   sbatch scripts/slurm/job.sbatch benchmark --models cell_rf
# Every (model, seed, fold) is cached, so a job that hits the wall-time limit simply resumes when resubmitted.
# Edit the #SBATCH lines / the environment activation for your cluster.
#SBATCH --job-name=psychad
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
mkdir -p logs
# source ~/.bashrc && conda activate psychad     # <- your env
cd "$(dirname "$0")/../.."
python -m psychad_ml --profile "${PSYCHAD_PROFILE:-default}" "$@"
