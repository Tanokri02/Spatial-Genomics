"""Donor-aware disease classification from PsychAD snRNA-seq."""

__version__ = "0.1.0"

# NOTE (macOS): torch, sklearn and xgboost each ship their own OpenMP runtime. Loading more than one into a process
# that then runs multi-threaded code can segfault (exit 139) or deadlock. Therefore:
#   * xgboost is imported lazily, only when an XGBoost model is built (models/sk_models.py);
#   * on macOS, commands that train no neural net never import torch (cli._ctx);
#   * `psychad benchmark/celltype` split XGBoost and torch models into separate processes (cli._runtime_groups).
# Do not import xgboost or torch at package import time.
