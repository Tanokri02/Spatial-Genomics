"""Sample weights that make every donor (within a class) and every class count equally."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """w_cell = 1 / (n_classes * n_donors_in_class * n_cells_of_donor), rescaled to mean 1.

    Prevents donors with many nuclei and the majority class (AD/Control vs PD/DLB) from dominating.
    """
    df = pd.DataFrame({"y": np.asarray(y), "g": np.asarray(groups)})
    n_cells = df.groupby("g")["g"].transform("size").values
    donors_per_class = df.drop_duplicates("g").groupby("y")["g"].size()
    n_donors_cls = df["y"].map(donors_per_class).values
    w = 1.0 / (len(donors_per_class) * n_donors_cls * n_cells)
    return (w * len(w) / w.sum()).astype(np.float32)
