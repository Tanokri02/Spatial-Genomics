import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def smoke(tmp_path_factory):
    """Synthetic h5ad -> built derived dataset. Returns (cfg, paths, ctx). Built once per test session."""
    from psychad_ml.config import get_paths
    from psychad_ml.data.build import run_build
    from psychad_ml.data.synthetic import make_smoke_config
    from psychad_ml.experiments.cv import Ctx

    w = tmp_path_factory.mktemp("smoke")
    cfg = make_smoke_config(w)
    paths = get_paths(cfg)
    run_build(cfg, paths)
    ctx = Ctx.create(cfg, paths)
    ctx.verbose = False
    return cfg, paths, ctx
