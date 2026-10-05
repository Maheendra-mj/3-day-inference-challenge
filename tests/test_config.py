from pathlib import Path

import pytest

from infer_lab.backends import create_backend
from infer_lab.config import BenchConfig
from infer_lab.telemetry.env import env_info

CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.mark.parametrize("name", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_all_configs_parse(name):
    cfg = BenchConfig.from_yaml(CONFIGS / name)
    assert cfg.sweep.batch_sizes and cfg.model.name


def test_vllm_configs_differ_only_in_cuda_graphs():
    graphs = BenchConfig.from_yaml(CONFIGS / "vllm_offline.yaml")
    eager = BenchConfig.from_yaml(CONFIGS / "vllm_offline_eager.yaml")
    assert graphs.backend_args["enforce_eager"] is False
    assert eager.backend_args["enforce_eager"] is True
    assert {**graphs.backend_args, "enforce_eager": True} == eager.backend_args
    assert graphs.sweep == eager.sweep == BenchConfig.from_yaml(CONFIGS / "bench.yaml").sweep


def test_unknown_backend_lists_available():
    with pytest.raises(ValueError, match="vllm_offline"):
        create_backend("nope")


def test_env_info_works_without_gpu():
    info = env_info()
    assert info["python"] and info["cpu_count"] >= 1 and "libs" in info
