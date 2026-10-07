"""Frozen selected R v2 public-input adapter for Sugiyama evaluation."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL_NAME = "Attack_R/v2"


def predict_world(**public):
    rt = _load("_rv2_plugin_runtime", "runtime.py")
    model = _load("_rv2_plugin_model", "model.py")
    selected = rt.selected_check()
    before = rt.upstream_check()
    result = model.selected_world(selected, **public)
    if rt.upstream_check() != before:
        raise ValueError("Upstream changed during inference")
    rt.selected_check()
    result["diagnostics"].update({"model": MODEL_NAME, "selected": {k: selected[k] for k in ("mia", "aia")}})
    return result
