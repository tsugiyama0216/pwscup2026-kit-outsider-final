"""Subprocess entry point: only authorized input tables reach the plugin API."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mock_TS_using_prelim.common import install_official, require, write_json
from mock_TS_using_prelim.runner import public_tables


def main():
    bundle, plugin, output = map(Path, sys.argv[1:])
    install_official()
    inputs = public_tables(bundle)
    spec = importlib.util.spec_from_file_location("prelim_attack_" + uuid4().hex, plugin)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    function = getattr(module, "predict_world", None)
    require(callable(function), "Plugin must expose predict_world(**public_tables)")
    signature = inspect.signature(function)
    accepts_all = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    arguments = inputs if accepts_all else {k: v for k, v in inputs.items() if k in signature.parameters}
    # No target truth, archived prediction, floor estimate or source data path is passed.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        result = function(**arguments)
    require(isinstance(result, dict), "Plugin must return a dictionary")
    for key in ("F_mia", "F_aia"):
        frame = result.get(key)
        require(hasattr(frame, "to_csv"), key + " must be a pandas DataFrame")
        frame.to_csv(output / (key + ".csv"), index=False, lineterminator="\n", float_format="%.17g")
    write_json(output / "plugin_metadata.json", {
        "name": getattr(module, "MODEL_NAME", plugin.stem),
        "arguments": list(arguments), "diagnostics": result.get("diagnostics", {}),
    })


if __name__ == "__main__":
    main()
