import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from diffusers import DDPMPipeline, DDPMScheduler, DiffusionPipeline, UNet2DModel
from diffusers.pipelines.pipeline_loading_utils import get_class_obj_and_candidates, maybe_raise_or_warn
from diffusers.utils import dynamic_modules_utils
from diffusers.utils.dynamic_modules_utils import get_class_from_dynamic_module


@pytest.fixture
def checkpoint(tmp_path):
    model = UNet2DModel(
        sample_size=8,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(8, 8),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        norm_num_groups=4,
    )
    pipe = DDPMPipeline(model, DDPMScheduler(num_train_timesteps=10))
    directory = tmp_path / "checkpoint"
    pipe.save_pretrained(directory)
    return directory, pipe


def custom_pipeline(directory, marker):
    directory.mkdir(exist_ok=True)
    filename = "pipeline_" + hashlib.sha256(str(directory).encode()).hexdigest()[:12] + ".py"
    module = directory / filename
    module.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        "from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel\n"
        "class TrustedPipeline(DDPMPipeline):\n"
        "    def __init__(self, unet: UNet2DModel, scheduler: DDPMScheduler):\n"
        "        super().__init__(unet, scheduler)\n"
    )
    return module


@pytest.mark.parametrize("route", ["from_pretrained", "from_pipe"])
def test_local_custom_pipelines_require_trust_and_preserve_inference(checkpoint, tmp_path, route):
    directory, original = checkpoint
    marker = tmp_path / "marker"
    custom = tmp_path / "custom"
    custom = custom_pipeline(custom, marker)

    def load(**kwargs):
        if route == "from_pretrained":
            return DiffusionPipeline.from_pretrained(directory, custom_pipeline=str(custom), **kwargs)
        return DiffusionPipeline.from_pipe(original, custom_pipeline=str(custom), **kwargs)

    with pytest.raises(ValueError, match="trust_remote_code"):
        load()
    assert not marker.exists()
    trusted = load(trust_remote_code=True)
    assert marker.read_text() == "executed"
    assert trusted(num_inference_steps=2, output_type="np").images.shape == (1, 8, 8, 3)


@pytest.mark.parametrize("device_map", [None, "balanced"])
def test_local_custom_components_cannot_bypass_trust(checkpoint, tmp_path, device_map):
    directory, original = checkpoint
    marker = tmp_path / "component-marker"
    library_name = "custom_scheduler_" + hashlib.sha256(str(directory).encode()).hexdigest()[:12]
    (directory / "scheduler" / (library_name + ".py")).write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        "from diffusers import DDPMScheduler\nclass CustomScheduler(DDPMScheduler):\n    pass\n"
    )
    config_path = directory / "model_index.json"
    config = json.loads(config_path.read_text())
    config["scheduler"] = [library_name, "CustomScheduler"]
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="trust_remote_code"):
        DiffusionPipeline.from_pretrained(directory, device_map=device_map)
    assert not marker.exists()
    trusted = DiffusionPipeline.from_pretrained(directory, device_map=device_map, trust_remote_code=True)
    assert marker.read_text() == "executed"
    assert trusted(num_inference_steps=2, output_type="np").images.shape == (1, 8, 8, 3)


def test_hub_custom_modules_require_trust_before_dependency_imports(tmp_path, monkeypatch):
    marker = tmp_path / "dependency-marker"
    dependency = "untrusted_security_dependency"
    (tmp_path / (dependency + ".py")).write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    )
    module = tmp_path / "pipeline.py"
    module.write_text(f"import {dependency}\nclass HubClass:\n    value = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(dynamic_modules_utils, "hf_hub_download", lambda *args, **kwargs: str(module))
    monkeypatch.setattr(dynamic_modules_utils, "model_info", lambda *args, **kwargs: SimpleNamespace(sha="a" * 40))
    with pytest.raises(ValueError, match="trust_remote_code"):
        get_class_from_dynamic_module("security-test/model", "pipeline.py", "HubClass")
    assert not marker.exists()
    loaded = get_class_from_dynamic_module("security-test/model", "pipeline.py", "HubClass", trust_remote_code=True)
    assert loaded.value == 42
    assert marker.read_text() == "executed"


@pytest.mark.parametrize("passed_component", [False, True])
def test_configuration_cannot_import_unknown_libraries(tmp_path, monkeypatch, passed_component):
    marker = tmp_path / "library-marker"
    library_name = "untrusted_security_library"
    (tmp_path / (library_name + ".py")).write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError, match="trust_remote_code"):
        if passed_component:
            maybe_raise_or_warn(library_name, None, "Unknown", {}, {"model": object()}, "model", False)
        else:
            get_class_obj_and_candidates(library_name, "Unknown", {}, None, False)
    assert not marker.exists()


def test_global_disable_rejects_explicit_trust_in_a_fresh_process(tmp_path):
    marker = tmp_path / "global-marker"
    module = custom_pipeline(tmp_path / "custom", marker)
    code = (
        "from diffusers.utils.dynamic_modules_utils import get_class_from_dynamic_module\n"
        f"get_class_from_dynamic_module({str(module.parent)!r}, {module.name!r}, trust_remote_code=True)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "DIFFUSERS_DISABLE_REMOTE_CODE": "1"},
        timeout=30,
    )
    assert result.returncode != 0
    assert "DIFFUSERS_DISABLE_REMOTE_CODE" in result.stderr
    assert not marker.exists()


def test_community_classification_cannot_switch_to_local_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "race-marker"
    official = tmp_path / "official.py"
    official.write_text("class CommunityClass:\n    value = 42\n")
    pipeline = tmp_path / "community_security_race" / "pipeline.py"
    original_isfile = os.path.isfile

    def replace_after_check(path):
        result = original_isfile(path)
        if str(path) == "community_security_race/pipeline.py":
            pipeline.parent.mkdir(exist_ok=True)
            pipeline.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n")
        return result

    monkeypatch.setattr(os.path, "isfile", replace_after_check)
    monkeypatch.setattr(dynamic_modules_utils, "get_diffusers_versions", lambda: ["0.32.2"])
    monkeypatch.setattr(dynamic_modules_utils, "hf_hub_download", lambda *args, **kwargs: str(official))
    loaded = get_class_from_dynamic_module("community_security_race", "pipeline.py", "CommunityClass")
    assert loaded.value == 42
    assert not marker.exists()


def test_trusted_hub_relative_imports_keep_the_trust_decision(tmp_path, monkeypatch):
    entry = tmp_path / "entry.py"
    helper = tmp_path / "helper.py"
    entry.write_text("from .helper import VALUE\nclass RelativeClass:\n    value = VALUE\n")
    helper.write_text("VALUE = 73\n")
    monkeypatch.setattr(
        dynamic_modules_utils, "hf_hub_download", lambda repo, filename, **kwargs: str(tmp_path / filename)
    )
    monkeypatch.setattr(dynamic_modules_utils, "model_info", lambda *args, **kwargs: SimpleNamespace(sha="b" * 40))
    loaded = get_class_from_dynamic_module(
        "security-test/relative", "entry.py", "RelativeClass", trust_remote_code=True
    )
    assert loaded.value == 73


def test_builtin_pipeline_conversion_consumes_the_trust_option(checkpoint):
    directory, original = checkpoint
    converted = DDPMPipeline.from_pipe(original, unet=original.unet, trust_remote_code=True)
    assert converted(num_inference_steps=2, output_type="np").images.shape == (1, 8, 8, 3)


def test_configuration_selected_custom_pipeline_requires_trust(checkpoint, tmp_path):
    directory, original = checkpoint
    marker = tmp_path / "config-marker"
    module = custom_pipeline(directory, marker)
    config_path = directory / "model_index.json"
    config = json.loads(config_path.read_text())
    config["_class_name"] = [module.stem, "TrustedPipeline"]
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="trust_remote_code"):
        DiffusionPipeline.from_pretrained(directory)
    assert not marker.exists()
    loaded = DiffusionPipeline.from_pretrained(directory, trust_remote_code=True)
    assert marker.read_text() == "executed"
    assert loaded(num_inference_steps=2, output_type="np").images.shape == (1, 8, 8, 3)
