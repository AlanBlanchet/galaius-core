"""A workspace's own registered model: exactly one origin branch set, resource needs shared with
the cloud placement scheduler, safetensors-only weights (threat-model
cloud-compute-and-sovereignty-2026-09-24.md #5) and bundled-code origins sandboxed like a script."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from interact_core import ResourceRequirement, UserModel, UserModelOrigin

CONNECTION = {"id": str(uuid4()), "revision": str(uuid4()), "capability": "read"}


def test_a_user_model_origin_sets_exactly_its_own_branch() -> None:
    UserModelOrigin.model_validate({"kind": "huggingface_repo", "repo_id": "org/model", "weight_files": ["model.safetensors"]})
    UserModelOrigin.model_validate({"kind": "docker_image", "image": "ghcr.io/org/model:latest"})
    UserModelOrigin.model_validate({"kind": "object_storage", "connection": CONNECTION, "path": "weights/model.safetensors"})
    with pytest.raises(ValidationError, match="Hugging Face"):
        UserModelOrigin.model_validate({"kind": "huggingface_repo", "repo_id": "org/model", "weight_files": ["m.safetensors"], "image": "also/set"})
    with pytest.raises(ValidationError, match="uploaded weights"):
        UserModelOrigin.model_validate({"kind": "uploaded_weights", "connection": CONNECTION, "path": "w.safetensors"})  # missing digest


def test_only_safetensors_weights_are_ever_accepted() -> None:
    with pytest.raises(ValidationError, match="safetensors"):
        UserModelOrigin.model_validate({"kind": "huggingface_repo", "repo_id": "org/model", "weight_files": ["pytorch_model.bin"]})
    with pytest.raises(ValidationError, match="safetensors"):
        UserModelOrigin.model_validate({"kind": "huggingface_repo", "repo_id": "org/model"})  # no weight_files at all
    with pytest.raises(ValidationError, match="safetensors"):
        UserModelOrigin.model_validate({"kind": "uploaded_weights", "connection": CONNECTION, "path": "w.bin", "digest": "a" * 64})
    with pytest.raises(ValidationError, match="safetensors"):
        UserModelOrigin.model_validate({"kind": "object_storage", "connection": CONNECTION, "path": "weights/model.pt"})
    with pytest.raises(ValidationError, match="Hugging Face origin"):
        UserModelOrigin.model_validate({"kind": "docker_image", "image": "ghcr.io/org/model:latest", "weight_files": ["x.safetensors"]})


def test_bundled_code_origins_route_to_the_script_sandbox_tier() -> None:
    weights_only = UserModelOrigin.model_validate({"kind": "huggingface_repo", "repo_id": "org/model", "weight_files": ["model.safetensors"]})
    assert weights_only.runs_bundled_code is False and weights_only.sandbox_tier == "none"
    trusted_code = UserModelOrigin.model_validate({"kind": "huggingface_repo", "repo_id": "org/model", "weight_files": ["model.safetensors"], "trusts_remote_code": True})
    assert trusted_code.runs_bundled_code is True and trusted_code.sandbox_tier == "gvisor"
    container = UserModelOrigin.model_validate({"kind": "docker_image", "image": "ghcr.io/org/model:latest"})
    assert container.runs_bundled_code is True and container.sandbox_tier == "gvisor"  # a container is arbitrary code by definition, never opt-out
    with pytest.raises(ValidationError, match="Hugging Face origin"):
        UserModelOrigin.model_validate({"kind": "object_storage", "connection": CONNECTION, "path": "w.safetensors", "trusts_remote_code": True})


def test_a_registered_model_is_runnable_wherever_its_resources_fit() -> None:
    model = UserModel.model_validate({
        "id": str(uuid4()), "revision": str(uuid4()), "name": "My fine-tune", "task": "text-generation",
        "licence": "apache-2.0", "origin": {"kind": "huggingface_repo", "repo_id": "org/model", "weight_files": ["model.safetensors"]},
        "resources": {"ram_mb": 16384, "vram_mb": 8192}, "created_at": "2026-09-24T00:00:00Z",
    })
    assert isinstance(model.resources, ResourceRequirement) and model.resources.vram_mb == 8192
    assert model.weight_format == "safetensors"
