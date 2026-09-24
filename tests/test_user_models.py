"""A workspace's own registered model: exactly one origin branch set, resource needs shared with
the cloud placement scheduler."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from interact_core import ResourceRequirement, UserModel, UserModelOrigin


def test_a_user_model_origin_sets_exactly_its_own_branch() -> None:
    UserModelOrigin.model_validate({"kind": "huggingface_repo", "repo_id": "org/model"})
    UserModelOrigin.model_validate({"kind": "docker_image", "image": "ghcr.io/org/model:latest"})
    connection = {"id": str(uuid4()), "revision": str(uuid4()), "capability": "read"}
    UserModelOrigin.model_validate({"kind": "object_storage", "connection": connection, "path": "weights/model.bin"})
    with pytest.raises(ValidationError, match="Hugging Face"):
        UserModelOrigin.model_validate({"kind": "huggingface_repo", "repo_id": "org/model", "image": "also/set"})
    with pytest.raises(ValidationError, match="uploaded weights"):
        UserModelOrigin.model_validate({"kind": "uploaded_weights", "connection": connection, "path": "w.bin"})  # missing digest


def test_a_registered_model_is_runnable_wherever_its_resources_fit() -> None:
    model = UserModel.model_validate({
        "id": str(uuid4()), "revision": str(uuid4()), "name": "My fine-tune", "task": "text-generation",
        "licence": "apache-2.0", "origin": {"kind": "huggingface_repo", "repo_id": "org/model"},
        "resources": {"ram_mb": 16384, "vram_mb": 8192}, "created_at": "2026-09-24T00:00:00Z",
    })
    assert isinstance(model.resources, ResourceRequirement) and model.resources.vram_mb == 8192
