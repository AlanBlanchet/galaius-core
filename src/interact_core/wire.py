"""Validation shared by immutable Interact wire contracts."""

from pydantic import BaseModel, ConfigDict


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
