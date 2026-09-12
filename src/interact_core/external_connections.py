"""Provider-independent external account and model connection contracts."""

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from .wire import WireModel


class GoogleConnectionStatus(WireModel):
    provider: Literal["google"] = "google"
    configured: bool
    connected: bool
    account_label: str | None = Field(default=None, max_length=320)
    scopes: tuple[str, ...] = Field(default=(), max_length=16)
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def consistent_state(self) -> Self:
        if not self.connected and (self.account_label is not None or self.scopes or self.expires_at is not None):
            raise ValueError("disconnected Google status cannot expose credential metadata")
        if self.connected and (not self.configured or self.account_label is None):
            raise ValueError("connected Google status requires configured account metadata")
        return self


class GoogleOAuthAuthorization(WireModel):
    authorization_url: str = Field(min_length=1, max_length=4096)
    state: str = Field(min_length=32, max_length=256)
    expires_at: datetime


class GoogleOAuthCallback(WireModel):
    state: str = Field(min_length=32, max_length=256)
    code: str = Field(min_length=1, max_length=4096)


class GoogleOAuthStart(WireModel):
    gmail: tuple[Literal["metadata", "read", "send"], ...] = Field(default=("metadata",), min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique_capabilities(self) -> Self:
        if len(set(self.gmail)) != len(self.gmail):
            raise ValueError("Gmail capabilities must be unique")
        return self


class GeminiConnectionConfiguration(WireModel):
    name: str = Field(min_length=1, max_length=120)
    models: tuple[str, ...] = Field(min_length=1, max_length=64)
