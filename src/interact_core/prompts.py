"""Typed wire contracts shared by local clients and the private prompt service."""

import hashlib
import re
from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from .wire import WireModel

_PART = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class PromptKey(WireModel):
    """Stable prompt identity independent of revisions and publication channels."""

    namespace: str
    slug: str

    @model_validator(mode="after")
    def validate_parts(self) -> Self:
        if not _PART.fullmatch(self.namespace) or not _PART.fullmatch(self.slug):
            raise ValueError("prompt namespace and slug must be lowercase kebab-case")
        return self


class PromptRevision(WireModel):
    """Immutable prompt content and its verifiable publication provenance."""

    key: PromptKey
    revision: UUID
    parent_digest: str | None = None
    digest: str
    content: str
    source_commit: str
    created_at: datetime

    @model_validator(mode="after")
    def verify_digest(self) -> Self:
        if self.parent_digest is not None and not _DIGEST.fullmatch(self.parent_digest):
            raise ValueError("parent digest must be lowercase SHA-256")
        if not _DIGEST.fullmatch(self.digest):
            raise ValueError("digest must be lowercase SHA-256")
        if hashlib.sha256(self.content.encode()).hexdigest() != self.digest:
            raise ValueError("prompt content digest does not match")
        return self


class PromptCreateRequest(WireModel):
    """Hostile authoring input for the first immutable Git-backed prompt revision."""

    key: PromptKey
    name: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=1 << 20)


class PromptChannelEntry(WireModel):
    """Compare-and-swap channel pointer to one immutable prompt revision."""

    key: PromptKey
    channel: str
    revision: UUID
    digest: str
    lock_version: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_wire_values(self) -> Self:
        if not _PART.fullmatch(self.channel):
            raise ValueError("prompt channel must be lowercase kebab-case")
        if not _DIGEST.fullmatch(self.digest):
            raise ValueError("digest must be lowercase SHA-256")
        return self


class _PromptIdentity(WireModel):
    key: PromptKey
    channel: str
    digest: str

    @model_validator(mode="after")
    def validate_wire_values(self) -> Self:
        if not _PART.fullmatch(self.channel):
            raise ValueError("prompt channel must be lowercase kebab-case")
        if not _DIGEST.fullmatch(self.digest):
            raise ValueError("digest must be lowercase SHA-256")
        return self


class PromptSelection(_PromptIdentity):
    """Untrusted channel and digest requested for server-backed resolution."""


class PromptExecutionRef(_PromptIdentity):
    """Server-verified immutable prompt identity persisted with one execution."""

    revision: UUID


class PromptCatalogPage(WireModel):
    """One typed, cursor-addressed delta from the prompt catalogue."""

    entries: tuple[PromptChannelEntry, ...]
    removed: tuple[PromptKey, ...] = ()
    cursor: str | None = None
    server_timestamp: datetime

    def by_key(self) -> dict[tuple[str, str], PromptChannelEntry]:
        return {(entry.key.namespace, entry.key.slug): entry for entry in self.entries}


class PromptPublicationRequest(WireModel):
    """One complete exact-commit prompt snapshot applied by global cursor CAS."""

    expected_cursor: str | None = None
    source_commit: str
    entries: tuple[PromptSelection, ...]
    revisions: tuple[PromptRevision, ...]

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if self.expected_cursor is not None and not _DIGEST.fullmatch(self.expected_cursor):
            raise ValueError("publication cursor must be lowercase SHA-256")
        if not _GIT_OBJECT.fullmatch(self.source_commit):
            raise ValueError("publication source commit must be a lowercase Git object id")
        entries = [(entry.key.namespace, entry.key.slug) for entry in self.entries]
        if entries != sorted(entries) or len(set(entries)) != len(entries):
            raise ValueError("publication entries must have unique ordered keys")
        if any(entry.channel != "stable" for entry in self.entries):
            raise ValueError("publication entries must select the stable channel")
        identities = [
            (revision.key.namespace, revision.key.slug, revision.digest)
            for revision in self.revisions
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("publication revision key and digest pairs must be unique")
        revision_ids = [revision.revision for revision in self.revisions]
        if len(set(revision_ids)) != len(revision_ids):
            raise ValueError("publication revision identifiers must be unique")
        selected = {
            (entry.key.namespace, entry.key.slug, entry.digest) for entry in self.entries
        }
        if any(
            (revision.key.namespace, revision.key.slug, revision.digest) not in selected
            for revision in self.revisions
        ):
            raise ValueError("publication revisions must be referenced by an entry")
        if any(revision.source_commit != self.source_commit for revision in self.revisions):
            raise ValueError("publication revisions must bind the source commit")
        return self


class PromptGitBundleManifest(WireModel):
    """Integrity and compare-and-swap metadata for one bounded Git bundle."""

    expected_head: str | None = None
    head: str
    manifest_digest: str
    bundle_digest: str
    bundle_size: int = Field(ge=1, le=16 * 1024 * 1024)

    @model_validator(mode="after")
    def validate_objects(self) -> Self:
        if self.expected_head is not None and not _GIT_OBJECT.fullmatch(self.expected_head):
            raise ValueError("expected prompt Git head must be a lowercase object id")
        if not _GIT_OBJECT.fullmatch(self.head):
            raise ValueError("prompt Git head must be a lowercase object id")
        if not _DIGEST.fullmatch(self.manifest_digest) or not _DIGEST.fullmatch(self.bundle_digest):
            raise ValueError("prompt Git bundle digests must be lowercase SHA-256")
        return self


class PromptSyncStatus(WireModel):
    """Safe server-side projection of the latest prompt Git synchronization."""

    server_head: str | None = None
    last_import_at: datetime | None = None
    last_export_at: datetime | None = None
    last_result: Literal["succeeded", "conflict", "unavailable"] | None = None
    error_code: Literal["head_changed", "non_fast_forward", "invalid_bundle", "repository_unavailable"] | None = None

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.server_head is not None and not _GIT_OBJECT.fullmatch(self.server_head):
            raise ValueError("prompt sync server head must be a lowercase Git object id")
        if (self.last_result in {None, "succeeded"}) != (self.error_code is None):
            raise ValueError("successful prompt sync status cannot include an error code")
        return self
