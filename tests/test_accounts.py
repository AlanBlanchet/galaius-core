"""Contract tests for account/workspace wire shapes: `WorkspaceDeleteRequest` is the confirmation
an owner-only, irreversible workspace delete requires (server's `accounts.py`
`delete_workspace`) — the wire contract precedes the server-side deletion it gates."""

import pytest
from pydantic import ValidationError

from interact_core import WorkspaceDeleteRequest


def test_workspace_delete_request_needs_a_nonblank_confirmation_name() -> None:
    with pytest.raises(ValidationError):
        WorkspaceDeleteRequest(confirm_name="")


def test_workspace_delete_request_carries_the_typed_name_verbatim() -> None:
    assert WorkspaceDeleteRequest(confirm_name="Acme Corp").confirm_name == "Acme Corp"
