"""Contract tests for the permission catalog and group/membership wire shapes
(the server's permission store is the runtime this precedes)."""

from datetime import UTC, datetime
from typing import get_args
from uuid import uuid4

import pytest
from pydantic import ValidationError

from galaius_core import PERMISSION_CATALOG, GroupMembership, Permission, PermissionGroup


def test_catalog_holds_one_row_per_permission_key() -> None:
    keys = [item.key for item in PERMISSION_CATALOG]
    assert sorted(keys) == sorted(get_args(Permission))


def test_company_rights_read_in_category_blocks() -> None:
    """The people screen renders one heading per category in catalog order: a category never splits."""
    categories = [item.group for item in PERMISSION_CATALOG if "company" in item.scopes]
    blocks = [group for index, group in enumerate(categories) if index == 0 or categories[index - 1] != group]
    assert len(blocks) == len(set(blocks))


def test_catalog_entries_all_carry_a_label_and_description() -> None:
    assert all(item.label and item.description for item in PERMISSION_CATALOG)


def test_permission_group_rejects_an_unknown_permission_key() -> None:
    with pytest.raises(ValidationError):
        PermissionGroup(id=uuid4(), name="Administrators", permissions=("admin.access", "not.a.real.permission"), created_at=datetime.now(UTC), updated_at=datetime.now(UTC))


def test_permission_group_rejects_duplicate_permissions() -> None:
    with pytest.raises(ValidationError):
        PermissionGroup(id=uuid4(), name="Administrators", permissions=("admin.access", "admin.access"), created_at=datetime.now(UTC), updated_at=datetime.now(UTC))


def test_group_membership_carries_kind_and_source() -> None:
    membership = GroupMembership(group_id=uuid4(), member_kind="workspace", member_id=uuid4(), source="subscription", added_at=datetime.now(UTC))
    assert membership.member_kind == "workspace"
    assert membership.source == "subscription"
