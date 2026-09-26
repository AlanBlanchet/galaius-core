"""Contract tests for the permission catalog and group/membership wire shapes
(server's `permissions.py` PermissionRepository is the runtime this precedes)."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from interact_core import PERMISSION_CATALOG, GroupMembership, PermissionGroup


def test_catalog_has_every_admin_and_product_permission_named_in_the_brief() -> None:
    keys = {item.key for item in PERMISSION_CATALOG}
    assert keys == {
        "admin.access", "admin.groups.manage", "admin.users.view", "admin.plans.manage", "admin.audit.view", "admin.contact.view",
        "workflows.run", "models.platform_paid", "machines.pool.use", "machines.cloud.launch",
        "machines.reserve", "credits.auto_topup",
        "company.members.manage", "company.groups.manage", "company.settings.manage", "company.delete",
        "api_keys.manage", "approvals.manage", "billing.view", "billing.manage", "workflows.edit",
        "connections.manage", "agents.edit", "machines.manage",
    }


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
