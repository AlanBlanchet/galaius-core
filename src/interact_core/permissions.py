"""Deny-by-default platform permissions: one flat catalog, groups hold a set of permissions,
accounts and workspaces join groups, effective permission = union of every group a member is in.
Two axes stay separate on purpose: ADMIN permissions (`admin.*`) are granted to ACCOUNTS (Alan
signs in with Google, the account is what carries platform-admin rights); PRODUCT permissions
(everything else — the capabilities a subscription unlocks) are granted to WORKSPACES (a
workspace runs workflows, spends credits, launches machines — never an individual account).
`GroupMembership.member_kind` carries this so ONE group/membership model serves both axes instead
of two parallel ones.

Company scope (Alan, 2026-09-25: "A company should also be able to manage it's users... Manage
members ? And assign some users into for instance a billing group, dev group etc... And they have
rights.") reuses the SAME group/membership model: a group whose `workspace_id` is set belongs to
that company (a workspace IS a company here — `CompanyProfile` hangs off it), holds only
`company`-grantable permissions, and has only that company's member ACCOUNTS as members. A
company member's effective rights = union of that company's groups they are in; membership of the
company alone grants reading it, nothing more. The old per-workspace roles
(owner/admin/member/viewer) are exactly `DEFAULT_COMPANY_GROUPS`, seeded into every company."""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator
from pydantic.experimental.missing_sentinel import MISSING

from .wire import WireModel

Permission = Literal[
    "admin.access",
    "admin.groups.manage",
    "admin.users.view",
    "admin.plans.manage",
    "admin.audit.view",
    "admin.contact.view",
    "workflows.run",
    "models.platform_paid",
    "machines.pool.use",
    "machines.cloud.launch",
    "machines.reserve",
    "credits.auto_topup",
    "company.members.manage",
    "company.groups.manage",
    "company.settings.manage",
    "company.delete",
    "api_keys.manage",
    "approvals.manage",
    "billing.view",
    "billing.manage",
    "workflows.edit",
    "connections.manage",
    "agents.edit",
    "machines.manage",
]
"""The complete permission vocabulary — a permission not in this tuple can never be attached to a
group or checked, so a typo is a validation error, never a silently-ignored no-op string."""

PermissionScope = Literal["platform", "company"]
"""WHERE a permission can be granted: `platform` = a platform group (no `workspace_id`) — admin
rights to accounts, product entitlements to workspaces; `company` = one company's own group, to
that company's member accounts. A key may be grantable in both (`workflows.run`: the platform
decides whether a workspace's plan may run workflows, the company decides which members may)."""

MemberKind = Literal["account", "workspace"]
MembershipSource = Literal["manual", "subscription"]


class PermissionInfo(WireModel):
    """One catalog row: the key a group stores, plus the human label + description an admin UI
    renders beside its checkbox — never bare enum values on screen."""

    key: Permission
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=240)
    label_fr: str = Field(min_length=1, max_length=80)
    description_fr: str = Field(min_length=1, max_length=240)
    scopes: tuple[PermissionScope, ...] = Field(min_length=1)


PERMISSION_CATALOG: tuple[PermissionInfo, ...] = (
    PermissionInfo(key="admin.access", label="Admin panel access", description="Sign into the admin panel at all; every other admin permission requires this too.", label_fr='Accès au panneau d’administration', description_fr='Ouvrir le panneau d’administration ; toute autre permission d’administration l’exige aussi.', scopes=("platform",)),
    PermissionInfo(key="admin.groups.manage", label="Manage groups", description="Create, rename and delete permission groups; edit which permissions and which accounts/workspaces belong to each.", label_fr='Gérer les groupes', description_fr='Créer, renommer et supprimer des groupes de permissions ; choisir leurs permissions et leurs comptes ou espaces membres.', scopes=("platform",)),
    PermissionInfo(key="admin.users.view", label="View accounts & workspaces", description="See every account, every workspace, service totals and run failures.", label_fr='Voir comptes et espaces', description_fr='Voir tous les comptes, tous les espaces, les totaux du service et les échecs d’exécution.', scopes=("platform",)),
    PermissionInfo(key="admin.plans.manage", label="Manage subscription plans", description="Create and edit subscription plans, and assign or change a workspace's plan.", label_fr='Gérer les formules d’abonnement', description_fr='Créer et modifier les formules, et attribuer ou changer la formule d’un espace.', scopes=("platform",)),
    PermissionInfo(key="admin.audit.view", label="View audit log", description="See the log of every admin action taken, by whom, on what.", label_fr='Voir le journal d’audit', description_fr='Voir chaque action d’administration : qui, quand, sur quoi.', scopes=("platform",)),
    PermissionInfo(key="admin.contact.view", label="Read contact messages", description="Read the messages visitors send through the public site's contact form, and mark them read.", label_fr='Lire les messages de contact', description_fr='Lire les messages envoyés par les visiteurs via le formulaire de contact du site public, et les marquer comme lus.', scopes=("platform",)),
    PermissionInfo(key="workflows.run", label="Run workflows", description="Execute saved workflows, their steps and triggers, conversations and model runs; cancel runs.", label_fr='Exécuter les workflows', description_fr='Lancer les workflows enregistrés, leurs étapes et déclencheurs, les conversations et les exécutions de modèles ; annuler des exécutions.', scopes=("platform", "company")),
    PermissionInfo(key="models.platform_paid", label="Use platform-paid models", description="Call a model through this platform's own provider keys, billed at cost plus the platform fee.", label_fr='Utiliser les modèles payés par la plateforme', description_fr='Appeler un modèle avec les clés de la plateforme, facturé au coût plus la commission.', scopes=("platform",)),
    PermissionInfo(key="machines.pool.use", label="Use pooled machines", description="Dispatch a run onto another workspace's shared machine, billed to this workspace's credits.", label_fr='Utiliser les machines partagées', description_fr='Exécuter sur la machine partagée d’un autre espace, facturé sur les crédits de cet espace.', scopes=("platform",)),
    PermissionInfo(key="machines.cloud.launch", label="Launch cloud machines", description="Have the platform provision a cloud instance (e.g. Scaleway) on this workspace's behalf.", label_fr='Lancer des machines cloud', description_fr='Laisser la plateforme louer une instance cloud (ex. Scaleway) pour cet espace.', scopes=("platform",)),
    PermissionInfo(key="machines.reserve", label="Reserve machines", description="Hold a machine (own, pooled or cloud) for a fixed number of hours, pre-paid from credits.", label_fr='Réserver des machines', description_fr='Bloquer une machine (propre, partagée ou cloud) pour un nombre d’heures, prépayé en crédits.', scopes=("platform",)),
    PermissionInfo(key="credits.auto_topup", label="Enable auto top-up", description="Let this workspace's saved card be charged automatically when its credit balance runs low.", label_fr='Activer la recharge automatique', description_fr='Autoriser le débit automatique de la carte enregistrée quand le solde de crédits est bas.', scopes=("platform",)),
    PermissionInfo(key="company.members.manage", label="Manage members", description="Invite people by email, remove members, cancel invitations, and add or remove members from this company's groups.", label_fr='Gérer les membres', description_fr='Inviter par e-mail, retirer des membres, annuler des invitations, et ajouter ou retirer des membres des groupes de l’organisation.', scopes=("company",)),
    PermissionInfo(key="company.groups.manage", label="Manage groups", description="Create, rename and delete this company's groups and choose the permissions each one grants.", label_fr='Gérer les groupes', description_fr='Créer, renommer et supprimer les groupes de l’organisation et choisir les permissions de chacun.', scopes=("company",)),
    PermissionInfo(key="company.settings.manage", label="Manage company settings", description="Rename the company and edit its profile and logo.", label_fr='Gérer les réglages de l’organisation', description_fr='Renommer l’organisation et modifier son profil et son logo.', scopes=("company",)),
    PermissionInfo(key="company.delete", label="Delete the company", description="Irreversibly delete this company and everything in it.", label_fr='Supprimer l’organisation', description_fr='Supprimer définitivement cette organisation et tout son contenu.', scopes=("company",)),
    PermissionInfo(key="api_keys.manage", label="Manage API keys", description="Create, list and revoke API keys that act on this company programmatically.", label_fr='Gérer les clés API', description_fr='Créer, lister et révoquer les clés API qui agissent sur cette organisation.', scopes=("company",)),
    PermissionInfo(key="approvals.manage", label="Approve agent actions", description="Approve or deny actions agents wait on (remote actions, outgoing emails) and set their approval policies.", label_fr='Approuver les actions des agents', description_fr='Approuver ou refuser les actions en attente des agents (actions distantes, e-mails sortants) et régler leurs règles d’approbation.', scopes=("company",)),
    PermissionInfo(key="billing.view", label="View billing", description="See the credit balance, ledger, invoices, subscription, budgets, reservations and cost usage.", label_fr='Voir la facturation', description_fr='Voir le solde de crédits, le journal, les factures, l’abonnement, les budgets, les réservations et les coûts.', scopes=("company",)),
    PermissionInfo(key="billing.manage", label="Manage billing", description="Buy credits, set auto top-up, subscribe or change plan, edit the tax profile, set budgets and reserve machines.", label_fr='Gérer la facturation', description_fr='Acheter des crédits, régler la recharge automatique, s’abonner ou changer de formule, modifier le profil fiscal, fixer les budgets et réserver des machines.', scopes=("company",)),
    PermissionInfo(key="workflows.edit", label="Edit workflows", description="Create, change and delete workflows, their triggers and the node library.", label_fr='Modifier les workflows', description_fr='Créer, modifier et supprimer les workflows, leurs déclencheurs et la bibliothèque de nœuds.', scopes=("company",)),
    PermissionInfo(key="connections.manage", label="Manage connections", description="Add, test and delete connections and their secrets, and link external accounts (Google, Microsoft).", label_fr='Gérer les connexions', description_fr='Ajouter, tester et supprimer les connexions et leurs secrets, et lier des comptes externes (Google, Microsoft).', scopes=("company",)),
    PermissionInfo(key="agents.edit", label="Edit agents", description="Create and change agents, the agent graph, prompts, model preferences and own models.", label_fr='Modifier les agents', description_fr='Créer et modifier les agents, le graphe d’agents, les prompts, les préférences de modèles et les modèles propres.', scopes=("company",)),
    PermissionInfo(key="machines.manage", label="Manage machines", description="Enroll and revoke machines, approve the scripts they run, set their cost rate and pool sharing.", label_fr='Gérer les machines', description_fr='Enrôler et révoquer des machines, approuver leurs scripts, régler leur coût et leur partage.', scopes=("company",)),
)


class PermissionGroup(WireModel):
    id: UUID
    workspace_id: UUID | None = None
    """The company this group belongs to; `None` = a platform group (admin panel)."""
    name: str = Field(min_length=1, max_length=120)
    permissions: tuple[Permission, ...] = Field(default=())
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def unique_permissions(self) -> Self:
        if len(set(self.permissions)) != len(self.permissions):
            raise ValueError("a group's permissions must be unique")
        return self


class GroupCreate(WireModel):
    name: str = Field(min_length=1, max_length=120)
    permissions: tuple[Permission, ...] = Field(default=())


class GroupUpdate(WireModel):
    name: str | MISSING = MISSING
    permissions: tuple[Permission, ...] | MISSING = MISSING


class GroupMembership(WireModel):
    group_id: UUID
    member_kind: MemberKind
    member_id: UUID
    source: MembershipSource
    added_at: datetime


class GroupMemberView(WireModel):
    """`GroupMembership` plus the one display fact a picker needs (an account's email, a
    workspace's name) — resolved server-side so the admin UI never fans out N+1 lookups."""

    membership: GroupMembership
    label: str = Field(min_length=1, max_length=320)


class EffectivePermissions(WireModel):
    member_kind: MemberKind
    member_id: UUID
    permissions: tuple[Permission, ...]
    groups: tuple[UUID, ...]
    """Which groups contributed — so a UI can explain "why do I have this" without a second call."""


DEFAULT_COMPANY_GROUPS: tuple[GroupCreate, ...] = (
    GroupCreate(name="Owners", permissions=tuple(item.key for item in PERMISSION_CATALOG if "company" in item.scopes)),
    GroupCreate(name="Admins", permissions=tuple(item.key for item in PERMISSION_CATALOG if "company" in item.scopes and item.key not in ("company.delete", "machines.manage"))),
    GroupCreate(name="Members", permissions=("billing.view", "workflows.edit", "workflows.run", "connections.manage", "agents.edit")),
    GroupCreate(name="Viewers", permissions=("billing.view",)),
)
"""Seeded into every company; the old per-workspace roles, one group each (owner-only rights —
deleting the company, enrolling machines and approving their scripts — stay Owners-only). A
company renames, edits or deletes them like any group it created itself."""
