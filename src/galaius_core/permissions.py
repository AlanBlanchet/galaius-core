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

from .wire import WireModel, WireRequest

Permission = Literal[
    "admin.access",
    "admin.groups.manage",
    "admin.users.view",
    "admin.plans.manage",
    "admin.audit.view",
    "admin.contact.view",
    "admin.signals.view",
    "agents.view",
    "agents.run",
    "workflows.run",
    "agents.edit",
    "prompts.edit",
    "tools.manage",
    "workflows.edit",
    "connections.manage",
    "accounts.connect_own",
    "machines.manage",
    "billing.view",
    "billing.manage",
    "approvals.manage",
    "api_keys.manage",
    "company.members.manage",
    "company.groups.manage",
    "company.settings.manage",
    "company.delete",
    "models.platform_paid",
    "machines.pool.use",
    "machines.cloud.launch",
    "machines.reserve",
    "credits.auto_topup",
]
"""The complete permission vocabulary — a permission not in this tuple can never be attached to a
group or checked, so a typo is a validation error, never a silently-ignored no-op string."""

PermissionScope = Literal["platform", "company"]
"""WHERE a permission can be granted: `platform` = a platform group (no `workspace_id`) — admin
rights to accounts, product entitlements to workspaces; `company` = one company's own group, to
that company's member accounts. A key may be grantable in both (`workflows.run`: the platform
decides whether a workspace's plan may run workflows, the company decides which members may)."""

PermissionCategory = Literal["agents", "pcs_data", "build", "billing", "admin", "platform"]
"""The section a permission sits in on screen (company rights in catalog order, one heading per
category); `platform`: the admin panel's own rights and plan entitlements."""

MemberKind = Literal["account", "workspace", "agent"]
"""`agent`: an agent placed in a company DEPARTMENT (never a rights-bearing member: a department
gives its agents no permission, it says which part of the company runs them)."""
MembershipSource = Literal["manual", "subscription"]

GroupKind = Literal["group", "department", "company"]
"""`group`: an access group (Owners, Admins, Billing...), permissions only. `department`: a part of
the company (Sales, Support...) — same permissions its PEOPLE inherit, plus a reporting line
(`parent_id`), an owner (transfers / deletes it) and a manager ("responsable": accountable for it,
adds and removes its people and the agents they own or manage). `company`: the company itself, the
root of its tree, one per company, made with it: what it gives every member gets, nobody sits at it,
it has no owner, manager or parent. Platform groups are always `group`."""


class PermissionInfo(WireModel):
    """One catalog row: the key a group stores, plus the human label + description an admin UI
    renders beside its checkbox — never bare enum values on screen. Catalog order is display order."""

    key: Permission
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=240)
    label_fr: str = Field(min_length=1, max_length=80)
    description_fr: str = Field(min_length=1, max_length=240)
    group: PermissionCategory
    scopes: tuple[PermissionScope, ...] = Field(min_length=1)


PERMISSION_CATALOG: tuple[PermissionInfo, ...] = (
    PermissionInfo(key="admin.access", label="Admin panel access", description="Sign into the admin panel at all; every other admin permission requires this too.", label_fr='Accès au panneau d’administration', description_fr='Ouvrir le panneau d’administration ; toute autre permission d’administration l’exige aussi.', group="platform", scopes=("platform",)),
    PermissionInfo(key="admin.groups.manage", label="Manage groups", description="Create, rename and delete permission groups; edit which permissions and which accounts/workspaces belong to each.", label_fr='Gérer les groupes', description_fr='Créer, renommer et supprimer des groupes de permissions ; choisir leurs permissions et leurs comptes ou espaces membres.', group="platform", scopes=("platform",)),
    PermissionInfo(key="admin.users.view", label="View accounts & workspaces", description="See every account, every workspace, service totals and run failures.", label_fr='Voir comptes et espaces', description_fr='Voir tous les comptes, tous les espaces, les totaux du service et les échecs d’exécution.', group="platform", scopes=("platform",)),
    PermissionInfo(key="admin.plans.manage", label="Manage subscription plans", description="Create and edit subscription plans, and assign or change a workspace's plan.", label_fr='Gérer les formules d’abonnement', description_fr='Créer et modifier les formules, et attribuer ou changer la formule d’un espace.', group="platform", scopes=("platform",)),
    PermissionInfo(key="admin.audit.view", label="View audit log", description="See the log of every admin action taken, by whom, on what.", label_fr='Voir le journal d’audit', description_fr='Voir chaque action d’administration : qui, quand, sur quoi.', group="platform", scopes=("platform",)),
    PermissionInfo(key="admin.contact.view", label="Read contact messages", description="Read the messages visitors send through the public site's contact form, mark them read, and delete them.", label_fr='Lire les messages de contact', description_fr='Lire les messages envoyés par les visiteurs via le formulaire de contact du site public, les marquer comme lus et les supprimer.', group="platform", scopes=("platform",)),
    PermissionInfo(key="admin.signals.view", label="Read and triage signals", description="Read what users signal from the app (their words, the screen and element, build, errors, screenshot), set new / triaged / fixed, and delete.", label_fr='Lire et trier les signalements', description_fr='Lire ce que les utilisateurs signalent depuis l’application (leurs mots, l’écran et l’élément, la version, les erreurs, la capture), passer en nouveau / trié / corrigé, et supprimer.', group="platform", scopes=("platform",)),
    PermissionInfo(key="agents.view", label="See agents", description="List and open the company's agents, their runs and transcripts.", label_fr="Voir les agents", description_fr="Lister et ouvrir les agents de l’organisation, leurs exécutions et leurs transcriptions.", group="agents", scopes=("company",)),
    PermissionInfo(key="agents.run", label="Run agents", description="Start chats and agent runs, attach places to a chat, and cancel those runs.", label_fr="Lancer les agents", description_fr="Démarrer des conversations et des exécutions d’agents, donner des emplacements à une conversation, et annuler ces exécutions.", group="agents", scopes=("company",)),
    PermissionInfo(key="workflows.run", label="Run workflows", description="Execute saved workflows, their steps and triggers, and model runs; cancel workflow runs.", label_fr='Exécuter les workflows', description_fr='Lancer les workflows enregistrés, leurs étapes et déclencheurs, et les exécutions de modèles ; annuler des exécutions de workflows.', group="agents", scopes=("platform", "company")),
    PermissionInfo(key="agents.edit", label="Edit agents", description="Create and change agent definitions, the agent graph, model preferences and own models.", label_fr='Modifier les agents', description_fr='Créer et modifier les définitions d’agents, le graphe d’agents, les préférences de modèles et les modèles propres.', group="build", scopes=("company",)),
    PermissionInfo(key="prompts.edit", label="Manage prompts", description="Create, change, publish and copy prompts and skills.", label_fr="Gérer les prompts", description_fr="Créer, modifier, publier et copier les prompts et les compétences.", group="build", scopes=("company",)),
    PermissionInfo(key="tools.manage", label="Manage tools", description="Add, approve, refresh and remove the tool servers (MCP) agents call.", label_fr="Gérer les outils", description_fr="Ajouter, approuver, actualiser et retirer les serveurs d’outils (MCP) que les agents appellent.", group="build", scopes=("company",)),
    PermissionInfo(key="workflows.edit", label="Edit workflows", description="Create, change and delete workflows, their triggers and the node library.", label_fr='Modifier les workflows', description_fr='Créer, modifier et supprimer les workflows, leurs déclencheurs et la bibliothèque de nœuds.', group="build", scopes=("company",)),
    PermissionInfo(key="connections.manage", label="Manage connections", description="Add, test and delete connections and their secrets, and link external accounts (Google, Microsoft).", label_fr='Gérer les connexions', description_fr='Ajouter, tester et supprimer les connexions et leurs secrets, et lier des comptes externes (Google, Microsoft).', group="pcs_data", scopes=("company",)),
    PermissionInfo(key="accounts.connect_own", label="Connect your own accounts", description="Sign in your own Google or Microsoft account and read it in Data; only you see it.", label_fr="Connecter ses propres comptes", description_fr="Connecter son propre compte Google ou Microsoft et le lire dans Données ; vous seul le voyez.", group="pcs_data", scopes=("company",)),
    PermissionInfo(key="machines.manage", label="Manage machines", description="Enroll and revoke machines, approve the scripts they run, set their cost rate and pool sharing.", label_fr='Gérer les machines', description_fr='Enrôler et révoquer des machines, approuver leurs scripts, régler leur coût et leur partage.', group="pcs_data", scopes=("company",)),
    PermissionInfo(key="billing.view", label="View billing", description="See the credit balance, ledger, invoices, subscription, budgets, reservations and cost usage.", label_fr='Voir la facturation', description_fr='Voir le solde de crédits, le journal, les factures, l’abonnement, les budgets, les réservations et les coûts.', group="billing", scopes=("company",)),
    PermissionInfo(key="billing.manage", label="Manage billing", description="Buy credits, set auto top-up, subscribe or change plan, edit the tax profile, set budgets and reserve machines.", label_fr='Gérer la facturation', description_fr='Acheter des crédits, régler la recharge automatique, s’abonner ou changer de formule, modifier le profil fiscal, fixer les budgets et réserver des machines.', group="billing", scopes=("company",)),
    PermissionInfo(key="approvals.manage", label="Approve agent actions", description="Approve or deny actions agents wait on (remote actions, outgoing emails) and set their approval policies.", label_fr='Approuver les actions des agents', description_fr='Approuver ou refuser les actions en attente des agents (actions distantes, e-mails sortants) et régler leurs règles d’approbation.', group="admin", scopes=("company",)),
    PermissionInfo(key="api_keys.manage", label="Manage API keys", description="Create, list and revoke API keys that act on this company programmatically.", label_fr='Gérer les clés API', description_fr='Créer, lister et révoquer les clés API qui agissent sur cette organisation.', group="admin", scopes=("company",)),
    PermissionInfo(key="company.members.manage", label="Manage members", description="Invite people by email, remove members, cancel invitations, and add or remove members from this company's groups.", label_fr='Gérer les membres', description_fr='Inviter par e-mail, retirer des membres, annuler des invitations, et ajouter ou retirer des membres des groupes de l’organisation.', group="admin", scopes=("company",)),
    PermissionInfo(key="company.groups.manage", label="Manage groups", description="Create, rename and delete this company's groups and choose the permissions each one grants.", label_fr='Gérer les groupes', description_fr='Créer, renommer et supprimer les groupes de l’organisation et choisir les permissions de chacun.', group="admin", scopes=("company",)),
    PermissionInfo(key="company.settings.manage", label="Manage company settings", description="Rename the company and edit its profile and logo.", label_fr='Gérer les réglages de l’organisation', description_fr='Renommer l’organisation et modifier son profil et son logo.', group="admin", scopes=("company",)),
    PermissionInfo(key="company.delete", label="Delete the company", description="Irreversibly delete this company and everything in it.", label_fr='Supprimer l’organisation', description_fr='Supprimer définitivement cette organisation et tout son contenu.', group="admin", scopes=("company",)),
    PermissionInfo(key="models.platform_paid", label="Use platform-paid models", description="Call a model through this platform's own provider keys, billed at cost plus the platform fee.", label_fr='Utiliser les modèles payés par la plateforme', description_fr='Appeler un modèle avec les clés de la plateforme, facturé au coût plus la commission.', group="platform", scopes=("platform",)),
    PermissionInfo(key="machines.pool.use", label="Use pooled machines", description="Dispatch a run onto another workspace's shared machine, billed to this workspace's credits.", label_fr='Utiliser les machines partagées', description_fr='Exécuter sur la machine partagée d’un autre espace, facturé sur les crédits de cet espace.', group="platform", scopes=("platform",)),
    PermissionInfo(key="machines.cloud.launch", label="Launch cloud machines", description="Have the platform provision a cloud instance (e.g. Scaleway) on this workspace's behalf.", label_fr='Lancer des machines cloud', description_fr='Laisser la plateforme louer une instance cloud (ex. Scaleway) pour cet espace.', group="platform", scopes=("platform",)),
    PermissionInfo(key="machines.reserve", label="Reserve machines", description="Hold a machine (own, pooled or cloud) for a fixed number of hours, pre-paid from credits.", label_fr='Réserver des machines', description_fr='Bloquer une machine (propre, partagée ou cloud) pour un nombre d’heures, prépayé en crédits.', group="platform", scopes=("platform",)),
    PermissionInfo(key="credits.auto_topup", label="Enable auto top-up", description="Let this workspace's saved card be charged automatically when its credit balance runs low.", label_fr='Activer la recharge automatique', description_fr='Autoriser le débit automatique de la carte enregistrée quand le solde de crédits est bas.', group="platform", scopes=("platform",)),
)


class PermissionGroup(WireModel):
    id: UUID
    workspace_id: UUID | None = None
    """The company this group belongs to; `None` = a platform group (admin panel)."""
    name: str = Field(min_length=1, max_length=120)
    permissions: tuple[Permission, ...] = Field(default=())
    created_at: datetime
    updated_at: datetime
    kind: GroupKind = "group"
    parent_id: UUID | None = None
    """Department it reports to (same company, never a cycle); None = reports to the company."""
    owner_account_id: UUID | None = None
    """Department owner (a company member); None for an access group."""
    manager_account_id: UUID | None = None
    """Department manager (a company member); None for an access group."""

    @model_validator(mode="after")
    def unique_permissions(self) -> Self:
        if len(set(self.permissions)) != len(self.permissions):
            raise ValueError("a group's permissions must be unique")
        return self


class GroupCreate(WireRequest):
    name: str = Field(min_length=1, max_length=120)
    permissions: tuple[Permission, ...] = Field(default=())
    kind: GroupKind = "group"
    parent_id: UUID | None = None
    owner_account_id: UUID | None = None
    manager_account_id: UUID | None = None

    @model_validator(mode="after")
    def department_has_stewards(self) -> Self:
        if self.kind == "company":
            raise ValueError("the company node exists once, made with the company")
        if self.kind == "department" and (self.owner_account_id is None or self.manager_account_id is None):
            raise ValueError("a department needs an owner and a manager")
        if self.kind == "group" and (self.parent_id, self.owner_account_id, self.manager_account_id) != (None, None, None):
            raise ValueError("only a department has a parent, an owner and a manager")
        return self


class DepartmentUpdate(WireRequest):
    """A department's reporting line, owner and manager; a missing field stays as it is,
    `parent_id: null` moves the department to the top level."""

    parent_id: UUID | None | MISSING = MISSING
    owner_account_id: UUID | MISSING = MISSING
    manager_account_id: UUID | MISSING = MISSING

    def changes(self) -> dict[str, UUID | None]:
        """Only the fields this update carries (column name -> value)."""
        return {key: getattr(self, key) for key in DepartmentUpdate.model_fields if getattr(self, key) is not MISSING}


class GroupUpdate(DepartmentUpdate):
    """A missing field stays as it is (the department fields: `DepartmentUpdate`)."""

    name: str | MISSING = MISSING
    permissions: tuple[Permission, ...] | MISSING = MISSING

    def department(self) -> DepartmentUpdate:
        return DepartmentUpdate(**self.changes())


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
    GroupCreate(name="Members", permissions=("agents.view", "agents.run", "workflows.run", "agents.edit", "prompts.edit", "tools.manage", "workflows.edit", "accounts.connect_own", "billing.view")),
    GroupCreate(name="Viewers", permissions=("agents.view", "billing.view")),
)
"""Seeded into every company; the old per-workspace roles, one group each (owner-only rights —
deleting the company, enrolling machines and approving their scripts — stay Owners-only). Members
connect their OWN accounts but do not read every company place (`connections.manage`: Owners +
Admins). A new company's groups equal an older company's after migration 106 (which split
`agents.edit` into prompts / tools and `workflows.run` into agents). A company renames, edits or
deletes them like any group it created itself."""


OverrideEffect = Literal["allow", "deny"]


class MemberOverride(WireModel):
    """One per-person exception on top of groups and departments: `allow` adds a right nobody's
    group gives them, `deny` removes one whatever gives it (deny beats every source)."""

    permission: Permission
    effect: OverrideEffect


class MemberOverridesUpdate(WireRequest):
    """Full replacement of one member's overrides (an empty tuple clears them)."""

    overrides: tuple[MemberOverride, ...] = Field(default=())

    @model_validator(mode="after")
    def one_per_permission(self) -> Self:
        keys = [item.permission for item in self.overrides]
        if len(set(keys)) != len(keys):
            raise ValueError("one override per permission")
        return self


RightSourceKind = Literal["group", "department", "company", "allow", "deny", "manager"]


class RightSource(WireModel):
    """WHERE one right comes from (or, `deny`, what takes it away): an access group, a department,
    the company itself (what every member gets), a per-person override, or (`manager`) the agent's
    manager it is capped by."""

    kind: RightSourceKind
    id: UUID | None = None
    name: str = Field(min_length=1, max_length=320)


class RightExplanation(WireModel):
    permission: Permission
    held: bool
    sources: tuple[RightSource, ...] = Field(default=())


class AgentStewards(WireModel):
    """Who answers for an agent: its OWNER (transfers, deletes, names the manager) and its MANAGER
    (gives it places, edits it), plus the one department it sits in (None = no department)."""

    agent_id: UUID
    owner_account_id: UUID
    manager_account_id: UUID
    department_id: UUID | None = None


class AgentStewardsUpdate(WireRequest):
    owner_account_id: UUID
    manager_account_id: UUID
