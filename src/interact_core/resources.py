"""The generic core every screen renders from: one read projection over every kind of thing a
workspace holds (`ResourceSummary`), who may do what with each (`Verb`, `Grant`, `AccessView`), and
projects that group them (`Project`).

Access is ONE rule set: a resource's verbs for a person come from its owner, the company's owners
(admins), its audience (private / company) and grants; a grant never gives more than its giver holds
at the moment it is used. The server computes it (`reach`); a client renders `my_verbs` /
`AccessView` and never re-derives them."""

from datetime import datetime
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from .wire import WireModel, WireRequest
from .workflows import GitUrl

#: What a person (or agent) may do with a resource. Each verb implies every verb before it.
Verb = Literal["see", "read", "use", "write_on_review", "write", "manage"]
VERBS: tuple[Verb, ...] = ("see", "read", "use", "write_on_review", "write", "manage")

ResourceKind = Literal["agent", "prompt", "workflow", "template", "place", "mcp_server", "project", "chat"]
#: The kinds a grant can name.
GrantResourceKind = Literal["workflow", "agent", "prompt", "project", "connection", "place"]
#: `group` covers access groups and departments.
GranteeKind = Literal["account", "group", "agent"]
#: `private`: its owner (and grants) only; `shared`: private with at least one grant; `company`:
#: everyone in the company.
Audience = Literal["private", "shared", "company"]
PlaceKind = Literal["account", "computer", "server", "company_drive"]
#: The client's closed status vocabulary (`frontend/src/plain.ts` keys).
StatusWord = Literal["working", "waiting", "done", "failed", "stopped", "lost", "idle", "ready", "low", "empty", "paused", "checking",
                     "online", "offline", "active", "trial", "cancelled", "verified", "saved", "revoked", "unknown"]


class PersonRef(WireModel):
    id: UUID
    name: str = Field(max_length=320)
    avatar_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ResourceSummary(WireModel):
    """One row of the generic collection, whatever its kind. `id` is the kind's own id (a UUID, a
    place's source id, a prompt's `namespace/slug`), never re-keyed."""

    kind: ResourceKind
    id: str = Field(min_length=1, max_length=512)
    subkind: str | None = Field(default=None, max_length=80)
    name: str = Field(max_length=240)
    name_fr: str = Field(default="", max_length=240)
    summary: str = Field(default="", max_length=120)
    summary_fr: str = Field(default="", max_length=120)
    mark: str = Field(max_length=80)
    status: StatusWord | None = None
    owner: PersonRef | None = None
    manager: PersonRef | None = None
    my_verbs: tuple[Verb, ...]
    audience: Audience
    shared_count: int = Field(default=0, ge=0)
    projects: tuple[UUID, ...] = ()
    #: None: the kind keeps no change time (a place, a built-in template); listed after dated rows.
    updated_at: datetime | None = None
    counts: dict[str, int] = Field(default_factory=dict)


class ResourcePage(WireModel):
    items: tuple[ResourceSummary, ...] = Field(max_length=200)
    next: str | None = Field(default=None, max_length=512)


class GrantRequest(WireRequest):
    resource_kind: GrantResourceKind
    resource_id: str = Field(min_length=1, max_length=512)
    grantee_kind: GranteeKind
    grantee_id: UUID
    verb: Verb
    #: Agents only: `team` also reaches the sub-agents this agent calls in the SAME run's call chain.
    covers: Literal["self", "team"] = "self"

    @model_validator(mode="after")
    def team_is_for_agents(self) -> Self:
        if self.covers == "team" and self.grantee_kind != "agent":
            raise ValueError("only a grant to an agent covers its team")
        return self


class Grant(WireModel):
    id: UUID
    workspace_id: UUID
    resource_kind: GrantResourceKind
    resource_id: str
    grantee_kind: GranteeKind
    grantee_id: UUID
    verb: Verb
    covers: Literal["self", "team"]
    given_by: UUID
    created_at: datetime


class Grantee(WireModel):
    kind: Literal["account", "group", "department", "agent", "company"]
    id: str
    name: str = Field(max_length=320)
    mark: str = Field(max_length=80)


class AccessRow(WireModel):
    """One way someone reaches the resource. `paused`: the row exists but gives nothing now (its
    giver lost the right, the agent changed manager), `paused_reason` says why."""

    grant_id: UUID | None = None
    grantee: Grantee
    verb: Verb
    source: Literal["grant", "department", "manager", "team", "owner", "admin", "company"]
    source_name: str | None = None
    paused: bool = False
    paused_reason: str | None = None


class AccessMay(WireModel):
    share: bool
    change_audience: bool
    transfer: bool


class AccessView(WireModel):
    """Who reaches one resource and how — `reach` output, rendered as is."""

    resource_kind: GrantResourceKind
    resource_id: str
    owner: PersonRef | None = None
    manager: PersonRef | None = None
    audience: Audience
    rows: tuple[AccessRow, ...]
    my_verbs: tuple[Verb, ...]
    may: AccessMay
    #: The caller sees this only because they are one of the company's owners.
    admin_view: bool


class AudienceUpdate(WireRequest):
    resource_kind: Literal["workflow", "agent", "prompt", "project"]
    resource_id: str = Field(min_length=1, max_length=512)
    audience: Literal["private", "company"]


# ---- projects ------------------------------------------------------------------------------------

#: A project colour: a key of the client's identity accent set (the client owns that set; the
#: server stores the key and never renders it).
ProjectColour = Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{0,23}$")]


class FolderRule(WireModel):
    """Runs prompted in this session root folder (any PC) or below it."""

    kind: Literal["folder"]
    path: str = Field(min_length=1, max_length=1024)


class ResourceRule(WireModel):
    """A workflow's runs, a chat, or every run of an agent (any origin)."""

    kind: Literal["workflow", "chat", "agent"]
    id: UUID


ProjectRule = FolderRule | ResourceRule


class ProjectCreate(WireRequest):
    name: str = Field(min_length=1, max_length=60)
    colour: ProjectColour
    rules: tuple[ProjectRule, ...] = Field(default=(), max_length=64)
    #: Where its code lives: a PC without it clones it from there ("prepare a workspace").
    repository: GitUrl | None = None


class ProjectUpdate(WireRequest):
    """Fields left out stay as they are."""

    name: str | None = Field(default=None, min_length=1, max_length=60)
    colour: ProjectColour | None = None
    rules: tuple[ProjectRule, ...] | None = Field(default=None, max_length=64)
    repository: GitUrl | None = None


class Project(WireModel):
    id: UUID
    workspace_id: UUID
    name: str = Field(min_length=1, max_length=60)
    colour: ProjectColour
    created_by: UUID
    created_at: datetime
    rules: tuple[ProjectRule, ...] = Field(max_length=64)
    repository: GitUrl | None = None

    def names(self, kind: Literal["workflow", "chat", "agent"], resource: UUID) -> bool:
        """A rule binds this workflow / chat / agent to the project."""
        return any(isinstance(rule, ResourceRule) and rule.kind == kind and rule.id == resource for rule in self.rules)

    def holds_folder(self, root: str) -> bool:
        """Work prompted in `root` (a session root folder) is at or below one of its folders."""
        root = root.rstrip("/")
        return bool(root) and any(isinstance(rule, FolderRule) and (root == rule.path.rstrip("/") or root.startswith(rule.path.rstrip("/") + "/")) for rule in self.rules)

    def claims_run(self, root: str, agent: UUID | None, workflow: UUID | None = None) -> bool:
        """A run belongs to every project whose rule matches it: its folder, its agent, its workflow."""
        return self.holds_folder(root) or agent is not None and self.names("agent", agent) or workflow is not None and self.names("workflow", workflow)


class ProjectSuggestion(WireModel):
    """A session root folder that runs came from and no project claims yet."""

    path: str
    runs: int = Field(ge=1)
    last_at: datetime
