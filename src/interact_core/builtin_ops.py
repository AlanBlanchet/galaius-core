"""Every builtin operation's contract: what it does (category, summary, search words), its settings
(a pydantic `Config`, published as JSON Schema so one generic form edits any of them) and the ports
it fixes. The server runs them (keyed by the same op name); the
registry itself (`BUILTIN_OPS`) lives beside `BuiltinImplementation`, which checks a node against
its spec.

Settings are typed (text, number, boolean, choice, a list of texts, a JSON object); the node
editor renders every op with one form driven by the schema."""

from typing import Annotated, ClassVar, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from .workflows import BuiltinOpSpec, Plumbing, PlumbingCondition, PortSpec, ValueType, WorkflowValue, _port_slug


def _port(name: str, direction: Literal["input", "output"], value_type: ValueType = "any", *, required: bool = True, multiple: bool = False) -> PortSpec:
    return PortSpec(name=name, direction=direction, value_type=value_type, required=required, multiple=multiple)


class _Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _path(title: str = "Field", description: str = "Dot path into the value (items.0.name); empty = the value itself.") -> object:
    return Field(default="", max_length=200, title=title, description=description)


Operator = Literal["equals", "not_equals", "contains", "not_contains", "starts_with", "ends_with", "greater", "greater_or_equal", "less", "less_or_equal", "is_empty", "is_not_empty", "is_true", "is_false", "matches"]


# ---- the ops that predate this registry: ports chosen where the node is built ----


class _Free(BuiltinOpSpec):
    """Ports set by the editor (an input's value type follows what it holds); config also carries
    connection refs and paths, checked by `BuiltinImplementation.check`."""

    fixed_ports = False


class HeldValue(BaseModel):
    """An input's constant; the rest of its config (connection, path) is read where it runs."""

    model_config = ConfigDict(extra="ignore")
    value: WorkflowValue | None = None


class InputOp(_Free):
    op, category, title = "input", "input_output", "Input"
    title_fr = "Entrée"
    summary = "A workflow input: a constant, or the value a run is started with."
    summary_fr = "Une entrée du workflow : une constante, ou la valeur donnée au lancement."
    keywords = ("constant", "parameter", "value", "start")
    required = ("value",)
    Config = HeldValue

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("result", "output", "text"),)


class OutputOp(_Free):
    op, category, title = "output", "input_output", "Result"
    title_fr = "Résultat"
    summary = "A workflow result: what the run answers."
    summary_fr = "Un résultat du workflow : ce que l’exécution renvoie."
    keywords = ("return", "answer", "end")

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input", "text"), _port("result", "output", "text"))


class _FileOp(_Free):
    category = "files"
    required = ("artifact_path",)
    placements = frozenset({"server", "machine"})


class WriteArtifactOp(_FileOp):
    op, title = "write_artifact", "Write file"
    title_fr = "Écrire un fichier"
    summary = "Writes a value or a file to workspace storage or a machine's folder."
    summary_fr = "Écrit une valeur ou un fichier dans le stockage de l’espace ou le dossier d’une machine."
    keywords = ("save", "store", "export", "file", "artifact")

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input", "text"), _port("result", "output", "artifact"))


class ReadFileOp(_FileOp):
    op, title = "read_file", "Read file"
    title_fr = "Lire un fichier"
    summary = "Reads a file from workspace storage or a machine's folder."
    summary_fr = "Lit un fichier du stockage de l’espace ou du dossier d’une machine."
    keywords = ("load", "open", "import", "file")

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("result", "output", "artifact"),)


class _TextMap(_Free):
    category = "text"

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input", "text"), _port("result", "output", "text"))


class UppercaseOp(_TextMap):
    op, title, summary, keywords = "uppercase", "Uppercase", "Upper-cases text (each item of a list).", ("capitals", "case")
    title_fr, summary_fr = "Majuscules", "Met le texte en majuscules (chaque élément d’une liste)."


class LowercaseOp(_TextMap):
    op, title, summary, keywords = "lowercase", "Lowercase", "Lower-cases text (each item of a list).", ("case",)
    title_fr, summary_fr = "Minuscules", "Met le texte en minuscules (chaque élément d’une liste)."


class IdentityOp(_TextMap):
    op, title, summary, keywords = "identity", "Pass through", "Passes its value on unchanged.", ("noop", "relay", "identity")
    title_fr, summary_fr = "Transmettre", "Transmet sa valeur sans la changer."
    plumbing = Plumbing(badge="passes on", badge_fr="transmet")


class HttpGetOp(_TextMap):
    op, category, title = "http_get", "web", "HTTP GET"
    title_fr = "Lecture HTTP (GET)"
    summary = "GETs a path on a configured HTTP connection."
    summary_fr = "Lit (GET) une adresse sur une connexion HTTP configurée."
    keywords = ("fetch", "api", "url", "download", "request")
    required = ("connection",)


# ---- flow: which path runs ----


class PredicateSettings(_Settings):
    path: str = _path()
    operator: Operator = Field(default="is_not_empty", title="Test")
    ignore_case: bool = Field(default=False, title="Ignore case")


class ConditionOp(BuiltinOpSpec):
    op, category, title = "condition", "flow", "If"
    title_fr = "Si"
    summary = "Sends its value down 'true' or 'false' after one test (equals, contains, greater, empty, regex...)."
    summary_fr = "Envoie sa valeur vers « true » ou « false » après un test (égal, contient, plus grand, vide, regex…)."
    keywords = ("if", "branch", "when", "test", "compare", "check", "else")
    Config = PredicateSettings

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input"), _port("compare_to", "input", required=False), _port("true", "output"), _port("false", "output"))


_DEFAULT_CASES = ["yes", "no"]


class SwitchSettings(_Settings):
    path: str = _path()
    cases: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(default=_DEFAULT_CASES, min_length=1, max_length=16, title="Cases", description="Each case gets its own output; a value matching none goes to 'other'.")
    mode: Literal["equals", "contains", "matches"] = Field(default="equals", title="Match")
    ignore_case: bool = Field(default=True, title="Ignore case")


def switch_cases(config: dict[str, WorkflowValue]) -> tuple[tuple[str, str], ...]:
    """(case, output port name) per case, port names unique."""
    taken, result = {"value", "other"}, []
    for line in SwitchSettings.model_validate({"cases": config.get("cases", _DEFAULT_CASES)}).cases:
        base = f"case_{_port_slug(line)}"[:70]
        name, counter = base, 2
        while name in taken:
            name, counter = f"{base}_{counter}", counter + 1
        taken.add(name)
        result.append((line, name))
    return tuple(result)


class SwitchOp(BuiltinOpSpec):
    op, category, title = "switch", "flow", "Switch"
    title_fr = "Aiguillage"
    summary = "Routes its value to the output of the first case it matches, else to 'other'."
    summary_fr = "Envoie sa valeur vers la sortie du premier cas qui correspond, sinon vers « other »."
    keywords = ("router", "route", "case", "paths", "branch", "classify")
    Config = SwitchSettings

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input"), *(_port(name, "output") for _case, name in switch_cases(config)), _port("other", "output"))


class MergeSettings(_Settings):
    # `options` / `fr`: the words the editor's settings form shows for each value, and in French.
    mode: Literal["append", "combine", "zip", "join"] = Field(default="append", title="How to merge", description="One list; one object (later keys win); items side by side; or the first two lists matched on a key.",
        json_schema_extra={"options": {"append": "One list, one after another", "combine": "One object (later keys win)", "zip": "Items side by side", "join": "Match two lists on a key"},
                           "fr": {"title": "Comment fusionner", "description": "Une liste ; un objet (les dernières clés gagnent) ; éléments côte à côte ; ou les deux premières listes appariées sur une clé.",
                                  "options": {"append": "Une liste, à la suite", "combine": "Un objet (les dernières clés gagnent)", "zip": "Éléments côte à côte", "join": "Apparier deux listes sur une clé"}}})
    key: str = Field(default="id", max_length=200, title="Join key", description="For 'Match two lists on a key': the field matched in both lists (a dot path).",
        json_schema_extra={"fr": {"title": "Clé d’appariement", "description": "Pour « Apparier deux listes sur une clé » : le champ comparé dans les deux listes (chemin avec des points)."}})
    keep_unmatched: bool = Field(default=False, title="Keep items with no match",
        json_schema_extra={"fr": {"title": "Garder les éléments sans correspondance"}})


class MergeOp(BuiltinOpSpec):
    op, category, title = "merge", "flow", "Merge"
    title_fr = "Fusionner"
    summary = "Waits until every wire into it has delivered, then passes their values on together: append lists, combine objects, zip by position, or join on a key."
    summary_fr = "Attend que chaque fil qui y arrive ait livré, puis transmet leurs valeurs ensemble : listes ajoutées, objets combinés, appariés par position ou joints par clé."
    keywords = ("join", "combine", "concat", "zip", "wait for all", "gather", "union")
    Config = MergeSettings
    plumbing = Plumbing(badge="waits for all, merges", badge_fr="attend tout, fusionne")

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("inputs", "input", multiple=True), _port("result", "output"))


class WaitSettings(_Settings):
    seconds: float = Field(default=5, ge=0, le=3600, title="Seconds")


class WaitOp(BuiltinOpSpec):
    op, category, title = "wait", "time", "Wait"
    title_fr = "Attendre"
    summary = "Waits a number of seconds, then passes its value on. Cancelling the run stops the wait."
    summary_fr = "Attend quelques secondes, puis transmet sa valeur. Annuler l’exécution arrête l’attente."
    keywords = ("delay", "sleep", "pause", "throttle")
    Config = WaitSettings

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input", required=False), _port("result", "output"))


class ApprovalSettings(_Settings):
    message: str = Field(default="Approve {{value}}?", min_length=1, max_length=2000, title="Question", description="What the approver reads; {{field}} inserts part of the value.")
    timeout_minutes: int = Field(default=60, ge=1, le=1440, title="Wait at most (minutes)", description="No answer by then: the value goes down 'rejected'.")


class ApprovalOp(BuiltinOpSpec):
    op, category, title = "approval", "people", "Ask for approval"
    title_fr = "Demander une validation"
    summary = "Pauses the run until a person approves or rejects it in Approvals; the value goes down 'approved' or 'rejected'."
    summary_fr = "Met l’exécution en pause jusqu’à ce qu’une personne valide ou refuse dans Validations ; la valeur part vers « approved » ou « rejected »."
    keywords = ("human in the loop", "review", "confirm", "sign off", "validate", "manual")
    Config = ApprovalSettings

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input"), _port("approved", "output"), _port("rejected", "output"))


class FailSettings(_Settings):
    message: str = Field(default="Stopped: {{value}}", min_length=1, max_length=400, title="Message", description="{{field}} inserts part of the value.")


class FailOp(BuiltinOpSpec):
    op, category, title = "fail", "flow", "Stop with error"
    title_fr = "Arrêter en erreur"
    summary = "Fails on purpose with a message; its error path or the run's failure shows it."
    summary_fr = "Échoue exprès avec un message ; son chemin d’erreur ou l’échec de l’exécution l’affiche."
    keywords = ("error", "abort", "raise", "throw", "stop")
    Config = FailSettings

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input", required=False),)


# ---- data: lists and objects ----


class _ListOp(BuiltinOpSpec):
    """Takes a list of anything (JSON items, a text list), answers a JSON list."""

    category = "data"

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input"), _port("result", "output", "json"))


class FilterSettings(PredicateSettings):
    compare_to: str = Field(default="", max_length=400, title="Compared to", description="Numbers compare as numbers.")


class FilterOp(BuiltinOpSpec):
    op, category, title = "filter", "data", "Filter"
    title_fr = "Filtrer"
    summary = "Keeps the list items that pass one test; the others go to 'dropped'."
    summary_fr = "Garde les éléments de la liste qui passent un test ; les autres vont vers « dropped »."
    keywords = ("where", "keep", "remove", "select", "exclude")
    Config = FilterSettings

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input"), _port("kept", "output", "json"), _port("dropped", "output", "json"))


class SortSettings(_Settings):
    path: str = _path("Sort by")
    order: Literal["ascending", "descending"] = Field(default="ascending", title="Order")


class SortOp(_ListOp):
    op, title = "sort", "Sort"
    title_fr = "Trier"
    summary = "Sorts a list by a field (numbers as numbers, text alphabetically, empty last)."
    summary_fr = "Trie une liste selon un champ (nombres comme nombres, texte par ordre alphabétique, vides à la fin)."
    keywords = ("order", "rank", "arrange")
    Config = SortSettings


class LimitSettings(_Settings):
    count: int = Field(default=10, ge=0, le=100_000, title="Keep")
    keep: Literal["first", "last"] = Field(default="first", title="Which")


class LimitOp(_ListOp):
    op, title = "limit", "Limit"
    title_fr = "Limiter"
    summary = "Keeps the first (or last) N items of a list."
    summary_fr = "Garde les N premiers (ou derniers) éléments d’une liste."
    keywords = ("top", "head", "tail", "first", "take", "max items")
    Config = LimitSettings


class DedupeSettings(_Settings):
    path: str = _path("Compare by", "Dot path of the key; empty = the whole item.")


class DedupeOp(_ListOp):
    op, title = "dedupe", "Remove duplicates"
    title_fr = "Retirer les doublons"
    summary = "Drops list items whose key was already seen (the first one stays)."
    summary_fr = "Retire les éléments dont la clé a déjà été vue (le premier reste)."
    keywords = ("unique", "distinct", "duplicates", "dedup")
    Config = DedupeSettings


class AggregateSettings(_Settings):
    operation: Literal["count", "sum", "average", "min", "max", "concatenate", "collect", "group"] = Field(default="count", title="Operation")
    path: str = _path(description="Dot path in each item; for 'group', the grouping key.")
    separator: str = Field(default=", ", max_length=40, title="Separator", description="For 'concatenate'.")


class AggregateOp(BuiltinOpSpec):
    op, category, title = "aggregate", "data", "Aggregate"
    title_fr = "Agréger"
    summary = "Reduces a list to one value: count, sum, average, min, max, joined text, a field's values, or groups by key."
    summary_fr = "Réduit une liste à une valeur : nombre, somme, moyenne, min, max, texte joint, valeurs d’un champ, ou groupes par clé."
    keywords = ("summarize", "total", "sum", "count", "group by", "reduce", "pluck", "gather")
    Config = AggregateSettings

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input"), _port("result", "output"))


class SetFieldsSettings(_Settings):
    fields: dict[str, WorkflowValue] = Field(default_factory=dict, max_length=64, title="Fields", description='Assigned onto the value; text may use {{field}} from the value, e.g. {"name": "{{first}} {{last}}"}.')
    remove: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(default_factory=list, max_length=64, title="Remove", description="Fields to drop.")
    keep_only_set: bool = Field(default=False, title="Keep only these fields")


class SetFieldsOp(BuiltinOpSpec):
    op, category, title = "set_fields", "data", "Set fields"
    title_fr = "Modifier des champs"
    summary = "Adds, overwrites or removes fields of an object (each item of a list)."
    summary_fr = "Ajoute, remplace ou retire des champs d’un objet (de chaque élément d’une liste)."
    keywords = ("edit fields", "assign", "rename", "map", "variable", "set", "object")
    Config = SetFieldsSettings
    plumbing = Plumbing(badge="set {fields} · drop {remove}", badge_fr="champs {fields} · retire {remove}")

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input", "json", required=False), _port("result", "output", "json"))


class TransformSettings(_Settings):
    expression: str = Field(default="@", min_length=1, max_length=4000, title="Expression (JMESPath)", description="e.g. items[?price > `10`].{name: name, price: price}")


class TransformOp(BuiltinOpSpec):
    op, category, title = "transform", "data", "Transform JSON"
    title_fr = "Transformer du JSON"
    summary = "Reshapes JSON with a JMESPath expression: pick, filter, project, flatten."
    summary_fr = "Remodèle du JSON avec une expression JMESPath : choisir, filtrer, projeter, aplatir."
    keywords = ("jmespath", "jq", "jsonata", "query", "select", "reshape", "extract", "pick field")
    Config = TransformSettings
    plumbing = Plumbing(badge="pick {expression}", badge_fr="choisit {expression}")

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input"), _port("result", "output"))


class SqlSettings(_Settings):
    query: str = Field(default="SELECT * FROM input", min_length=1, max_length=8000, title="SQL")


class SqlOp(BuiltinOpSpec):
    op, category, title = "sql", "data", "SQL query"
    title_fr = "Requête SQL"
    summary = "Runs SQL (SQLite) over JSON tables: a list is table 'input'; an object of lists is one table per key."
    summary_fr = "Exécute du SQL (SQLite) sur des tables JSON : une liste est la table « input » ; un objet de listes, une table par clé."
    keywords = ("select", "join tables", "group by", "database", "query", "sqlite")
    Config = SqlSettings

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("tables", "input", "json"), _port("result", "output", "json"))


#: A data format's name in words (the settings form's choices, a folded step's words), EN and FR.
FORMAT_WORDS = {"json": "JSON", "csv": "CSV", "tsv": "TSV", "xml": "XML", "yaml": "YAML", "lines": "lines", "xlsx": "Excel", "markdown_table": "Markdown table"}
FORMAT_WORDS_FR = {**FORMAT_WORDS, "lines": "lignes", "markdown_table": "tableau Markdown"}


def _format_words(*formats: str) -> dict[str, object]:
    return {"options": {name: FORMAT_WORDS[name] for name in formats}, "fr": {"options": {name: FORMAT_WORDS_FR[name] for name in formats}}}


ParseFormat = Literal["json", "csv", "tsv", "xml", "yaml", "lines", "xlsx"]
TextFormat = Literal["json", "csv", "tsv", "xml", "yaml", "lines", "markdown_table"]


class ParseSettings(_Settings):
    format: ParseFormat = Field(default="json", title="Format", json_schema_extra=_format_words(*get_args(ParseFormat)))
    header: bool = Field(default=True, title="First row is the header", description="CSV, TSV and spreadsheets.")
    sheet: str = Field(default="", max_length=120, title="Sheet", description="Spreadsheets; empty = the first sheet.")


class ParseOp(BuiltinOpSpec):
    op, category, title = "parse", "data", "Parse"
    title_fr = "Lire un format"
    summary = "Turns text or a file into JSON: JSON, CSV, TSV, XML, YAML, lines, or an Excel sheet."
    summary_fr = "Transforme un texte ou un fichier en JSON : JSON, CSV, TSV, XML, YAML, lignes ou feuille Excel."
    keywords = ("read csv", "read xml", "spreadsheet", "excel", "xlsx", "decode", "extract from file", "yaml")
    Config = ParseSettings
    plumbing = Plumbing(badge="{format} → data", badge_fr="{format} → données")

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input"), _port("result", "output", "json"))


class FormatSettings(_Settings):
    format: TextFormat = Field(default="json", title="Format", json_schema_extra=_format_words(*get_args(TextFormat)))
    pretty: bool = Field(default=True, title="Indented")


class FormatOp(BuiltinOpSpec):
    op, category, title = "format", "data", "Convert to text"
    title_fr = "Convertir en texte"
    summary = "Writes JSON as text: JSON, CSV, TSV, XML, YAML, lines or a Markdown table ('Write file' then saves it)."
    summary_fr = "Écrit du JSON en texte : JSON, CSV, TSV, XML, YAML, lignes ou tableau Markdown (« Écrire un fichier » l’enregistre ensuite)."
    keywords = ("serialize", "to csv", "to json", "to xml", "export", "stringify", "table")
    Config = FormatSettings
    plumbing = Plumbing(badge="→ {format} text", badge_fr="→ texte {format}")

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input"), _port("result", "output", "text"))


class CalculateSettings(_Settings):
    expression: str = Field(default="value * 2", min_length=1, max_length=1000, title="Formula", description="+ - * / // % **, round, min, max, abs, sqrt; names are fields of the value ('value' = the value itself).")


class CalculateOp(BuiltinOpSpec):
    op, category, title = "calculate", "data", "Calculate"
    title_fr = "Calculer"
    summary = "Computes a number from a formula over the value's fields."
    summary_fr = "Calcule un nombre avec une formule sur les champs de la valeur."
    keywords = ("math", "formula", "arithmetic", "number", "compute", "expression")
    Config = CalculateSettings

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input", required=False), _port("result", "output", "number"))


class EncodeSettings(_Settings):
    operation: Literal["sha256", "sha512", "sha1", "md5", "base64_encode", "base64_decode", "url_encode", "url_decode", "hex_encode"] = Field(default="sha256", title="Operation")


class EncodeOp(BuiltinOpSpec):
    op, category, title = "encode", "data", "Hash or encode"
    title_fr = "Empreinte ou encodage"
    summary = "Hashes (SHA-256, SHA-512, SHA-1, MD5) or encodes / decodes (Base64, URL, hex) text."
    summary_fr = "Calcule une empreinte (SHA-256, SHA-512, SHA-1, MD5) ou encode / décode (Base64, URL, hex) un texte."
    keywords = ("crypto", "hash", "checksum", "base64", "digest", "url encode")
    Config = EncodeSettings

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input", "text"), _port("result", "output", "text"))


# ---- text ----


class TemplateSettings(_Settings):
    #: One `{{field}}` of a template (group 1 = its path, spaces around it stripped by the reader):
    #: what the server fills and what makes a template only plumbing. One quantifier inside, so a
    #: long run of spaces never backtracks; a class both Python and the editor's engines read alike.
    PLACEHOLDER: ClassVar[str] = r"\{\{([^{}]*)\}\}"
    template: str = Field(default="Hello {{name}}", min_length=1, max_length=16000, title="Template", description="{{field}} inserts a field of 'values' (dot path); {{value}} the whole value.")
    missing: Literal["error", "empty"] = Field(default="error", title="Missing field")


class TemplateOp(BuiltinOpSpec):
    op, category, title = "template", "text", "Text template"
    title_fr = "Modèle de texte"
    summary = "Builds text from a template filled with the fields of its input."
    summary_fr = "Compose un texte à partir d’un modèle rempli avec les champs de son entrée."
    keywords = ("format", "compose", "prompt", "message", "interpolate", "string")
    Config = TemplateSettings
    # Only while the template holds nothing but {{fields}}: words around them are content, a step.
    plumbing = Plumbing(badge="→ text", badge_fr="→ texte", only_if=PlumbingCondition(field="template", pattern=rf"^\s*(?:{TemplateSettings.PLACEHOLDER}\s*)+$"))

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("values", "input", required=False), _port("result", "output", "text"))


class _TextOp(BuiltinOpSpec):
    category = "text"

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input", "text"), _port("result", "output", "text"))


class ReplaceSettings(_Settings):
    find: str = Field(default=r"\s+", min_length=1, max_length=1000, title="Find")
    replace_with: str = Field(default=" ", max_length=4000, title="Replace with", description="With 'regex', \\1 inserts a group.")
    regex: bool = Field(default=True, title="Regular expression")
    ignore_case: bool = Field(default=False, title="Ignore case")


class ReplaceOp(_TextOp):
    op, title = "replace_text", "Replace text"
    title_fr = "Remplacer du texte"
    summary = "Replaces every occurrence of a text or regular expression."
    summary_fr = "Remplace chaque occurrence d’un texte ou d’une expression régulière."
    keywords = ("substitute", "regex", "sed", "clean")
    Config = ReplaceSettings


class ExtractSettings(_Settings):
    pattern: str = Field(default=r"[\w.+-]+@[\w-]+\.[\w.-]+", min_length=1, max_length=1000, title="Regular expression", description="Named groups (?P<name>...) give objects; one group gives its text.")
    all: bool = Field(default=True, title="Every match", description="Off: the first match only.")
    ignore_case: bool = Field(default=False, title="Ignore case")


class ExtractOp(BuiltinOpSpec):
    op, category, title = "extract_text", "text", "Extract with regex"
    title_fr = "Extraire par regex"
    summary = "Pulls matches of a regular expression out of text (emails, numbers, ids...)."
    summary_fr = "Extrait d’un texte ce qui correspond à une expression régulière (e-mails, nombres, identifiants…)."
    keywords = ("regex", "match", "find", "scrape", "pattern")
    Config = ExtractSettings

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input", "text"), _port("result", "output"))


#: A split's separators by name; `custom` reads `custom`.
SEPARATORS = {"new_line": "\n", "comma": ",", "semicolon": ";", "tab": "\t", "space": " "}


class SplitSettings(_Settings):
    separator: Literal["new_line", "comma", "semicolon", "tab", "space", "custom"] = Field(default="new_line", title="Separator",
        json_schema_extra={"options": {name: name.replace("_", " ") for name in [*SEPARATORS, "custom"]},
                           "fr": {"title": "Séparateur", "options": {"new_line": "retour à la ligne", "comma": "virgule", "semicolon": "point-virgule", "tab": "tabulation", "space": "espace", "custom": "personnalisé"}}})
    custom: str = Field(default="|", min_length=1, max_length=40, title="Custom separator")
    trim: bool = Field(default=True, title="Trim spaces")
    drop_empty: bool = Field(default=True, title="Drop empty parts")


class SplitOp(BuiltinOpSpec):
    op, category, title = "split_text", "text", "Split text"
    title_fr = "Découper un texte"
    summary = "Splits text into a list (by line, comma or any separator) to loop or filter over."
    summary_fr = "Découpe un texte en liste (par ligne, virgule ou tout séparateur) pour la parcourir ou la filtrer."
    keywords = ("lines", "tokenize", "explode", "split out", "list")
    Config = SplitSettings
    plumbing = Plumbing(badge="split ({separator})", badge_fr="découpe ({separator})")

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input", "text"), _port("result", "output", "text", multiple=True))


# ---- time ----


class DateTimeSettings(_Settings):
    operation: Literal["now", "format", "add", "difference", "parts"] = Field(default="now", title="Operation", description="difference: seconds from 'value' to 'until'.")
    output_format: str = Field(default="iso", max_length=80, title="Output format", description="'iso', 'epoch', 'date', or strftime like %d/%m/%Y %H:%M.")
    timezone: str = Field(default="UTC", max_length=64, title="Time zone", description="IANA name, e.g. Europe/Paris.")
    amount: float = Field(default=1, ge=-1_000_000, le=1_000_000, title="Amount", description="For 'add'.")
    unit: Literal["seconds", "minutes", "hours", "days", "weeks"] = Field(default="days", title="Unit")


class DateTimeOp(BuiltinOpSpec):
    op, category, title = "date_time", "time", "Date & time"
    title_fr = "Date et heure"
    summary = "Current time, reformat a date, add a duration, the time between two dates, or a date's parts."
    summary_fr = "Heure actuelle, date reformatée, durée ajoutée, temps entre deux dates, ou parties d’une date."
    keywords = ("date", "time", "timestamp", "now", "format date", "timezone", "duration")
    Config = DateTimeSettings

    @classmethod
    def ports(cls, config: dict[str, WorkflowValue]) -> tuple[PortSpec, ...]:
        return (_port("value", "input", required=False), _port("until", "input", required=False), _port("result", "output"))


#: Every op, in palette order.
BUILTIN_OP_SPECS: tuple[type[BuiltinOpSpec], ...] = (
    InputOp, OutputOp, ReadFileOp, WriteArtifactOp, HttpGetOp,
    ConditionOp, SwitchOp, MergeOp, WaitOp, ApprovalOp, FailOp,
    SetFieldsOp, TransformOp, FilterOp, SortOp, LimitOp, DedupeOp, AggregateOp, SqlOp, ParseOp, FormatOp, CalculateOp, EncodeOp,
    TemplateOp, ReplaceOp, ExtractOp, SplitOp, UppercaseOp, LowercaseOp, IdentityOp,
    DateTimeOp,
)
