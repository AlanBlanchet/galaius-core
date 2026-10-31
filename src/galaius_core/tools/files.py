"""The agent tools whose answer names the files they kept for their run (`workflows.file_token`), each with the formats it
writes (suffix, the content type it is served with): `create_file` here, each media job's in `media.MEDIA_TOOLS`; together
they make `workflows.FILE_TOOL_REGISTRY`, which the chat's file cards and every kept-file rule read."""

from galaius_core.wire import WireModel


class FileTool(WireModel):
    """One file tool: its name, and each format it writes (suffix, the content type it is served with)."""

    tool: str
    writes: tuple[tuple[str, str], ...]


#: The formats an agent writes with `create_file`, one row per format: its suffix and the content type it is served with.
CREATE_FILE_TYPES: tuple[tuple[str, str], ...] = (
    (".txt", "text/plain"), (".md", "text/markdown"), (".csv", "text/csv"), (".json", "application/json"),
    (".eml", "message/rfc822"), (".pdf", "application/pdf"),
    (".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    (".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    (".pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
)
CREATE_FILE = FileTool(tool="create_file", writes=CREATE_FILE_TYPES)
