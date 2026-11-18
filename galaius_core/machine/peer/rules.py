"""The numbers every side of cross-PC dispatch shares: the server enforces them, the runner and the CLI know them."""

from typing import Final

from galaius_core.workflows.contracts import AgentTouchScope

#: A peer start's brief and a peer message, in characters of text.
PEER_TEXT_MAX: Final = 32 * 1024
PEER_MESSAGE_MAX: Final = 16 * 1024
#: Idempotency key, as the HTTP header's own rule: letters, digits, `._:-`.
PEER_KEY_PATTERN: Final = r"^[A-Za-z0-9._:-]+$"
PEER_KEY_MAX: Final = 160
#: Starts a source PC may place per hour and per day, and all the PCs of one owner per hour (counted in the database).
STARTS_PER_PC_HOUR: Final = 30
STARTS_PER_PC_DAY: Final = 100
STARTS_PER_OWNER_HOUR: Final = 60
#: Peer messages per target run per hour, and per source PC per hour.
MESSAGES_PER_RUN_HOUR: Final = 60
MESSAGES_PER_PC_HOUR: Final = 600
#: Peer requests one PC's connection serves at once; over it the answer is `dispatch_rate`.
PEER_IN_FLIGHT: Final = 4
#: A start the target PC never answered is `failed` after this many seconds, and its run asked to stop when it shows.
PENDING_TTL_SECONDS: Final = 600
#: How long a connection of another process instance is read as a takeover, not a reconnect, after a socket held alive.
TAKEOVER_WINDOW_SECONDS: Final = 30
#: What a peer-placed run may do at most, whatever the PCs' own settings allow.
PEER_CEILING: Final[AgentTouchScope] = "workspace_write"
#: The local endpoint of a runner: one request frame at most, read within this many seconds, this many connections.
BROKER_FRAME_MAX: Final = 64 * 1024
BROKER_READ_SECONDS: Final = 5.0
BROKER_CONNECTIONS: Final = 16
#: Seconds the source waits for the server, and (a start) for the target PC's answer.
PEER_ANSWER_SECONDS: Final = 20.0
PEER_START_SECONDS: Final = 60.0
#: What may cross one PC's link in an hour, every class but a project copy (which has its own one-time cap): past it the peer calls of that PC answer
#: `link_ceiling` until the hour moves on. His own web requests are never counted against it.
LINK_CEILING_BYTES_PER_HOUR: Final = 50 * 1024 * 1024
