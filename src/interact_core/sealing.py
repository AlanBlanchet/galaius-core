"""Project secrets sealed for one machine on their way from the server's vault to a PC
(`SealedSecrets`): both sides derive the same AES-256-GCM key from that machine's signing key
(SHA-256 of its token), so a websocket frame log or a proxy in between sees ciphertext only."""

import json
import os
from hashlib import sha256
from uuid import UUID

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .workflows import SECRETS_TOTAL, ProjectSecretName, ProjectSecretText, SealedSecrets

_SECRETS: TypeAdapter[dict[ProjectSecretName, ProjectSecretText]] = TypeAdapter(dict[ProjectSecretName, ProjectSecretText], config=ConfigDict(hide_input_in_errors=True))


class SecretsSeal(BaseModel):
    """Seals / opens one project's secrets for the machine whose signing key this is."""

    model_config = ConfigDict(frozen=True)
    signing_key: bytes = Field(repr=False)

    @classmethod
    def for_token(cls, token: str) -> "SecretsSeal":
        """The PC's side: its signing key is SHA-256 of its token (as the server stores it)."""
        return cls(signing_key=sha256(token.encode()).digest())

    def _cipher(self) -> AESGCM:
        return AESGCM(HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"interact project secrets v1").derive(self.signing_key))

    @staticmethod
    def _bound(request: UUID, project: UUID, origin: str) -> bytes:
        return f"{request}:{project}:{origin}".encode()

    @staticmethod
    def plain(values: dict[str, str]) -> bytes:
        """The canonical form sealed (and measured against `SECRETS_TOTAL`); names and values checked."""
        return json.dumps(_SECRETS.validate_python(values), sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def fits(cls, values: dict[str, str]) -> bool:
        return len(cls.plain(values)) <= SECRETS_TOTAL

    def seal(self, values: dict[str, str], *, request: UUID, project: UUID, origin: str) -> SealedSecrets:
        plain = self.plain(values)
        if len(plain) > SECRETS_TOTAL:
            raise ValueError(f"a project's secrets together exceed {SECRETS_TOTAL // 1024} KB")
        nonce = os.urandom(12)
        return SealedSecrets(project=project, origin=origin, nonce=nonce.hex(), count=len(values),
                             ciphertext=self._cipher().encrypt(nonce, plain, self._bound(request, project, origin)).hex())

    def open(self, sealed: SealedSecrets, *, request: UUID) -> dict[str, str]:
        """The secrets, each name and value re-checked (`ProjectSecretName`, `ProjectSecretText`)."""
        plain = self._cipher().decrypt(bytes.fromhex(sealed.nonce), bytes.fromhex(sealed.ciphertext), self._bound(request, sealed.project, sealed.origin))
        return _SECRETS.validate_json(plain)
