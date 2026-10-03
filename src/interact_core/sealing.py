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
from pydantic import BaseModel, ConfigDict, TypeAdapter

from .workflows import ProjectSecretName, ProjectSecretText, SealedSecrets

#: The most one delivery carries, all values together.
SECRETS_LIMIT = 64 * 1024
_SECRETS: TypeAdapter[dict[ProjectSecretName, ProjectSecretText]] = TypeAdapter(dict[ProjectSecretName, ProjectSecretText])


class SecretsSeal(BaseModel):
    """Seals / opens one project's secrets for the machine whose signing key this is."""

    model_config = ConfigDict(frozen=True)
    signing_key: bytes

    @classmethod
    def for_token(cls, token: str) -> "SecretsSeal":
        """The PC's side: its signing key is SHA-256 of its token (as the server stores it)."""
        return cls(signing_key=sha256(token.encode()).digest())

    def _cipher(self) -> AESGCM:
        return AESGCM(HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"interact project secrets v1").derive(self.signing_key))

    @staticmethod
    def _bound(request: UUID, project: UUID) -> bytes:
        return f"{request}:{project}".encode()

    @staticmethod
    def revision(values: dict[str, str]) -> str:
        """What a set of secrets is, without them: sha256 of their canonical form."""
        return sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def seal(self, values: dict[str, str], *, request: UUID, project: UUID, origin: str) -> SealedSecrets:
        plain = json.dumps(_SECRETS.validate_python(values), sort_keys=True, separators=(",", ":")).encode()
        if len(plain) > SECRETS_LIMIT:
            raise ValueError(f"a project's secrets together exceed {SECRETS_LIMIT // 1024} KB")
        nonce = os.urandom(12)
        return SealedSecrets(project=project, origin=origin, revision=self.revision(values), nonce=nonce.hex(), count=len(values),
                             ciphertext=self._cipher().encrypt(nonce, plain, self._bound(request, project)).hex())

    def open(self, sealed: SealedSecrets, *, request: UUID) -> dict[str, str]:
        """The secrets, each name re-checked against the reserved list (`ProjectSecretName`)."""
        plain = self._cipher().decrypt(bytes.fromhex(sealed.nonce), bytes.fromhex(sealed.ciphertext), self._bound(request, sealed.project))
        return _SECRETS.validate_json(plain)
