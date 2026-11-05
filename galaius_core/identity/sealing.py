"""Project secrets sealed for one machine on their way from the server's vault to a PC
(`SealedSecrets`): both sides derive the same AES-256-GCM key from that machine's signing key
(SHA-256 of its token), so a websocket frame log or a proxy in between sees ciphertext only."""

import base64
import json
import os
from hashlib import sha256
from uuid import UUID

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from galaius_core.workflows.contracts import SECRETS_TOTAL, ProjectSecretName, ProjectSecretText, SealedHeld, SealedLogin, SealedSecrets

_OAEP = padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)

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
        return AESGCM(HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"galaius project secrets v1").derive(self.signing_key))

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


def seal_to(seal_key: str, plain: str) -> SealedHeld:
    """`plain` sealed to `seal_key` (RSA-OAEP SHA-256, SubjectPublicKeyInfo DER base64): a fresh AES-256-GCM key wrapped
    by it, `plain` under that key. Only the private key's holder opens it (`open_with`)."""
    public = serialization.load_der_public_key(base64.b64decode(seal_key))
    key, nonce = AESGCM.generate_key(bit_length=256), os.urandom(12)
    return SealedHeld(key=base64.b64encode(public.encrypt(key, _OAEP)).decode(), nonce=nonce.hex(),
                      ciphertext=AESGCM(key).encrypt(nonce, plain.encode(), None).hex())


def open_with(private: rsa.RSAPrivateKey, sealed: SealedHeld) -> str:
    """`seal_to`'s `plain`, opened with the private key `sealed` was made for (ValueError / InvalidTag otherwise)."""
    key = private.decrypt(base64.b64decode(sealed.key), _OAEP)
    return AESGCM(key).decrypt(bytes.fromhex(sealed.nonce), bytes.fromhex(sealed.ciphertext), None).decode()


class LoginSeal(BaseModel):
    """Seals / opens a subscription's held login (`SealedLogin`) for the machine whose signing key this is: the same
    derivation as `SecretsSeal`, its own label, bound to the request, the subscription and its provider."""

    model_config = ConfigDict(frozen=True)
    signing_key: bytes = Field(repr=False)

    @classmethod
    def for_token(cls, token: str) -> "LoginSeal":
        """The PC's side: its signing key is SHA-256 of its token (as the server stores it)."""
        return cls(signing_key=sha256(token.encode()).digest())

    def _cipher(self) -> AESGCM:
        return AESGCM(HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"galaius held login v1").derive(self.signing_key))

    @staticmethod
    def _bound(request: UUID, subscription: UUID, provider: str) -> bytes:
        return f"{request}:{subscription}:{provider}".encode()

    def seal(self, login: str, *, request: UUID, subscription: UUID, provider: str) -> SealedLogin:
        nonce = os.urandom(12)
        return SealedLogin(nonce=nonce.hex(), ciphertext=self._cipher().encrypt(nonce, login.encode(), self._bound(request, subscription, provider)).hex())

    def open(self, sealed: SealedLogin, *, request: UUID, subscription: UUID, provider: str) -> str:
        return self._cipher().decrypt(bytes.fromhex(sealed.nonce), bytes.fromhex(sealed.ciphertext), self._bound(request, subscription, provider)).decode()
