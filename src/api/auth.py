import base64
import binascii
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sqlite3
import time
from typing import Annotated, Any, cast
from uuid import uuid4

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

DATABASE_PATH = Path("data/app.sqlite3")
PASSWORD_HASH_ITERATIONS = 600_000
ACCESS_TOKEN_LIFETIME_SECONDS = 60 * 60
_DUMMY_SALT = b"knowledge-assistant-dummy-salt"
_DUMMY_PASSWORD_HASH = hashlib.pbkdf2_hmac(
    "sha256",
    b"not-a-real-password",
    _DUMMY_SALT,
    PASSWORD_HASH_ITERATIONS,
)
_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class UserIdentity:
    user_id: str
    username: str


def _database_connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    with closing(_database_connection()) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL REFERENCES users(user_id),
                    size_bytes INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS documents_owner_id "
                "ON documents(owner_id)"
            )


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )


def _normalize_username(username: str) -> str:
    return username.strip().casefold()


def register_user(username: str, password: str) -> UserIdentity:
    initialize_database()
    normalized_username = _normalize_username(username)
    if not 3 <= len(normalized_username) <= 64:
        raise ValueError("Username must contain between 3 and 64 characters.")

    salt = secrets.token_bytes(16)
    user = UserIdentity(user_id=uuid4().hex, username=normalized_username)

    try:
        with closing(_database_connection()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO users (user_id, username, password_salt, password_hash)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        user.user_id,
                        user.username,
                        salt,
                        _hash_password(password, salt),
                    ),
                )
    except sqlite3.IntegrityError as exc:
        raise ValueError("Username is already registered.") from exc

    return user


def authenticate_user(username: str, password: str) -> UserIdentity | None:
    initialize_database()
    with closing(_database_connection()) as connection:
        row = connection.execute(
            """
            SELECT user_id, username, password_salt, password_hash
            FROM users
            WHERE username = ?
            """,
            (_normalize_username(username),),
        ).fetchone()

    if row is None:
        hmac.compare_digest(
            _hash_password(password, _DUMMY_SALT),
            _DUMMY_PASSWORD_HASH,
        )
        return None

    user_id, stored_username, salt, expected_hash = row
    if not hmac.compare_digest(_hash_password(password, salt), expected_hash):
        return None

    return UserIdentity(user_id=user_id, username=stored_username)


def _signing_key() -> bytes:
    secret = os.environ.get("JWT_SECRET_KEY", "")
    if len(secret.encode("utf-8")) < 32:
        raise RuntimeError(
            "JWT_SECRET_KEY must be configured with at least 32 bytes."
        )
    return secret.encode("utf-8")


def validate_signing_key() -> None:
    _signing_key()


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def create_access_token(user: UserIdentity) -> str:
    now = int(time.time())
    header = _base64url_encode(
        json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode()
    )
    payload = _base64url_encode(
        json.dumps(
            {
                "sub": user.user_id,
                "username": user.username,
                "iat": now,
                "exp": now + ACCESS_TOKEN_LIFETIME_SECONDS,
            },
            separators=(",", ":"),
        ).encode()
    )
    signing_input = f"{header}.{payload}".encode("ascii")
    signature = hmac.new(_signing_key(), signing_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_base64url_encode(signature)}"


def _decode_base64url(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_bearer_scheme),
    ],
) -> UserIdentity:
    if credentials is None or credentials.scheme.casefold() != "bearer":
        raise _unauthorized()

    token_parts = credentials.credentials.split(".")
    if len(token_parts) != 3:
        raise _unauthorized()

    header_part, payload_part, signature_part = token_parts
    try:
        decoded_header: Any = json.loads(_decode_base64url(header_part))
        if not isinstance(decoded_header, dict):
            raise _unauthorized()
        header = cast(dict[str, object], decoded_header)
        if header.get("alg") != "HS256":
            raise _unauthorized()

        signing_input = f"{header_part}.{payload_part}".encode("ascii")
        expected_signature = hmac.new(
            _signing_key(),
            signing_input,
            hashlib.sha256,
        ).digest()
        provided_signature = _decode_base64url(signature_part)
        if not hmac.compare_digest(expected_signature, provided_signature):
            raise _unauthorized()

        decoded_payload: Any = json.loads(_decode_base64url(payload_part))
        if not isinstance(decoded_payload, dict):
            raise _unauthorized()
        payload = cast(dict[str, object], decoded_payload)
        subject = payload.get("sub")
        username = payload.get("username")
        expiration = payload.get("exp")
        if (
            not isinstance(subject, str)
            or not subject
            or not isinstance(username, str)
            or not username
            or not isinstance(expiration, int)
            or isinstance(expiration, bool)
            or expiration <= int(datetime.now(timezone.utc).timestamp())
        ):
            raise _unauthorized()
    except (ValueError, TypeError, KeyError, binascii.Error, json.JSONDecodeError):
        raise _unauthorized() from None

    return UserIdentity(user_id=subject, username=username)


def add_document_owner(
    document_id: str,
    owner_id: str,
    size_bytes: int,
) -> None:
    initialize_database()
    with closing(_database_connection()) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO documents (document_id, owner_id, size_bytes)
                VALUES (?, ?, ?)
                """,
                (document_id, owner_id, size_bytes),
            )


def remove_document_owner(document_id: str, owner_id: str) -> None:
    initialize_database()
    with closing(_database_connection()) as connection:
        with connection:
            connection.execute(
                "DELETE FROM documents WHERE document_id = ? AND owner_id = ?",
                (document_id, owner_id),
            )


def list_user_documents(owner_id: str) -> list[dict[str, str | int]]:
    initialize_database()
    with closing(_database_connection()) as connection:
        rows = connection.execute(
            """
            SELECT document_id, size_bytes
            FROM documents
            WHERE owner_id = ?
            ORDER BY document_id COLLATE NOCASE
            """,
            (owner_id,),
        ).fetchall()
    return [
        {"document_id": document_id, "size_bytes": size_bytes}
        for document_id, size_bytes in rows
    ]
