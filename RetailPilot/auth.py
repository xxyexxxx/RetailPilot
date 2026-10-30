"""Same-origin browser sessions and identity-bound service authentication."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

from fastapi import HTTPException, Request

from .config import settings
from .database import db_session

COOKIE = "retailpilot_auth"
_attempts: OrderedDict[str, list[float]] = OrderedDict()
_lock = Lock()


@dataclass(frozen=True)
class Principal:
    user_id: str
    mode: str
    csrf_token: str = ""


def password_hash(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 310000
    ).hex()
    return f"pbkdf2_sha256$310000${salt}${digest}"


def _password_matches(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt, digest = encoded.split("$")
        if algorithm != "pbkdf2_sha256" or not 100000 <= int(rounds) <= 1000000:
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), int(rounds)
        ).hex()
        return hmac.compare_digest(digest, actual)
    except (ValueError, TypeError):
        return False


def same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin and origin != f"{request.url.scheme}://{request.url.netloc}":
        raise HTTPException(403, "cross-origin request denied")


def authenticate(request: Request) -> Principal:
    if settings.allow_open_api:
        principal = Principal("demo-user", "demo")
    else:
        supplied = request.headers.get("x-api-key", "")
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
        if supplied:
            if not settings.api_key or not hmac.compare_digest(
                supplied.encode(), settings.api_key.encode()
            ):
                raise HTTPException(401, "invalid API credential")
            principal = Principal(settings.api_user_id, "service")
        else:
            token = request.cookies.get(COOKIE, "")
            with db_session() as connection:
                row = (
                    connection.execute(
                        "SELECT user_id,csrf_token FROM web_sessions WHERE token_hash=? AND expires_at>?",
                        (hashlib.sha256(token.encode()).hexdigest(), time.time()),
                    ).fetchone()
                    if token
                    else None
                )
            if not row:
                if not settings.api_key and not settings.auth_users_json:
                    raise HTTPException(
                        503,
                        "API access is disabled; configure browser users or a service key",
                    )
                raise HTTPException(401, "login required")
            principal = Principal(row["user_id"], "cookie", row["csrf_token"])
            if request.method not in {"GET", "HEAD", "OPTIONS"}:
                same_origin(request)
                if not hmac.compare_digest(
                    request.headers.get("x-csrf-token", "").encode(),
                    principal.csrf_token.encode(),
                ):
                    raise HTTPException(403, "invalid CSRF token")
    request.state.principal = principal
    return principal


def request_user(request: Request, requested: str | None = None) -> str:
    principal = request.state.principal
    if principal.mode == "demo":
        return requested or principal.user_id
    if requested and requested != principal.user_id:
        raise HTTPException(403, "user does not match authenticated identity")
    return principal.user_id


def login(request: Request, user_id: str, password: str) -> tuple[str, Principal]:
    same_origin(request)
    address = request.client.host if request.client else "unknown"
    now = time.time()
    with _lock:
        attempts = [stamp for stamp in _attempts.pop(address, []) if stamp > now - 300]
        _attempts[address] = attempts
        while len(_attempts) > 1024:
            _attempts.popitem(last=False)
        if len(attempts) >= 5:
            raise HTTPException(429, "too many login attempts; try later")
        attempts.append(now)
    try:
        users = json.loads(settings.auth_users_json or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(503, "browser login is not configured") from exc
    encoded = users.get(user_id, "") if isinstance(users, dict) else ""
    dummy = "pbkdf2_sha256$310000$00000000000000000000000000000000$" + "0" * 64
    verified = _password_matches(
        password, encoded if isinstance(encoded, str) and encoded else dummy
    )
    if not encoded or not verified:
        raise HTTPException(401, "invalid login")
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    with db_session() as connection:
        connection.execute("DELETE FROM web_sessions WHERE expires_at<=?", (now,))
        previous = request.cookies.get(COOKIE)
        if previous:
            connection.execute(
                "DELETE FROM web_sessions WHERE token_hash=?",
                (hashlib.sha256(previous.encode()).hexdigest(),),
            )
        connection.execute(
            "INSERT INTO web_sessions VALUES(?,?,?,?)",
            (
                hashlib.sha256(token.encode()).hexdigest(),
                user_id,
                csrf,
                now + settings.auth_session_seconds,
            ),
        )
    return token, Principal(user_id, "cookie", csrf)
