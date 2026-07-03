from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from secrets import token_urlsafe
from typing import Any

API_KEY_PREFIX_LENGTH = 12


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    display_name: str | None
    email: str | None
    plan: str | None
    locale: str | None
    preferences: dict[str, Any]

    def to_user_profile_metadata(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "email": self.email,
            "plan": self.plan,
            "locale": self.locale,
            "preferences": dict(self.preferences),
        }


def generate_api_key() -> str:
    return f"cbk_{token_urlsafe(24)}"


def hash_api_key(api_key: str) -> str:
    return sha256(api_key.encode("utf-8")).hexdigest()


def build_api_key_prefix(api_key: str) -> str:
    return api_key[:API_KEY_PREFIX_LENGTH]


def owner_user_id_of(authenticated_user: AuthenticatedUser | None) -> str | None:
    return None if authenticated_user is None else authenticated_user.user_id


def is_forbidden_owner(
    authenticated_user: AuthenticatedUser | None,
    requested_user_id: str,
) -> bool:
    return authenticated_user is not None and requested_user_id != authenticated_user.user_id
