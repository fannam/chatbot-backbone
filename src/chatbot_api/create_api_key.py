from __future__ import annotations

import argparse
import asyncio
import json

from chatbot_api.database import session_scope
from chatbot_api.repositories import SqlAlchemyAuthRepository
from chatbot_api.settings import get_settings


async def run() -> None:
    parser = argparse.ArgumentParser(description="Create or update a user and mint an API key.")
    parser.add_argument("--user-id", required=True, help="Stable user identifier.")
    parser.add_argument("--name", default="default", help="Human-readable API key name.")
    parser.add_argument("--display-name", default=None, help="Display name for the user.")
    parser.add_argument("--email", default=None, help="Email address for the user.")
    parser.add_argument("--plan", default=None, help="Plan label for the user profile.")
    parser.add_argument("--locale", default=None, help="Locale for the user profile.")
    parser.add_argument(
        "--preferences-json",
        default="{}",
        help="JSON object for user preferences.",
    )
    args = parser.parse_args()

    preferences = json.loads(args.preferences_json)
    if not isinstance(preferences, dict):
        raise ValueError("--preferences-json must decode to a JSON object")

    settings = get_settings()
    async with session_scope(settings.database_url) as session_factory:
        async with session_factory() as session:
            repository = SqlAlchemyAuthRepository(session)
            user = await repository.upsert_user(
                user_id=args.user_id,
                display_name=args.display_name,
                email=args.email,
                plan=args.plan,
                locale=args.locale,
                preferences_json=preferences,
            )
            created = await repository.create_api_key(user_id=user.id, name=args.name)

    print(f"user_id={created.user.id}")
    print(f"api_key_name={args.name}")
    print(f"api_key_prefix={created.key_prefix}")
    print(f"api_key={created.api_key}")


if __name__ == "__main__":
    asyncio.run(run())
