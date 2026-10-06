"""The current user's global configuration, kept outside the repository.

    ~/.config/proofs/config        (or $XDG_CONFIG_HOME/proofs/config)

is a JSON object holding the user's identity, named git-style (see
.comments, "User identity"):

    {
        "user.id": "eoin",
        "user.name": "Eoin Mackall",
        "user.email": "eoin@example.com"
    }

The keys are set and read through the config subcommand, the way git's are
set and read through `git config`: `proofs config --global user.id ID` sets,
`proofs config user.id` prints, and an empty value unsets. It is per-user by
design and lives in the home directory, so a clone of the repository never
carries one user's identity into another's working copy — the file is never
committed (see .comments, "User identity").
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_SUBDIR = "proofs"
CONFIG_FILENAME = "config"

# The keys the config holds: the user's identity, named section.field the
# way git's [user] section is named. user.id goes into file names (dags/
# <user_id>_dag.json, certificates/<user_id>.jsonl, refutation file names),
# so it must stay a plain identifier; user.name (a full name) and
# user.email are free-form, as git's are.
USER_ID_KEY = "user.id"
USER_NAME_KEY = "user.name"
USER_EMAIL_KEY = "user.email"
KEYS = (USER_ID_KEY, USER_NAME_KEY, USER_EMAIL_KEY)

# A user.id is reused in file names (dags/<user_id>_dag.json,
# certificates/<user_id>.jsonl, refutation file names), so it must stay a
# plain identifier rather than an arbitrary string.
_USER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

# The pre-rename name of user.id, kept so a config file written before the
# rename keeps its identity: load() reads it as user.id, and the next save
# upgrades the file in place.
_LEGACY_USER_ID_KEY = "user_id"


class UserConfigError(Exception):
    """The config file is corrupt, or no user.id could be resolved."""


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / CONFIG_SUBDIR / CONFIG_FILENAME


def load() -> Dict[str, Any]:
    """The config object, or {} when the file does not exist yet.

    A file that exists but is not a JSON object is an error, not an empty
    config: it is the user's data, and overwriting it would be worse than
    failing loudly. A file that still holds the legacy "user_id" key is
    read as "user.id", so it keeps its identity and is upgraded on the
    next save.
    """
    p = config_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise UserConfigError(f"{p} is not a readable config file: {e}") from e
    if not isinstance(data, dict):
        raise UserConfigError(f"{p} must hold a JSON object")
    if data.get(USER_ID_KEY) is None and data.get(_LEGACY_USER_ID_KEY) is not None:
        data[USER_ID_KEY] = data.pop(_LEGACY_USER_ID_KEY)
    return data


def save(data: Dict[str, Any]) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp, p)


def validate_key(key: str) -> str:
    """Return key if it is one the config holds, else ValueError."""
    if key not in KEYS:
        raise ValueError(
            f"{key!r} is not a config key; the keys are {' '.join(KEYS)}"
        )
    return key


def validate_value(key: str, value: str) -> str:
    """Return value if it may be stored under key, else ValueError."""
    validate_key(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} needs a non-empty value")
    if key == USER_ID_KEY and not _USER_ID_RE.match(value):
        raise ValueError(
            f"{value!r} is not a usable user.id: start with a letter or "
            f"digit and use only letters, digits, dots, underscores and "
            f"hyphens — it becomes part of file names"
        )
    return value


def get(key: str) -> Optional[str]:
    """The value the config holds under key, or None when it holds none."""
    value = load().get(key)
    return value if isinstance(value, str) and value else None


def set(key: str, value: str) -> str:
    """Store value under key and return it.

    An empty value unsets the key, as git's `git config user.name ""`
    unsets. Validation runs before anything is written, so a bad value
    leaves the file untouched.
    """
    validate_key(key)
    data = load()
    if not isinstance(value, str) or value == "":
        data.pop(key, None)
    else:
        data[key] = validate_value(key, value)
    save(data)
    return data.get(key, "")


def unset(key: str) -> None:
    """Delete key from the config (a no-op when it is not set)."""
    validate_key(key)
    data = load()
    if key in data:
        del data[key]
        save(data)


def items() -> Dict[str, str]:
    """Every key the config holds, as {key: value}, in KEYS' order."""
    data = load()
    return {
        key: data[key]
        for key in KEYS
        if isinstance(data.get(key), str) and data[key]
    }


def ensure_user_id() -> str:
    """The user.id every run acts as, resolved once per process.

    Like git, proofs does not ask: a run without a user.id fails and
    names the command that sets it (`proofs config --global user.id ID`),
    rather than prompting on the terminal.
    """
    uid = get(USER_ID_KEY)
    if uid is None:
        raise UserConfigError(
            "user.id is not set, and every run acts as one.\n"
            "Run\n\n"
            '  proofs config --global user.id "YOUR_USER_ID"\n\n'
            f"(it is stored in {config_path()}, outside the repository, "
            "and is never committed)"
        )
    try:
        validate_value(USER_ID_KEY, uid)
    except ValueError:
        raise UserConfigError(
            f"user.id {uid!r} in {config_path()} is not usable (start with "
            "a letter or digit; letters, digits, dots, underscores and "
            "hyphens only — it becomes part of file names); set a new one "
            "with proofs config --global user.id ID"
        ) from None
    return uid
