"""Isolation backstop for the whole suite.

`secondbrain.config` does two things at import time: it calls `load_dotenv()`
with no argument, and it builds the module-level `cfg`. `load_dotenv()` walks
*up* the directory tree for the nearest `.env`, which on a developer machine
resolves to the owner's real one — so importing `secondbrain` anywhere in a test
would pick up the live Qdrant URL, the live API key, and the real corpus paths.
25 of 26 test modules import `secondbrain`, so that fires at collection time,
before any test body runs.

Until now the hermeticity lived entirely *outside* the repository, in the CI
workflow's `env:` block. That is why CI was green and a local run was not
isolated at all: nothing in the repo enforced it, and the failure is silent —
tests pass either way, they just pass against production.

This module moves that guarantee inside the repo. pytest imports `conftest.py`
before it collects any test module, and `load_dotenv()` does not override
variables that already exist, so the values pinned here win over the real
`.env`. The CI block is now a belt-and-braces duplicate rather than the only
copy.

Two rules keep this working:

1. **Nothing here may import `secondbrain` at module scope.** The pinning has to
   land in `os.environ` before that package is first imported; an import at the
   top of this file would defeat the entire mechanism.
2. **The pin is unconditional.** Honouring a pre-existing `QDRANT_URL` would let
   the exact ambient state this guards against leak back in. A test that wants
   different values sets them per-test, not through the environment.

`tests/test_isolation.py` asserts all of this holds, so a regression here is
caught by the suite rather than discovered against live infrastructure.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

# One throwaway tree per session, removed at interpreter exit. Created before the
# pins below so every path can point inside it.
SANDBOX = Path(tempfile.mkdtemp(prefix="second-brain-tests-"))
atexit.register(shutil.rmtree, SANDBOX, ignore_errors=True)

# Storage is redirected so a test that constructs a store with no arguments
# cannot reach the real corpus; the network points at a closed port so a
# regression that adds a live call fails loudly instead of quietly succeeding
# against the home lab. Port 9 is `discard` — reliably refused, never routed.
PINNED_ENV: dict[str, str] = {
    # storage
    "SB_STORE": "chroma",
    "SB_DATA": str(SANDBOX / "chroma"),
    "SB_MEMORY_DIR": str(SANDBOX / "memory"),
    "SB_STATE_DB": str(SANDBOX / "state.sqlite3"),
    "SB_COLLECTION": "second_brain_tests",
    # network — unreachable on purpose
    "OPENAI_API_KEY": "test-key",
    "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
    "QDRANT_URL": "http://127.0.0.1:9",
    "QDRANT_API_KEY": "",
}

os.environ.update(PINNED_ENV)


@pytest.fixture(scope="session")
def sandbox_root() -> Path:
    """The throwaway tree every default storage path is pinned into."""
    return SANDBOX


@pytest.fixture(scope="session")
def pinned_env() -> dict[str, str]:
    """The exact pins applied at import time, for assertions in test_isolation."""
    return dict(PINNED_ENV)
