"""The suite's own safety net: proof that the tests cannot reach production.

`tests/conftest.py` pins storage and network before `secondbrain` is imported.
That mechanism is invisible — if it regresses, every other test keeps passing,
just against the owner's live Qdrant, live gateway and real corpus. These tests
make the regression loud.

They assert the *resolved* configuration rather than the environment variables,
because what matters is what `Config` actually computed, not what was set.
"""

from __future__ import annotations

import os
from pathlib import Path


def test_config_storage_resolves_inside_the_sandbox(sandbox_root: Path) -> None:
    """No default storage path may point at the real corpus."""
    from secondbrain.config import cfg

    for name in ("persist_dir", "memory_dir", "state_db"):
        resolved = Path(getattr(cfg, name)).resolve()
        assert resolved.is_relative_to(sandbox_root.resolve()), (
            f"cfg.{name} resolved to {resolved}, outside the test sandbox "
            f"{sandbox_root} — the isolation backstop in conftest.py is not "
            f"taking effect and this suite is running against real storage."
        )


def test_config_network_points_at_a_closed_port() -> None:
    """A live call must fail loudly rather than reach the home lab."""
    from secondbrain.config import cfg

    assert cfg.base_url == "http://127.0.0.1:9/v1", cfg.base_url
    assert cfg.qdrant_url == "http://127.0.0.1:9", cfg.qdrant_url
    assert cfg.qdrant_api_key is None
    # A real key would mean the owner's .env won the race.
    assert cfg.api_key == "test-key", "cfg.api_key is not the pinned test value"


def test_collection_is_not_the_live_collection() -> None:
    """Writing to `second_brain` from a test would corrupt the real memory."""
    from secondbrain.config import cfg

    assert cfg.collection == "second_brain_tests", cfg.collection
    assert cfg.collection != "second_brain"


def test_pins_survived_dotenv_load(pinned_env: dict[str, str]) -> None:
    """`load_dotenv()` must not have overridden the pins.

    `python-dotenv` defaults to `override=False`, so pre-set variables win. This
    test is what notices if that default ever changes, or if some module calls
    `load_dotenv(override=True)`.
    """
    for key, expected in pinned_env.items():
        assert os.environ.get(key) == expected, (
            f"{key} is {os.environ.get(key)!r}, expected {expected!r} — "
            f"something overrode the conftest pin after import."
        )


def test_no_plaintext_env_is_readable_from_the_repo_root() -> None:
    """Document the ancestor-walk hazard that made this file necessary.

    A `.env` at the repo root is normal on the owner's machine and absent in CI.
    This test does not fail on its presence — it asserts the pins hold *despite*
    it, which is the property that actually matters.
    """
    from secondbrain.config import cfg

    repo_root = Path(__file__).resolve().parents[1]
    real_env = repo_root / ".env"
    if not real_env.is_file():
        return  # CI, or a clean checkout: nothing to shadow.

    # The real .env exists and was walked into by load_dotenv(). The pins must
    # still be the values in effect.
    assert Path(cfg.persist_dir).resolve() != (repo_root / "data" / "chroma").resolve()
    assert cfg.qdrant_url == "http://127.0.0.1:9"
