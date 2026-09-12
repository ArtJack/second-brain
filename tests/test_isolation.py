"""The suite's own safety net: proof that the tests cannot reach production.

`tests/conftest.py` pins storage and network before `secondbrain` is imported.
That mechanism is invisible — if it regresses, every other test keeps passing,
just against the owner's live Qdrant, live gateway and real corpus.

**Nothing in this file may put a live value into an assertion.** These tests fire
exactly when the real `.env` won, so at that moment `cfg.api_key` holds the real
key and `cfg.qdrant_url` the real lab address — and pytest's assertion rewriting
prints both sides of a failed comparison. A custom message does not suppress it.
The suite is run by coding agents whose transcripts are not a secret store, and
this repository is public, so its Actions logs are too.

So every assertion below compares a pre-computed boolean, and every failure
message describes the value instead of quoting it: `_describe()` returns a
length and a short digest, which is enough to tell "the pin held" from "something
else is here" without disclosing what.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def _describe(value: object) -> str:
    """A value-free fingerprint, safe to print in a failing assertion."""
    if value is None:
        return "<None>"
    text = str(value)
    if not text:
        return "<empty>"
    digest = hashlib.sha256(text.encode()).hexdigest()[:8]
    return f"<{len(text)} chars, sha256:{digest}>"


def test_config_storage_resolves_inside_the_sandbox(sandbox_root: Path) -> None:
    """No default storage path may point at the real corpus."""
    from secondbrain.config import cfg

    root = sandbox_root.resolve()
    for name in ("persist_dir", "memory_dir", "state_db"):
        resolved = Path(getattr(cfg, name)).resolve()
        inside = resolved.is_relative_to(root)
        assert inside, (
            f"cfg.{name} resolved outside the test sandbox {root} "
            f"(actual {_describe(resolved)}) — the isolation backstop in "
            f"conftest.py is not taking effect and this suite is running "
            f"against real storage."
        )


def test_config_network_points_at_a_closed_port(pinned_env: dict[str, str]) -> None:
    """A live call must fail loudly rather than reach the home lab."""
    from secondbrain.config import cfg

    base_url_pinned = cfg.base_url == pinned_env["OPENAI_BASE_URL"]
    assert base_url_pinned, f"cfg.base_url is not the pinned value: {_describe(cfg.base_url)}"

    qdrant_pinned = cfg.qdrant_url == pinned_env["QDRANT_URL"]
    assert qdrant_pinned, f"cfg.qdrant_url is not the pinned value: {_describe(cfg.qdrant_url)}"

    no_qdrant_key = cfg.qdrant_api_key is None
    assert no_qdrant_key, f"cfg.qdrant_api_key is set: {_describe(cfg.qdrant_api_key)}"

    # A real key here would mean the owner's .env won the race. Never quoted.
    api_key_pinned = cfg.api_key == pinned_env["OPENAI_API_KEY"]
    assert api_key_pinned, f"cfg.api_key is not the pinned test value: {_describe(cfg.api_key)}"


def test_collection_is_not_the_live_collection(pinned_env: dict[str, str]) -> None:
    """Writing to `second_brain` from a test would corrupt the real memory."""
    from secondbrain.config import cfg

    pinned = cfg.collection == pinned_env["SB_COLLECTION"]
    assert pinned, f"cfg.collection is not the pinned test collection: {_describe(cfg.collection)}"
    assert cfg.collection != "second_brain"


def test_behaviour_knobs_are_pinned_to_their_defaults(pinned_env: dict[str, str]) -> None:
    """Ambient tuning values change outcomes, not just credentials (SB-F-9).

    `SB_CHUNK_SIZE` alone reaches an upsert-count assertion in
    `test_collection_routing.py` and flips it. These knobs are harmless to
    disclose, so they may be compared directly.
    """
    from secondbrain.config import cfg

    assert str(cfg.chunk_size) == pinned_env["SB_CHUNK_SIZE"]
    assert str(cfg.chunk_overlap) == pinned_env["SB_CHUNK_OVERLAP"]
    assert str(cfg.top_k) == pinned_env["SB_TOP_K"]
    assert cfg.embed_model == pinned_env["EMBED_MODEL"]
    assert cfg.chat_model == pinned_env["CHAT_MODEL"]
    assert cfg.vision_model == pinned_env["VISION_MODEL"]
    assert cfg.weather_location is None
    assert cfg.hybrid_enabled is True


def test_the_web_app_under_test_has_no_ambient_cors_middleware() -> None:
    """`server.py` reads its origins at import, before any fixture can help.

    A legitimate `SB_WEB_ALLOWED_ORIGINS` in the developer's `.env` silently
    gives the local app CORS middleware that CI's app does not have — the object
    under test stops being the object CI tests. A `*` value raises at import and
    takes 17 tests down with it.
    """
    import secondbrain.server as server

    has_cors = any("CORS" in str(m) for m in server.app.user_middleware)
    assert not has_cors, (
        "the app under test carries CORS middleware, so SB_WEB_ALLOWED_ORIGINS "
        "leaked in from the environment and this app differs from CI's"
    )


def test_pins_survived_dotenv_load(pinned_env: dict[str, str]) -> None:
    """`load_dotenv()` must not have overridden the pins.

    `python-dotenv` defaults to `override=False`, so pre-set variables win. This
    notices if that default changes, or if some module calls
    `load_dotenv(override=True)`. Reports names and fingerprints, never values —
    `OPENAI_API_KEY` is in this dict.
    """
    # Force the import that triggers load_dotenv(); without it this test can
    # pass vacuously when selected on its own (SB-F-16).
    import secondbrain.config  # noqa: F401

    drifted = [key for key, expected in pinned_env.items() if os.environ.get(key) != expected]
    assert not drifted, (
        "these pins were overridden after conftest set them: "
        + ", ".join(f"{k} now {_describe(os.environ.get(k))}" for k in drifted)
    )


def test_no_plaintext_env_is_readable_from_the_repo_root(pinned_env: dict[str, str]) -> None:
    """Document the ancestor-walk hazard that made this file necessary.

    A `.env` at the repo root is normal on the owner's machine and absent in CI.
    This does not fail on its presence — it asserts the pins hold *despite* it,
    which is the property that actually matters.
    """
    from secondbrain.config import cfg

    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / ".env").is_file():
        return  # CI, or a clean checkout: nothing to shadow.

    not_repo_data = Path(cfg.persist_dir).resolve() != (repo_root / "data" / "chroma").resolve()
    assert not_repo_data, "cfg.persist_dir points at the repo's real data dir"

    qdrant_pinned = cfg.qdrant_url == pinned_env["QDRANT_URL"]
    assert qdrant_pinned, f"cfg.qdrant_url is not the pinned value: {_describe(cfg.qdrant_url)}"
