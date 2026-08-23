"""The .env setup helper.

Everything that touches the network or the terminal is injected, so these run
with no API key and no prompts. What is actually asserted here is the part that
can lose data: merging into an existing .env without dropping the keys and
comments that are already in it.
"""

import pytest

import setup_api


# --- masking ---------------------------------------------------------------


def test_a_masked_key_shows_only_its_ends():
    masked = setup_api.mask("sk-ant-api03-abcdefghijklmnop4f2a")

    assert masked.startswith("sk-ant-")
    assert masked.endswith("4f2a")
    assert "abcdefghijklmnop" not in masked


def test_a_short_key_is_not_partially_revealed():
    """Below a useful length there is nothing to show that is not the key itself."""
    assert setup_api.mask("abcd") == "****"
    assert setup_api.mask("") == "****"


# --- .env parsing and merging ----------------------------------------------


EXISTING = """# Claude API key (do NOT commit a real key)
ANTHROPIC_API_KEY=old-claude-key

# Optional: override the Claude model
CLAUDE_MODEL=claude-sonnet-5
"""


def test_parsing_ignores_comments_and_blank_lines():
    assert setup_api.parse_env(EXISTING) == {
        "ANTHROPIC_API_KEY": "old-claude-key",
        "CLAUDE_MODEL": "claude-sonnet-5",
    }


def test_parsing_keeps_equals_signs_inside_a_value():
    assert setup_api.parse_env("OPENAI_BASE_URL=https://x/v1?a=b\n") == {
        "OPENAI_BASE_URL": "https://x/v1?a=b"
    }


def test_merging_replaces_a_value_in_place_and_keeps_the_comments():
    merged = setup_api.merge_env(EXISTING, {"ANTHROPIC_API_KEY": "new-claude-key"})

    assert "# Claude API key (do NOT commit a real key)" in merged
    assert "ANTHROPIC_API_KEY=new-claude-key" in merged
    assert "old-claude-key" not in merged
    assert "CLAUDE_MODEL=claude-sonnet-5" in merged


def test_merging_appends_keys_that_are_not_there_yet():
    merged = setup_api.merge_env(EXISTING, {"OPENAI_API_KEY": "new", "LLM_PROVIDER": "openai"})

    # The Claude entries survive — adding a provider must not remove one.
    assert setup_api.parse_env(merged) == {
        "ANTHROPIC_API_KEY": "old-claude-key",
        "CLAUDE_MODEL": "claude-sonnet-5",
        "OPENAI_API_KEY": "new",
        "LLM_PROVIDER": "openai",
    }


def test_merging_into_nothing_produces_a_valid_file():
    merged = setup_api.merge_env("", {"LLM_PROVIDER": "openai"})

    assert setup_api.parse_env(merged) == {"LLM_PROVIDER": "openai"}
    assert merged.endswith("\n")


def test_merging_does_not_duplicate_a_key_it_already_replaced():
    merged = setup_api.merge_env(EXISTING, {"CLAUDE_MODEL": "claude-opus-5"})

    assert merged.count("CLAUDE_MODEL=") == 1


# --- the command -----------------------------------------------------------


def fake_verifier(models=("gpt-test", "gpt-other")):
    calls = []

    def verify(provider, key, base_url=None):
        calls.append((provider, key, base_url))
        return list(models)

    verify.calls = calls
    return verify


def run(tmp_path, argv, *, verify=None, answers=(), ignored=True, existing=None):
    env = tmp_path / ".env"
    if existing is not None:
        env.write_text(existing, encoding="utf-8")
    printed = []
    replies = list(answers)
    code = setup_api.main(
        [*argv, "--env", str(env)],
        verify=verify or fake_verifier(),
        ask=lambda prompt, default=None: replies.pop(0),
        ask_secret=lambda prompt: replies.pop(0),
        is_ignored=lambda path: ignored,
        out=printed.append,
    )
    return code, env, "\n".join(printed)


def test_a_non_interactive_run_writes_the_provider_key_and_model(tmp_path):
    verify = fake_verifier()

    code, env, _ = run(
        tmp_path,
        ["--provider", "openai", "--model", "gpt-test", "--key", "test-key"],
        verify=verify,
        existing=EXISTING,
    )

    assert code == 0
    assert setup_api.parse_env(env.read_text(encoding="utf-8")) == {
        "ANTHROPIC_API_KEY": "old-claude-key",
        "CLAUDE_MODEL": "claude-sonnet-5",
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-key",
        "OPENAI_MODEL": "gpt-test",
    }
    assert verify.calls == [("openai", "test-key", None)]


def test_openrouter_is_verified_against_its_own_host(tmp_path):
    verify = fake_verifier(models=["vendor/model-test"])

    run(
        tmp_path,
        ["--provider", "openrouter", "--model", "vendor/model-test", "--key", "k"],
        verify=verify,
    )

    assert "openrouter.ai" in str(verify.calls[0][2])


def test_the_key_is_never_printed(tmp_path):
    _, _, printed = run(
        tmp_path, ["--provider", "openai", "--model", "gpt-test", "--key", "supersecret123456"]
    )

    assert "supersecret123456" not in printed


def test_a_rejected_key_leaves_the_file_untouched(tmp_path):
    def verify(provider, key, base_url=None):
        raise setup_api.VerificationError("401 Unauthorized")

    code, env, printed = run(
        tmp_path,
        ["--provider", "openai", "--model", "gpt-test", "--key", "bad"],
        verify=verify,
        existing=EXISTING,
    )

    assert code == 1
    assert env.read_text(encoding="utf-8") == EXISTING
    assert "401" in printed


def test_a_model_the_key_cannot_reach_is_refused(tmp_path):
    """Catching the typo here beats a 404 on the first real question."""
    code, env, printed = run(
        tmp_path,
        ["--provider", "openai", "--model", "gpt-typo", "--key", "k"],
        verify=fake_verifier(models=["gpt-test"]),
        existing=EXISTING,
    )

    assert code == 1
    assert env.read_text(encoding="utf-8") == EXISTING
    assert "gpt-typo" in printed


def test_an_alias_counts_as_reachable(tmp_path):
    """`claude-sonnet-5` is an alias; the models endpoint only lists dated IDs."""
    code, env, _ = run(
        tmp_path,
        ["--provider", "anthropic", "--model", "claude-sonnet-5", "--key", "k"],
        verify=fake_verifier(models=["claude-sonnet-5-20260101", "claude-opus-5-20260101"]),
    )

    assert code == 0
    assert setup_api.parse_env(env.read_text(encoding="utf-8"))["CLAUDE_MODEL"] == "claude-sonnet-5"


def test_a_prefix_that_is_not_a_version_boundary_is_still_a_typo():
    # gpt-4 must not be accepted just because gpt-4o exists.
    assert setup_api.model_is_reachable("gpt-4", ["gpt-4o"]) is False
    assert setup_api.model_is_reachable("gpt-4", ["gpt-4-turbo"]) is True
    assert setup_api.model_is_reachable("gpt-4o", ["gpt-4o"]) is True


def test_writing_is_refused_when_env_is_not_git_ignored(tmp_path):
    """A .env that git can see is one commit away from a leaked key."""
    code, env, printed = run(
        tmp_path,
        ["--provider", "openai", "--model", "gpt-test", "--key", "k"],
        ignored=False,
    )

    assert code == 1
    assert not env.exists()
    assert ".gitignore" in printed


def test_verification_can_be_skipped_for_an_offline_run(tmp_path):
    def explode(*args, **kwargs):
        raise AssertionError("verification should not run")

    code, env, _ = run(
        tmp_path,
        ["--provider", "openai", "--model", "gpt-test", "--key", "k", "--no-verify"],
        verify=explode,
    )

    assert code == 0
    assert setup_api.parse_env(env.read_text(encoding="utf-8"))["OPENAI_MODEL"] == "gpt-test"


# --- --check ---------------------------------------------------------------


def test_check_reports_the_active_provider_without_writing(tmp_path):
    existing = "LLM_PROVIDER=openai\nOPENAI_API_KEY=k\nOPENAI_MODEL=gpt-test\n"

    code, env, printed = run(tmp_path, ["--check"], existing=existing)

    assert code == 0
    assert env.read_text(encoding="utf-8") == existing
    assert "openai" in printed
    assert "gpt-test" in printed


def test_check_fails_when_the_active_provider_has_no_key(tmp_path):
    code, _, printed = run(tmp_path, ["--check"], existing="LLM_PROVIDER=openai\n")

    assert code == 1
    assert "OPENAI_API_KEY" in printed


def test_check_on_a_missing_file_says_so(tmp_path):
    code, _, printed = run(tmp_path, ["--check"])

    assert code == 1
    assert ".env" in printed


# --- interactive -----------------------------------------------------------


def test_the_interactive_run_asks_for_provider_key_and_model(tmp_path):
    code, env, _ = run(
        tmp_path,
        [],
        verify=fake_verifier(models=["gpt-test", "gpt-other"]),
        answers=["2", "typed-key", "1"],  # provider #2, the key, model #1
    )

    assert code == 0
    written = setup_api.parse_env(env.read_text(encoding="utf-8"))
    assert written["LLM_PROVIDER"] == "openai"
    assert written["OPENAI_API_KEY"] == "typed-key"
    assert written["OPENAI_MODEL"] == "gpt-test"


def test_an_unknown_provider_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, ["--provider", "gemini", "--model", "x", "--key", "k"])
