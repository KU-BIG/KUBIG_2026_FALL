"""Configure an LLM provider in `.env`.

    uv run python setup_api.py              # 대화형
    uv run python setup_api.py --check      # 지금 설정이 동작하는지만 확인
    uv run python setup_api.py --provider openrouter --model vendor/model --key ...

The key is verified against the provider's models endpoint before anything is
written. That call returns the list of models the key can actually reach and
costs no tokens, so a typo is caught here instead of at the first question.

Nothing is overwritten: the existing `.env` is merged line by line, keeping the
comments and the keys of providers you are not touching. The key is read with
`getpass`, is never echoed, and is never printed back — not even on success.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from getpass import getpass
from pathlib import Path

from generation.llm import OPENROUTER_BASE_URL, PROVIDERS

DEFAULT_ENV = Path(__file__).resolve().parent / ".env"
VISIBLE_PREFIX = 7
VISIBLE_SUFFIX = 4
MIN_MASKABLE = 12
MODEL_LIST_LIMIT = 30


class VerificationError(RuntimeError):
    """The provider rejected the key, or could not be reached."""


# --- pure helpers ----------------------------------------------------------


def mask(key: str) -> str:
    """`sk-ant-…4f2a`. Short strings are hidden outright — there is nothing safe to show."""
    if len(key) < MIN_MASKABLE:
        return "****"
    return f"{key[:VISIBLE_PREFIX]}…{key[-VISIBLE_SUFFIX:]}"


def parse_env(text: str) -> dict[str, str]:
    """`KEY=VALUE` pairs, ignoring comments and blank lines."""
    values = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        values[name.strip()] = value.strip()
    return values


def merge_env(text: str, updates: dict[str, str]) -> str:
    """Apply `updates` to an existing .env, in place where the key already exists.

    Rewriting the file from a parsed dict would drop every comment and reorder
    the rest, so existing lines are edited where they stand and only genuinely
    new keys are appended.
    """
    remaining = dict(updates)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        name = stripped.partition("=")[0].strip()
        if stripped and not stripped.startswith("#") and "=" in stripped and name in remaining:
            lines.append(f"{name}={remaining.pop(name)}")
        else:
            lines.append(line)

    if remaining:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("# setup_api.py")
        lines.extend(f"{name}={value}" for name, value in remaining.items())

    return "\n".join(lines).lstrip("\n") + "\n"


def model_is_reachable(model: str, models: list[str]) -> bool:
    """Whether a model ID is covered by what the key can see.

    An exact match, or an alias of one: providers list dated IDs
    (`claude-sonnet-5-20260101`) while people configure the alias
    (`claude-sonnet-5`). The trailing dash keeps `gpt-4` from matching `gpt-4o`.
    """
    return model in models or any(listed.startswith(f"{model}-") for listed in models)


def base_url_for(provider: str) -> str | None:
    """Where this provider's API lives, when it is not the SDK default."""
    if provider == "openrouter":
        return OPENROUTER_BASE_URL
    if provider == "openai":
        return os.getenv("OPENAI_BASE_URL")
    return None


# --- side effects ----------------------------------------------------------


def verify_key(provider: str, key: str, base_url: str | None = None) -> list[str]:
    """List the models this key can reach. Uses the models endpoint — zero tokens."""
    try:
        if provider == "anthropic":
            from anthropic import Anthropic

            return [model.id for model in Anthropic(api_key=key).models.list(limit=100).data]

        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
        return [model.id for model in client.models.list().data]
    except Exception as exc:
        raise VerificationError(str(exc)) from exc


def git_ignores(path: Path) -> bool:
    """Whether git is set to ignore this path. A tracked .env is a leaked key."""
    try:
        done = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            capture_output=True,
            cwd=str(Path(path).resolve().parent),
        )
    except OSError:
        return False
    return done.returncode == 0


def ask_line(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    reply = input(f"{prompt}{suffix}: ").strip()
    return reply or (default or "")


# --- command ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM 프로바이더 API 설정을 .env에 기록합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", choices=sorted(PROVIDERS), help="설정할 프로바이더")
    parser.add_argument("--model", help="사용할 모델 ID")
    parser.add_argument("--key", help="API 키. 생략하면 화면에 안 보이게 입력받습니다")
    parser.add_argument("--env", default=str(DEFAULT_ENV), help="기록할 .env 경로")
    parser.add_argument("--check", action="store_true", help="현재 설정만 확인하고 아무것도 고치지 않습니다")
    parser.add_argument("--no-verify", action="store_true", help="키 검증 호출을 건너뜁니다")
    return parser


def _choose_provider(ask, out) -> str:
    names = sorted(PROVIDERS)
    out("\n프로바이더를 고르세요:")
    for index, name in enumerate(names, 1):
        out(f"  {index}. {name}")
    reply = ask("번호 또는 이름", names[0])
    if reply.isdigit() and 1 <= int(reply) <= len(names):
        return names[int(reply) - 1]
    if reply in PROVIDERS:
        return reply
    raise ValueError(f"알 수 없는 프로바이더: {reply}")


def _choose_model(models: list[str], ask, out) -> str:
    if not models:
        return ask("모델 ID")
    shown = models[:MODEL_LIST_LIMIT]
    out(f"\n이 키로 쓸 수 있는 모델 {len(models)}개" + (f" (앞 {len(shown)}개)" if len(models) > len(shown) else ""))
    for index, name in enumerate(shown, 1):
        out(f"  {index}. {name}")
    reply = ask("번호 또는 모델 ID", shown[0])
    if reply.isdigit() and 1 <= int(reply) <= len(shown):
        return shown[int(reply) - 1]
    return reply


def _run_check(env_path: Path, verify, no_verify: bool, out) -> int:
    if not env_path.is_file():
        out(f"{env_path} 파일이 없습니다. 먼저 실행하세요: uv run python setup_api.py")
        return 1

    values = parse_env(env_path.read_text(encoding="utf-8"))
    provider = (values.get("LLM_PROVIDER") or "anthropic").strip().lower()
    out(f"LLM_PROVIDER = {provider}")

    if provider not in PROVIDERS:
        out(f"알 수 없는 프로바이더입니다. 가능한 값: {', '.join(sorted(PROVIDERS))}")
        return 1

    config = PROVIDERS[provider]
    key_env, model_env = str(config["key_env"]), str(config["model_env"])
    key = values.get(key_env) or os.getenv(key_env)
    model = values.get(model_env) or os.getenv(model_env) or config["default_model"]

    out(f"{key_env} = {mask(key) if key else '(없음)'}")
    out(f"{model_env} = {model or '(없음)'}")

    if not key:
        out(f"{key_env}가 설정되지 않았습니다.")
        return 1
    if not model:
        out(f"{model_env}가 설정되지 않았습니다.")
        return 1
    if no_verify:
        return 0

    try:
        models = verify(provider, key, base_url_for(provider))
    except VerificationError as exc:
        out(f"키 검증 실패: {exc}")
        return 1

    out(f"키 정상 · 사용 가능 모델 {len(models)}개")
    if models and not model_is_reachable(str(model), models):
        out(f"경고: {model} 은(는) 이 키로 보이지 않습니다.")
        return 1
    return 0


def main(
    argv: list[str] | None = None,
    *,
    verify=verify_key,
    ask=ask_line,
    ask_secret=getpass,
    is_ignored=git_ignores,
    out=print,
) -> int:
    args = build_parser().parse_args(argv)
    env_path = Path(args.env)

    if args.check:
        return _run_check(env_path, verify, args.no_verify, out)

    # Checked before the key is asked for: writing a secret into a file git can
    # see is the one failure here that cannot be undone.
    if not is_ignored(env_path):
        out(f"{env_path} 이(가) .gitignore에 없습니다. 키가 커밋될 수 있어 중단합니다.")
        return 1

    try:
        provider = args.provider or _choose_provider(ask, out)
    except ValueError as exc:
        out(str(exc))
        return 1

    config = PROVIDERS[provider]
    key_env, model_env = str(config["key_env"]), str(config["model_env"])
    key = args.key or ask_secret(f"{key_env} (입력은 화면에 보이지 않습니다): ")
    if not key.strip():
        out("키가 비어 있습니다.")
        return 1
    key = key.strip()

    models: list[str] = []
    if not args.no_verify:
        try:
            models = verify(provider, key, base_url_for(provider))
        except VerificationError as exc:
            out(f"키 검증 실패: {exc}")
            return 1
        out(f"키 정상 · 사용 가능 모델 {len(models)}개")

    model = args.model or _choose_model(models, ask, out)
    if not model.strip():
        out("모델이 비어 있습니다.")
        return 1
    model = model.strip()

    if models and not model_is_reachable(model, models):
        out(f"{model} 은(는) 이 키로 보이지 않습니다. 모델 ID를 확인하세요.")
        return 1

    existing = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
    updates = {"LLM_PROVIDER": provider, key_env: key, model_env: model}
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(merge_env(existing, updates), encoding="utf-8")

    out(f"\n{env_path} 기록 완료")
    out(f"  LLM_PROVIDER = {provider}")
    out(f"  {key_env} = {mask(key)}")
    out(f"  {model_env} = {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
