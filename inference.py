from __future__ import annotations

import os

import requests
from openai import OpenAI


TASKS = ["easy", "medium", "hard"]


def build_openai_client(api_base_url: str, hf_token: str) -> OpenAI:
    base_url = api_base_url.rstrip("/")
    if base_url.endswith("/v1"):
        openai_base_url = base_url
    else:
        openai_base_url = f"{base_url}/v1"

    return OpenAI(
        api_key=hf_token or "dummy-token",
        base_url=openai_base_url,
    )


def normalize_environment_base_url(api_base_url: str) -> str:
    base_url = api_base_url.rstrip("/")

    blocked_hosts = (
        "router.huggingface.co",
        "api.openai.com",
    )
    if any(host in base_url for host in blocked_hosts):
        return "http://127.0.0.1:7860"

    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")

    return base_url


def ensure_environment_available(api_base_url: str) -> str:
    candidates = [
        normalize_environment_base_url(api_base_url),
        "http://127.0.0.1:7860",
        "http://localhost:7860",
    ]

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            response = requests.get(f"{candidate}/", timeout=5)
            if response.status_code == 200 and response.json() == {"status": "running"}:
                return candidate
        except (requests.RequestException, ValueError):
            continue

    raise RuntimeError(
        "OpenEnv server is not reachable. Start the server on http://127.0.0.1:7860 or set API_BASE_URL to that address."
    )


def run_task(api_base_url: str, task: str) -> float:
    reset_response = requests.post(
        f"{api_base_url}/reset",
        json={"task_id": task},
        timeout=30,
    )
    reset_response.raise_for_status()
    reset_payload = reset_response.json()

    if reset_payload != {
        "observation": "task started",
        "reward": 0.0,
        "done": False,
        "info": {},
    }:
        raise RuntimeError("Invalid /reset response")

    action = "analyze"
    print(f"[START] task={task}")
    print(f"[STEP] action={action}")

    step_response = requests.post(
        f"{api_base_url}/step",
        json={"action": {"type": action}},
        timeout=30,
    )
    step_response.raise_for_status()
    step_payload = step_response.json()

    score = float(step_payload.get("reward", 0.0))
    if step_payload.get("observation") != "task finished":
        raise RuntimeError("Invalid /step observation")
    if step_payload.get("done") is not True:
        raise RuntimeError("Invalid /step done flag")
    if not 0.0 <= score <= 1.0:
        raise RuntimeError("Score out of range")

    print(f"[END] score={score:.2f}")
    return score


def main() -> None:
    raw_api_base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:7860").rstrip("/")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    HF_TOKEN = os.getenv("HF_TOKEN")
    api_base_url = ensure_environment_available(raw_api_base_url)
    client = build_openai_client(api_base_url, hf_token)

    if not model_name:
        raise RuntimeError("MODEL_NAME must be set")
    if client.base_url is None:
        raise RuntimeError("OpenAI client initialization failed")

    for task in TASKS:
        run_task(api_base_url, task)


if __name__ == "__main__":
    main()
