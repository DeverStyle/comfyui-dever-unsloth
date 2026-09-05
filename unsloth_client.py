"""Pure HTTP client for the Unsloth Desktop OpenAI-compatible backend.

No ComfyUI imports here — this module must stay unit-testable in isolation.
"""
from __future__ import annotations

import requests

REQUEST_TIMEOUT_S = 300.0


class UnslothError(Exception):
    """Base error for all Unsloth client failures."""


class ConnectionError_(UnslothError):
    """Server unreachable."""


class AuthError(UnslothError):
    """401 / invalid API key."""


class ModelNotLoadedError(UnslothError):
    """400 'no model loaded' style — auto-switch did not load the model."""


class ApiError(UnslothError):
    """Other non-2xx response."""

    def __init__(self, status: int, body: str):
        super().__init__(f"Unsloth API error {status}: {body}")
        self.status = status
        self.body = body


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _raise_for_status(resp, url: str) -> None:
    if resp.status_code == 401:
        raise AuthError(f"Invalid API key for {url} — check Unsloth Settings → API.")
    if resp.status_code == 400:
        body = resp.text or ""
        if "model" in body.lower() and ("load" in body.lower() or "not loaded" in body.lower()):
            raise ModelNotLoadedError(
                f"Model is not loaded and auto-switch did not load it — load it in "
                f"Unsloth Desktop, or enable Settings → \"Model auto-switch (OpenAI API)\"."
            )
    if resp.status_code >= 400:
        raise ApiError(resp.status_code, resp.text or "")


def validate_connection(url: str, api_key: str, timeout: float = 10.0) -> None:
    base = url.rstrip("/")
    try:
        resp = requests.get(base + "/v1/models", headers=_headers(api_key), timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise ConnectionError_(f"Cannot reach Unsloth at {url} — is Unsloth Desktop running? ({e})")
    _raise_for_status(resp, url)


def list_models(url: str, api_key: str, timeout: float = 15.0) -> list:
    base = url.rstrip("/")
    try:
        resp = requests.get(base + "/v1/models", headers=_headers(api_key), timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise ConnectionError_(f"Cannot reach Unsloth at {url} — is Unsloth Desktop running? ({e})")
    _raise_for_status(resp, url)
    data = resp.json().get("data", [])
    return [m.get("id") for m in data if m.get("id")]


def list_available_models(url: str, api_key: str, timeout: float = 15.0) -> list:
    """Model ids actually downloaded on disk (usable via auto-switch).

    GET /api/models/local returns {"models": [{id, partial, source, ...}]}.
    Incomplete downloads (`partial` true) are excluded. Falls back to the
    loaded models from /v1/models on error or an empty list.

    (Not /api/models/list — that returns the full catalog of models Unsloth
    can use, including ones not yet downloaded.)
    """
    base = url.rstrip("/")
    try:
        resp = requests.get(base + "/api/models/local", headers=_headers(api_key), timeout=timeout)
        _raise_for_status(resp, url)
        data = resp.json()
    except (requests.exceptions.RequestException, UnslothError, ValueError):
        return list_models(url, api_key, timeout=timeout)
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        return list_models(url, api_key, timeout=timeout)
    out = []
    for m in data["models"]:
        if not isinstance(m, dict) or m.get("partial"):
            continue  # skip incomplete downloads — not usable
        if m.get("id"):
            out.append(m["id"])
    return out or list_models(url, api_key, timeout=timeout)


def chat(url: str, api_key: str, payload: dict, timeout: float = REQUEST_TIMEOUT_S) -> dict:
    base = url.rstrip("/")
    try:
        resp = requests.post(
            base + "/v1/chat/completions",
            headers=_headers(api_key),
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.Timeout as e:
        raise UnslothError(f"Timed out after {timeout:.0f} s — the model may still be loading in Unsloth Desktop.")
    except requests.exceptions.RequestException as e:
        raise ConnectionError_(f"Cannot reach Unsloth at {url} — is Unsloth Desktop running? ({e})")
    _raise_for_status(resp, url)
    return resp.json()


def unload(url: str, api_key: str, model: str, timeout: float = 30.0) -> bool:
    """Best-effort unload: never raises; returns True only on a 2xx response.

    The caller (the keep-alive manager) logs failures when False is returned.
    """
    base = url.rstrip("/")
    try:
        resp = requests.post(
            base + "/v1/unload",
            headers=_headers(api_key),
            json={"model_path": model},
            timeout=timeout,
        )
    except requests.exceptions.RequestException:
        return False
    return 200 <= resp.status_code < 300
