"""Ollama-style keep-alive, emulated client-side for Unsloth.

Unsloth's OpenAI API has no per-request keep_alive. We emulate it with a
per-URL background heartbeat that pings the active model and calls
POST /v1/unload when the idle grace period expires.
"""
from __future__ import annotations

import threading
import time

import unsloth_client


def _grace_seconds(keep_alive: int, unit: str) -> float:
    mult = 3600 if unit == "hours" else 60
    return keep_alive * mult


def _default_pinger(url: str, model: str, api_key: str) -> None:
    unsloth_client.chat(
        url,
        api_key,
        {"model": model, "messages": [{"role": "user", "content": "."}], "max_tokens": 1},
        timeout=30.0,
    )


def _default_unloader(url: str, model: str, api_key: str) -> None:
    ok = unsloth_client.unload(url, api_key, model)
    if not ok:
        print(f"[unsloth] keep-alive: failed to unload '{model}' at {url} — model may still be loaded (best-effort).")


class KeepAliveManager:
    def __init__(self, pinger=None, unloader=None, clock=None, tick_interval=30.0, timer_factory=None):
        self._entries = {}
        self._lock = threading.Lock()
        self._pinger = pinger or _default_pinger
        self._unloader = unloader or _default_unloader
        self._clock = clock or time.time
        self._tick_interval = tick_interval
        self._timer_factory = timer_factory or threading.Timer

    def on_generation(self, url, model, keep_alive, unit, api_key):
        key = url.rstrip("/")
        if keep_alive == 0:
            # Unload immediately after the response (Ollama keep_alive=0).
            self._unloader(key, model, api_key)
            with self._lock:
                self._drop(key)
            return
        with self._lock:
            self._drop(key)  # cancel any existing heartbeat for this url
            entry = {
                "model": model,
                "api_key": api_key,
                "keep_alive": keep_alive,
                "unit": unit,
                "last_generation_ts": self._clock(),
                "timer": None,
            }
            self._entries[key] = entry
            timer = self._timer_factory(self._tick_interval, self._tick, args=(key,))
            timer.daemon = True
            entry["timer"] = timer
            timer.start()

    def _drop(self, key):
        """Caller must hold self._lock."""
        entry = self._entries.pop(key, None)
        if entry and entry["timer"] is not None:
            entry["timer"].cancel()

    def _reschedule(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            if entry["timer"] is not None:
                entry["timer"].cancel()
            timer = self._timer_factory(self._tick_interval, self._tick, args=(key,))
            timer.daemon = True
            entry["timer"] = timer
            timer.start()

    def _tick(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            model = entry["model"]
            api_key = entry["api_key"]
            keep_alive = entry["keep_alive"]
            unit = entry["unit"]
            last_ts = entry["last_generation_ts"]
        # Idle past the grace period (and not indefinite) -> unload and stop.
        if keep_alive != -1 and self._clock() - last_ts >= _grace_seconds(keep_alive, unit):
            self._unloader(key, model, api_key)
            with self._lock:
                self._drop(key)
            return
        # Keep the model warm. A failed ping means the server/model is gone.
        try:
            self._pinger(key, model, api_key)
        except Exception:
            with self._lock:
                self._drop(key)
            return
        self._reschedule(key)

    def stop(self, url):
        key = url.rstrip("/")
        with self._lock:
            self._drop(key)


# Shared singleton used by the nodes.
manager = KeepAliveManager()
