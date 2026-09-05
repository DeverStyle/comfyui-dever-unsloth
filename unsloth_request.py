"""Pure builders for the Unsloth chat-completions request/response.

No I/O, no ComfyUI imports — unit-testable in isolation.
"""
from __future__ import annotations

OPTION_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "seed",
    "stop",
    "max_tokens",
)


def filter_enabled_options(options: dict | None) -> dict | None:
    """Return only the option params whose `enable_<field>` flag is True."""
    if not options:
        return None
    out = {}
    for field in OPTION_FIELDS:
        if options.get(f"enable_{field}", False):
            value = options[field]
            if field == "stop" and isinstance(value, str):
                value = [value]
            out[field] = value
    return out or None


def build_chat_payload(
    model: str,
    system: str,
    prompt: str,
    images_b64: list | None,
    options: dict | None,
    think: bool,
    fmt: str,
) -> dict:
    """Build the full /v1/chat/completions JSON body."""
    if images_b64:
        user_content = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            user_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )
    else:
        user_content = prompt

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    }

    enabled = filter_enabled_options(options)
    if enabled:
        payload.update(enabled)

    if think:
        payload["enable_thinking"] = True

    if fmt == "json":
        payload["response_format"] = {"type": "json_object"}

    return payload


def parse_chat_response(response: dict, think: bool) -> tuple:
    """Extract (result, thinking) from a chat-completions response."""
    message = response["choices"][0]["message"]
    result = message.get("content") or ""
    thinking = (message.get("reasoning_content") or "") if think else ""
    return result, thinking
