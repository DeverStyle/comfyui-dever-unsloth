"""ComfyUI node classes + server endpoint for comfyui-dever-unsloth.

This is the only module that imports ComfyUI's `server` / `aiohttp` / `PIL` /
`numpy`. All real logic lives in the pure modules (unsloth_client,
unsloth_request, unsloth_keep_alive) so it stays unit-testable.
"""
from __future__ import annotations

import base64
import io
import random
from pprint import pprint

import numpy as np
from PIL import Image

from server import PromptServer
from aiohttp import web

import unsloth_client
import unsloth_request
import unsloth_keep_alive


def _resolve_meta(connectivity, options, meta):
    """Merge direct connectivity/options inputs over a chained `meta`.

    Mirrors OllamaGenerateV2's override rules: a directly-connected input
    wins over whatever is in `meta`. Raises if no connectivity remains.
    """
    if meta is not None:
        if connectivity is not None:
            meta["connectivity"] = connectivity
        if options is not None:
            meta["options"] = options
    else:
        meta = {"options": options, "connectivity": connectivity}

    if meta.get("connectivity") is None:
        raise Exception("Required input connectivity or connectivity in meta.")
    return meta


def images_to_b64(images):
    """Convert a ComfyUI IMAGE batch (list of float tensors) to base64 PNG strings."""
    out = []
    for image in images:
        arr = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        out.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    return out


@PromptServer.instance.routes.post("/unsloth/get_models")
async def get_models_endpoint(request):
    data = await request.json()
    url = data.get("url")
    api_key = data.get("api_key", "")
    try:
        models = unsloth_client.list_available_models(url, api_key)
        return web.json_response(models)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=502)


class UnslothConnectivity:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:8888",
                    "tooltip": "Base URL of the Unsloth Desktop server. Default is the local instance on its default port.",
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Bearer API key from Unsloth Desktop (Settings -> API). Required — the OpenAI API rejects requests without it.",
                }),
                "model": ((), {
                    "tooltip": "Select a model. Populated by the Reconnect button from the models available in Unsloth Desktop. Any listed model can be requested — Unsloth auto-switches to it if it is not already loaded.",
                }),
                "keep_alive": ("INT", {
                    "default": 5, "min": -1, "max": 120, "step": 1,
                    "tooltip": "How long to keep the model loaded after the last generation. -1 = keep loaded indefinitely, 0 = unload immediately after the response, N = unload N (unit) after the last generation.",
                }),
                "keep_alive_unit": (["minutes", "hours"],),
            },
        }

    RETURN_TYPES = ("UNSLOTH_CONNECTIVITY",)
    RETURN_NAMES = ("connection",)
    FUNCTION = "connect"
    CATEGORY = "Unsloth"
    DESCRIPTION = "Connection to an Unsloth Desktop (OpenAI-compatible) server: URL, API key, model, and keep-alive behavior. Use the Reconnect button to load the model list."

    def connect(self, url, api_key, model, keep_alive, keep_alive_unit):
        return ({
            "url": url,
            "api_key": api_key,
            "model": model,
            "keep_alive": keep_alive,
            "keep_alive_unit": keep_alive_unit,
        },)


class UnslothOptions:
    @classmethod
    def INPUT_TYPES(cls):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "enable_temperature": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 2, "step": 0.05, "tooltip": "Higher = more creative/less focused."}),
                "enable_top_p": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0, "max": 1, "step": 0.05, "tooltip": "Nucleus sampling. Lower = more focused."}),
                "enable_top_k": ("BOOLEAN", {"default": False}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1, "tooltip": "Limit sampling to the top-k most likely tokens."}),
                "enable_min_p": ("BOOLEAN", {"default": False}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05, "tooltip": "Minimum probability relative to the most likely token."}),
                "enable_repeat_penalty": ("BOOLEAN", {"default": False}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0, "max": 2, "step": 0.05, "tooltip": "Penalize repeated tokens. >1 discourages repetition."}),
                "enable_presence_penalty": ("BOOLEAN", {"default": False}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": 0, "max": 2, "step": 0.05, "tooltip": "Penalize tokens already present in the output."}),
                "enable_seed": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1, "tooltip": "Deterministic sampling seed (best-effort)."}),
                "enable_stop": ("BOOLEAN", {"default": False}),
                "stop": ("STRING", {"default": "", "multiline": False, "tooltip": "Stop sequence. Generation halts when this text appears."}),
                "enable_max_tokens": ("BOOLEAN", {"default": False}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1, "tooltip": "Maximum tokens to generate."}),
                "debug": ("BOOLEAN", {"default": False, "tooltip": "Node-side debug printing only — no effect on the API."}),
            },
        }

    RETURN_TYPES = ("UNSLOTH_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "options"
    CATEGORY = "Unsloth"
    DESCRIPTION = "Advanced sampling options for the Unsloth backend. Only enabled (toggled) values are sent in the request."

    def options(self, **kargs):
        if kargs.get("debug"):
            print("--- unsloth options dump ---")
            pprint(kargs)
            print("--------------------------------")
        return (kargs,)


class UnslothGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an AI artist.",
                    "tooltip": "System prompt — sets the role and general behavior of the model.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is art?",
                    "tooltip": "User prompt. For vision tasks you can refer to the input image as 'this image'.",
                }),
                "think": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If enabled, the model does a thinking pass first. The thinking is returned as a separate output. Some models do not support this.",
                }),
                "format": (["text", "json"], {
                    "tooltip": "'json' requests a JSON object response (response_format). Show the model example outputs in the system prompt for best results.",
                }),
            },
            "optional": {
                "connectivity": ("UNSLOTH_CONNECTIVITY", {
                    "forceInput": False,
                    "tooltip": "Connection to set the server, key, model, and keep-alive. Required unless provided via 'meta'.",
                }),
                "options": ("UNSLOTH_OPTIONS", {
                    "forceInput": False,
                    "tooltip": "Advanced sampling options.",
                }),
                "images": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Optional image or batch for vision tasks. The model must support vision.",
                }),
                "meta": ("UNSLOTH_META", {
                    "forceInput": False,
                    "tooltip": "Chain multiple Generate nodes; connectivity and options are passed along (and can be overridden).",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "UNSLOTH_META")
    RETURN_NAMES = ("result", "thinking", "meta")
    FUNCTION = "generate"
    CATEGORY = "Unsloth"
    DESCRIPTION = "Text (and vision) generation via an Unsloth Desktop OpenAI-compatible backend. Connect an Unsloth Connectivity node to set the server, key, and model."

    def generate(self, system, prompt, think, format,
                 connectivity=None, options=None, images=None, meta=None):
        meta = _resolve_meta(connectivity, options, meta)
        conn = meta["connectivity"]
        url = conn["url"]
        model = conn["model"]
        api_key = conn.get("api_key", "")
        keep_alive = conn["keep_alive"]
        keep_alive_unit = conn["keep_alive_unit"]

        debug = bool(options and options.get("debug"))
        if debug:
            print(f"--- unsloth generate request\nurl: {url}\nmodel: {model}\n"
                  f"system: {system}\nprompt: {prompt}\nimages: "
                  f"{0 if images is None else len(images)}\nthink: {think}\n"
                  f"keep_alive: {keep_alive} {keep_alive_unit}\nformat: {format}\n"
                  f"options: {unsloth_request.filter_enabled_options(options)}\n"
                  f"---------------------------------------------------------")

        unsloth_client.validate_connection(url, api_key)

        images_b64 = images_to_b64(images) if images is not None else None
        payload = unsloth_request.build_chat_payload(
            model, system, prompt, images_b64, options, think, format)

        if debug:
            print("--- unsloth generate payload ---")
            pprint(payload)
            print("--------------------------------")

        response = unsloth_client.chat(url, api_key, payload)
        result, thinking = unsloth_request.parse_chat_response(response, think)

        if debug:
            print("--- unsloth generate response ---")
            pprint(response)
            print("--------------------------------")

        # Emulate Ollama keep-alive client-side (ping + idle-grace unload).
        unsloth_keep_alive.manager.on_generation(url, model, keep_alive, keep_alive_unit, api_key)

        return result, thinking, meta


NODE_CLASS_MAPPINGS = {
    "UnslothConnectivity": UnslothConnectivity,
    "UnslothOptions": UnslothOptions,
    "UnslothGenerate": UnslothGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnslothConnectivity": "Unsloth Connectivity",
    "UnslothOptions": "Unsloth Options",
    "UnslothGenerate": "Unsloth Generate",
}
