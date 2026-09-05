import base64
import sys
import types
import asyncio

# ---- Stub ComfyUI `server` and `aiohttp` so UnslothNodes imports in isolation ----
server_stub = types.ModuleType("server")
class _Routes:
    def post(self, path):
        def deco(fn):
            return fn
        return deco
class _PromptServer:
    instance = None
    routes = _Routes()
_PromptServer.instance = _PromptServer()
server_stub.PromptServer = _PromptServer
sys.modules["server"] = server_stub

aiohttp_stub = types.ModuleType("aiohttp")
aiohttp_web = types.ModuleType("aiohttp.web")
def _json_response(data, status=200):
    class R:
        pass
    r = R()
    r.status = status
    r.body = data
    return r
aiohttp_web.json_response = _json_response
aiohttp_stub.web = aiohttp_web
sys.modules["aiohttp"] = aiohttp_stub
sys.modules["aiohttp.web"] = aiohttp_web

import numpy as np
import UnslothNodes
import unsloth_client

# ---------------------------------------------------------------------------

def _conn(url="http://x", model="m", key="k", keep_alive=5, unit="minutes"):
    return {"url": url, "model": model, "api_key": key,
            "keep_alive": keep_alive, "keep_alive_unit": unit}


def test_resolve_meta_none_raises():
    import pytest
    with pytest.raises(Exception):
        UnslothNodes._resolve_meta(None, None, None)


def test_resolve_meta_from_connectivity():
    c = _conn()
    meta = UnslothNodes._resolve_meta(c, {"a": 1}, None)
    assert meta["connectivity"] is c
    assert meta["options"] == {"a": 1}


def test_resolve_meta_uses_meta_connectivity():
    c = _conn()
    meta_in = {"connectivity": c, "options": {"old": True}}
    meta = UnslothNodes._resolve_meta(None, None, meta_in)
    assert meta["connectivity"] is c
    assert meta["options"] == {"old": True}


def test_resolve_meta_connectivity_overrides():
    direct = _conn(url="http://direct")
    meta_in = {"connectivity": _conn(url="http://meta"), "options": None}
    meta = UnslothNodes._resolve_meta(direct, None, meta_in)
    assert meta["connectivity"]["url"] == "http://direct"


def test_resolve_meta_options_overrides():
    meta_in = {"connectivity": _conn(), "options": {"old": True}}
    meta = UnslothNodes._resolve_meta(None, {"new": True}, meta_in)
    assert meta["options"] == {"new": True}


def test_images_to_b64():
    class FakeTensor:
        def __init__(self, arr):
            self._arr = arr
        def cpu(self):
            return self
        def numpy(self):
            return self._arr
    images = [FakeTensor(np.zeros((4, 4, 3), dtype=np.float32))]
    out = UnslothNodes.images_to_b64(images)
    assert len(out) == 1
    raw = base64.b64decode(out[0])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_endpoint_returns_models(monkeypatch):
    monkeypatch.setattr(unsloth_client, "list_available_models", lambda url, key: ["a", "b"])
    class FakeReq:
        async def json(self):
            return {"url": "http://x", "api_key": "k"}
    resp = asyncio.get_event_loop().run_until_complete(
        UnslothNodes.get_models_endpoint(FakeReq())
    )
    assert resp.status == 200
    assert resp.body == ["a", "b"]


def test_endpoint_error_returns_502(monkeypatch):
    def boom(url, key):
        raise RuntimeError("cannot reach")
    monkeypatch.setattr(unsloth_client, "list_available_models", boom)
    class FakeReq:
        async def json(self):
            return {"url": "http://x", "api_key": "k"}
    resp = asyncio.get_event_loop().run_until_complete(
        UnslothNodes.get_models_endpoint(FakeReq())
    )
    assert resp.status == 502
    assert "cannot reach" in resp.body["error"]


def test_node_classes_registered():
    assert set(UnslothNodes.NODE_CLASS_MAPPINGS) == {
        "UnslothConnectivity", "UnslothOptions", "UnslothGenerate"}
    assert UnslothNodes.NODE_CLASS_MAPPINGS["UnslothGenerate"].RETURN_TYPES == (
        "STRING", "STRING", "UNSLOTH_META")
    assert UnslothNodes.NODE_CLASS_MAPPINGS["UnslothConnectivity"].CATEGORY == "Unsloth"
