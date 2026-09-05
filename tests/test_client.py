import json
import pytest
import unsloth_client as uc


class _FakeResp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_headers():
    h = uc._headers("sk-abc")
    assert h == {"Authorization": "Bearer sk-abc", "Content-Type": "application/json"}


def test_validate_connection_success(monkeypatch):
    monkeypatch.setattr(uc.requests, "get", lambda *a, **k: _FakeResp(200, {"data": []}))
    assert uc.validate_connection("http://x", "k") is None


def test_validate_connection_unreachable(monkeypatch):
    import requests
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("nope")
    monkeypatch.setattr(uc.requests, "get", boom)
    with pytest.raises(uc.ConnectionError_):
        uc.validate_connection("http://x", "k")


def test_validate_connection_bad_key(monkeypatch):
    monkeypatch.setattr(uc.requests, "get", lambda *a, **k: _FakeResp(401, {"error": "x"}))
    with pytest.raises(uc.AuthError):
        uc.validate_connection("http://x", "k")


def test_list_models_parses_data(monkeypatch):
    payload = {"data": [{"id": "a"}, {"id": "b"}]}
    monkeypatch.setattr(uc.requests, "get", lambda *a, **k: _FakeResp(200, payload))
    assert uc.list_models("http://x", "k") == ["a", "b"]


def test_list_available_parses_local_models(monkeypatch):
    payload = {"models": [
        {"id": "a/b", "partial": False},
        {"id": "c/d", "partial": True},   # incomplete download -> excluded
        {"id": "e/f", "partial": False},
    ]}
    monkeypatch.setattr(uc.requests, "get", lambda *a, **k: _FakeResp(200, payload))
    assert uc.list_available_models("http://x", "k") == ["a/b", "e/f"]


def test_list_available_falls_back_to_v1(monkeypatch):
    calls = []
    def fake_get(url, **k):
        calls.append(url)
        if "/api/models/local" in url:
            import requests
            raise requests.exceptions.HTTPError("403")
        return _FakeResp(200, {"data": [{"id": "only"}]})
    monkeypatch.setattr(uc.requests, "get", fake_get)
    assert uc.list_available_models("http://x", "k") == ["only"]
    assert any("/api/models/local" in c for c in calls)


def test_chat_returns_json(monkeypatch):
    payload = {"choices": [{"message": {"content": "hi"}}]}
    monkeypatch.setattr(uc.requests, "post", lambda *a, **k: _FakeResp(200, payload))
    assert uc.chat("http://x", "k", {"model": "m"}) == payload


def test_chat_model_not_loaded(monkeypatch):
    # A real 400 response's .text is the JSON body string — mirror that.
    def fake_post(url, **k):
        body = {"detail": "No model loaded. Call POST /inference/load first."}
        return _FakeResp(400, payload=body, text=json.dumps(body))
    monkeypatch.setattr(uc.requests, "post", fake_post)
    with pytest.raises(uc.ModelNotLoadedError):
        uc.chat("http://x", "k", {"model": "m"})


def test_chat_api_error_carries_status(monkeypatch):
    monkeypatch.setattr(uc.requests, "post", lambda *a, **k: _FakeResp(500, {"error": "boom"}))
    with pytest.raises(uc.ApiError) as ei:
        uc.chat("http://x", "k", {"model": "m"})
    assert ei.value.status == 500


def test_unload_posts_model_path(monkeypatch):
    seen = {}
    def fake_post(url, headers=None, json=None, timeout=None):
        seen["url"] = url
        seen["json"] = json
        return _FakeResp(200, {})
    monkeypatch.setattr(uc.requests, "post", fake_post)
    ok = uc.unload("http://x", "k", "my-model")
    assert seen["url"] == "http://x/v1/unload"
    assert seen["json"] == {"model_path": "my-model"}
    assert ok is True


def test_unload_returns_false_on_connection_error(monkeypatch):
    import requests
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("nope")
    monkeypatch.setattr(uc.requests, "post", boom)
    assert uc.unload("http://x", "k", "my-model") is False


def test_unload_returns_false_on_non_2xx(monkeypatch):
    monkeypatch.setattr(uc.requests, "post", lambda *a, **k: _FakeResp(500, {"error": "boom"}))
    assert uc.unload("http://x", "k", "my-model") is False
