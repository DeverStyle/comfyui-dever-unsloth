import unsloth_request as ur


def test_filter_enabled_options_none():
    assert ur.filter_enabled_options(None) is None


def test_filter_enabled_options_empty():
    assert ur.filter_enabled_options({"enable_temperature": False}) is None


def test_filter_enabled_options_strips_prefix():
    opts = {"enable_temperature": True, "temperature": 0.5, "enable_top_k": False, "top_k": 40}
    assert ur.filter_enabled_options(opts) == {"temperature": 0.5}


def test_build_payload_text_no_images():
    p = ur.build_chat_payload("m", "sys", "hello", None, None, False, "text")
    assert p["model"] == "m"
    assert p["messages"][0] == {"role": "system", "content": "sys"}
    assert p["messages"][1] == {"role": "user", "content": "hello"}
    assert "enable_thinking" not in p
    assert "response_format" not in p


def test_build_payload_json_format():
    p = ur.build_chat_payload("m", "s", "p", None, None, False, "json")
    assert p["response_format"] == {"type": "json_object"}


def test_build_payload_think():
    p = ur.build_chat_payload("m", "s", "p", None, None, True, "text")
    assert p["enable_thinking"] is True


def test_build_payload_with_options():
    opts = {"enable_temperature": True, "temperature": 0.2, "enable_max_tokens": True, "max_tokens": 64}
    p = ur.build_chat_payload("m", "s", "p", None, opts, False, "text")
    assert p["temperature"] == 0.2
    assert p["max_tokens"] == 64


def test_build_payload_stop_as_list():
    opts = {"enable_stop": True, "stop": "END"}
    p = ur.build_chat_payload("m", "s", "p", None, opts, False, "text")
    assert p["stop"] == ["END"]


def test_build_payload_vision_multimodal():
    p = ur.build_chat_payload("m", "s", "describe", ["AAA", "BBB"], None, False, "text")
    user = p["messages"][1]["content"]
    assert user[0] == {"type": "text", "text": "describe"}
    assert user[1]["type"] == "image_url"
    assert user[1]["image_url"]["url"] == "data:image/png;base64,AAA"
    assert user[2]["image_url"]["url"] == "data:image/png;base64,BBB"


def test_parse_response_text():
    resp = {"choices": [{"message": {"content": "hi", "reasoning_content": "thought"}}]}
    result, thinking = ur.parse_chat_response(resp, think=False)
    assert result == "hi"
    assert thinking == ""


def test_parse_response_think():
    resp = {"choices": [{"message": {"content": "hi", "reasoning_content": "thought"}}]}
    result, thinking = ur.parse_chat_response(resp, think=True)
    assert result == "hi"
    assert thinking == "thought"


def test_parse_response_missing_reasoning():
    resp = {"choices": [{"message": {"content": "hi"}}]}
    result, thinking = ur.parse_chat_response(resp, think=True)
    assert result == "hi"
    assert thinking == ""
