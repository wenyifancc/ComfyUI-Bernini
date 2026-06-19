from bernini.prompt_enhancer import (
    API_FORMAT_OLLAMA,
    API_FORMAT_OPENAI_COMPAT,
    API_FORMAT_ZHIPU,
    OPENAI_COMPAT_MODE_LLAMA_SWAP,
    OPENAI_COMPAT_MODE_STANDARD,
    _LEGACY_OPENAI_FORMAT,
    _extract_llm_raw,
    _parse_enhanced_text,
    _prepare_zhipu_images,
    _strip_think_blocks,
    infer_api_format,
    llama_swap_unload_endpoint,
    llm_chat_endpoint,
    llm_headers,
    llm_models_endpoint,
    normalize_openai_compat_mode,
    openai_compat_root,
    zhipu_supports_vision,
)


def test_strip_think_blocks():
    think_open = "<" + "think>"
    think_close = "</" + "think>"
    raw = think_open + "reasoning" + think_close + "Final prompt"
    assert _strip_think_blocks(raw) == "Final prompt"


def test_parse_json_rewritten():
    assert _parse_enhanced_text('{"rewritten_text": "hello"}') == "hello"
    assert _parse_enhanced_text("```json\n{\"rewritten_text\": \"x\"}\n```") == "x"


def test_parse_long_json_rewritten_chinese():
    inner = (
        "将视频中身穿深绿长袍的男子替换为 image1 中的女子。该女子乌黑长发及腰、发间佩戴精致金色步摇；"
        "身着浅青色交领广袖汉服、淡绿内衬；腰束金色镂空雕花腰带；颈戴双层金项圈。"
        "保留原视频室内古风背景与镜头构图不变。"
    )
    raw = '{"rewritten_text": "' + inner + '"}'
    assert _parse_enhanced_text(raw) == inner


def test_parse_json_with_unescaped_newlines():
    inner = "将视频中男子替换为 image1 中的女子。\n该女子黑色长发。"
    raw = '{"rewritten_text": "' + inner + '"}'
    assert "image1" in _parse_enhanced_text(raw)
    assert "rewritten_text" not in _parse_enhanced_text(raw)


def test_parse_plain_then_json_suffix():
    plain = "将视频中男子替换为 image1 中的女子，保留背景不变。"
    raw = plain + '\n\n{"rewritten_text": "另一段扩写内容"}'
    assert _parse_enhanced_text(raw) == plain


def test_parse_never_leaves_json_wrapper():
    inner = "将视频中身穿深绿长袍的男子替换为 image1 中的女子，保留背景不变。"
    raw = '{"rewritten_text": "' + inner + '"}'
    out = _parse_enhanced_text(raw)
    assert out == inner
    assert not out.startswith("{")
    assert "rewritten_text" not in out


def test_extract_ollama_thinking_fallback():
    ollama = {"message": {"content": "", "thinking": "thought only"}}
    assert _extract_llm_raw(ollama, API_FORMAT_OLLAMA) == "thought only"


def test_extract_zhipu_reasoning():
    zhipu = {"choices": [{"message": {"content": "", "reasoning_content": "answer"}}]}
    assert _extract_llm_raw(zhipu, API_FORMAT_ZHIPU) == "answer"


def test_extract_openai_compatible_content():
    result = {"choices": [{"message": {"content": "enhanced prompt"}}]}
    assert _extract_llm_raw(result, API_FORMAT_OPENAI_COMPAT) == "enhanced prompt"


def test_openai_compatible_format_and_endpoints():
    assert infer_api_format("http://127.0.0.1:8080/v1", API_FORMAT_OPENAI_COMPAT) == API_FORMAT_OPENAI_COMPAT
    assert infer_api_format("http://127.0.0.1:8080/v1", _LEGACY_OPENAI_FORMAT) == API_FORMAT_OPENAI_COMPAT
    assert llm_chat_endpoint("http://127.0.0.1:8080", API_FORMAT_OPENAI_COMPAT) == "http://127.0.0.1:8080/v1/chat/completions"
    assert llm_chat_endpoint("http://127.0.0.1:8080/v1", API_FORMAT_OPENAI_COMPAT) == "http://127.0.0.1:8080/v1/chat/completions"
    assert llm_models_endpoint("http://127.0.0.1:8080/v1", API_FORMAT_OPENAI_COMPAT) == "http://127.0.0.1:8080/v1/models"


def test_llama_swap_unload_endpoint_uses_root_and_escaped_model():
    assert openai_compat_root("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080"
    assert llama_swap_unload_endpoint("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080/api/models/unload"
    assert (
        llama_swap_unload_endpoint("http://127.0.0.1:8080/v1", "author/model q4")
        == "http://127.0.0.1:8080/api/models/unload/author%2Fmodel%20q4"
    )


def test_openai_compat_mode_and_optional_auth_header():
    assert normalize_openai_compat_mode("llama-swap") == OPENAI_COMPAT_MODE_LLAMA_SWAP
    assert normalize_openai_compat_mode("anything") == OPENAI_COMPAT_MODE_STANDARD
    assert "Authorization" not in llm_headers(API_FORMAT_OPENAI_COMPAT, include_json=False)
    assert llm_headers(API_FORMAT_OPENAI_COMPAT, include_json=False, api_key="abc")["Authorization"] == "Bearer abc"


def test_zhipu_vision_model_detection():
    assert zhipu_supports_vision("glm-4.6v-flash") is True
    assert zhipu_supports_vision("glm-4v-flash") is True
    assert zhipu_supports_vision("glm-4-flash-250414") is False


def test_zhipu_text_model_strips_images():
    imgs, err = _prepare_zhipu_images("glm-4-flash-250414", ["abc123"])
    assert imgs == []
    assert err is None
