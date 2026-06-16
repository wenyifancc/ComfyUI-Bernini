from bernini.prompt_enhancer import (
    API_FORMAT_OLLAMA,
    API_FORMAT_ZHIPU,
    _extract_llm_raw,
    _parse_enhanced_text,
    _prepare_zhipu_images,
    _strip_think_blocks,
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


def test_zhipu_vision_model_detection():
    assert zhipu_supports_vision("glm-4.6v-flash") is True
    assert zhipu_supports_vision("glm-4v-flash") is True
    assert zhipu_supports_vision("glm-4-flash-250414") is False


def test_zhipu_text_model_strips_images():
    imgs, err = _prepare_zhipu_images("glm-4-flash-250414", ["abc123"])
    assert imgs == []
    assert err is None
