from bernini.prompt_enhance_templates import (
    DETAILED_MIN_APPEARANCE_HAN,
    DETAILED_MIN_TOTAL_HAN,
    build_character_detail_directive,
    count_han_chars,
    get_enhance_template,
    is_character_feature_enhance_enabled,
    normalize_character_feature_enhance,
    patch_rv2v_vision_intro,
    resolve_enhance_system_prompt,
    uses_t2v_system_prompt,
)
from bernini.official_pe_templates import VR2V_TEMPLATE


def test_rv2v_zh_localizes_official_template():
    tpl = get_enhance_template("rv2v", output_language="中文")
    assert "简体中文" in tpl
    assert "image0" in tpl
    assert "reference-image-guided" in tpl or "reference image" in tpl


def test_rv2v_en_uses_official_template():
    tpl = get_enhance_template("rv2v", output_language="English")
    assert "You are an expert" in tpl
    assert "reference image" in tpl.lower()


def test_v2v_has_no_image_slot_rule():
    tpl = get_enhance_template("v2v", output_language="中文")
    assert "参考图 slot" not in tpl


def test_t2v_uses_a14b_system_prompt():
    assert uses_t2v_system_prompt("t2v")
    sys_prompt = resolve_enhance_system_prompt("t2v", output_language="English")
    assert "电影导演" in sys_prompt
    tpl = get_enhance_template("t2v", output_language="English")
    assert tpl == "{user_prompt}"


def test_character_feature_enhance_disabled():
    assert not normalize_character_feature_enhance(False)
    assert not is_character_feature_enhance_enabled(False)
    d = build_character_detail_directive(False, output_language="中文", task_key="rv2v")
    assert d == ""
    d = build_character_detail_directive("一般", output_language="中文", task_key="rv2v")
    assert d == ""
    assert build_character_detail_directive(False, output_language="中文", task_key="v2v") == ""


def test_character_feature_enhance_enabled():
    assert normalize_character_feature_enhance(True)
    assert is_character_feature_enhance_enabled("详尽")
    d = build_character_detail_directive(True, output_language="中文", task_key="rv2v")
    assert "角色特征增强" in d
    assert str(DETAILED_MIN_APPEARANCE_HAN) in d
    assert str(DETAILED_MIN_TOTAL_HAN) in d
    assert "面容" in d
    d_legacy = build_character_detail_directive("详尽", output_language="中文", task_key="rv2v")
    assert "角色特征增强" in d_legacy
    assert build_character_detail_directive(True, output_language="中文", task_key="v2v") == ""


def test_count_han_chars():
    assert count_han_chars("image1 中文") == 2


def test_patch_rv2v_vision_intro_refs_first():
    tpl = patch_rv2v_vision_intro(
        VR2V_TEMPLATE,
        source_count=2,
        ref_slots=[0],
        ref_images_first=True,
        output_language="中文",
    )
    assert "前 3 张为源视频" not in tpl or "忽略" in tpl
    assert "image0" in tpl
    assert "frame0" in tpl
    assert tpl.index("image0") < tpl.index("frame0") or "参考图 image0" in tpl
