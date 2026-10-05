from llm_abm_ga.prompts.components import (
    MANDATORY_SYSTEM_LINE,
    MANDATORY_USER_LINE,
    system_template_phrases,
    user_template_phrases,
)
from llm_abm_ga.prompts.templates import decode_individual

N_SYSTEM = len(system_template_phrases)


def test_component_library_has_the_documented_size():
    assert N_SYSTEM == 8
    assert len(user_template_phrases) == 1


def test_all_zero_genome_keeps_only_the_mandatory_lines():
    out = decode_individual(([0] * N_SYSTEM, [0]))
    assert out["system_template"].strip() == MANDATORY_SYSTEM_LINE.strip()
    assert out["user_template"].strip() == MANDATORY_USER_LINE.strip()


def test_selected_components_are_included_in_library_order():
    bits = [1, 0, 0, 1, 1, 1, 0, 0]  # the genome evolved for gpt-4o-mini
    system = decode_individual((bits, [0]))["system_template"]
    for bit, phrase in zip(bits, system_template_phrases):
        assert (phrase in system) == bool(bit)
    positions = [system.index(p) for b, p in zip(bits, system_template_phrases) if b]
    assert positions == sorted(positions)


def test_mandatory_lines_are_always_appended():
    for bits in ([0] * N_SYSTEM, [1] * N_SYSTEM):
        out = decode_individual((bits, [0]))
        assert MANDATORY_SYSTEM_LINE in out["system_template"]
        assert MANDATORY_USER_LINE in out["user_template"]


def test_placeholders_are_filled_by_the_firm():
    system = decode_individual(([1] * N_SYSTEM, [0]))["system_template"]
    filled = system.format(n_competitors=9, n_buyers=1000, cost_index=0.75)
    assert "composed by 9 other firms and 1000 buyers" in filled
    assert "0.75 times the quality level" in filled
