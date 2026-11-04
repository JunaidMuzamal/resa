"""Fast unit tests (no GPU, no network). Tests needing the IntelliSA hub are skipped if absent."""
from __future__ import annotations

import pytest

from resa.candidates import generate_candidates
from resa.config import load_config
from resa.context import extract_context, student_input
from resa.data.oracle import hub_dir
from resa.smells import SMELLS, normalize_category

CFG = load_config()
HAS_HUB = hub_dir(CFG).exists()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hardcoded-username", "Hard-coded secret"),
        ("hardcoded-password", "Hard-coded secret"),
        ("improper ip address binding", "Unrestricted IP Address"),
        ("use of http", "Use of HTTP without SSL/TLS"),
        ("missing default case statement", "Missing Default in Case Statement"),
        ("weak cryptography algorithms", "Use of weak cryptography algorithms"),
        ("none", None),
        ("sec_hard_secr", "Hard-coded secret"),
        ("sec_no_default_switch", "Missing Default in Case Statement"),
    ],
)
def test_normalize_category(raw, expected):
    assert normalize_category(raw) == expected


def test_canonical_smells_map_to_themselves():
    for smell in SMELLS:
        assert normalize_category(smell) == smell


def test_context_window_marks_target():
    lines = [f"line {i}" for i in range(1, 21)]
    ctx = extract_context(lines, 10, window=2)
    assert ctx.splitlines()[2].startswith(">> 10:")
    assert len(ctx.splitlines()) == 5
    assert extract_context(lines, 10, window=0) == "line 10"
    assert extract_context(lines, 1, window=3).splitlines()[0].startswith(">> 1:")


def test_student_input_conditions_on_smell():
    text = student_input("chef", "Admin by default", "recover", "x")
    assert text.startswith("stage: recover | tech: chef | smell: Admin by default")


@pytest.mark.parametrize(
    "tech,line,smell",
    [
        ("chef", 'cmd = "mysql -uroot -p#{pw} -e select"', "Admin by default"),
        ("ansible", "  url: http://example.com/file.tgz", "Use of HTTP without SSL/TLS"),
        ("ansible", "  gpgcheck: no", "No integrity check"),
        ("puppet", "  password => '',", "Empty password"),
        ("puppet", "  bind_address => '0.0.0.0',", "Unrestricted IP Address"),
        ("puppet", "# TODO: remove this hack before release", "Suspicious comment"),
        ("chef", "  checksum_type 'md5'", "Use of weak cryptography algorithms"),
    ],
)
def test_candidates_detect_relaxed_patterns(tech, line, smell):
    found = {c.smell for c in generate_candidates(tech, [line])}
    assert smell in found


def test_candidates_respect_exclusions_and_variables():
    lines = ["  password => 'hunter2',"]
    assert any(c.smell == "Hard-coded secret" for c in generate_candidates("puppet", lines))
    assert not any(c.smell == "Hard-coded secret" for c in generate_candidates("puppet", lines, {(1, "Hard-coded secret")}))
    assert not any(c.smell == "Hard-coded secret" for c in generate_candidates("puppet", ["  password => $db_password,"]))


def test_puppet_case_without_default():
    lines = ["case $os {", "  'Debian': { $x = 1 }", "  'RedHat': { $x = 2 }", "}"]
    assert any(c.smell == "Missing Default in Case Statement" for c in generate_candidates("puppet", lines))
    lines_ok = ["case $os {", "  'Debian': { $x = 1 }", "  default: { $x = 2 }", "}"]
    assert not any(c.smell == "Missing Default in Case Statement" for c in generate_candidates("puppet", lines_ok))


@pytest.mark.skipif(not HAS_HUB, reason="IntelliSA hub not fetched")
def test_published_table2_reproduces():
    from resa.eval.accuracy import check_published

    report = check_published(CFG)
    assert report and all(r["ok"] for r in report.values())
    assert abs(report["IntelliSA"]["macro_f1"] - 0.831) < 0.001


@pytest.mark.skipif(not HAS_HUB, reason="IntelliSA hub not fetched")
def test_published_table3_reproduces():
    from resa.eval.effort import check_published_effort

    report = check_published_effort(CFG)
    assert report and all(report.values())


@pytest.mark.skipif(not HAS_HUB, reason="IntelliSA hub not fetched")
def test_puppet_reconstruction_matches_intellisa_loc():
    from resa.data.oracle import expected_nonblank_loc, oracle_file_list, reconstruct_from_prompt

    total = 0
    for name in oracle_file_list(CFG, "puppet"):
        text = reconstruct_from_prompt(CFG, "puppet", name)
        assert text is not None, name
        total += sum(1 for line in text.splitlines() if line.strip())
    assert total == expected_nonblank_loc(CFG, "puppet") == 3964
