"""Decision and gate logic of the automatic retraining (no training is run here)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import daily_update as du  # noqa: E402

LIVE = {"key_fact_coverage_pct": 22.2, "citation_rate_pct": 70.8, "refusal_correct_pct": 25.0,
        "invented_rate_pct": 46.7, "false_refusal_pct": 25.0, "mean_repetition_pct": 0.5}


def test_decide_needs_enough_new_material():
    info = {"passage_blocks": 500, "fact_blocks": 50}
    assert du.decide({}, {"passage_blocks": 0, "fact_blocks": 0})[0] is False                      # nothing at all
    assert du.decide({}, info)[0] is True                                                          # first training: everything is new
    state = {"last_training": {"export": {"passage_blocks": 450, "fact_blocks": 40}}}
    train, why = du.decide(state, info)
    assert train is False and "only 60 new blocks" in why
    assert du.decide(state, info, min_new=50)[0] is True and du.decide(state, info, force=True)[0] is True
    assert du.decide({}, {"passage_blocks": 0, "fact_blocks": 0}, force=True)[0] is False           # even forced: no data, no training


def test_gate_passes_an_equal_or_better_candidate():
    better = {**LIVE, "key_fact_coverage_pct": 30.0, "invented_rate_pct": 35.0, "refusal_correct_pct": 40.0}
    assert du.gate(LIVE, better) == (True, [])
    assert du.gate(LIVE, LIVE)[0] is True


def test_gate_tolerates_small_noise_but_blocks_real_regressions():
    noisy = {**LIVE, "key_fact_coverage_pct": 21.5, "invented_rate_pct": 48.0}                      # within the slack
    assert du.gate(LIVE, noisy)[0] is True
    worse = {**LIVE, "key_fact_coverage_pct": 15.0, "invented_rate_pct": 60.0}
    ok, reasons = du.gate(LIVE, worse)
    assert ok is False and len(reasons) == 2 and any("key_fact_coverage_pct" in r for r in reasons) and any("invented_rate_pct" in r for r in reasons)


def test_gate_covers_every_measure_the_evaluation_reports():
    import json
    summary = list(json.loads((Path(__file__).resolve().parent.parent / "eval" / "results" / "rag_eval_final.json").read_text(encoding="utf-8")).values())[1]["summary"]
    assert set(du.GATE) <= set(summary)                                                              # every gated measure exists in real results
