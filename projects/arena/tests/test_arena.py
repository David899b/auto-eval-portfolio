"""Tests for arena rating math (pure python, no dependencies)."""
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena.rating import (
    win_rate,
    bradley_terry,
    bootstrap_elo,
    ranking_with_ci,
    run_battles,
)


def _judge(prompt, a, b):
    # "b" always wins -> ranking must be [b, a]
    return "b"


def test_win_rate_ties():
    assert win_rate(3, 2, 0) == 0.6
    assert win_rate(0, 0, 0) == 0.0


def test_bt_orders_clear_winner():
    counts = {"a": {"b": 2}, "b": {"a": 0}}
    s = bradley_terry(counts)
    assert s["a"] > s["b"]


def test_bt_symmetric_is_tied():
    counts = {"a": {"b": 1}, "b": {"a": 1}}
    s = bradley_terry(counts)
    assert abs(s["a"] - s["b"]) < 0.05


def test_run_battles_deterministic():
    models = {"a": lambda p: "sa", "b": lambda p: "sb"}
    prompts = ["x", "y"]
    c1 = run_battles(prompts, models, _judge, rounds_per_prompt=2, rng=random.Random(1))
    c2 = run_battles(prompts, models, _judge, rounds_per_prompt=2, rng=random.Random(1))
    assert c1 == c2
    # b should have won every battle
    assert c1["b"]["a"] > c1["a"]["b"]


def test_bootstrap_and_ranking_shape():
    counts = {"a": {"b": 3}, "b": {"a": 1}}
    samples = bootstrap_elo(counts, n_samples=100, rng=random.Random(2))
    assert set(samples) == {"a", "b"}
    ranking = ranking_with_ci(samples, z=1.96)
    assert ranking[0]["model"] == "a"
    assert ranking[0]["elo"] >= ranking[1]["elo"]
    assert ranking[0]["ci_lo"] <= ranking[0]["elo"] <= ranking[0]["ci_hi"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")