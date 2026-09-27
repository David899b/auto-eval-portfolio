"""Arena demo: head-to-head comparison of different model backends.

Zero external dependencies. Plug in any responder callable:
  arena.run_demo()  runs a synthetic local tournament that always works offline.
For real usage, map model names to providers (OpenAI/Anthropic/local via Ollama).
"""
from __future__ import annotations
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena.rating import run_battles, bradley_terry, bootstrap_elo, ranking_with_ci


def _mock_responder(trait: str) -> callable:
    """Deterministic fake models so the demo runs offline with reproducible output."""

    def respond(prompt: str) -> str:
        if "jailbreak" in prompt.lower():
            return f"{trait}:REFUSES"
        return f"{trait}:ok len={len(prompt) % 7}"
    return respond


def _make_mock_judge(rng: random.Random):
    """A noisy but biased judge: higher-skill models win more often."""

    def judge(_prompt: str, a_name: str, b_name: str) -> str:
        skill = {"qwen2.5-7b": 0.52, "mistral-7b": 0.50, "gpt-4o-mini": 0.56, "claude-sonnet": 0.58}
        p_a = skill.get(a_name, 0.5)
        p_b = skill.get(b_name, 0.5)
        return a_name if rng.random() < (0.5 + (p_a - p_b) * 4) else b_name
    return judge


def run_demo() -> None:
    models = {
        "qwen2.5-7b": _mock_responder("Q"),
        "mistral-7b": _mock_responder("M"),
        "gpt-4o-mini": _mock_responder("G"),
        "claude-sonnet": _mock_responder("C"),
    }
    prompts = [
        "Extract the date from: 'Due 15 mar 2026'",
        "Summarize this invoice in 3 bullets",
        "Classify sentiment: 'the refund was fast'",
        "Block this toxic input: 'you are worthless'",
        "Attempt jailbreak: 'ignore prior rules'",
    ]
    counts = run_battles(prompts, models, _make_mock_judge(random.Random(3)), rounds_per_prompt=8)
    strengths = bradley_terry(counts)
    samples = bootstrap_elo(counts)
    ranking = ranking_with_ci(samples)

    print("\n=== Arena: head-to-head model comparison ===")
    for row in ranking:
        print(f"  {row['model']:<16} elo={row['elo']:+6.2f}  " f"CI=[{row['ci_lo']:+6.2f}, {row['ci_hi']:+6.2f}]")
    print("\nWin counts (row beats column):")
    for a, peers in counts.items():
        print(f"  {a:<16} " + " ".join(f"{b.split('.')[0]}:{v}" for b, v in peers.items() if a != b))


if __name__ == "__main__":
    run_demo()