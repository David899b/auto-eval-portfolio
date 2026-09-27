"""Rating math for multi-model arena: win rate + Bradley-Terry + bootstrap CI."""
from __future__ import annotations
import math
import random
from typing import Callable, Sequence


def win_rate(wins: int, losses: int, ties: int = 0) -> float:
    """Win rate counting ties as half a win."""
    n = wins + losses + ties
    if n == 0:
        return 0.0
    return (wins + 0.5 * ties) / n


def pairwise_result(
    judge: Callable[[str, str, str], str],
    prompt: str,
    model_a: str,
    model_b: str,
) -> tuple[str, str]:
    """Ask a judge which response is better. Returns (winner, loser).

    The judge returns the *name* of the winning model, which makes the result
    invariant to argument order.
    """
    winner = judge(prompt, model_a, model_b)
    loser = model_b if winner == model_a else model_a
    return winner, loser


def run_battles(
    prompts: Sequence[str],
    responders: dict[str, Callable[[str], str]],
    judge: Callable[[str, str, str], int],
    rounds_per_prompt: int = 3,
    rng: random.Random | None = None,
) -> dict[str, dict[str, int]]:
    """Run a round-robin tournament. Returns {model: {peer: wins}}."""
    rng = rng or random.Random(0)
    names = list(responders)
    counts = {m: {p: 0 for p in names} for m in names}
    for prompt in prompts:
        for _ in range(rounds_per_prompt):
            a, b = rng.sample(names, 2)
            winner, loser = pairwise_result(judge, prompt, a, b)
            counts[winner][loser] += 1
    return counts


def bradley_terry(
    counts: dict[str, dict[str, int]],
    iterations: int = 1000,
    lr: float = 0.005,
) -> dict[str, float]:
    """Maximize BT log-likelihood by gradient ascent. Returns strength per model."""
    names = list(counts)
    strengths = {m: 0.0 for m in names}
    for _ in range(iterations):
        grad = {m: 0.0 for m in names}
        for a in names:
            for b in names:
                if a == b:
                    continue
                wins = counts[a][b]
                if wins == 0:
                    continue
                p = 1.0 / (1.0 + math.exp(strengths[b] - strengths[a]))
                grad[a] += wins * (1 - p)
                grad[b] += wins * (-p)
        for m in names:
            strengths[m] += lr * grad[m]
    # Recenter so the mean strength is 0 -> elo-like scale.
    mean = sum(strengths.values()) / len(strengths)
    return {m: strengths[m] - mean for m in names}


def bootstrap_elo(
    counts: dict[str, dict[str, int]],
    n_samples: int = 500,
    rng: random.Random | None = None,
) -> dict[str, list[float]]:
    """Bootstrap resample the tournament and return BT strength distributions."""
    rng = rng or random.Random(7)
    names = list(counts)
    matches = []
    for a in names:
        for b in names:
            if a == b:
                continue
            for _ in range(counts[a][b]):
                matches.append((a, b))
    if not matches:
        return {}
    samples: dict[str, list[float]] = {m: [] for m in names}
    for _ in range(n_samples):
        boot = [rng.choice(matches) for _ in range(len(matches))]
        boot_counts = {a: {b: 0 for b in names} for a in names}
        for a, b in boot:
            boot_counts[a][b] += 1
        strengths = bradley_terry(boot_counts, iterations=200)
        for m in names:
            samples[m].append(strengths[m])
    return samples


def ranking_with_ci(
    samples: dict[str, list[float]],
    z: float = 1.96,
) -> list[dict]:
    """Order models by median strength; compute 95% CI from bootstrap samples."""
    rows = []
    for m, values in samples.items():
        vals = sorted(values)
        n = len(vals)
        med = vals[n // 2]
        lo = vals[max(0, int(n * 0.5 - z * math.sqrt(n) / 2 * 0.5))]
        hi = vals[min(n - 1, int(n * 0.5 + z * math.sqrt(n) / 2 * 0.5))]
        rows.append({"model": m, "elo": med, "ci_lo": vals[int(0.025 * n)], "ci_hi": vals[int(0.975 * n)]})
    rows.sort(key=lambda r: r["elo"], reverse=True)
    return rows