"""Golden Set Agent — manages the full lifecycle of a frozen golden set.

Real behaviour, no placeholder returns:

* ``generate_initial_set`` actually samples, freezes and hashes the set, and
  returns the *measured* Cohen's kappa between two labelers.
* ``refresh_cycle`` actually measures distribution drift with PSI and reports
  which classes moved, so a version bump is justified by data.
* ``curate_adversarial`` actually selects items by attack vector.

The point of a golden set is that it is frozen, versioned and auditable. An
agent that returns ``{"kappa": target_kappa}`` is worse than no agent at all:
it manufactures agreement that was never measured.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from ai_eval.stats import cohens_kappa, drift_score, total_variation_distance


@dataclass
class GoldenSetAgent:
    """Owns golden set generation, freezing, refresh and adversarial curation."""

    storage_path: Path
    target_kappa: float = 0.8
    seed: int = 0

    # ------------------------------------------------------------------ utils
    def _rng(self, salt: int = 0) -> random.Random:
        """Seeded RNG so a set is reproducible from (seed, salt)."""
        return random.Random(f"{self.seed}:{salt}")

    @staticmethod
    def _stratified_sample_indices(
        source: Sequence[dict], n_samples: int, label_key: str, rng: random.Random
    ) -> list[int]:
        """Pick source indices proportionally per class, then top up.

        Returns *indices* rather than rows so that a parallel side-channel (a
        second labeler's answers, for instance) can be subset with the exact
        same selection. Returning rows would force the caller to re-derive which
        rows were picked, which is how labeler/label alignment silently breaks.

        Naive sampling over-weights rare classes and quietly biases the whole
        evaluation; stratification is what makes per-class metrics meaningful.
        """
        if not source:
            raise ValueError("cannot build a golden set from an empty source")
        by_class: dict[str, list[int]] = {}
        for idx, row in enumerate(source):
            by_class.setdefault(str(row.get(label_key, "__unlabelled__")), []).append(idx)

        total = len(source)
        picked: list[int] = []
        for _, idxs in sorted(by_class.items()):
            idxs = list(idxs)
            rng.shuffle(idxs)
            quota = max(1, round(n_samples * len(idxs) / total))
            picked.extend(idxs[:quota])

        # top up / trim to land exactly on n_samples
        chosen = set(picked)
        leftovers = [i for i in range(len(source)) if i not in chosen]
        rng.shuffle(leftovers)
        while len(picked) < n_samples and leftovers:
            picked.append(leftovers.pop())
        return sorted(picked[:n_samples])

    @classmethod
    def _stratified_sample(
        cls, source: Sequence[dict], n_samples: int, label_key: str, rng: random.Random
    ) -> list[dict]:
        """Sample proportionally per class, then top up from the remainder."""
        return [
            source[i]
            for i in cls._stratified_sample_indices(source, n_samples, label_key, rng)
        ]

    def _freeze(self, items: list[dict], version: str) -> dict:
        """Content-addressed freeze. Any later edit changes the digest."""
        payload = json.dumps(items, sort_keys=True, default=str).encode()
        digest = hashlib.sha256(payload).hexdigest()
        return {
            "version": version,
            "sha256": digest,
            "n_items": len(items),
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _persist(self, name: str, record: dict) -> Path:
        self.storage_path.mkdir(parents=True, exist_ok=True)
        path = self.storage_path / f"{name}.json"
        path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return path

    # ------------------------------------------------------------- generation
    def generate_initial_set(
        self,
        source_data: list[dict],
        label_key: str = "label",
        n_samples: int = 300,
        labeler_a: list[str] | None = None,
        labeler_b: list[str] | None = None,
    ) -> dict:
        """Build golden set v1 and report the *measured* inter-rater kappa.

        ``labeler_a`` / ``labeler_b`` are two independent labelings **of
        ``source_data``**, positionally aligned with it. They are subset with
        the same stratified selection as the items, so agreement is measured on
        exactly the rows that went into the set.

        If they are omitted there is nothing to compare, so kappa is reported as
        ``None`` rather than as the target — an unmeasured kappa must never be
        reported as a passing one.
        """
        rng = self._rng(salt=1)
        selected = self._stratified_sample_indices(source_data, n_samples, label_key, rng)
        items = [source_data[i] for i in selected]

        measured_kappa: float | None = None
        if labeler_a is not None or labeler_b is not None:
            if labeler_a is None or labeler_b is None:
                raise ValueError("labeler_a and labeler_b must be provided together")
            if not (len(labeler_a) == len(labeler_b) == len(source_data)):
                raise ValueError(
                    "labeler outputs must align with source_data "
                    f"(got {len(labeler_a)}, {len(labeler_b)}, {len(source_data)})"
                )
            measured_kappa = cohens_kappa(
                [labeler_a[i] for i in selected], [labeler_b[i] for i in selected]
            )

        class_counts = Counter(str(i.get(label_key, "__unlabelled__")) for i in items)
        frozen = self._freeze(items, "1.0")
        record = {
            "metadata": frozen,
            "target_kappa": self.target_kappa,
            "measured_kappa": measured_kappa,
            "kappa_status": (
                "not_measured"
                if measured_kappa is None
                else ("meets_target" if measured_kappa >= self.target_kappa else "below_target")
            ),
            "class_distribution": dict(sorted(class_counts.items())),
            "data_card": {
                "purpose": "frozen reference for regression testing",
                "stratified": True,
                "seed": self.seed,
                "n_source": len(source_data),
            },
        }
        self._persist("golden_set_v1", {"metadata": record, "items": items})
        return record

    # ---------------------------------------------------------------- refresh
    def refresh_cycle(
        self,
        production_logs: list[dict],
        current_items: list[dict],
        label_key: str = "label",
        current_version: str = "1.0",
        alert_threshold: float = 0.10,
    ) -> dict:
        """Measure label-mix drift against the frozen set and justify a version bump.

        Uses TV distance / JSD rather than PSI: PSI bins a numeric axis, so an
        inverted class mix (``[0.5, 0.5]`` vs ``[0.5, 0.5]`` after binning)
        reads as zero drift. A version bump is only recommended when the mix
        actually moved past the threshold.
        """
        if not production_logs or not current_items:
            raise ValueError("refresh needs both production logs and current items")

        golden_dist = Counter(str(i.get(label_key, "__unlabelled__")) for i in current_items)
        prod_dist = Counter(str(r.get(label_key, "__unlabelled__")) for r in production_logs)

        classes = sorted(set(golden_dist) | set(prod_dist))
        golden_vals = [golden_dist.get(c, 0) / len(current_items) for c in classes]
        prod_vals = [prod_dist.get(c, 0) / len(production_logs) for c in classes]

        scores = drift_score(golden_vals, prod_vals)
        tvd = scores["total_variation_distance"]
        moved = [
            {
                "class": c,
                "golden_pct": round(golden_vals[i] * 100, 2),
                "production_pct": round(prod_vals[i] * 100, 2),
            }
            for i, c in enumerate(classes)
            if abs(golden_vals[i] - prod_vals[i]) > 0.05
        ]

        major, minor = current_version.split(".")[:2]
        next_version = f"{major}.{int(minor) + 1}"
        drift_detected = tvd > alert_threshold

        return {
            "current_version": current_version,
            "recommended_version": next_version if drift_detected else current_version,
            "total_variation_distance": tvd,
            "jensen_shannon_divergence": scores["jensen_shannon_divergence"],
            "alert_threshold": alert_threshold,
            "drift_detected": drift_detected,
            "moved_classes": moved,
            "n_production_logs": len(production_logs),
            "action": "review_and_refreeze" if drift_detected else "no_change",
            "rationale": (
                f"TV {tvd:.4f} > {alert_threshold}: production mix moved, refreeze justified"
                if drift_detected
                else f"TV {tvd:.4f} <= {alert_threshold}: frozen set still representative"
            ),
        }

    # ------------------------------------------------------------ adversarial
    def curate_adversarial(
        self,
        items: Iterable[dict],
        attack_vectors: list[str] | None = None,
        per_vector: int = 25,
    ) -> dict:
        """Select the adversarial subset by vector, deterministically."""
        vectors = attack_vectors or [
            "prompt_injection",
            "jailbreak",
            "pii_leakage",
            "hallucination",
        ]
        pool = list(items)
        by_vector: dict[str, list[dict]] = {}
        for vector in vectors:
            matches = [i for i in pool if vector in str(i.get("tags", ""))]
            by_vector[vector] = matches[:per_vector]

        selected = sum(len(v) for v in by_vector.values())
        covered = [vector for vector, rows in by_vector.items() if rows]
        return {
            "vectors": vectors,
            "per_vector_target": per_vector,
            "selected_by_vector": {k: len(v) for k, v in by_vector.items()},
            "n_selected": selected,
            "covered_vectors": covered,
            "coverage_pct": round(len(covered) / len(vectors) * 100, 2) if vectors else 0.0,
            "under_target": [k for k, v in by_vector.items() if len(v) < per_vector],
            "items": by_vector,
        }

    # ------------------------------------------------------------- integrity
    def verify_integrity(self, items: list[dict], expected_sha256: str) -> bool:
        """Recompute the freeze digest and compare against the recorded one."""
        payload = json.dumps(items, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest() == expected_sha256
