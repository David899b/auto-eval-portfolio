"""Drift Agent — PSI/KL en embeddings + alertas + auto-retraining triggers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.stats import entropy

@dataclass
class DriftAgent:
    """Monitorea drift en producción y dispara alertas/retraining."""
    
    psi_threshold: float = 0.2
    kl_threshold: float = 0.15
    reference_window_days: int = 30
    
    def compute_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Population Stability Index."""
        exp_hist, _ = np.histogram(expected, bins=bins, density=True)
        act_hist, _ = np.histogram(actual, bins=bins, density=True)
        exp_hist = np.clip(exp_hist, 1e-6, None)
        act_hist = np.clip(act_hist, 1e-6, None)
        return float(np.sum((act_hist - exp_hist) * np.log(act_hist / exp_hist)))
    
    def compute_kl(self, p: np.ndarray, q: np.ndarray) -> float:
        """KL divergence."""
        p = np.clip(p, 1e-6, None)
        q = np.clip(q, 1e-6, None)
        return float(entropy(p, q))
    
    def check_drift(self, 
                   reference_embeddings: np.ndarray,
                   current_embeddings: np.ndarray) -> dict:
        psi = self.compute_psi(reference_embeddings, current_embeddings)
        kl = self.compute_kl(
            np.histogram(reference_embeddings, bins=20, density=True)[0],
            np.histogram(current_embeddings, bins=20, density=True)[0]
        )
        return {
            "psi": psi,
            "kl": kl,
            "psi_alert": psi > self.psi_threshold,
            "kl_alert": kl > self.kl_threshold,
            "retrain_recommended": psi > self.psi_threshold or kl > self.kl_threshold
        }

