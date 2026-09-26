"""Gate Synthesis Agent — analiza scores → propone thresholds pesimista/base/optimista."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class GateSynthesisAgent:
    """Sintetiza gates medibles a partir de distribución de scores."""
    
    confidence_level: float = 0.95
    n_bootstrap: int = 1000
    
    def synthesize_gates(self, 
                        scores: dict[str, list[float]],
                        business_costs: dict[str, dict] = None) -> dict:
        """
        scores: {metric_name: [values_per_item]}
        business_costs: {metric: {"fn_cost": 100, "fp_cost": 10}}
        """
        gates = {}
        for metric, values in scores.items():
            arr = np.array(values)
            boots = [np.mean(np.random.choice(arr, len(arr), replace=True)) 
                    for _ in range(self.n_bootstrap)]
            
            pessimistic = np.percentile(boots, (1-self.confidence_level)/2 * 100)
            base = np.median(boots)
            optimistic = np.percentile(boots, (1-(1-self.confidence_level)/2) * 100)
            
            # Ajuste por costos de negocio si se proveen
            if business_costs and metric in business_costs:
                # Optimización de threshold via cost-sensitive learning
                pass
            
            gates[metric] = {
                "pessimistic": float(pessimistic),
                "base": float(base),
                "optimistic": float(optimistic),
                "gate_type": "min" if "accuracy" in metric or "f1" in metric else "max"
            }
        return gates

