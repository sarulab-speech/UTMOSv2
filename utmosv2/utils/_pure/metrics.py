from __future__ import annotations


def print_metrics(metrics: dict[str, float]):
    print(", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))