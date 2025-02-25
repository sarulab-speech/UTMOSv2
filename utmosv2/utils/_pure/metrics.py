from __future__ import annotations


def print_metrics(metrics: dict[str, float]) -> None:
    """
    Print the given metrics in a formatted string.

    Args:
        metrics (dict[str, float]):
            A dictionary of metric names and their corresponding values.

    Returns:
        None: This function prints the metrics to the console in the format "metric_name: value".
    """
    print(", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
