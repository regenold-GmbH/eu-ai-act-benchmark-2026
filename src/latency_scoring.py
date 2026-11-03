"""Pure response-speed scoring helpers."""


MAX_SCORED_LATENCY_SECONDS = 100.0


def compute_response_speed_score(latency_seconds: float) -> float:
    """Return 100 minus latency in seconds, clipped at zero."""
    return max(0.0, MAX_SCORED_LATENCY_SECONDS - float(latency_seconds))
