class MonitorAgent:
    """Watches live KPI streams and emits anomaly events."""

    def run(self) -> None:
        raise NotImplementedError
