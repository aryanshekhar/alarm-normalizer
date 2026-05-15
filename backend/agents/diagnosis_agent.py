class DiagnosisAgent:
    """Performs root-cause analysis on anomaly events."""

    def diagnose(self, event: dict) -> dict:
        raise NotImplementedError
