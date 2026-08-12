"""Core audio processing and classification modules."""

__all__ = ["YAMNetClassifier", "calculate_rms"]


def __getattr__(name: str):
    """Lazy-load submodules so lightweight imports avoid TensorFlow."""
    if name == "YAMNetClassifier":
        from .classifier import YAMNetClassifier

        return YAMNetClassifier
    if name == "calculate_rms":
        from .metrics import calculate_rms

        return calculate_rms
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
