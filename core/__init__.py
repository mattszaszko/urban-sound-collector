"""Core audio processing and classification modules."""

__all__ = ["YamnetTFLiteClassifier", "LoudnessEngine"]


def __getattr__(name: str):
    if name == "YamnetTFLiteClassifier":
        from .classifier_tflite import YamnetTFLiteClassifier

        return YamnetTFLiteClassifier
    if name == "LoudnessEngine":
        from .loudness import LoudnessEngine

        return LoudnessEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
