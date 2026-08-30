"""YAMNet sound-event classification via TensorFlow Lite."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from core.audio_constants import YAMNET_CHUNK_SAMPLES

DEFAULT_TOP_K = 3
MODEL_NAME = "yamnet"
MODEL_VERSION = "tflite/1"

# Repo-root relative defaults.
_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = _ROOT / "models" / "yamnet.tflite"
DEFAULT_CLASS_MAP_PATH = _ROOT / "models" / "yamnet_class_map.csv"


def _load_interpreter(model_path: Path):
    """Load a TFLite Interpreter (Pi: tflite-runtime or ai-edge-litert)."""
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter  # type: ignore
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "No TFLite interpreter found. On Raspberry Pi install "
                    "tflite-runtime (Python 3.11) or ai-edge-litert (3.12+): "
                    "pip install ai-edge-litert"
                ) from exc
    return Interpreter


def load_class_names(class_map_path: Path) -> List[str]:
    """Load YAMNet display names from the AudioSet class-map CSV."""
    names: List[str] = []
    with class_map_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        # Expected columns: index, mid, display_name
        for row in reader:
            name = (row.get("display_name") or "").strip().strip('"')
            if name:
                names.append(name)
    if not names:
        raise RuntimeError(f"Class map empty or unreadable: {class_map_path}")
    return names


class YamnetTFLiteClassifier:
    """Run YAMNet TFLite inference on fixed-length 16 kHz mono windows."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        class_map_path: Path | str = DEFAULT_CLASS_MAP_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_map_path = Path(class_map_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"YAMNet TFLite model not found: {self.model_path}")
        if not self.class_map_path.is_file():
            raise FileNotFoundError(f"YAMNet class map not found: {self.class_map_path}")

        Interpreter = _load_interpreter(self.model_path)
        self.interpreter = Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()

        self._input = self.interpreter.get_input_details()[0]
        self._output = self.interpreter.get_output_details()[0]
        self.class_names = load_class_names(self.class_map_path)

        expected = int(np.prod(self._input["shape"]))
        if expected != YAMNET_CHUNK_SAMPLES:
            raise RuntimeError(
                f"TFLite input length is {expected}, expected {YAMNET_CHUNK_SAMPLES}."
            )
        if len(self.class_names) < int(self._output["shape"][-1]):
            raise RuntimeError(
                "Class map has fewer labels than model output classes."
            )

    def predict(
        self,
        waveform_16k: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict[str, Any]]:
        """Classify one mono float32 window and return top-k predictions."""
        wave = np.asarray(waveform_16k, dtype=np.float32).reshape(-1)
        if wave.size != YAMNET_CHUNK_SAMPLES:
            raise ValueError(
                f"Expected {YAMNET_CHUNK_SAMPLES} samples, got {wave.size}"
            )

        input_index = self._input["index"]
        # Some builds want shape (15600,), others (1, 15600).
        in_shape = tuple(int(x) for x in self._input["shape"])
        if len(in_shape) == 1:
            self.interpreter.set_tensor(input_index, wave)
        else:
            self.interpreter.set_tensor(input_index, wave.reshape(in_shape))

        self.interpreter.invoke()
        scores = np.asarray(
            self.interpreter.get_tensor(self._output["index"]),
            dtype=np.float32,
        ).reshape(-1)

        k = min(top_k, scores.size, len(self.class_names))
        top_indices = np.argsort(scores)[::-1][:k]

        predictions: List[Dict[str, Any]] = []
        for index in top_indices:
            predictions.append(
                {
                    "label": self.class_names[int(index)],
                    "confidence": float(scores[int(index)]),
                }
            )
        return predictions
