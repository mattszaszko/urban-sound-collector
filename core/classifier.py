"""YAMNet-based sound event classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# TF Hub migrated to Kaggle Models; prefer the Kaggle handle.
YAMNET_MODEL_URL = (
    "https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1"
)
DEFAULT_TOP_K = 3


class YAMNetClassifier:
    """Load Google's YAMNet model and classify short audio windows.

    YAMNet expects mono float32 PCM at 16 kHz. Each call to :meth:`predict`
    should receive a ~0.975-second buffer (15,600 samples).
    """

    def __init__(
        self,
        model_handle: Union[str, Path] = YAMNET_MODEL_URL,
    ) -> None:
        """Load YAMNet from a Kaggle/TF-Hub URL or a local SavedModel directory.

        Args:
            model_handle: Remote model URL or filesystem path to a SavedModel.

        Raises:
            RuntimeError: If the model or class map cannot be loaded.
            KeyboardInterrupt: If the user cancels during download/load.
        """
        self.model_handle = str(model_handle)
        try:
            self.model = hub.load(self.model_handle)
            self.class_names = self._load_class_names()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load YAMNet from '{self.model_handle}': {exc}"
            ) from exc

    def _load_class_names(self) -> List[str]:
        """Resolve YAMNet class display names from the model's class map CSV."""
        class_map_path = self.model.class_map_path().numpy().decode("utf-8")
        class_names: List[str] = []

        with tf.io.gfile.GFile(class_map_path) as csv_file:
            # Skip header: index,mid,display_name
            next(csv_file)
            for line in csv_file:
                # CSV fields: index, mid, display_name (display_name may contain commas)
                parts = line.strip().split(",", 2)
                if len(parts) < 3:
                    continue
                display_name = parts[2].strip().strip('"')
                class_names.append(display_name)

        if not class_names:
            raise RuntimeError("YAMNet class map is empty or unreadable.")

        return class_names

    def predict(
        self,
        audio_chunk: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict[str, Any]]:
        """Run YAMNet inference and return the top-k sound predictions.

        Args:
            audio_chunk: Mono float32 waveform for one analysis window.
            top_k: Number of top predictions to return (default: 3).

        Returns:
            A list of dicts with keys ``label`` and ``confidence``, sorted by
            confidence descending.

        Raises:
            ValueError: If the audio chunk is invalid.
            RuntimeError: If inference fails.
        """
        if not isinstance(audio_chunk, np.ndarray):
            raise ValueError("audio_chunk must be a numpy.ndarray")

        if audio_chunk.ndim != 1:
            raise ValueError(
                f"Expected a 1-D audio array, got shape {audio_chunk.shape}"
            )

        if audio_chunk.size == 0:
            raise ValueError("audio_chunk is empty")

        waveform = audio_chunk.astype(np.float32, copy=False)

        try:
            # YAMNet returns (scores, embeddings, log_mel_spectrogram)
            scores, _embeddings, _spectrogram = self.model(waveform)
            # Average frame-level scores over the window.
            mean_scores = tf.reduce_mean(scores, axis=0).numpy()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"YAMNet inference failed: {exc}") from exc

        k = min(top_k, mean_scores.shape[0], len(self.class_names))
        top_indices = np.argsort(mean_scores)[::-1][:k]

        predictions: List[Dict[str, Any]] = []
        for index in top_indices:
            predictions.append(
                {
                    "label": self.class_names[int(index)],
                    "confidence": float(mean_scores[int(index)]),
                }
            )
        return predictions
