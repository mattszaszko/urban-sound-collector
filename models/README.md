# Bundled ML assets

- `yamnet.tflite` — Google YAMNet audio event classifier (TF Lite),
  from TensorFlow Lite Task Library / TF Hub lite-model
  (`lite-model_yamnet_classification_tflite_1`).
  Input: float32 waveform, length 15600 @ 16 kHz.
  Output: float32 scores for 521 AudioSet classes.

- `yamnet_class_map.csv` — AudioSet class index → display name mapping
  from the TensorFlow Models AudioSet YAMNet package.
