# Bundled ML assets

- `yamnet.tflite` — Google YAMNet audio event classifier (TF Lite),
  from TensorFlow Lite Task Library / TF Hub lite-model
  (`lite-model_yamnet_classification_tflite_1`).
  Input: float32 waveform, length 15600 @ 16 kHz.
  Output: float32 scores for 521 AudioSet classes.

- `yamnet_class_map.csv` — AudioSet class index → display name mapping
  from the TensorFlow Models AudioSet YAMNet package.

## CLAP (`models/clap/`)

Quantized ONNX towers from
[Xenova/clap-htsat-unfused](https://huggingface.co/Xenova/clap-htsat-unfused)
(LAION `clap-htsat-unfused`). **Not committed** (~160 MB); download once:

```bash
python scripts/download_clap_models.py
```

Expected files:

| File | Role |
|---|---|
| `audio_model_quantized.onnx` | Live audio embeds (~34 MB) |
| `text_model_quantized.onnx` | Prompt embeds on rebuild (~127 MB) |
| `tokenizer.json` | Xenova RoBERTa tokenizer (must match this export) |
| `preprocessor_config.json` | Log-mel / 10 s window settings |

After download, rebuild text embeddings (web **Prompts → Rebuild** or
`python scripts/rebuild_clap_embeddings.py`) before `--enable-clap`.
