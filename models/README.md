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

## DASM (`models/dasm/`)

HTS-AT Detect Any Sound **text-query** ORT graph from the Phase 0 spike
([docs/DASM-SPIKE.md](../docs/DASM-SPIKE.md)). **Not committed** (~155 MB ONNX).

### One-time model install (Pi)

Copy from the spike tree (`~/dasm-spike/Transformer4SED/`):

```bash
mkdir -p models/dasm
cp …/onnx_export/dasm_text_fp32.onnx models/dasm/
# Base AudioSet query rows (407 × 1024) from text_query.pt — see docs/DASM-SPIKE.md
cp …/base_query.npy models/dasm/
# Optional but recommended (mel BN stats):
#   scripts/dasm_spike/extract_dasm_bn0.py on the spike venv → bn0.npz
cp …/bn0.npz models/dasm/
```

### Prompt embeddings (PC → git → Pi)

There is **no in-app Rebuild** (MGA-CLAP stays off the collector). Workflow:

1. Edit prompts in the web UI (**Settings → DASM**) or `config/dasm_prompts.json`.
2. On a **PC** (or any bake machine) that has the Transformer4SED spike + MGA-CLAP:

```bash
# Prefer the spike venv (torch + MGA-CLAP deps):
/path/to/Transformer4SED/.venv/bin/python scripts/bake_dasm_embeddings.py \
  --spike-root /path/to/Transformer4SED
```

3. Commit and push the small npz (and prompts JSON if changed):

```bash
git add config/dasm_prompts.json config/dasm_query_embeddings.npz
git commit -m "Update DASM prompt embeddings"
git push
```

4. On the Pi: `git pull`, then Start → **Heavy ML: DASM**.

`config/dasm_query_embeddings.npz` is meant to be versioned. Mutually exclusive
with `--enable-clap` / Heavy ML CLAP.
