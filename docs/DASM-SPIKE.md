# DASM Phase 0 spike notes

Provisional go/no-go for adding Detect Any Sound (DASM) as an exclusive heavy-ML backend (CLAP **xor** DASM) in urban-sound-collector. YAMNet stays always-on.

**Host:** Raspberry Pi 4 **4GB**, aarch64, Python 3.13, CPU-only PyTorch 2.14 (`matt@192.168.178.11`).  
**Upstream:** [cai525/Transformer4SED](https://github.com/cai525/Transformer4SED) + [CPF2/detect_any_sound](https://huggingface.co/CPF2/detect_any_sound) (`text_query` only).  
**Query encoder (bake only):** MGA-CLAP + `bert-base-uncased` + `htast_mga.pt`.  
**Spike tree on Pi:** `~/dasm-spike/Transformer4SED`

---

## What the spike was

A **go/no-go technical spike**: prove open-vocab SED can run on a Pi for **triggered ±5 s (10 s) windows**, not continuous inference.

**Product constraints (from the plan)**

- Shared Heavy ML Triggers; separate DASM prompts / window settings
- Multilabel Live / Inspect; reports ignore DASM
- Design ceiling: Pi 4 **2GB**; this spike ran on Pi 4 **4GB**
- Prefer **ONNX** (+ INT8 if it helps) and **offline query vectors** so the Pi runtime never needs MGA-CLAP / BERT

---

## What we did (chronological)

1. **Freed disk / backed up** USC recordings + `.env` to the PC; spike under `~/dasm-spike/Transformer4SED`.
2. **Downloaded text-query weights** + cloned MGA-CLAP; created a Python 3.13 venv; installed CPU PyTorch 2.14 (not upstream’s ancient pins).
3. **Fought env issues:** old `numpy` pins fail on 3.13; missing deps; `torchvision` mismatch; CUDA `torch.load` / Sparsemax `device='cuda'`; commented unused `tools.utils` import in MGA-CLAP.
4. **Jupyter notebook smoke:** loaded CLAP + DASM, ran an example wav, got a multilabel heatmap (alarm / sirens / etc.).
5. **Benchmarked PyTorch** (RSS + latency) — too heavy for product runtime.
6. **Exported ONNX** (`scripts/dasm_spike/export_dasm_onnx.py`): mel + query → strong / weak; optional dynamic INT8.
7. **Clean ORT-only RSS** (numpy + onnxruntime only, no torch model in the process).
8. **Baked offline queries** (`scripts/dasm_spike/bake_dasm_queries.py`): MGA-CLAP once → `.npy` / `.npz` pack; ORT smoke with baked vectors.
9. **Documented** this file + kept helper scripts under `scripts/dasm_spike/` in the product repo.

---

## Verdict (provisional)

| Target | Verdict | Notes |
|--------|---------|--------|
| Pi 4 **4GB** | **Go** (triggered ±5 s windows) | ORT FP32 ~2.9 s / ~428 MB peak |
| Pi 4 **2GB** | **Plausible — re-validate** | Needs load → infer → unload; not validated on 2GB hardware yet |
| Continuous always-on DASM | **No** | ~3 s / 10 s window; design is trigger-armed only |
| Dynamic INT8 (ORT) | **No for now** | 47 MB but **slower** (5.8 s) and large numeric drift |
| Runtime MGA-CLAP / BERT on Pi | **No** | Bake query vectors offline |

Product runtime must run **ORT + numpy only**.

---

## What we proved

1. Full PyTorch notebook smoke (text prompts → multilabel frame scores) works on Pi 4 4GB.
2. Monolith ONNX export of DASM (mel + query → strong / weak) works; ORT matches torch (~1e-5).
3. Clean ORT-only RSS (~428 MB peak) is far below the PyTorch stack (~1.9 GB).
4. Offline query bake + ORT smoke works end-to-end (weak scores up to ~0.98 on the example).

---

## Results / benchmarks

All numbers are for a **10 s** window, **413** queries = 407 checkpoint base + 6 open-vocab.

### PyTorch (MGA-CLAP + DASM in one process) — not product-viable

| Metric | Value |
|--------|------:|
| RSS before / after | 1842 → 1906 MB |
| Text encode (6 prompts) | 1.41 s |
| Forward mean (×3) | 3.63 s |
| Output | strong `(1, 413, 500)`, weak `(1, 413)` |

500 frames ≈ 10 s @ 50 fps.

### ONNX export

| Artifact | Size | ORT latency | max \|torch−ort\| |
|----------|-----:|------------:|------------------:|
| `dasm_text_fp32.onnx` | 155.4 MB | 3.34 s | ~2e-5 |
| `dasm_text_int8.onnx` (dynamic) | 47.1 MB | 5.83 s | ~0.19 (strong) |

**Keeper:** FP32. Dynamic INT8 was a regression on this Pi (slower + drift).

### ORT-only (fresh process, numpy + onnxruntime only) — product path

| Stage | RSS |
|-------|----:|
| Start | 44 MB |
| After npy load | 47 MB |
| After ORT session load | **277 MB** |
| After inference | **428 MB** |
| Mean latency | **2.88 s** |

### Offline query bake

| Metric | Value |
|--------|------:|
| Encode 6 prompts | ~8.6 s (bake time only) |
| Open-vocab shape | `(6, 1024)` |
| Full pack | `(413, 1024)` + mask `(413, 413)` |
| ORT smoke with baked pack | 3.13 s; weak min/max ~0.0009 / 0.9820 |

---

## What “done” means for Phase 0

**Done**

- Proven: open-vocab DASM runs on Pi 4 4GB via **ORT FP32 + prebaked queries**.
- Export recipe, bake script, and these notes exist in-repo.

**Not done yet**

- 2GB hardware re-validation
- Numpy / ORT-friendly mel frontend (today mel still uses torch HTS-AT `wav2mel` to build inputs)
- Product wiring (Phase 1+)

---

## Graph cut

**Kept for Phase 0:** single ONNX graph  

`mel [B,1,1001,64] + query [Nq,1024] + tgt_mask [Nq,Nq] → strong [B,Nq,T], weak [B,Nq]`

**Out of graph (offline):** MGA-CLAP text encoder → normalized 1024-d query rows; concatenated with checkpoint `text_query.pt` base (407).

**Split sub-graphs:** not required yet (monolith export succeeded). Revisit if ORT size / limits bite on 2GB.

---

## Export / bake recipe (Pi)

Scripts live in this repo under `scripts/dasm_spike/` and are copied into the spike tree:

```bash
cd ~/dasm-spike/Transformer4SED
source .venv/bin/activate

# 1) Export DASM (no CLAP in the graph)
python export_dasm_onnx.py --quantize   # INT8 optional; FP32 is the keeper

# 2) Bake open-vocab query vectors (uses MGA-CLAP once)
python bake_dasm_queries.py --ort-smoke --prompts alarm sirens "air raid sirens" dog music bird

# 3) Clean ORT RSS check (after reboot recommended; numpy + ORT only)
python - <<'PY'
import time, resource, numpy as np, onnxruntime as ort
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
pack = np.load("onnx_export/dasm_query_pack.npz")
mel = np.load("onnx_export/smoke_mel.npy")
sess = ort.InferenceSession("onnx_export/dasm_text_fp32.onnx", providers=["CPUExecutionProvider"])
feeds = {"mel": mel, "query": pack["query"], "tgt_mask": pack["tgt_mask"]}
for _ in range(2): sess.run(None, feeds)
t=[]
for _ in range(3):
    t0=time.perf_counter(); sess.run(None, feeds); t.append(time.perf_counter()-t0)
print("RSS", round(rss()), "MB  mean", round(sum(t)/len(t),2), "s")
PY
```

### Pi env notes (Python 3.13)

- Do **not** use upstream `requirements.txt` pins (`numpy==1.23.5`, etc.).
- Torch / torchaudio / torchvision from `https://download.pytorch.org/whl/cpu`.
- Comment out unused `from tools.utils import *` in `third_parties/MGA-CLAP/models/ase_model.py` (pulls sed_eval / dcase_util).
- Always `torch.load(..., map_location="cpu")` (or patch globally).
- MGA-CLAP `Sparsemax` hardcodes `device='cuda'` — patch to `input.device` for CPU.

---

## Artifacts

### On the spike Pi

```
~/dasm-spike/Transformer4SED/
  export_dasm_onnx.py
  bake_dasm_queries.py
  onnx_export/
    dasm_text_fp32.onnx          # keeper
    dasm_text_int8.onnx          # not recommended
    smoke_mel.npy
    dasm_query.npy
    dasm_tgt_mask.npy
    dasm_query_pack.npz
    dasm_query_meta.json
```

Keep `onnx_export/` if you want Phase 1 without re-exporting. The spike venv / Jupyter can be deleted later to reclaim disk.

### In this product repo

- `scripts/dasm_spike/export_dasm_onnx.py`
- `scripts/dasm_spike/bake_dasm_queries.py`
- `docs/DASM-SPIKE.md` (this file)

**Commit guidance:** committing the **scripts + this doc** is useful. Do **not** commit Pi weights / ONNX blobs into the app repo unless you intentionally vendor them under a gitignored `models/dasm/`.

---

## Open items

- [ ] Re-run ORT RSS / latency on a **2GB** Pi (or document DASM as 4GB+ only).
- [ ] Mel frontend parity: product uses numpy log-mel + optional `bn0.npz`; extract bn0 on the spike with `scripts/dasm_spike/extract_dasm_bn0.py`.
- [ ] Better INT8 (static PTQ / QDQ) only if FP32 RSS fails 2GB — dynamic quant was a regression on this Pi.
- [x] Phase 1 core: `core/dasm_*.py`, `classifier_dasm.py`, `--enable-dasm` xor `--enable-clap`, nested JSONL `dasm` payload.
- [x] Phase 2: Heavy ML Settings UI + Live/Inspect overlays (local; not committed yet).

## Phase 1 runtime (product)

```bash
# Install assets into models/dasm/ (see models/README.md), then:
python main.py --enable-dasm --record-audio …
```

Mutually exclusive with `--enable-clap`. Shared arming via `evaluate_arm` / `config/clap_triggers.json`.

### Prompt bake workflow (PC → git → Pi)

Open-vocab vectors are **not** rebuilt on the Pi. Edit prompts in the UI /
`config/dasm_prompts.json`, bake on a machine with the spike tree, commit the npz:

```bash
# On PC (spike venv + --spike-root):
python scripts/bake_dasm_embeddings.py --spike-root /path/to/Transformer4SED
git add config/dasm_prompts.json config/dasm_query_embeddings.npz
git commit && git push
# On Pi: git pull
```

Details: `models/README.md`. Spike-only pack script remains
`scripts/dasm_spike/bake_dasm_queries.py` (Phase 0); product uses
`scripts/bake_dasm_embeddings.py`.
