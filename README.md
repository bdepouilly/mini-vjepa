# Mini V-JEPA — Self-Supervised Video Prediction in Latent Space

A small-scale, single-GPU reproduction of a **Joint-Embedding Predictive Architecture (JEPA)** for video, together with an evaluation suite that probes whether the learned representations actually capture **physical dynamics** — not just appearance.

The guiding idea: a useful world model should predict the **future state of the world in a learned representation space**, rather than reconstructing raw pixels. This repo is a minimal, readable implementation of that thesis, sized to run on one consumer or cloud GPU.

> Status: **in active development.** See the [Roadmap](#roadmap) for what's done and what's next.

---

## Motivation

Pixel-reconstruction objectives waste capacity modelling unpredictable, low-level detail (exact textures, lighting noise). JEPA-style methods instead:

1. Encode context and target into a latent space with a learned encoder.
2. Mask part of the spatiotemporal volume.
3. Train a **predictor** to infer the *representations* of the masked region from the visible context.
4. Use an **EMA (exponential moving average) target encoder** to provide prediction targets, avoiding representation collapse without contrastive negatives or a pixel decoder.

This repo reproduces that recipe at a deliberately small scale and asks a concrete question: **do the resulting representations encode the arrow of time and short-horizon dynamics?**

---

## Architecture

```
                  ┌─────────────────────┐
   masked clip →  │   Context Encoder    │ ──┐         (online, ViT-S over tubelets)
                  └─────────────────────┘   │
                                            ▼
                                  ┌──────────────────┐
                                  │    Predictor     │ → predicted target reps
                                  │ (narrow ViT)     │
                                  └──────────────────┘
                                            ▲
                  ┌─────────────────────┐   │  L2 loss in latent space
   full clip →    │   Target Encoder     │ ──┘         (EMA of context encoder, stop-grad)
                  └─────────────────────┘
```

- **Encoder:** ViT-S (~22M params) over **tubelet** patches (e.g. 2×16×16 spatiotemporal patches).
- **Predictor:** a narrow, shallow transformer that maps visible-context tokens + mask tokens (with positional/temporal embeddings) to predicted target representations.
- **Target encoder:** parameters updated as an EMA of the context encoder; gradients stopped. Targets are layer-normalised representations of the *masked* region.
- **Loss:** smooth-L1 / L2 between predicted and target representations of masked tubelets.
- **Masking:** multi-block spatiotemporal masking (mask contiguous tubes across several frames) to force temporal reasoning rather than per-frame interpolation.

---

## Data

| Dataset | Why | Scale used here |
|---|---|---|
| **Something-Something-V2** (default) | Motion-centric; labels describe *actions on objects*, so appearance shortcuts don't work. Forces real dynamics. | ~50–100k short clips |
| Kinetics-400 subset | Broader scenes; optional. | curated subset |
| UCF101 | Tiny; useful for the first smoke tests. | full |

Clips are decoded to **16 frames @ 112–128 px**, with random temporal crops and standard spatial augmentation. A curation script filters corrupt/short clips and caches frame indices for fast loading.

---

## Training (single GPU)

The whole point is that this runs **without a research cluster**:

- ViT-S encoder, 16 frames, 112–128 px.
- Mixed precision (`bf16`/`fp16`) + gradient accumulation.
- Fits on a single A100 (40GB) or a 24GB consumer card (4090/3090) with reduced batch.
- EMA momentum schedule ramped from ~0.996 → 1.0 over training.
- Cosine LR schedule with warmup; AdamW.

```bash
# install
pip install -r requirements.txt

# (1) curate the dataset
python scripts/prepare_ssv2.py --root /path/to/ssv2 --out data/ssv2_index.json

# (2) pretrain (self-supervised)
python -m src.training.pretrain --config configs/mini_vjepa_vits.yaml

# (3) evaluate
python -m src.evaluation.linear_probe --checkpoint checkpoints/last.pt
python -m src.evaluation.dynamics_probe --checkpoint checkpoints/last.pt
```

Dealing with throughput, mixed precision, and the EMA schedule on real hardware is part of the deliverable — it's the concrete "large-scale / accelerator training" experience the repo is meant to demonstrate, at a scale one person can actually run.

---

## Evaluation suite

Most reproductions stop at a decreasing pretraining loss. The interesting contribution here is the **eval suite**, mirroring the three things a world model should do — *understand*, *predict*, *plan*.

### 1. Frozen-feature linear probe (understanding)
Freeze the encoder, train a linear classifier on SSv2 action labels. Report accuracy vs. a from-scratch supervised baseline of equal compute. Answers: *did self-supervision learn anything useful?*

### 2. Dynamics-sensitivity probe (prediction) — the novel bit
Does the model encode the **arrow of time**? Feed:

- correctly-ordered clips,
- temporally **shuffled** clips,
- **reversed** clips,

and measure whether the predictor's latent prediction error separates these conditions. A genuine dynamics model should find shuffled/reversed futures harder to predict. Reported as an AUC of "real vs. corrupted-order" separability.

### 3. Latent rollout (planning, stretch)
Autoregressively predict target representations N steps ahead and plot latent prediction error vs. horizon. A flat-ish, slowly-growing curve indicates the model has captured stable short-horizon dynamics — the substrate for model-based planning.

---

## Repo layout

```
mini-vjepa/
├── configs/            # YAML experiment configs
├── scripts/            # data prep & curation
├── src/
│   ├── data/           # dataset, decoding, masking, augmentation
│   ├── models/         # ViT encoder, predictor, EMA target
│   ├── training/       # pretraining loop, schedules, AMP
│   └── evaluation/     # linear probe, dynamics probe, rollout
├── notebooks/          # exploration & figures
├── requirements.txt
└── README.md
```

---

## Roadmap

- [ ] **Week 1** — Data pipeline: decoding, tubelet patching, masking; ViT-S forward pass end to end.
- [ ] **Week 2** — EMA target encoder + predictor; SSL loss decreasing on a small subset.
- [ ] **Week 3** — Scale to full subset; linear-probe eval + baseline comparison.
- [ ] **Week 4** — Dynamics-sensitivity probe + technical write-up.
- [ ] **Stretch** — Latent rollout eval; Kinetics subset; ViT-B.

---

## References & inspiration

- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)*, 2023.
- Bardes et al., *V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video*, 2024.
- LeCun, *A Path Towards Autonomous Machine Intelligence*, 2022.

---

*Built by Baptiste Depouilly as an exploration of video-based world models. Not affiliated with Meta AI or the original V-JEPA authors.*
