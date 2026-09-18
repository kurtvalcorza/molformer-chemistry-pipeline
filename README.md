# MoLFormer Chemistry Pipeline

DIMER-oriented pipeline for **MoLFormer-XL both-10pct** (`ibm-research/MoLFormer-XL-both-10pct`), pinned to an immutable Hugging Face revision. The repository exposes mean-pooled molecule embeddings from SMILES, a labelled-dataset contract with explicit ceilings, bounded gradient fine-tuning of a sequence-classification head, held-out classification metrics with trivial baselines, and a safetensors adapter artifact that is digest-verified before it is loaded.

## Upstream alignment

- Model: `ibm-research/MoLFormer-XL-both-10pct`
- Revision: `361063d0ad524ef77cf39b08469f6be770dc550f`
- Upstream weight license: Apache-2.0
- Upstream task: chemical language modelling on SMILES (masked LM); this repository uses the encoder for molecule embeddings and molecular-property classification
- Repository adaptation: **E2E** (bounded fine-tuning of the classification head plus the last *n* encoder layers, with a portable safetensors adapter)

## Two upstream properties you need to know before you start

**This checkpoint requires `trust_remote_code=True`.** Its `config.json` declares `model_type: "molformer"`, for which the installed Transformers has no native implementation, and its linear-attention architecture exists only in the upstream `modeling_molformer.py`. `AutoModel.from_pretrained(..., trust_remote_code=False)` refuses it with a `ValueError` naming the custom code. This pipeline therefore puts `configuration_molformer.py` and `modeling_molformer.py` **inside the snapshot manifest** and verifies their SHA-256 before importing them — `verify_snapshot()` refuses a manifest that omits them. That narrows the trust boundary; it does not remove it, and no security review of the upstream code has been performed. See *DIMER deployment notes* in `MODEL_CARD.md`.

**This repository pins `transformers==5.17.0`, not the fleet's 4.57.6.** The pinned upstream code calls `transformers.masking_utils.create_bidirectional_mask(inputs_embeds=...)`, whose current signature exists from Transformers 5.5.0. Use a dedicated virtual environment.

**Inference is deterministic here, but is not upstream.** MoLFormer's linear attention draws random feature maps, and the pinned `config.json` ships `deterministic_eval: false`, which redraws them on every forward pass — two identical calls return different embeddings. The loader overrides it to `True`, and `save_artifact()` exports the 12 `.feature_map.weight` buffers as declared serving state so a reloaded adapter reproduces the adapted model exactly.

## Quick start

```python
from molformer_chemistry_pipeline import MolformerPipeline, generate_sample_dataset, split_dataset

pipe = MolformerPipeline.from_pretrained()        # verifies the snapshot under weights/ first
emb = pipe.embed(["CCCCCCCO"])
print(emb["dimension"], len(emb["embeddings"][0]))   # 768 768

splits = split_dataset(generate_sample_dataset())
pipe.adapt(splits["train"], splits["validation"])    # bounded AdamW fine-tuning
print(pipe.evaluate(splits["test"])["accuracy"])
print(pipe.classify(["CCCC(C)CCO"])["predictions"][0]["label"])
```

`embed()` and `classify()` take 1..64 molecules (`MAX_MOLECULES_PER_CALL`) as SMILES strings of at most 200 tokens each (`MAX_TOKENS`, the checkpoint's 202 position embeddings minus `<bos>`/`<eos>`); a longer molecule is **refused, not truncated**, because a truncated SMILES is a different molecule. SMILES are checked **syntactically only** — accepted character set, balanced parentheses and brackets, paired ring-closure digits — and this repository ships no cheminformatics toolkit, so a chemically impossible but syntactically valid string is embedded without complaint. No canonicalisation, standardisation, desalting or tautomer handling is performed: the same compound written two ways is two different inputs. `classify()` requires a prior `adapt()` or `from_artifact()`. Datasets are `{id, smiles, label}` records: at least 8 rows and 3 per class, at most 5,000 rows and 20 classes, unique ids and unique SMILES.

## Weights layout

```
weights/molformer-xl-both-10pct/   config.json  model.safetensors  tokenizer.json  tokenizer_config.json
                                   configuration_molformer.py  modeling_molformer.py  README.md
                                   dimer-base-manifest.json
```

`from_pretrained()` calls `stage_missing_files()` then `verify_snapshot()` (byte size + SHA-256 of every manifest entry, refusing a manifest without the two Python files; staging fetches only absent entries, only at the pinned revision, and only with `allow_download=True`), then loads the tokenizer with `trust_remote_code=False` and the model with `trust_remote_code=True`, `deterministic_eval=True` and `local_files_only=True`. Neither `.safetensors` nor `.py` files under `weights/` are tracked by Git; the repository vendors neither the checkpoint nor the executable upstream code. See `docs/WEIGHTS.md`.

## Adapter artifacts

`save_artifact(dir)` writes `adapter.safetensors` (the trained tensors — head, final layer norm and the unfrozen encoder layers — **plus** the 12 linear-attention `.feature_map.weight` buffers as declared `serving_state_tensors`; about 33 MB with the default two layers) and a `manifest.json` recording the artifact format, the exact base model id and revision, `requires_remote_code`, `deterministic_eval`, the class order, the tensor names, the file size and SHA-256, and the full adaptation configuration. `MolformerPipeline.from_artifact(dir)` re-verifies the base snapshot, then checks the artifact manifest, the base identity and every digest **before** deserialising, and refuses any tensor the base architecture does not have. Without the feature buffers a reloaded adapter does not reproduce the adapted model (measured: maximum absolute score difference 0.0023 without them, 0.0 with them).

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: injected backends and temporary manifests, never the weights.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/molformer-chemistry-pipeline/blob/main/tutorials/molformer_chemistry_colab.ipynb)

`tutorials/molformer_chemistry_colab.ipynb` is declared `E2E` and is **standalone** (DIMER Notebook Specification 2.0 §4): it is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and embeds the 3 package modules (`pipeline.py`, `samples.py`, `metrics.py`) verbatim in dependency order, the pinned model identity, the snapshot manifest and the exact runtime pins, so the exported `.ipynb` keeps working without this repository being reachable — no clone, no repository install, no repository import on its primary path. Do not edit it by hand; change the package or the template and regenerate (`python tools/build_notebook.py`; `--check` is enforced by the validator and CI). Its default path generates and validates a deterministic 64-molecule **constitutional-isomer** dataset — the two classes are built so that every molecular formula appears in both, which is why the formula baseline sits near chance and the fine-tuned result means something — splits it 36/12/16 stratified by class, stages and digest-verifies the pinned snapshot including the two Python files it will execute, extracts mean-pooled embeddings and asserts a repeated call returns the identical vector, measures majority-class and molecular-formula baselines, runs a 4-epoch bounded fine-tuning, evaluates accuracy/macro-F1/AUROC on the held-out test split, classifies six freshly generated molecules, exports the safetensors adapter with its serving state and verifies reload parity. BYOD (CSV/JSON/JSONL) is optional and gated off by default. See `tutorials/README.md`.

## Release status

**Candidate.** Static/unit checks — including the standalone generator parity checks (`tools/build_notebook.py --check`, `tests/test_notebook_parity.py`) — do not constitute clean-runtime notebook evidence. One local CPU pre-flight execution of the committed notebook is recorded in `docs/release-verification.md`; complete the supported clean-runtime procedure in that document against the exact release revision before calling the notebook release-grade. DIMER hosting has a separate open gate — the remote-code requirement — recorded in `MODEL_CARD.md`.

## Licensing

- Upstream weights and upstream model code: Apache-2.0 (`ibm-research/MoLFormer-XL-both-10pct`), staged unmodified from the pinned revision.
- This repository's code and documentation: Apache-2.0 (`LICENSE`).
- The upstream licence governs your use of the weights, including commercial use and redistribution; this repository grants no rights beyond it.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
