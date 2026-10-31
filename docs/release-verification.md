# Release verification

**Current status (2026-10-26): Candidate.** The current blob `26019328` (code commit `e96b94d`) shows a refused BYOD input as one `Refused: <file>: <reason>` line with no traceback, removes the earlier run's results and `outputs/` files and stops Run all at that cell, while genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20); its default path (2026-10-30; 14 code cells, exit 0, 0 errors) and its scripted DAT20 refusal (2026-10-30; `USE_BYOD=True;BYOD_PATH=/content/sample_data/california_housing_test.csv`: exactly one `Refused:` line at cell `molformer_chemistry-16`, no traceback, earlier cells clean, VM released) both passed on a hosted Colab Tesla T4 (Colab CLI `colab exec -f`, not a browser `Run all`), and the hosted runs of earlier blobs, including `6eafc6a0`, are history for it (REL14).

**Previous status (2026-10-25): Candidate.** The previous notebook `tutorials/molformer_chemistry_colab.ipynb` is blob `6eafc6a0dc1183b2b28dcf81027a2ff9e328b9c0` (PR #10 head `e03afaf`, regenerated from generator commit `cfadd43` for the tier C review findings of row 53). It passed a Colab CLI sequential execution on a fresh Colab Tesla T4 at `e03afaf` on 2026-10-25 (14/14 code cells, one pass, no restart, 0 errors, 99.2 s; test accuracy / macro-F1 / AUROC 1.0, reload parity 0.0; see "Recorded executions"). That run covers the **default path** only. Remaining gates: REL12 (a hosted BYOD run, one input accepted and one refused) and REL2 (a browser run of the interactive journeys). Every run of an earlier blob recorded here is history for that blob (REL14).

`tutorials/molformer_chemistry_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate record.

DIMER hosting has a **separate** gate that this document does not cover: the checkpoint requires
`trust_remote_code=True`, which MODEL_ASSET_SPEC §12 RC6 treats as not ordinarily qualified. A clean-runtime PASS here
promotes the *tutorial*, not the model profile. See the *DIMER deployment notes* section of `../MODEL_CARD.md`.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the inline
  `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the single kernel cell that builds (or reuses, by lock digest) the isolated hash-locked uv environment and routes every later cell to it, with no `pip install` into the kernel and no restart request; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions;
- **the remote-code boundary** (`validate_remote_code_boundary`, this repository's one documented divergence from the
  fleet validator): `trust_remote_code=True` appears in the carried module cell and **nowhere else** in the notebook;
  `src/molformer_chemistry_pipeline/pipeline.py` is where it is enabled and names both executed files;
  `configuration_molformer.py` and `modeling_molformer.py` are entries in the snapshot manifest; and
  `verify_snapshot` runs **before** the model libraries are imported; and the notebook markdown never claims that no
  remote model code is executed, while Section 3 names `trust_remote_code=True`, both executed files and the
  `deterministic_eval` override (review finding MOL-M2);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `MolformerPipeline.from_pretrained(weights_dir=...)`, `validate_dataset`, `split_dataset`, `write_dataset_csv`,
  the shared-formula assertion, `pipe.token_count`, `validate_inputs`, `pipe.embed` with its asserted
  like-for-like repeated-call determinism check, `majority_baseline`, `formula_baseline`, `pipe.adapt` with its explicit hyperparameters,
  `pipe.evaluate` on both the validation and the test split, `pipe.classify`, `pipe.save_artifact` with its
  `serving_state_tensors` report, `MolformerPipeline.from_artifact` and the reload-parity assertion), the six
  expected `outputs/` paths, the learner-facing statements (scores are not calibrated probabilities, validation is
  monitoring only, embeddings are representations, SMILES validation is syntactic, scaffold- or cluster-aware
  splitting) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` /
  repository import on the primary path, a mutable `revision='main'`, direct `transformers` / `huggingface_hub` /
  `safetensors` use **outside the carried module cells**, `pickle.load`, `torch.load(` without `weights_only=True`,
  `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected backends and temporary manifests, no
weights). These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/molformer-xl-both-10pct/` (the standalone path writes the manifest itself and stages every
   listed file, so the directory may not be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `VAL_FRACTION = 0.2`, `TEST_FRACTION = 0.25`, `SEED = 42`, `EPOCHS = 4`,
   `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 8`, `TRAINABLE_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, **`transformers==5.17.0`**,
   `huggingface-hub==1.32.0`, `tokenizers==0.23.2`, `safetensors==0.8.0`, `numpy==2.5.3`, installed by Section 1 into
   an isolated environment built from the hash-locked `tutorials/requirements-colab.lock.txt`. Nothing is installed
   into the kernel and no restart is expected; a run that needed a restart is not a one-pass Run all and does not
   count as release evidence (REL2/REL11). A runtime that resolved Transformers 4.x has not exercised the supported
   path and its result does not count;
5. verify every default-path stage completes:
   - the isolated environment built (or reused, by lock digest) from the hash lock with no GitHub access;
   - the three carried module cells execute (defining `MolformerPipeline`, `verify_snapshot`, `stage_missing_files`,
     `validate_inputs`, `check_smiles_syntax`, `validate_dataset`, `split_dataset`, `generate_sample_dataset`,
     `write_dataset_csv`, `molecular_formula`, `formula_string`, `classification_metrics`, `majority_baseline`,
     `formula_baseline`) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(..., allow_download=True)`
     reporting the 7 entries fetched from `ibm-research/MoLFormer-XL-both-10pct` at the immutable revision — including
     `configuration_molformer.py` and `modeling_molformer.py` — and `verify_snapshot` reporting 7 verified files
     **before** the model loads and before the remote code is imported;
   - the Section 3 load summary printing `remote_code_executed: True`, both verified remote-code files, the upstream
     `deterministic_eval: False` beside the package's `True`, and the base encoder's parameter count (the
     classification head is newly initialised in Section 8, whose report gives 45,557,762 total parameters);
   - the dataset manifest printed with 64 records, classes `['branched', 'linear']`, 32/32 class counts, 32 distinct
     molecular formulas with **all 32 shared across both classes**, the ceilings and the digest, and the splits
     36 / 12 / 16 (`split_dataset` refuses a split whose validation or test part falls below 8 records or 3 per class);
   - `pipe.embed` reporting 768-dimensional vectors, writing `outputs/molformer_chemistry_embeddings.csv`, and the
     determinism cell embedding the same eight molecules twice, printing
     `identical_vectors_on_a_second_identical_call: True` and asserting it; the single-molecule vs padded-batch
     difference is printed against a `1e-4` tolerance for inspection only;
   - both baselines reported on the test split (majority accuracy 0.5 on the balanced split; the molecular-formula
     baseline at or below chance, by construction), and the one-character rule (`(` in the SMILES means branched)
     printed for the default sample;
   - `pipe.adapt` reporting the trainable/total parameter counts and a four-epoch history with per-epoch validation
     metrics;
   - `pipe.evaluate` reporting validation and test accuracy, macro-F1, AUROC and per-class rows, and writing
     `outputs/molformer_chemistry_evaluation_report.json` with both baselines and the delta against the majority
     baseline;
   - `validate_inputs` accepting the six new molecules and `pipe.classify` writing
     `outputs/molformer_chemistry_predictions.csv` with per-class scores;
   - `pipe.save_artifact` writing `outputs/molformer_chemistry_adapter/{adapter.safetensors,manifest.json}` with 12
     `serving_state_tensors`, and `MolformerPipeline.from_artifact` reloading it with identical labels and a maximum
     absolute score difference below `1e-5` (the cell asserts both);
   - `outputs/molformer_chemistry_result.json` written with `NOTEBOOK_SOURCE`, the model identity, revision and
     licence, `remote_code_executed` and `remote_code_files`, the dataset manifest, the evaluation report, the
     predictions, the artifact manifest and the runtime versions;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache and the weights directory were clean, outcome,
   produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable `SHOULD`
   deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `molformer_chemistry_colab.ipynb` | `7c2ab16` / `dfb259ef4801` | 2026-09-18 | Kaggle batch kernel `dimer-nb2-molformer-chemistry` v1 (Python 3.12.13, Tesla T4, empty Hugging Face cache, no repository checkout) | **Restart-dependent — not a one-pass Run all (REL2/REL11)** — the in-kernel `pip install` of Section 1 required a fresh-process restart before the remaining cells ran; after that restart, 13/13 code cells; download, digest verification and the pinned remote-code boundary exercised. History, not release evidence |
| `molformer_chemistry_colab.ipynb` | `daaa53f` / `dfb259ef` | 2026-09-18 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/molformer_chemistry_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/molformer_chemistry_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-18 | `7c2ab16` / `dfb259ef4801` | Kaggle batch kernel `dimer-nb2-molformer-chemistry` v1 (Python 3.12.13, Tesla T4, clean cache, `transformers 5.17.0`) | Default sample path (download and verify 7 files including both remote-code modules → validate → split → embed + determinism check → baselines → adapt → evaluate → classify → export → reload) | 208.7 s | **Restart-dependent — not a one-pass Run all (REL2/REL11)** — the in-kernel `pip install` of Section 1 required a fresh-process restart before the remaining cells ran; after that restart, 13/13 code cells; test accuracy/macro-F1/AUROC 1.0 (n=16) against majority 0.5/0.3333 and formula 0.25/0.2; reload parity 0.0. History, not release evidence |
| 2026-10-08 (09:36:18 UTC start) | commit `9c8440b` / blob `7b0975db8ddd` (`NOTEBOOK_SOURCE.repository_revision` `6cde8bd`, the source revision the notebook was generated from, equal to `metadata.dimer.generated_from.revision`; `6cde8bd..9c8440b` is the review-fix commit itself; embedded `module_sha256` `1783c35ea4be…`, generator `build_notebook.py/2.2`, `notebook_spec` 2.2) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-molformer-9c8440b-80f0`), committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout, empty Hugging Face cache; kernel Python 3.13.15; isolated `uv` environment Python 3.12.12, 56 locked packages, built in 56 s; torch 2.14.0+cu130, transformers 5.17.0, safetensors 0.8.0; device `cuda:0` | default settings only (`USE_BYOD = False`, `EPOCHS = 4`, `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 8`, `TRAINABLE_LAYERS = 2`, `SEED = 42`); all 7 snapshot files fetched from `ibm-research/MoLFormer-XL-both-10pct` at `361063d0ad52` and verified before the remote code was imported | 99.8 s | **One pass, no restart, 0 errors** — 14/14 code cells in order (`exec.log`); code cells 3–5 (the carried modules) print nothing by design; load summary `remote_code_executed: True`, both remote-code files verified, upstream `deterministic_eval: False` vs package `True`, 44,375,040 encoder parameters; splits 36 / 12 / 16 with 32 of 32 formulas shared; four syntax rejections and the 201-token refusal; Section 6 `identical_vectors_on_a_second_identical_call: True` (asserted), single-molecule vs padded-batch difference 1.31e-06 within the 1e-4 tolerance; baselines majority 0.5 / 0.3333, formula 0.25 / 0.2 (9 test formulas unseen), one-character rule 16 / 16; 8,275,970 of 45,557,762 parameters trainable, train loss 0.4829 → 0.0073 over 4 epochs, validation accuracy 1.0 from epoch 1; test accuracy / macro-F1 / AUROC 1.0 / 1.0 / 1.0 (n = 16); 6 / 6 new molecules matched; adapter 33,208,144 bytes with 12 serving-state tensors, reload parity labels equal, max abs score difference 0.0; stderr carries the Hub's unauthenticated-request notice, Transformers' `molformer` model-type note and its load reports (`lm_head.*` UNEXPECTED; the new classifier keys MISSING when the classifier is built), as Section 3 describes. Evidence: `docs/execution-evidence/2026-10-08-9c8440b/` (executed notebook sha256 `0ced64efe8bb…`, `run_summary.json` `b2d8a99ba023…`, `exec.log` `050996c33b48…`). BYOD and the optional experiments not exercised. |
| 2026-10-25 (09:24:23 UTC start) | commit `e03afaf` (PR #10 head) / blob `6eafc6a0dc11` (tier C review fixes, generated at `cfadd43`) | Colab CLI sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4 (session `suite-molformer-e03afaf-80f0`); isolated hash-locked environment | Default sample path, every form field at its default | 99.2 s | **PASSED** — 14/14 code cells in order in `exec.log`, one pass, no restart, 0 error outputs, input notebook unchanged; input manifest `accepted` (4 molecules); baselines majority 0.5 (macro-F1 0.3333) and molecular-formula lookup 0.25 (macro-F1 0.2); one-character rule 16/16; 8,275,970 of 45,557,762 parameters trainable on `cuda:0`, 4.51 s; train loss 0.482904 → 0.182822 → 0.024827 → 0.007268, validation accuracy 1.0 from epoch 1; test (n = 16) accuracy, macro-F1 and AUROC 1.0; reload parity PASS, labels equal, max abs score difference 0.0; 6 of 6 freshly generated molecules matched. Every number equals the `9c8440b` run and the numbers the notebook quotes. Evidence `docs/execution-evidence/2026-10-25-e03afaf/`: SHA-256 captured → committed: `molformer_chemistry_colab_e03afaf_colab-cli-t4_default.ipynb` `0d718dd6e086c712163b1c095eb42a2673ed67270b77886585d8cd72b345dde0` → unchanged (no local path); `exec.log` `bcfcfe8e03b47969b163c3d8353e88d69cc54f0e7ce53329d2ee0ff0954d6fe1` → `7c4f3d48b5d77137e09fed4bac07bd0b9ab7c4f79f53e19ad8b77de52bc47867` (2 local-path replacements); `run_summary.json` `dedfa4c72153a4b8e86021cb22dc42a532983b2acba86d80a06f3f3d71034264` → unchanged (no local path) |
| 2026-09-18 | `daaa53f` / `dfb259ef` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `transformers 5.17.0`) | Default sample path (validate → split → embed + determinism check → baselines → adapt → evaluate → classify → export → reload); weights pre-staged, so `stage_missing_files` fetched 0 of 7 entries and `verify_snapshot` verified all 7 | 10.5 s | **PASSED** — 13/13 code cells; test accuracy/macro-F1/AUROC 1.0 (n=16) against majority 0.5/0.3333 and formula 0.25/0.2; reload parity 0.0. Pre-flight; hosted clean-runtime run still required |

### Hosted Colab T4 runs of blob `26019328` (2026-10-30)

These runs executed the PR head `b21deaaa4c7d7183e4c14329b43936f945dca5ec`, notebook `tutorials/molformer_chemistry_colab.ipynb` blob `26019328b4381f1e3b433d9faa5185fb74087b90`, fetched at the 40-character SHA and Git-blob verified, with the Colab CLI (`colab exec -f`) on a fresh Colab Tesla T4 session (not a browser `Run all`). Each `run_summary.json` records `verdict` `PASS`, `exec_exit_code` 0, `input_unchanged` true and `stop_verified` true (the VM was released), and the SHA-256 of each executed notebook equals the summary's `executed_sha256`.

**Default path (2026-10-30 18:10:17 UTC start).** Default settings, no overrides; session `suite-molformer-b21deaa-80f0`; wall 109.8 s (`wall_s`); `code_cells` 14, `executed_cells` 11, 0 errors. Retained evidence: [`docs/execution-evidence/2026-10-30-b21deaa/`](execution-evidence/2026-10-30-b21deaa/).

Printed by the default run (verbatim lines from the executed outputs):

```text
{'verified_files': 7, 'revision': '361063d0ad524ef77cf39b08469f6be770dc550f'}
{'baseline': 'majority-class', 'predicted_label': 'branched', 'accuracy': 0.5, 'macro_f1': 0.3333}
{'baseline': 'molecular-formula lookup', 'distinct_train_formulas': 25, 'ambiguous_train_formulas': 11, 'eval_formulas_unseen_in_train': 9, 'accuracy': 0.25, 'macro_f1': 0.2}
{'rule': 'one-character rule (default sample only): a ( in the SMILES means branched', 'correct': 16, 'of': 16, 'accuracy': 1.0}
{'method': 'gradient fine-tuning (AdamW) of the classification head and the last 2 encoder layer(s)', 'trainable_parameters': 8275970, 'total_parameters': 45557762, 'precision': 'float32', 'device': 'cuda:0', 'seconds': 4.58}
{'split': 'test', 'n': 16, 'accuracy': 1.0, 'macro_f1': 1.0, 'auroc': 1.0}
{'reload_parity': 'PASS', 'labels_equal': True, 'max_abs_score_diff': 0.0}
```

**DAT20 refusal record (2026-10-30, blob `26019328`).** Overrides, set by `--set` on a copy of the blob: `USE_BYOD` `False` → `True` and `BYOD_PATH` `''` → `'/content/sample_data/california_housing_test.csv'` (the run copy differs from the blob in exactly those 2 source lines). The suite cut the run copy right after the refusing cell `molformer_chemistry-16`, dropping 20 cells (7 of them code cells), which reproduces `Run all` stopping at the first error; the cut copy is `run_copy.ipynb`, SHA-256 `191a4e3e68eeaf218496f0d3ff92fd549e67c5b85e9dfed51153e495ea777e8f` (Git blob `8d085ec5`), equal to the summary's `run_copy_sha256`. The cut copy's earlier code cells completed without an error output; the refusing cell's only error output has `ename` `IsolatedCellRefusal` and no traceback, and its output holds exactly one `Refused:` line:

```text
Refused: california_housing_test.csv: ValueError: CSV header ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value'] is missing required column(s) ['id', 'smiles', 'label']; fix the input and run this cell again (the earlier results of this session and their outputs/ files were removed).
```

This run says nothing about the 20 cut cells, which did not run in it. Session `suite-molformer-b21deaa-80f0`, wall 91.7 s. Retained evidence: [`docs/execution-evidence/2026-10-30-b21deaa-refusal/`](execution-evidence/2026-10-30-b21deaa-refusal/).

**Local-path redaction.** Local paths were replaced byte-wise by `<local-run-dir>` (and the Colab CLI install path by `<colab-cli-tool-dir>/`) before the first commit; each executed notebook and run copy was checked against its `run_summary.json` before any edit.

| File (under `docs/execution-evidence/2026-10-30-b21deaa/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `molformer_chemistry_colab_b21deaa_colab-cli-t4_default.ipynb` | `5309a6a82118689d0edcba26133a26c9a6a305dd0f1ec778be3eca4f7bac564e` | `5309a6a82118689d0edcba26133a26c9a6a305dd0f1ec778be3eca4f7bac564e` (unchanged, no local path) |
| `exec.log` | `e2491797bf0ebb4020226d68561b3b0fab9ebd054c6b69d5443c50262198f52a` | `f362ce626a36d2bf9604c95c8fcb5abc98fe7fd072a5f2088bfc44dfea77fa92` (2 local-path replacements) |
| `run_summary.json` | `1e6b66ec2330438f6cdfabfcf6c25d24e45a58e5c450bf49f7248a229c3df837` | `1e6b66ec2330438f6cdfabfcf6c25d24e45a58e5c450bf49f7248a229c3df837` (unchanged, no local path) |

| File (under `docs/execution-evidence/2026-10-30-b21deaa-refusal/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `molformer_chemistry_colab_b21deaa_colab-cli-t4_refusal.ipynb` | `4d3d6f70718cfd5c57e528da1dcad4c84a0489e6da78f5c08e88edaff5028a2f` | `4d3d6f70718cfd5c57e528da1dcad4c84a0489e6da78f5c08e88edaff5028a2f` (unchanged, no local path) |
| `exec.log` | `faa5ccdb3ff53a3df9c3911c351aef063c4bdc0833a3540ce888e37fe03c490c` | `3f95e4b9b28743e04c96416da933c4cd95966645fc7dd89290a519faf3645e37` (2 local-path replacements) |
| `run_summary.json` | `31ca7ebbd915efbe96da5f674010f5343c2a358d2f01d8ba04dd6306b66a4c4b` | `31ca7ebbd915efbe96da5f674010f5343c2a358d2f01d8ba04dd6306b66a4c4b` (unchanged, no local path) |
| `run_copy.ipynb` | `191a4e3e68eeaf218496f0d3ff92fd549e67c5b85e9dfed51153e495ea777e8f` | `191a4e3e68eeaf218496f0d3ff92fd549e67c5b85e9dfed51153e495ea777e8f` (unchanged, no local path) |

Not exercised by these runs: an accepted BYOD input (REL12), a browser `Run all` of the interactive journeys (REL2), and every other open gate named in this file; these runs close none of them.

## Current status

**Candidate.** The current blob `26019328` (code commit `e96b94d`) shows a refused BYOD input as one `Refused: <file>: <reason>` line with no traceback, removes the earlier run's results and `outputs/` files and stops Run all at that cell, while genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20); its default path (2026-10-30; 14 code cells, exit 0, 0 errors) and its scripted DAT20 refusal (2026-10-30; `USE_BYOD=True;BYOD_PATH=/content/sample_data/california_housing_test.csv`: exactly one `Refused:` line at cell `molformer_chemistry-16`, no traceback, earlier cells clean, VM released) both passed on a hosted Colab Tesla T4 (Colab CLI `colab exec -f`, not a browser `Run all`), and the hosted runs of earlier blobs, including `6eafc6a0`, are history for it (REL14). Before that change: The notebook now builds an isolated, hash-locked environment in Section 1 (generator `/2.2`) instead of
installing into the kernel, and the review fixes (MOL-M1..M4, MOL-m1..m5) changed the carried modules and the
notebook, so its blob is new; the review-fix blob `7b0975db8ddd` (commit `9c8440b`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-08 (Colab CLI sequential execution, 14/14 code cells, 99.8 s; test accuracy / macro-F1 / AUROC 1.0 (n = 16) against majority 0.5 and formula 0.25, repeated embeddings identical, reload parity 0.0) — record below. The 2026-09-18 Kaggle T4 run of blob
`dfb259ef4801` needed a restart after the install cell, so it was never a one-pass Run all and is kept above as
history, not as release evidence. Release step REL12 (BYOD on a hosted runtime) has not been run, so the status
stays **Candidate**. Execution records remain sample-sanity evidence, not a benchmark or production-readiness claim.
