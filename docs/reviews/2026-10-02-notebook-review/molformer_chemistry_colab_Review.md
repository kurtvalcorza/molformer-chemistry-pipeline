# MoLFormer-XL Molecular-Property E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 4 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/molformer-chemistry-pipeline`  
**Notebook:** `tutorials/molformer_chemistry_colab.ipynb`  
**Reviewed commit:** `12ba808bf448fad713ac70480f12ae393097ba90` (`main`, confirmed with `gh api repos/kurtvalcorza/molformer-chemistry-pipeline/commits/main`)  
**Notebook Git blob:** `dfb259ef480142ff4080f61696fbb09ba716442c`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-18 (commit `7c2ab16`). The notebook has not changed since; it carries the modules of `b7421af203e7`, and only `tools/validate_release_assets.py` has changed in `src/` + `tools/` since then.  
**Finding prefix:** `MOL`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`. The notebook declares 2.0.

## Executive assessment

The default sample path works and is carefully engineered. The notebook stages the pinned 7-file snapshot, digest-verifies the two upstream Python files before `trust_remote_code=True` imports them, generates 64 constitutional-isomer molecules, runs both baselines on the training split only, fine-tunes the head plus two encoder layers, evaluates on an independent test split, and exports an adapter whose reload parity is exact. This review's clean CPU run reproduced the recorded numbers: test accuracy, macro-F1 and AUROC all 1.0 (n = 16), against majority 0.5 / 0.3333 and formula 0.25 / 0.2, with reload parity 0.0.

What fails is what the notebook tells the learner around that path:

1. **No one-pass `Run all` (MOL-M1).** The recorded Kaggle run needed a restart after the install cell. The release record still calls it PASSED / Release-grade and describes the restart as "expected".
2. **Section 3 says no remote code runs (MOL-M2).** The model it loads executes two upstream Python files. The opening, the Prerequisites and Section 6 all send the learner to Section 3 for the remote-code and `deterministic_eval` explanation, and Section 3 says the opposite of the opening and does not explain either. The sentence comes from the generator, and the repository's remote-code boundary validator does not catch it.
3. **The determinism demonstration prints `False` (MOL-M3).** Section 6 tells the learner that a second call returns exactly the same vector. It then compares a 1-molecule call with the same molecule inside a padded 8-molecule batch, and prints `same_vector_on_a_second_call: False` (difference up to 2.7e-6). The model is in fact deterministic: the same call repeated is identical, and an equal-length batch matches exactly. The release record says the cell "asserts" equality, but no assertion exists.
4. **The stated BYOD minimum is wrong (MOL-M4).** Stated: 8 records, 3 per class. The validation and test splits are then checked against that same whole-dataset minimum, so a balanced 2-class dataset needs **36** records. A 24-record CSV passes Sections 4–7 and then stops in Section 8 with `dataset has 4 records; at least 8 are required`, which describes the validation split, not the learner's dataset.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites assume SMILES, constitutional isomers, and accuracy / macro-F1 / AUROC |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU is enough, CUDA used when present; float32 |
| Promised outcomes | pinned install; carried modules; staged and digest-verified snapshot including the two executed Python files; explanation of `trust_remote_code` and the `deterministic_eval` override; 64-molecule isomer dataset validated and split 36/12/16; tokenization and ceiling refusal demo; 768-d embeddings with a repeated-call determinism check; majority and formula baselines; bounded AdamW fine-tune; validation and test metrics; inference on six new molecules; safetensors adapter export with serving state and fresh-reload parity; optional BYOD (CSV/JSON/JSONL) through every stage |
| Learning objectives | install; inspect modules; stage/verify a snapshot including executed code; "understand" `trust_remote_code` and `deterministic_eval`; validate SMILES and split without leakage; read tokenization; extract embeddings; measure baselines; fine-tune; evaluate on an independent split; classify new molecules; export and reload an adapter |

### Existing execution evidence

- `docs/release-verification.md`: Kaggle Tesla T4, commit `7c2ab16` / blob `dfb259ef4801` (= the reviewed blob), 2026-09-18, **PASSED — 13/13 code cells; "One expected fresh-process restart followed the install cell."** Test 1.0/1.0/1.0, reload parity 0.0, 208.7 s.
- A local pre-flight on the same blob (weights pre-staged), labelled not promotion evidence.
- No executed notebook or cell log of the Kaggle run is in the repository or the workspace backups, so the hosted Section 6 output (MOL-M3) is **not verified**.
- No hosted run covers BYOD or either optional experiment (REL12).

### Evidence obtained by this review

- **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=4`, 4 torch threads), build venv `dimer-molformer` (Python 3.12.10, torch 2.14.0+cpu, transformers 5.17.0, huggingface-hub 1.32.0, tokenizers 0.23.2, safetensors 0.8.0, numpy 2.5.3 = `PINS`; torchvision/torchaudio absent). The install cell ran with the notebook's own `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, so nothing was installed. `HF_HOME` and the working directory were empty scratch directories, so all 7 snapshot files were fetched from the Hub at the pinned revision and verified. `google.colab.files.upload` was a shim returning prepared bytes (`{}` for a cancelled dialog). **This is not a Colab run.**
- **P1 static:** JSON parses (nbformat 4.5); 13 code cells compile; no persisted outputs; blob equals the recorded-run blob. `tools/build_notebook.py --check` exit 0, `tools/validate_release_assets.py` exit 0 (PASS including `remote-code-boundary`), offline `pytest` 49 passed.
- **P2 default path (direct, CPU):** 13/13 cells ok in about 50 s including the download. Fetched 7 / verified 7; 8,275,970 of 45,557,762 parameters trainable; validation 1.0 from epoch 1; test 1.0 / 1.0 / 1.0 (n = 16); baselines majority 0.5 / 0.3333 and formula 0.25 / 0.2; 6/6 new molecules matched; reload parity 0.0; 7 output files. Section 6 printed `same_vector_on_a_second_call: False`. Section 5 printed **four** syntax rejections, not three.
- **P3 exercises (rerun Sections 8 and 9 only):** `EPOCHS = 1` gives test 1.0 / 1.0 / 1.0. `TRAINABLE_LAYERS = 0` (1,182,722 trainable) also gives 1.0 / 1.0 / 1.0. Cell 23 rewrites `evaluation_report.json`, but `result.json` (epochs 4) and the adapter manifest (2 layers) still describe the default run.
- **P4 BYOD:** a 40-record CSV (20 carboxylic acids, 20 amines; aromatic rings, `Cl`, `[NH3+]`) ran Sections 4–11 with test 1.0, formula baseline 0.5 (all 10 test formulas unseen) and reload parity 0.0. Its "new molecules" were its first six test-split records. A 24-record CSV that meets the stated contract stopped in Section 8 (`ValueError: dataset has 4 records; at least 8 are required`). Invalid inputs: a missing `label` column, an unpaired ring digit, a single class and a `.xlsx` file are each rejected in Section 4 with a message naming the problem. A cancelled upload raises a bare `StopIteration`.
- **P5 BYOD minimum arithmetic (no model):** for balanced 2-class sets of 8–34 records, `split_dataset` succeeds, but the validation and/or test split fails `validate_dataset` (minimum 8 records, 3 per class), which `pipe.adapt` and `pipe.evaluate` call. 36 is the smallest balanced set that reaches evaluation.
- **P6 Section 6 check decomposed:** the same 8-molecule call repeated is identical; the same 1-molecule call repeated is identical; molecule vs itself inside the padded batch differs by 0.95e-6–2.7e-6 for all 8; inside an equal-length (unpadded) batch the difference is 0.0. Classifier scores, batched vs one at a time, differ by at most 6.7e-10, with 0 label flips.
- **P7 one-character rule:** `'(' in smiles → branched` classifies 64/64 of the sample, 16/16 of the test split and 6/6 of the new molecules.

## 2. Separate judgments

| Judgment | Assessment |
|---|---|
| Technical correctness | The default path is correct and reproducible, and the snapshot and remote-code verification are sound. The install step forces a restart on hosted images (MOL-M1). The Section 6 check compares two different computations and reports a false negative (MOL-M3). `adapt`/`evaluate` apply the whole-dataset minimum to each split, so the BYOD contract is mis-stated (MOL-M4). |
| Promise fulfilment | The default promises are delivered and evidenced. The promised remote-code / `deterministic_eval` explanation in Section 3 is missing and contradicted (MOL-M2). "Running this cell twice gives exactly the same vectors" is shown as `False` (MOL-M3). BYOD from 8 records is not delivered (MOL-M4). "Run all" is not one pass (MOL-M1). |
| Scientific validity | The split is explicit and stratified, the baselines are fitted on train only, validation is monitoring only, and limits are stated. The model's 1.0 is matched by a one-character rule; Section 4 says so in a caveat, but the interpretation section does not carry it forward (MOL-m5). |
| Learner experience | Clear section prose and "Look for" notes, and an honest interpretation section. No audience, how-to-use, roadmap, prediction prompts, checkpoints, troubleshooting or conclusion template; three long carried-module cells are not labelled infrastructure (MOL-m1). The exercises produce no visible effect (MOL-m2). |
| Spec conformance | Fails RUN1/RUN10/ENV6 (restart) and REL2/REL11 (restart-dependent run recorded as a pass); MOD6 trust-boundary documentation contradicted in Section 3, SRC3 (knowingly wrong text); DAT12/DAT19/REL12 (BYOD minimum, late failure, no BYOD verification). SHOULD gaps: GDL1–3, GDL7, GDL9–11, GDL13, GDL14, EXE5, UX5, UX10. Declares spec 2.0, not 2.2. |

## 3. Promise and objective tracing

| Claim | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| Run all completes in a fresh runtime | cell 3 pip install + stale-import guard | Kaggle T4: restart after cell 3 (documented) | The opening's Run-all paragraph does not mention a restart | **Not met** (MOL-M1) |
| Snapshot staged and digest-verified, including the two executed `.py` files | cell 11 → `stage_missing_files`, `verify_snapshot`, `from_pretrained` | fetched 7, verified 7 | Section 3 markdown says "no remote model code is executed" | Implementation met; **explanation wrong** (MOL-M2) |
| Explain `trust_remote_code` and why `deterministic_eval` is overridden ("explained in Section 3") | Only in the opening and the carried module docstring | Section 3 mentions neither | — | **Not met** (MOL-M2) |
| 64 isomer molecules, validated, split 36/12/16 | cell 13 | as stated; formula equality asserted for one pair; character-level caveat printed | "Look for" note matches | Met |
| Tokenization, refusal of > 200 tokens, "three syntax rejections" | cell 15 | four rejections plus the 201-token refusal | Count in prose is off by one | Met (MOL-m4) |
| Deterministic embeddings: "running this cell twice gives exactly the same vectors" | cell 17 `repeat == vectors[0]` | `same_vector_on_a_second_call: False` | Contradicts the prose just above | **Not met as shown** (MOL-M3) |
| Majority and formula baselines on the test split | cell 19 | 0.5 / 0.25 | Explained, including why formula ≤ chance | Met |
| Bounded fine-tune with explicit hyperparameters | cell 21 `pipe.adapt` | 4 epochs, 8.28 M trainable, history printed | Monitoring-only validation explained | Met |
| Independent test evaluation with report | cell 23 | 1.0/1.0/1.0, report JSON | EVAL6 labelling present | Met |
| Inference on new molecules | cell 25 | 6 freshly generated molecules (default); first 6 **test-split** records (BYOD) | "sanity check, not an evaluation" | Met (default); BYOD wording off (MOL-m3) |
| Adapter export + fresh reload parity | cell 25 `save_artifact`, `from_artifact` | 12 serving-state tensors, parity 0.0, asserted | Explained | Met |
| BYOD ≥ 8 records / ≥ 3 per class through every stage | cells 13–27 | 24 records stop in Section 8; 40 records complete | — | **Not met for 8–35 records** (MOL-M4) |

| Objective | Learner activity | Evidence it was exercised |
|---|---|---|
| Understand what `trust_remote_code=True` buys and costs | Read Section 3 | Section 3 says the opposite (MOL-M2) |
| Why `deterministic_eval` is overridden | Read Section 6 output | Output says `False` (MOL-M3) |
| Measure baselines / evaluate on an independent split | Read printed tables | Explained in prose; no prediction prompt or checkpoint |
| Run a bounded fine-tune and compare settings | Optional experiments | Both produce the same 1.0 as the default (MOL-m2) |
| Export and reload | Section 10 | Parity asserted |

## 4. Journeys

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Well-written section prose and "Look for" notes. Section 3 contradicts the opening on remote code, and Section 6's output contradicts its own prose (MOL-M2, MOL-M3). Guided-layer gaps (MOL-m1). The install restart is not mentioned where Run all is promised (MOL-M1). |
| Clean default | Documented (Kaggle T4, reviewed blob) + direct (local CPU) | Passes after one restart (documented). Passes 13/13 on CPU with the install skipped (direct, not Colab). |
| Active learning | Direct (local CPU) | Both documented experiments run, and both give test 1.0/1.0/1.0, the same as the default. Exports other than the evaluation report stay at the default configuration (MOL-m2). |
| Reuse and recovery | Direct (local CPU, upload shim) | A 40-record CSV completes every stage with parity 0.0. A 24-record CSV that meets the stated contract fails in Section 8 (MOL-M4). Four invalid files are rejected clearly in Section 4; a cancelled upload gives a bare `StopIteration` (MOL-m3). Not verified on a hosted runtime. |

## 5. Findings

### Major

#### MOL-M1 — `Run all` needs a manual restart after the install cell, and the release record counts the restarted run as a pass

- **Cell/section:** Section 1 (cell 3); opening "Run all" paragraph; `docs/release-verification.md` (Manual clean-runtime evidence, Recorded executions, Current status); `STATUS.md`; `tutorials/README.md`; `README.md` "Release status".
- **Observed issue:** cell 3 `pip install`s `torch==2.14.0`, `numpy==2.5.3`, `transformers==5.17.0` … into the running kernel, then raises `RuntimeError … Restart the runtime, then rerun from the top` when a distribution that was already loaded changed. The Kaggle T4 run of this blob completed "after the expected fresh-process restart following dependency installation". The record still reports **PASSED — 13/13 code cells**, and the repository is Release-grade on that basis. The opening Run-all paragraph promises completion with no mention of a restart.
- **Consequence:** a learner's first Run all on a hosted image stops at cell 3. A run that depends on a restart is reported as a release pass.
- **Evidence:** documented (`release-verification.md`, both evidence tables). Source: cell 3; `tools/build_notebook.py` install cell (lines ~40–70, `pip install` at the `subprocess.run` line).
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11.
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass. The setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in `tools/build_notebook.py`, regenerate, re-qualify with a one-pass hosted Run all, and correct the release record (and `STATUS.md`, `README.md`, `tutorials/README.md`) so a restart-dependent run is not reported as a `Run all` PASS; return the status to Candidate until the one-pass run is recorded. The pinned `transformers==5.17.0` (needed by the upstream remote code) goes into the locked environment unchanged.
- **Acceptance check:** a fresh hosted Colab or Kaggle runtime executes every code cell in one pass with no restart and no error output. The release record cites that run's blob and no longer calls a restart expected.

#### MOL-M2 — Section 3 says "no remote model code is executed" and omits the remote-code and `deterministic_eval` explanation that three other places point to

- **Cell/section:** Section 3 markdown (cell 10); opening cell ("…explained in Section 3"); Prerequisites ("Section 3 explains why that is unavoidable"); Section 6 markdown ("Under the upstream default it would not — see Section 3"); `tools/build_notebook.py` line 465.
- **Observed issue:** Section 3's markdown ends "There is no fallback to a different download and no remote model code is executed." The cell it introduces calls `MolformerPipeline.from_pretrained`, which loads the model with `trust_remote_code=True` and executes the checkpoint's `configuration_molformer.py` and `modeling_molformer.py` (carried module cell 5; `remote_code_executed: True` in `result.json`). Section 3's markdown mentions neither `trust_remote_code` nor `deterministic_eval`, although the opening, the Prerequisites and Section 6 all send the learner there for that explanation. The explanation exists only in the opening and in the 34,000-character carried-module docstring. The sentence is generic generator text (`build_notebook.py:465`). The repository's `validate_remote_code_boundary` passes, because it checks where `trust_remote_code=True` appears in code, not what the markdown claims.
- **Consequence:** at the one step where the trust boundary is crossed, the learner reads that it is not crossed. A learner who reads only the section prose leaves with the wrong security model for this checkpoint, and the "understand what `trust_remote_code=True` buys and costs" objective has no section that delivers it.
- **Evidence:** source inspection; P1 (`section3_says_no_remote_code: true`, `section3_mentions_trust_remote_code: false`, `section3_mentions_deterministic_eval: false`, the three cross-references present). Direct: P2 executed both files after verification.
- **Spec:** MOD6 (trust boundary documented), SRC3 (knowingly stale instruction), UX1, UX2.
- **Recommended correction:** let the template override the Section 3 prose (or make the generator emit "no remote code" only when the pipeline does not require it). In `tools/notebook_template.py`, add Section 3 text that says the loader executes the two verified files, why that is unavoidable (`model_type: "molformer"`, no native implementation), what the digest check does and does not protect against (no security review of the upstream code), and why `deterministic_eval` is set to `True`. Extend `validate_remote_code_boundary` to fail on the phrase "no remote model code is executed" when `trust_remote_code=True` is present. Regenerate.
- **Acceptance check:** the regenerated Section 3 markdown contains no "no remote model code" claim, names both executed files and `trust_remote_code=True`, and explains the `deterministic_eval` override. The validator fails on a notebook that reintroduces the old sentence.

#### MOL-M3 — The determinism check compares two different computations and prints `False`, contradicting the prose; the release record describes an assertion that does not exist

- **Cell/section:** Section 6 (cell 17, template lines 203–214); `docs/release-verification.md` step 5 ("the determinism cell asserting that a second call returns the identical vector"); `tutorials/README.md` "Determinism (UNC5)".
- **Observed issue:** the markdown says "running this cell twice gives exactly the same vectors". The code embeds 8 validation molecules in one padded batch, then embeds the first molecule alone, and prints `repeat == vectors[0]`. On the default run this printed `'same_vector_on_a_second_call': False`. The model is deterministic: the same call repeated is identical (8-molecule and 1-molecule), and the molecule inside an equal-length, unpadded batch matches its single-call vector exactly (0.0). The difference (0.95e-6–2.7e-6 per molecule) comes from padding/batch shape in float32, not from redrawn random features. There is no `assert`, so the run continues, while the release procedure says the cell "asserts" identity. No hosted cell log exists to show what the Kaggle T4 run printed.
- **Consequence:** the notebook's central reproducibility lesson (why `deterministic_eval` is overridden) ends with output telling the learner that it does not work. A careful learner concludes that the override fails, or that the prose is wrong. The release record overstates what was checked.
- **Evidence:** direct (P2 cell 17 stdout; P6 decomposition). Source: cell 17; `release-verification.md` line 98.
- **Spec:** ENV8, ENV9, UX4, VER5 (load vs equivalent outputs), REL8 (the record describes a check that was not run as stated).
- **Recommended correction:** compare like with like: call `pipe.embed` on the same 8-molecule list twice and compare all vectors exactly, then `assert` it. Separately, report the single-vs-batched difference with an explicit tolerance and a one-line note that batch padding changes float rounding at about 1e-6, which is not the random-feature effect (upstream's `deterministic_eval: false` gives about 1e-3 per the module docstring). Optionally show the upstream-default difference for contrast. Fix in `tools/notebook_template.py` (Section 6 cell), regenerate, and correct `release-verification.md` step 5 and `tutorials/README.md`.
- **Acceptance check:** on the default path, Section 6 prints `True` for an identical repeated call and asserts it. The single-vs-batch difference, if shown, is printed with its tolerance and an explanation. The release procedure describes the check that the cell performs.

#### MOL-M4 — The stated BYOD minimum (8 records, 3 per class) is wrong; datasets of 8–35 records pass Section 4 and fail in Section 8 with a misleading message

- **Cell/section:** Prerequisites ("at least 8 records and 3 per class"); Section 4 (cell 13: `validate_dataset`, `split_dataset`); Section 8 (`pipe.adapt` → `validate_dataset(val_records, …)`, `pipeline.py:473`); Section 9 (`pipe.evaluate` → `validate_dataset(records, …)`, `pipeline.py:429`); `samples.py` `MIN_RECORDS = 8`, `MIN_RECORDS_PER_CLASS = 3`.
- **Observed issue:** `split_dataset` cuts 20 % validation and 25 % test per class. `adapt` then validates the validation split, and `evaluate` validates both evaluated splits, against the whole-dataset minimums (≥ 8 records, ≥ 3 per class). A balanced 2-class dataset therefore needs **36 records**; every size from 8 to 34 passes Section 4 and the split, then fails. With 24 records, Sections 4–7 ran (baselines included), and Section 8 stopped with `ValueError: dataset has 4 records; at least 8 are required`, while Section 4 had just reported a 24-record dataset. More classes or imbalance raise the real minimum further (each class needs ≥ 3 in validation and test, so ≥ 15 records in its smallest class).
- **Consequence:** the BYOD promise (validate → split → baselines → adapt → evaluate → infer → export) fails for most datasets the documentation admits. The failure comes late and names "dataset" where it means the validation split, so the learner cannot tell what to change.
- **Evidence:** direct (P4 24-record run; P5 arithmetic for 8–50 records); a 40-record set completed every stage.
- **Spec:** DAT12, DAT14, DAT19, VAL1, UX10, REL12.
- **Recommended correction:** decide the contract and make Section 4 enforce it before any model runs. Either validate the splits in Section 4 right after `split_dataset` (or inside it) with a message naming the split, its size and the minimum, and state the real minimum in the Prerequisites; or validate splits with split-appropriate minimums (for example ≥ 1 per class in validation, ≥ 2 per class in test) passed explicitly by `adapt`/`evaluate`. Change `samples.py`/`pipeline.py` and `tools/notebook_template.py` (Prerequisites line 108, Section 4 cell), regenerate, and record a hosted BYOD run (REL12).
- **Acceptance check:** a balanced 24-record CSV is either refused in Section 4 with a message naming the split and the real minimum, or completes Sections 4–11. The Prerequisites state a minimum that a balanced dataset of exactly that size satisfies end to end.

### Minor

#### MOL-m1 — Guided layer is partial

- **Cell/section:** opening cells; Sections 2–10; Interpretation and limits.
- **Observed issue:** "Look for" notes (Sections 1 and 4) and the interpretation section are strong. Missing: an intended-audience statement and **How to use this notebook** (GDL1–2), a roadmap (GDL3), an Input → Model → Output contract near the top (GDL4), prediction prompts before the baselines and the fine-tune (GDL7), interpretation checkpoints with sample answers (GDL9), troubleshooting for download, remote-code, memory and BYOD failures (GDL13), and a conclusion template (GDL14). The three carried-module cells (~56,000 characters) are introduced, but not labelled as optional **Infrastructure** that learners may run without studying (GDL11). The objectives use "understand" rather than observable verbs (GDL5).
- **Consequence:** self-paced learners get explanation but few points where they have to think before they read the answer, and the infrastructure looks like required reading.
- **Evidence:** source inspection; P1 keyword scan (no "how to use", "roadmap", "troubleshoot", "infrastructure", "audience", "conclusion").
- **Recommended correction:** add these in `tools/notebook_template.py` (intro, section markdown, closing) and regenerate.
- **Acceptance check:** the regenerated notebook has an audience / how-to-use block, a roadmap, at least two prediction prompts with collapsible answers, an Infrastructure label on the carried modules, a troubleshooting section and a conclusion template.

#### MOL-m2 — The optional experiments produce no visible effect, and the rerun leaves exports stale

- **Cell/section:** Interpretation and limits, "Optional experiments" (template line 445); Sections 8–11.
- **Observed issue:** "lower `EPOCHS` to 1 to see an under-trained head where AUROC may be high while accuracy sits near 0.5" — with `EPOCHS = 1`, test accuracy, macro-F1 and AUROC are all 1.0. "Set `TRAINABLE_LAYERS = 0` … and compare" also gives 1.0 / 1.0 / 1.0. Neither exercise says which cells to re-run. Re-running Sections 8–9 rewrites `evaluation_report.json`, while `result.json` and the adapter manifest keep the default configuration unless Sections 10–11 are re-run too.
- **Consequence:** the GUIDED comparison (UX5, GDL10) yields nothing to observe or explain, and the predicted outcome does not occur.
- **Evidence:** direct (P3).
- **Recommended correction:** choose experiments that move on this sample (for example, a deliberately mislabelled or harder control, or comparing against the character-level rule, MOL-m5), phrase them as Predict → Change → Run → Observe → Explain, and list the cells to re-run (8 → 11). Fix in `tools/notebook_template.py`.
- **Acceptance check:** each documented experiment, run as instructed on the default sample, produces a result that differs from the default run or that the text predicts correctly, and the instructions name the cells to re-run.

#### MOL-m3 — BYOD branch rough edges

- **Cell/section:** Section 4 (cell 13); Section 10 (cell 25).
- **Observed issue:** (a) cancelling the upload dialog raises a bare `StopIteration` from `next(iter(uploaded.items()))` (UX10). (b) The user's data is written to `outputs/molformer_chemistry_sample_dataset.csv`. (c) Under BYOD, "Inference on new molecules" classifies the first six **test-split** records, which were already scored in Section 9. They are disjoint from training (INF2 holds), but they are not new data, and the heading says otherwise. The default path also exposes no `BYOD_PATH` form field (EXE2).
- **Evidence:** direct (P4); source inspection.
- **Recommended correction:** check for an empty upload with a message saying to re-run the cell and choose a file; name the dataset export after the data source; let BYOD users supply separate new molecules (or label the output "held-out test molecules"); add a path field per EXE2.
- **Acceptance check:** a cancelled upload prints an actionable message; the BYOD export name does not say "sample"; the Section 10 output under BYOD names its source accurately.

#### MOL-m4 — Declarations and documentation drift

- **Observed issue:** the notebook declares NOTEBOOK_SPEC 2.0 (current 2.2). The Prerequisites render a literal `{{id, smiles, label}}` (template line 108 doubles braces in a string that is not `.format()`-ed). The install cell reads `DIMER_NOTEBOOK_CI_PREINSTALLED` without documenting it (EXE5). Section 5 promises "three syntax rejections" and prints four. Cell 15 prints `pipe.token_count` (which includes `<bos>`/`<eos>`) as `tokens` next to a ceiling that excludes them. `release-verification.md` step 5 lists a load report (remote-code requirement, 45,557,762 parameters) that cell 11 does not print; it appears only in Transformers' own log, together with an unexplained "You are using a model of type `molformer` to instantiate a model of type ``" warning and UNEXPECTED `lm_head.*` keys.
- **Evidence:** source inspection (P1); direct (P2 stdout).
- **Recommended correction:** update the spec declaration after a 2.2 conformance pass; fix the braces; document the variable in Section 1; correct the rejection count; label the token count; print `pipe.load_warnings` or a one-line load summary in Section 3 and explain the expected Transformers warnings, or describe the record accurately.
- **Acceptance check:** the regenerated notebook shows `{id, smiles, label}`, names the environment variable, and its Section 3 and Section 5 output matches the prose and the release procedure.

#### MOL-m5 — The perfect score is matched by a one-character rule, and the interpretation does not carry that caveat forward

- **Cell/section:** Section 4 caveat (cell 13 print); Section 7 (baselines); Interpretation and limits.
- **Observed issue:** Section 4 correctly prints that "a character-level baseline would separate these classes trivially". On this sample, `'(' in smiles → branched` scores 64/64 overall, 16/16 on the test split and 6/6 on the new molecules — the same as the fine-tuned model. The interpretation then states "That is the claim: the model's representation carries structure, and a bounded adaptation can read it", comparing only against the formula baseline. The saturation also explains MOL-m2.
- **Consequence:** a learner can conclude that MoLFormer's representation, rather than a trivially visible token, explains the result. The experiment shows that the adaptation contract works; it does not separate "reads structure" from "reads the branch token".
- **Evidence:** direct (P7, P2, P3).
- **Spec:** EVAL10 (meaningful trivial baseline), EVAL15.
- **Recommended correction:** add the character-level rule as a third baseline in `metrics.py` / Section 7, or rephrase the interpretation to say the sample cannot distinguish the two, and point to the BYOD path (with its formula baseline) for a real test.
- **Acceptance check:** Section 7 or the interpretation shows or states that a one-token rule reaches the same score on this sample.

### Suggestions

- **MOL-S1:** in Section 3, print a one-line load summary (remote code executed: yes, files verified, `deterministic_eval`, parameter counts) so the trust boundary is visible in output, not only in Transformers' log.
- **MOL-S2:** add a second seed or a bootstrap interval to the test metrics, so learners see that n = 16 gives a coarse estimate.
- **MOL-S3:** in Section 6, show the upstream `deterministic_eval=False` behaviour (two calls differing by about 1e-3) next to the override, so the reason for the override is observed rather than asserted.

## 6. Readiness

**Needs revision.** Four Major findings are open: a restart-dependent Run all (MOL-M1), a Section 3 that misstates the remote-code boundary (MOL-M2), a determinism demonstration that prints `False` (MOL-M3), and a BYOD minimum that fails as documented (MOL-M4). The applicable MUSTs RUN1/RUN10/ENV6/REL2/REL11, MOD6, SRC3, DAT12/DAT19 and REL12 are unmet. The default sample path itself is technically sound and evidenced. Remaining gates after fixes: a one-pass hosted Run all of the new blob (with its Section 6 output recorded), and a recorded hosted BYOD run.

## 7. Verified versus inferred

- **Verified (direct, local CPU, not Colab):** default path 13/13 with the recorded metrics after a fresh download; Section 6 printing `False` and its cause (padding, not redrawn features); both exercises saturating at 1.0; BYOD at 40 records completing and at 24 records failing in Section 8; the 36-record minimum arithmetic; invalid-file refusals; the one-character rule at 64/64.
- **Verified (documented):** the Kaggle T4 run of the reviewed blob needed a restart after cell 3.
- **Verified (source):** Section 3's "no remote model code is executed" sentence and the generator line that emits it.
- **Inferred:** that hosted Colab images also trigger the restart (no Colab run of this blob is recorded); that Section 6 also prints `False` on the Kaggle T4 GPU (no cell log is archived; GPU batched kernels differ from CPU, so the size and sign of the effect may differ).
- **Only Kurt can confirm:** whether BYOD is release-gated for this row (REL12), and learner-facing effectiveness (no learner observation).
- **Most likely to be wrong:** MOL-M3's severity on the hosted runtime. It is observed on CPU only. If the T4 path happens to give bit-identical vectors, the visible `False` would not appear there, and the finding would reduce to a mis-specified check plus an overstated release record (Minor).

*Probes: `molformer_chemistry_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).*
