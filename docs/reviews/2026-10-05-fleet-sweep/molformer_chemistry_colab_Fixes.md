# molformer_chemistry_colab — fleet-sweep fixes (2026-10-05)

Targeted fix of the 2026-10-05 fleet sweep findings. There is no full Notebook Review Framework v1 report for this
notebook; each flag was first confirmed in the cell source on `main` (`12ba808`). All changes are made in the
generator (`tools/build_notebook.py`, `tools/notebook_template*.py`) and the notebook is regenerated. STATUS and the
release labels are unchanged. **Readiness: Verification pending** (hosted Run all not yet done).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R (restart guard) | Fixed — hosted confirmation pending | Confirmed (the recorded Kaggle T4 run of 2026-09-18 notes "one expected fresh-process restart followed the install cell"): Section 1 ran `pip install` into the kernel and raised "Restart the runtime" on stale modules. Generator upgraded to `build_notebook.py/2.2` (the fleet isolated runtime): one kernel cell downloads the pinned `uv` 0.12.15 wheel (size + SHA-256), builds a managed CPython 3.12.12 environment from the new hash lock `tutorials/requirements-colab.lock.txt` (`--require-hashes --only-binary :all:`), and routes every later cell to one persistent worker. The environment folder is keyed on the lock digest and reused by a re-run or a second Run all; re-running Section 1 keeps the live worker and its variables; the worker gets `MPLBACKEND=Agg` and no `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. | Section 1 (kernel cell + "Record the runtime"); `tools/build_notebook.py`; `tutorials/requirements-colab.lock.txt`; `tools/validate_release_assets.py` (install markers, bootstrap check, kernel cell excluded from the library-use scan); `docs/release-verification.md` (the line describing that check) | `test_swp_r_no_pip_install_or_restart_in_any_cell`, `test_swp_r_lock_is_carried_hash_locked_and_matches_pins`, `test_swp_r_environment_keyed_on_lock_and_child_env_cleaned`, `test_swp_r_section1_reuses_environment_and_worker_when_rerun` (executes the notebook's own kernel cell with a stand-in IPython shell; the worker runs on the test interpreter) |
| SWP-G (guided layer) | Fixed | Confirmed: GUIDED mode with 1 of 9 guided markers. Added a model-specific guided layer: audience and Input → Model → Output table, How to use this notebook, roadmap, a Learner prerequisite, five **Predict before running** prompts with **Check your reasoning** answers (Sections 4, 5, 7, 9, 10) quoting the recorded runs (Kaggle T4 2026-09-18 and the CPU pre-flight: splits 36/12/16, token counts 10/6/13, baselines 0.5/0.3333 and 0.25/0.2, test accuracy / macro-F1 / AUROC 1.0 at n = 16, reload difference 0.0) and naming the saturation, Troubleshooting, Glossary and a Conclusion template. Sections 1–3 labelled Infrastructure and collapsed. The literal `{{id, smiles, label}}` in the data-contract prerequisite now renders as `{id, smiles, label}`. | Template opening, prerequisites, Sections 4–10, closing | `test_swp_g_guided_layer_present`, `test_swp_g_infrastructure_cells_labelled_and_collapsed`, `test_swp_g_no_leftover_placeholders` |
| SWP-A (quality asserts) | Not fixed — finding does not reproduce | The sweep counted one quality assert; the notebook's asserts are the isomer-pair formula equality in Section 4 (a property of the generated data) and the reload-parity checks in Section 10 (`labels equal`, `max_score_diff < 1e-5`). Both are contract-integrity checks, which the brief keeps; no assert compares a metric with a baseline. | — | `test_swp_a_remaining_asserts_are_contract_checks` |
| SWP-F (frozen re-run) | Not applicable | `pipe.adapt` builds a fresh `AutoModelForSequenceClassification` from the verified checkpoint on every call; the base encoder used for embeddings is never trained in place. Stated in the closing. | Closing (one sentence) | `src/.../pipeline.py` `adapt` |
| SWP-B (BYOD) | Fixed | BYOD worked only through `files.upload()`, and an empty upload raised a bare `StopIteration`. Added a `BYOD_PATH` form field that works on Colab, Kaggle and Jupyter, with the Colab upload as a guarded fallback; refusals name the path, the expected suffix (`.csv`, `.json`, `.jsonl`) and the rule; `load_byod_dataset` / `validate_dataset` keep naming the record and rule. | Section 4, BYOD declaration, prerequisites | `test_swp_b_byod_path_reads_file_and_refuses_with_names`, `test_swp_b_cancelled_colab_upload_gives_a_clear_message`, `test_swp_b_byod_path_fields_default_off` |

## User-visible changes

- Section 1 no longer installs into the notebook's Python and never asks for a restart; it builds (first run) or reuses `dimer_isolated_env_<lock digest>/` and every later code cell runs there. Linux x86_64 runtimes only.
- New form field `BYOD_PATH` (Section 4); on Colab an empty path still opens the upload dialog.
- Guided material added; Sections 1–3 collapsed.

## Verification (offline; not clean-runtime evidence)

- Real input: none of the model stages could run here (the Hugging Face Hub is unreachable and torch is not installed).
- Stand-in: the kernel-cell test runs the generated bootstrap against a pre-built environment folder whose `python` is the test interpreter (routing, reuse and idempotence are real; the managed CPython and locked packages are stand-ins).
- `python tools/build_notebook.py --check`: OK. `python tools/validate_release_assets.py`: PASS (including remote-code-boundary). `ruff check src tests tools`: clean.
- `pytest` with CI's dependencies (pytest and ruff only): 49 passed before → 61 passed after.
- Every code cell of the regenerated notebook parses (`test_swp_r_every_code_cell_parses`).

## Remaining gates

- A hosted **Run all in one pass** on a fresh runtime (expected: no restart prompt; Section 1 builds the environment; a second Run all reports `'reused': True`).
- The REL12 BYOD run with the BYOD gates and path fields set.
- A full Notebook Review Framework v1 review has not been done.
