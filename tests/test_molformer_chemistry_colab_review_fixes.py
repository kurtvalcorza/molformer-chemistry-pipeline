"""Regression tests for the Notebook Review Framework v1 findings on molformer_chemistry_colab (MOL-M1..M4,
MOL-m1..m5; review of 2026-10-02, docs/reviews/2026-10-02-notebook-review/). CI dependencies only: no torch, no
transformers, no weights. Notebook cells that need a model run against a stand-in pipeline object."""
# ruff: noqa: E501  -- notebook source fragments are kept on single lines

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

from molformer_chemistry_pipeline import samples

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook_mol_review", TOOLS / "build_notebook.py")
validator = _load("validate_release_assets_mol_review", TOOLS / "validate_release_assets.py")
TEMPLATE = _load("notebook_template_mol_review", TOOLS / "notebook_template.py").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]
LOCK = ROOT / TEMPLATE["lock"]


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _code_with(nb: dict, needle: str) -> str:
    cells = [_src(c) for c in nb["cells"] if c["cell_type"] == "code" and needle in _src(c)]
    assert len(cells) == 1, needle
    return cells[0]


def _section_md(nb: dict, heading: str) -> str:
    cells = [_src(c) for c in nb["cells"] if c["cell_type"] == "markdown" and _src(c).startswith(heading)]
    assert len(cells) == 1, heading
    return cells[0]


# --- MOL-M1: one-pass Run all; the lock covers every runtime import -------------------------------------------

# Top-level imports the default path executes: the carried modules, the notebook cells, the checkpoint's two
# remote-code files (configuration_molformer.py: transformers; modeling_molformer.py: math, typing, torch,
# transformers -- read at revision 361063d0, no requires_backends, no RDKit), and the native fast tokenizer.
RUNTIME_DISTRIBUTIONS = ("torch", "transformers", "tokenizers", "safetensors", "huggingface-hub", "numpy")


def test_mol_m1_no_kernel_install_no_restart(nb: dict) -> None:
    code = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "code")
    assert "pip install" not in code and "'-m', 'pip', 'install'" not in code
    assert "Restart the runtime" not in code
    assert TEMPLATE["isolated_runtime"] is True


def test_mol_m1_lock_covers_every_runtime_distribution() -> None:
    locked = build.lock_packages(LOCK.read_text(encoding="utf-8"))
    missing = [d for d in RUNTIME_DISTRIBUTIONS if build._canonical(d) not in locked]
    assert not missing, f"lock misses runtime distributions {missing}"
    assert locked["transformers"] == "5.17.0"


def test_mol_m1_lock_covers_every_src_import() -> None:
    locked = build.lock_packages(LOCK.read_text(encoding="utf-8"))
    import_to_dist = {"huggingface_hub": "huggingface-hub"}
    stdlib = set(__import__("sys").stdlib_module_names)
    for module in (ROOT / "src" / "molformer_chemistry_pipeline").glob("*.py"):
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                if top in stdlib or top == "__future__":
                    continue
                dist = build._canonical(import_to_dist.get(top, top))
                assert dist in locked, f"{module.name} imports {top}, which the lock does not install"


def test_mol_m1_release_record_no_longer_counts_the_restart_run() -> None:
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "expected fresh-process restart" not in record
    assert "Restart-dependent — not a one-pass Run all" in record
    assert "Current status: **Candidate**" in (ROOT / "STATUS.md").read_text(encoding="utf-8")


# --- MOL-M2: Section 3 states the trust boundary it crosses -------------------------------------------------


def test_mol_m2_section3_explains_remote_code_and_deterministic_eval(nb: dict) -> None:
    section3 = _section_md(nb, "## 3. Pin, stage and verify the model")
    assert "no remote model code is executed" not in _markdown(nb)
    for needed in ("trust_remote_code=True", "configuration_molformer.py", "modeling_molformer.py", "deterministic_eval", "does **not** prove"):
        assert needed in section3, needed
    load = _code_with(nb, "fetched = stage_missing_files(WEIGHTS_DIR, allow_download=True)")
    assert "'remote_code_executed': True" in load and "'upstream_deterministic_eval': upstream_deterministic_eval" in load


def test_mol_m2_generator_default_sentence_unchanged_without_the_key() -> None:
    template = {k: v for k, v in TEMPLATE.items() if k not in ("load_trust_note", "load_summary")}
    rendered = build.render(ROOT, template, revision="0" * 40)
    section3 = next(_src(c) for c in rendered["cells"] if c["cell_type"] == "markdown" and _src(c).startswith("## 3."))
    assert "There is no fallback to a different download and no remote model code is executed." in section3


def test_mol_m2_validator_rejects_the_old_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    real_read = validator._read
    text = NOTEBOOK.read_text(encoding="utf-8")
    nb = json.loads(text)
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown" and _src(cell).startswith("## 3."):
            cell["source"] = _src(cell) + " There is no fallback to a different download and no remote model code is executed."
    tampered = json.dumps(nb)
    monkeypatch.setattr(validator, "_read", lambda path: tampered if Path(path).name == NOTEBOOK.name else real_read(path))
    with pytest.raises(validator.ValidationError, match="no remote model code is executed"):
        validator.validate_remote_code_boundary()
    monkeypatch.setattr(validator, "_read", real_read)
    validator.validate_remote_code_boundary()


# --- MOL-M3: the determinism check compares like with like and asserts it ------------------------------------


class _StandInPipe:
    def __init__(self, drift: float = 0.0) -> None:
        self.calls = 0
        self.drift = drift

    def embed(self, smiles, names=None):
        self.calls += 1
        noise = self.drift * self.calls
        vectors = [[float(len(s)) + 0.5 * k + noise for k in range(4)] for s in smiles]
        if len(smiles) == 1:
            vectors[0][0] += 1e-6  # the padded-batch effect: tiny, not random features
        return {"embeddings": vectors, "n_molecules": len(smiles), "dimension": 4, "tokens": [len(s) for s in smiles], "pooling": "mean"}


def _run_section6(nb: dict, pipe: _StandInPipe, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    import contextlib
    import io

    code = _code_with(nb, "second_call = pipe.embed(")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    records = [{"id": f"m{i}", "smiles": "C" * (5 + i) + ("(C)" if i % 2 else ""), "label": "branched" if i % 2 else "linear"} for i in range(8)]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "section6", "exec"), {"pipe": pipe, "val_records": records, "DETERMINISTIC_EVAL": True})
    return out.getvalue()


def test_mol_m3_identical_repeated_call_prints_true_and_asserts(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = _run_section6(nb, _StandInPipe(), tmp_path, monkeypatch)
    assert "'identical_vectors_on_a_second_identical_call': True" in out
    assert "'tolerance': 0.0001" in out and "'within_tolerance': True" in out
    assert "same_vector_on_a_second_call" not in out


def test_mol_m3_redrawn_features_fail_the_assertion(nb: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(AssertionError, match="deterministic_eval=True"):
        _run_section6(nb, _StandInPipe(drift=1e-3), tmp_path, monkeypatch)


def test_mol_m3_release_procedure_describes_the_check() -> None:
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "identical_vectors_on_a_second_identical_call: True" in record
    assert "a second call returns the identical vector" not in record


# --- MOL-M4: the BYOD minimum is real and enforced before any model runs -------------------------------------


def _balanced(per_class: int) -> list[dict]:
    rows = samples.generate_sample_dataset(seed=3, size=40)
    lin = [r for r in rows if r["label"] == "linear"][:per_class]
    bra = [r for r in rows if r["label"] == "branched"][:per_class]
    return lin + bra


def test_mol_m4_minimum_is_36_for_two_balanced_classes() -> None:
    assert samples.minimum_records_per_class(2) == 18


@pytest.mark.parametrize("per_class", [4, 12, 17])
def test_mol_m4_too_small_dataset_refused_at_split_naming_split_and_minimum(per_class: int) -> None:
    records = _balanced(per_class)
    samples.validate_dataset(records)  # passes the whole-dataset contract, as in the review's 24-record probe
    with pytest.raises(ValueError, match=r"the (validation|test) split would have \d+ records.*at least 18 records per class \(36 in total\)"):
        samples.split_dataset(records)


def test_mol_m4_stated_minimum_survives_adapt_and_evaluate_checks() -> None:
    splits = samples.split_dataset(_balanced(18))
    for part in ("validation", "test"):
        samples.validate_dataset(splits[part], classes=["branched", "linear"])  # what adapt / evaluate run
    assert {k: len(v) for k, v in splits.items()} == {"train": 20, "validation": 8, "test": 8}


def test_mol_m4_default_split_unchanged() -> None:
    splits = samples.split_dataset(samples.generate_sample_dataset())
    assert {k: len(v) for k, v in splits.items()} == {"train": 36, "validation": 12, "test": 16}


def test_mol_m4_prerequisites_state_the_real_minimum(nb: dict) -> None:
    md = _markdown(nb)
    assert "at least 36 records (18 per class)" in md
    assert "at least 8 records and 3 per class, 2..20 classes" not in md


# --- Minor findings ---------------------------------------------------------------------------------------


def test_mol_m1_minor_guided_layer_and_objective_verbs(nb: dict) -> None:
    md = _markdown(nb).lower()
    for marker in ("who this notebook is for", "how to use this notebook", "roadmap", "predict before running", "check your reasoning", "troubleshooting", "conclusion", "infrastructure"):
        assert marker in md, marker
    objectives = TEMPLATE["learning_objectives"]
    assert "understand" not in objectives


def test_mol_m2_minor_experiments_move_and_name_the_cells(nb: dict) -> None:
    closing = _section_md(nb, "## Interpretation and limits")
    assert "`LEARNING_RATE = 1e-6`" in closing
    assert "Sections 8, 9, 10 and 11 in order" in closing
    assert "AUROC may be high while accuracy sits near 0.5" not in closing
    assert "Lower `EPOCHS` to 1 to see a less saturated run" not in _markdown(nb)


def test_mol_m3_minor_byod_names(nb: dict) -> None:
    section4 = _code_with(nb, "dataset_manifest = validate_dataset(records)")
    assert "'outputs/molformer_chemistry_byod_dataset.csv' if USE_BYOD else 'outputs/molformer_chemistry_sample_dataset.csv'" in section4
    assert "if len(uploaded) != 1:" in section4  # a cancelled dialog no longer raises a bare StopIteration
    section10 = _code_with(nb, "inference_result = pipe.classify(")
    assert "already scored in Section 9; not new data" in section10


def test_mol_m4_minor_declarations_and_counts(nb: dict) -> None:
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.2"
    md = _markdown(nb)
    assert "{{id, smiles, label}}" not in md and "{id, smiles, label}" in md
    assert "DIMER_NOTEBOOK_CI_PREINSTALLED" in md
    assert "four syntax rejections" in md and "three syntax rejections" not in md
    section5 = _code_with(nb, "for probe in [")
    probes = ast.literal_eval(re.search(r"for probe in (\[.*?\]):", section5).group(1))
    assert len(probes) == 4
    assert "'tokens_incl_bos_eos': pipe.token_count(smiles)" in section5


def test_mol_m5_minor_one_character_rule_shown_and_carried_forward(nb: dict) -> None:
    section7 = _code_with(nb, "baseline_formula = formula_baseline(")
    assert "rule_correct = sum(('(' in r['smiles']) == (r['label'] == 'branched') for r in test_records)" in section7
    records = samples.generate_sample_dataset()
    test = samples.split_dataset(records)["test"]
    assert sum(("(" in r["smiles"]) == (r["label"] == "branched") for r in test) == len(test)  # 16/16, as the prose says
    closing = _section_md(nb, "## Interpretation and limits")
    assert "one-character rule" in closing
