"""Offline tests for the molecule dataset contract, isomer generator, splits, metrics and artifacts."""

from __future__ import annotations

import json

import pytest

from molformer_chemistry_pipeline import (
    ARTIFACT_FORMAT,
    ARTIFACT_MANIFEST_NAME,
    MAX_CARBONS,
    MIN_CARBONS,
    MIN_RECORDS,
    MIN_RECORDS_PER_CLASS,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_CLASSES,
    SAMPLE_SEED,
    SAMPLE_SIZE,
    MolformerPipeline,
    auroc,
    branch_positions,
    classification_metrics,
    dataset_digest,
    formula_baseline,
    formula_string,
    generate_sample_dataset,
    load_byod_dataset,
    majority_baseline,
    make_isomer_pair,
    molecular_formula,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)

# --- the isomer generator -------------------------------------------------------------------------


def test_make_isomer_pair_produces_constitutional_isomers():
    linear, branched = make_isomer_pair(8, 3, "O")
    assert linear == "CCCCCCCCO"
    assert branched == "CCC(C)CCCCO"
    assert molecular_formula(linear) == molecular_formula(branched) == {"C": 8, "O": 1}
    assert linear != branched


def test_make_isomer_pair_validates_its_arguments():
    with pytest.raises(ValueError, match=f"{MIN_CARBONS}..{MAX_CARBONS}"):
        make_isomer_pair(3, 2)
    with pytest.raises(ValueError, match="branch_at must be one of"):
        make_isomer_pair(8, 0)
    with pytest.raises(ValueError, match="terminus must be one of"):
        make_isomer_pair(8, 3, "Zz")


def test_branch_positions_exclude_the_ends():
    assert branch_positions(8) == [2, 3, 4, 5]
    assert all(0 < p < 8 for p in branch_positions(8))


def test_molecular_formula_handles_two_letter_and_bracket_atoms():
    assert molecular_formula("ClCCBr") == {"Cl": 1, "C": 2, "Br": 1}
    assert molecular_formula("C[C@H](N)C(=O)O") == {"C": 3, "O": 2, "N": 1}
    assert molecular_formula("c1ccccc1O") == {"C": 6, "O": 1}
    assert formula_string("CCCCCCCO") == "C7O1"


def test_sample_dataset_is_deterministic_balanced_and_valid():
    a = generate_sample_dataset()
    assert a == generate_sample_dataset(seed=SAMPLE_SEED, size=SAMPLE_SIZE)
    assert generate_sample_dataset(seed=7) != a
    manifest = validate_dataset(a)
    assert manifest["verdict"] == "accepted"
    assert manifest["classes"] == list(SAMPLE_CLASSES)
    assert manifest["class_counts"] == {c: SAMPLE_SIZE // 2 for c in SAMPLE_CLASSES}
    assert manifest["digest"] == dataset_digest(a)
    # The point of the sample: every formula appears in BOTH classes, so formula carries no label.
    assert manifest["distinct_formulas"] == SAMPLE_SIZE // 2
    assert manifest["formulas_shared_across_classes"] == SAMPLE_SIZE // 2
    assert "no cheminformatics toolkit" in manifest["validation"]


def test_sample_pairs_share_a_formula_and_differ_in_structure():
    records = generate_sample_dataset()
    pairs = [(records[2 * i], records[2 * i + 1]) for i in range(len(records) // 2)]
    for linear, branched in pairs:
        assert linear["label"] == "linear" and branched["label"] == "branched"
        assert molecular_formula(linear["smiles"]) == molecular_formula(branched["smiles"])
        assert "(" not in linear["smiles"] and "(C)" in branched["smiles"]


def test_sample_dataset_refuses_a_size_it_cannot_fill_uniquely():
    with pytest.raises(ValueError, match="distinct"):
        generate_sample_dataset(size=1000)
    with pytest.raises(ValueError, match="even number"):
        generate_sample_dataset(size=7)


# --- dataset validation ---------------------------------------------------------------------------


def _records(n_per_class=MIN_RECORDS, classes=("x", "y")):
    out = []
    for c_i, c in enumerate(classes):
        for i in range(n_per_class):
            out.append({"id": f"{c}-{i}", "smiles": "C" * (i + 4) + ("O" if c_i else "N"), "label": c})
    return out


def test_validate_dataset_rejections_are_actionable():
    good = _records()
    assert validate_dataset(good)["classes"] == ["x", "y"]
    with pytest.raises(TypeError, match="list of"):
        validate_dataset({"id": 1})
    with pytest.raises(ValueError, match=f"at least {MIN_RECORDS}"):
        validate_dataset(good[:2])
    bad = [dict(r) for r in good]
    del bad[0]["label"]
    with pytest.raises(ValueError, match=r"missing required column\(s\) \['label'\]"):
        validate_dataset(bad)
    bad = [dict(r) for r in good]
    bad[1]["id"] = bad[0]["id"]
    with pytest.raises(ValueError, match="duplicates id"):
        validate_dataset(bad)
    bad = [dict(r) for r in good]
    bad[2]["smiles"] = "CC(C"
    with pytest.raises(ValueError, match="failed SMILES syntax checks"):
        validate_dataset(bad)
    bad = [dict(r) for r in good]
    bad[3]["smiles"] = bad[4]["smiles"]
    with pytest.raises(ValueError, match="duplicates the SMILES"):
        validate_dataset(bad)
    bad = [dict(r) for r in good]
    bad[4]["smiles"] = " CCO"
    with pytest.raises(ValueError, match="whitespace"):
        validate_dataset(bad)
    one_class = [dict(r, label="x") for r in good]
    with pytest.raises(ValueError, match="at least 2 classes"):
        validate_dataset(one_class)
    thin = good + [{"id": "z-0", "smiles": "CCCCCCCCCCCCCS", "label": "z"}]
    with pytest.raises(ValueError, match=f"fewer than {MIN_RECORDS_PER_CLASS}"):
        validate_dataset(thin)
    with pytest.raises(ValueError, match="not in the class list"):
        validate_dataset(good, classes=["x"])


def test_split_is_stratified_disjoint_and_seeded():
    records = generate_sample_dataset()
    s1 = split_dataset(records, seed=42)
    assert s1 == split_dataset(records, seed=42)
    assert split_dataset(records, seed=1) != s1
    ids = [r["id"] for part in s1.values() for r in part]
    assert len(ids) == len(set(ids)) == len(records)
    for part in s1.values():
        labels = [r["label"] for r in part]
        assert labels.count("linear") == labels.count("branched")
    assert {k: len(v) for k, v in s1.items()} == {"train": 36, "validation": 12, "test": 16}
    with pytest.raises(ValueError, match="sum to less than 1"):
        split_dataset(records, val_fraction=0.6, test_fraction=0.5)


# --- metrics and baselines --------------------------------------------------------------------------


def test_auroc_handles_ties_and_degenerate_labels():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc([0, 1, 0, 1], [0.5] * 4) == 0.5
    assert auroc([1, 1], [0.1, 0.2]) is None


def test_classification_metrics_binary():
    y = ["branched", "linear", "linear", "branched"]
    p = ["branched", "linear", "branched", "branched"]
    scores = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.5, 0.5]]
    m = classification_metrics(y, p, scores, list(SAMPLE_CLASSES))
    assert m["n"] == 4 and m["accuracy"] == 0.75
    assert m["per_class"]["linear"]["recall"] == 0.5
    assert "positive class 'linear'" in m["auroc_definition"]
    with pytest.raises(ValueError, match="outside the class list"):
        classification_metrics(y, ["other"] * 4, None, list(SAMPLE_CLASSES))


def test_formula_baseline_cannot_separate_constitutional_isomers():
    records = generate_sample_dataset()
    splits = split_dataset(records, seed=42)
    classes = list(SAMPLE_CLASSES)
    maj = majority_baseline(splits["train"], splits["test"], classes)
    assert maj["baseline"] == "majority-class" and maj["accuracy"] == 0.5
    fb = formula_baseline(splits["train"], splits["test"], classes)
    assert fb["baseline"] == "molecular-formula lookup"
    # Every training formula that appears at all appears in both classes, so the lookup is a coin
    # flip where it applies and a fallback everywhere else.
    assert fb["ambiguous_train_formulas"] > 0
    assert fb["accuracy"] <= 0.6


# --- BYOD loaders -----------------------------------------------------------------------------------


def test_byod_csv_roundtrip_and_rejections(tmp_path):
    records = generate_sample_dataset(size=16)
    path = write_dataset_csv(records, tmp_path / "molecules.csv")
    assert load_byod_dataset(path) == records
    (tmp_path / "bad.csv").write_text("id,structure,label\nm1,CCO,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing required column\(s\) \['smiles'\]"):
        load_byod_dataset(tmp_path / "bad.csv")
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_byod_dataset(tmp_path / "empty.csv")
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "nope.csv")
    sdf = tmp_path / "molecules.sdf"
    sdf.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported BYOD file type"):
        load_byod_dataset(sdf)


def test_byod_json_and_jsonl(tmp_path):
    records = generate_sample_dataset(size=16)
    (tmp_path / "d.json").write_text(json.dumps(records), encoding="utf-8")
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "d.json") == records
    assert load_byod_dataset(tmp_path / "d.jsonl") == records
    (tmp_path / "obj.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(TypeError, match="top-level array"):
        load_byod_dataset(tmp_path / "obj.json")
    (tmp_path / "broken.jsonl").write_text('{"id": 1}\n{bad\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 is not valid JSON"):
        load_byod_dataset(tmp_path / "broken.jsonl")


# --- artifact manifest checks -------------------------------------------------------------------------


def _fake_loaded(tmp_path):
    pipe = MolformerPipeline(lambda mols: [[0.0] * 768 for _ in mols], "cpu")
    pipe.model = object()
    pipe.weights_dir = tmp_path
    return pipe


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path):
    pipe = _fake_loaded(tmp_path)
    art = tmp_path / "adapter"
    art.mkdir()
    base = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION}
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        pipe.load_artifact(art)
    (art / ARTIFACT_MANIFEST_NAME).write_text(json.dumps({"format": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(art)
    (art / ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps({"format": ARTIFACT_FORMAT, "base_model": {**base, "model_revision": "0" * 40}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="this package pins"):
        pipe.load_artifact(art)
    (art / ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps({"format": ARTIFACT_FORMAT, "base_model": base, "classes": ["only"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="at least two unique classes"):
        pipe.load_artifact(art)
    (art / ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "format": ARTIFACT_FORMAT,
                "base_model": base,
                "classes": list(SAMPLE_CLASSES),
                "files": [{"path": "adapter.safetensors", "bytes": 1, "sha256": "0" * 64}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="artifact file missing"):
        pipe.load_artifact(art)
    (art / "adapter.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        pipe.load_artifact(art)
