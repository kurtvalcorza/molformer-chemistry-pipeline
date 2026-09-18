"""Offline tests for the public validation-stage helpers (input manifest and dataset manifest)."""

from __future__ import annotations

from molformer_chemistry_pipeline import (
    INPUT_SCHEMA,
    MAX_MOLECULES_PER_CALL,
    MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    generate_sample_dataset,
    validate_dataset,
    validate_inputs,
)


def test_validate_inputs_returns_manifest_with_schema_and_identity():
    manifest = validate_inputs(["CCCCCCCO", "CCCC(C)CCO"], names=["m1", "m2"])
    assert manifest["verdict"] == "accepted" and manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["molecules"] == [1, MAX_MOLECULES_PER_CALL]
    assert manifest["schema"]["tokens_per_molecule"] == [1, MAX_TOKENS]
    assert "no cheminformatics toolkit" in manifest["schema"]["validation"]
    assert manifest["n_molecules"] == 2
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert manifest["requires_remote_code"] is True


def test_validate_inputs_default_ids():
    manifest = validate_inputs(["CCO"])
    assert [row["id"] for row in manifest["inputs"]] == ["mol-0"]


def test_dataset_manifest_reports_ceilings_and_formula_overlap():
    manifest = validate_dataset(generate_sample_dataset())
    assert manifest["ceilings"]["max_tokens"] == MAX_TOKENS
    assert manifest["formulas_shared_across_classes"] == manifest["distinct_formulas"]
    assert len(manifest["digest"]) == 64
