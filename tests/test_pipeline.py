"""Offline pipeline tests: identity, snapshot verification (including the remote-code files),
SMILES syntax checks, input validation, and the embed/classify/artifact contracts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from molformer_chemistry_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    DETERMINISTIC_EVAL,
    HIDDEN_SIZE,
    MANIFEST_NAME,
    MAX_MOLECULES_PER_CALL,
    MAX_POSITION_EMBEDDINGS,
    MAX_TOKENS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    REMOTE_CODE_FILES,
    VOCAB_SIZE,
    MolformerPipeline,
    check_smiles_syntax,
    stage_missing_files,
    validate_inputs,
    verify_snapshot,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "weights" / MODEL_KEY / MANIFEST_NAME
SMILES = ["CCCCCCCO", "CCCC(C)CCO"]


def _fake(classes=None, logits=None, token_counter=None):
    pipe = MolformerPipeline(lambda mols: [[float(len(m))] * HIDDEN_SIZE for m in mols], "cpu")
    pipe._token_counter = token_counter or (lambda smiles: len(smiles) + 2)
    if classes is not None:
        pipe.classes = list(classes)
        rows = logits or [[0.0, 1.0]] * MAX_MOLECULES_PER_CALL
        pipe._classifier = lambda mols: [rows[i] for i in range(len(mols))]
    return pipe


def test_identity_constants_and_manifest():
    assert re.fullmatch(r"[0-9a-f]{40}", MODEL_REVISION)
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert DETERMINISTIC_EVAL is True
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert manifest["modelId"] == MODEL_ID
        assert manifest["revision"] == MODEL_REVISION
        paths = {f["path"] for f in manifest["files"]}
        assert "model.safetensors" in paths
        # The loader executes these; their digests must be inside the verified perimeter.
        assert set(REMOTE_CODE_FILES) <= paths
    config = REPO / "weights" / MODEL_KEY / "config.json"
    if config.is_file():
        cfg = json.loads(config.read_text(encoding="utf-8"))
        assert cfg["hidden_size"] == HIDDEN_SIZE
        assert cfg["vocab_size"] == VOCAB_SIZE
        assert cfg["max_position_embeddings"] == MAX_POSITION_EMBEDDINGS
        assert MAX_TOKENS == MAX_POSITION_EMBEDDINGS - 2
        # Upstream ships stochastic evaluation; this package overrides it and says so.
        assert cfg["deterministic_eval"] is False


def _write_snapshot(tmp_path: Path, content: bytes, sha256: str, revision: str = MODEL_REVISION,
                    include_code: bool = True) -> Path:
    files = [{"path": "config.json", "bytes": len(content), "sha256": sha256}]
    (tmp_path / "config.json").write_bytes(content)
    if include_code:
        for name in REMOTE_CODE_FILES:
            (tmp_path / name).write_bytes(content)
            files.append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"modelId": MODEL_ID, "revision": revision, "files": files}
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_verify_snapshot_accepts_and_rejects(tmp_path):
    content = b'{"hidden_size": 768}'
    good = hashlib.sha256(content).hexdigest()
    assert verify_snapshot(_write_snapshot(tmp_path, content, good))["revision"] == MODEL_REVISION
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(_write_snapshot(tmp_path, content, "0" * 64))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(_write_snapshot(tmp_path, content, good, revision="0" * 40))
    root = _write_snapshot(tmp_path, content, good)
    (root / "config.json").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_snapshot(root)


def test_verify_snapshot_refuses_a_manifest_without_the_remote_code_files(tmp_path):
    content = b"{}"
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest(), include_code=False)
    with pytest.raises(ValueError, match="remote-code files"):
        verify_snapshot(root)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path):
    content = b"{}"
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest())
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["files"].append(
        {"path": "tokenizer.json", "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    )
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    fetched = []

    def downloader(rel, dst):
        fetched.append(rel)
        (dst / rel).write_bytes(b"abc")

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == ["tokenizer.json"]
    assert fetched == ["tokenizer.json"]
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == []


def test_stage_refuses_manifest_for_another_model(tmp_path):
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps({"modelId": "other/model", "revision": MODEL_REVISION, "files": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True)


# --- SMILES syntax checks ------------------------------------------------------------------------


def test_check_smiles_syntax_accepts_ordinary_molecules():
    for smiles in ("CCCCCCCO", "CCCC(C)CCO", "c1ccccc1O", "ClCCBr", "C[C@H](N)C(=O)O", "CC(=O)[O-]"):
        assert check_smiles_syntax(smiles) == [], smiles


def test_check_smiles_syntax_reports_each_kind_of_defect():
    assert "unclosed" in check_smiles_syntax("CCCC(CCO")[0]
    assert "closing" in check_smiles_syntax("CCCC)CCO")[0]
    assert "unclosed" in check_smiles_syntax("CC[C@HCC")[0]
    assert "unpaired" in check_smiles_syntax("c1ccccO")[0]
    assert "outside the accepted SMILES set" in check_smiles_syntax("CC*C")[0]


def test_check_smiles_syntax_does_not_claim_chemical_validity():
    # A syntactically fine but chemically impossible molecule passes: the checker is a string check.
    assert check_smiles_syntax("CCCCCCCCCC(C)(C)(C)(C)C") == []


# --- input validation ----------------------------------------------------------------------------


def test_validate_inputs_manifest_and_rejections():
    manifest = validate_inputs(SMILES, names=["a", "b"])
    assert manifest["verdict"] == "accepted"
    assert manifest["requires_remote_code"] is True
    assert manifest["token_ceiling_checked"] is False
    assert [row["id"] for row in manifest["inputs"]] == ["a", "b"]
    assert manifest["inputs"][0]["characters"] == len(SMILES[0])
    with pytest.raises(TypeError, match="list of SMILES"):
        validate_inputs("CCO")
    with pytest.raises(TypeError, match="must be str"):
        validate_inputs([123])
    with pytest.raises(ValueError, match="is empty"):
        validate_inputs([""])
    with pytest.raises(ValueError, match="whitespace"):
        validate_inputs([" CCO"])
    with pytest.raises(ValueError, match="failed SMILES syntax checks"):
        validate_inputs(["CC(C"])
    with pytest.raises(ValueError, match=f"1..{MAX_MOLECULES_PER_CALL}"):
        validate_inputs(["CCO"] * (MAX_MOLECULES_PER_CALL + 1))
    with pytest.raises(ValueError, match="exactly one id per molecule"):
        validate_inputs(SMILES, names=["a"])
    with pytest.raises(ValueError, match="names must be unique"):
        validate_inputs(SMILES, names=["a", "a"])


def test_validate_inputs_checks_the_token_ceiling_when_given_a_counter():
    manifest = validate_inputs(["CCO"], token_counter=lambda s: len(s) + 2)
    assert manifest["token_ceiling_checked"] is True
    assert manifest["inputs"][0]["tokens"] == 3
    with pytest.raises(ValueError, match=f"ceiling is {MAX_TOKENS}"):
        validate_inputs(["C" * (MAX_TOKENS + 1)], token_counter=lambda s: len(s) + 2)


# --- embed / classify / artifact contracts --------------------------------------------------------


def test_embed_contract_with_injected_backend():
    out = _fake().embed(SMILES, names=["a", "b"])
    assert out["ids"] == ["a", "b"] and out["dimension"] == HIDDEN_SIZE
    assert len(out["embeddings"][0]) == HIDDEN_SIZE
    assert out["tokens"] == [len(SMILES[0]), len(SMILES[1])]
    assert out["model_revision"] == MODEL_REVISION


def test_embed_refuses_a_molecule_past_the_token_ceiling():
    with pytest.raises(ValueError, match="refused rather than cut"):
        _fake().embed(["C" * (MAX_TOKENS + 1)])


def test_embed_rejects_backend_shape_drift():
    pipe = MolformerPipeline(lambda mols: [[0.0] * 3 for _ in mols], "cpu")
    pipe._token_counter = lambda smiles: len(smiles) + 2
    with pytest.raises(RuntimeError, match="wrong shape"):
        pipe.embed(SMILES)


def test_token_count_requires_a_loaded_pipeline():
    pipe = MolformerPipeline(lambda mols: [], "cpu")
    with pytest.raises(RuntimeError, match="from_pretrained"):
        pipe.token_count("CCO")


def test_classify_requires_adaptation():
    with pytest.raises(RuntimeError, match="adapt"):
        _fake().classify(SMILES)


def test_classify_contract_preserves_class_order():
    pipe = _fake(classes=["branched", "linear"], logits=[[2.0, 0.0], [0.0, 2.0]])
    out = pipe.classify(SMILES, names=["a", "b"])
    assert [p["label"] for p in out["predictions"]] == ["branched", "linear"]
    assert list(out["predictions"][0]["scores"]) == ["branched", "linear"]
    assert out["predictions"][0]["scores"]["branched"] == pytest.approx(0.8808, abs=1e-3)
    assert out["predictions"][0]["smiles"] == SMILES[0]
    assert "not calibrated" in out["decision_rule"]


def test_classify_rejects_logits_that_do_not_match_classes():
    pipe = _fake(classes=["a", "b", "c"], logits=[[0.0, 1.0]] * 4)
    with pytest.raises(RuntimeError, match="does not match the class list"):
        pipe.classify(SMILES)


def test_save_and_load_artifact_require_a_loaded_model(tmp_path):
    pipe = _fake()
    with pytest.raises(RuntimeError, match="adapt"):
        pipe.save_artifact(tmp_path / "adapter")
    with pytest.raises(RuntimeError, match="from_pretrained"):
        pipe.load_artifact(tmp_path / "adapter")
