"""Import-boundary contract (fleet RTM-001).

Rejected requests never import model libraries; the snapshot — including the two Python files the
loader will execute — is verified before anything is imported.
"""

import hashlib
import json

import pytest

from molformer_chemistry_pipeline.pipeline import (
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_REVISION,
    REMOTE_CODE_FILES,
    MolformerPipeline,
    validate_inputs,
)

_CONFIG = json.dumps({"model_type": "molformer", "hidden_size": 768}).encode()


def _snapshot(root, tamper=False):
    files = []
    for name in ("config.json", *REMOTE_CODE_FILES):
        (root / name).write_bytes(_CONFIG)
        digest = "0" * 64 if (tamper and name == "config.json") else hashlib.sha256(_CONFIG).hexdigest()
        files.append({"path": name, "bytes": len(_CONFIG), "sha256": digest})
    manifest = {"modelId": MODEL_ID, "revision": MODEL_REVISION, "files": files}
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_from_pretrained_refuses_without_manifest_before_model_imports(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="no snapshot manifest"):
        MolformerPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_from_pretrained_refuses_tampered_snapshot_before_model_imports(tmp_path, forbid_model_imports):
    _snapshot(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        MolformerPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_invalid_smiles_are_rejected_before_model_imports(forbid_model_imports):
    with pytest.raises(ValueError, match="failed SMILES syntax checks"):
        validate_inputs(["CC(C"])
