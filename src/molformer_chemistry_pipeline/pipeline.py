"""MoLFormer-XL (`ibm-research/MoLFormer-XL-both-10pct`) DIMER pipeline: verified snapshot, SMILES
embeddings, and bounded molecular-property classification fine-tuning with a portable adapter.

Two things about this checkpoint shape the module and are stated rather than hidden:

* **It requires remote code.** `config.json` declares `model_type: "molformer"`, for which the
  installed Transformers has no native implementation, and the architecture is a linear-attention
  transformer that the generic classes do not implement. The loader therefore passes
  `trust_remote_code=True` — but only after `verify_snapshot` has checked the SHA-256 of the two
  Python files it will import, which are manifest entries for exactly that reason.
* **Its pinned code needs a recent Transformers.** The upstream revision pinned here calls
  `transformers.masking_utils.create_bidirectional_mask(inputs_embeds=...)`, which exists only in
  Transformers 5.5.0 and later, so this package pins `transformers==5.17.0` rather than the fleet's
  4.57.6. See `docs/WEIGHTS.md`.
* **Its attention is stochastic unless told otherwise.** MoLFormer uses linear attention with
  random feature maps. The upstream config ships `deterministic_eval: false`, which redraws those
  features on *every* forward pass, so two identical calls return different embeddings (measured:
  ~1e-3 on a pooled vector). This package loads with `deterministic_eval=True`, which keeps the
  feature weights the checkpoint itself stores, and exports those buffers with any adapter because
  they are serving state, not incidental.

Everything model-related is imported lazily so that snapshot verification and input validation run
(and can refuse) before `torch` or `transformers` are imported (fleet RTM-001).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"
MODEL_REVISION = "361063d0ad524ef77cf39b08469f6be770dc550f"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "molformer-xl-both-10pct"
ARTIFACT_FORMAT = "org.valcorza.molformer-chemistry.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
# The two files the loader executes. They are manifest entries, so their bytes are digest-verified
# before `trust_remote_code=True` imports them (MODEL_ASSET_SPEC RC3).
REMOTE_CODE_FILES = ("configuration_molformer.py", "modeling_molformer.py")
# Linear-attention random-feature buffers. They live in the checkpoint, they change during training,
# and inference depends on them, so an exported adapter carries them (ART3).
FEATURE_MAP_SUFFIX = ".feature_map.weight"
# Upstream defaults to redrawing the random features on every forward pass (`deterministic_eval:
# false` in config.json). DIMER needs a reproducible answer to the same question, so this package
# overrides that to True, which uses the feature weights stored in the pinned checkpoint.
DETERMINISTIC_EVAL = True

# Ceilings. config.json declares max_position_embeddings = 202, and the tokenizer wraps every SMILES
# in <bos> ... <eos>, so 200 SMILES tokens is the most a molecule can carry. Longer input is refused
# rather than truncated: a truncated SMILES is a different molecule, not a shorter one.
MAX_POSITION_EMBEDDINGS = 202
MAX_TOKENS = MAX_POSITION_EMBEDDINGS - 2
MAX_MOLECULES_PER_CALL = 64
HIDDEN_SIZE = 768  # config.json hidden_size; the width of every `embed` row
VOCAB_SIZE = 2362  # config.json vocab_size; equals the tokenizer's vocabulary
# A syntactic character set for SMILES, not a chemical validity check: this package has no cheminformatics
# toolkit, so it checks characters, bracket balance and ring-digit pairing and says so. Use RDKit to
# establish that a string is a real molecule.
SMILES_CHARS = frozenset(
    "BCNOPSFIHbcnopsfi"  # organic subset, aromatic lowercase, explicit H in brackets
    "lr"  # the second letters of Cl and Br
    "eaigAKLMTZ"  # letters that appear inside bracket atoms (Se, Na, Si, Mg, ...)
    "0123456789"  # ring-closure digits
    "()[]"  # branches and bracket atoms
    "=#$:/\\"  # bond symbols
    "+-@"  # charges and stereochemistry
    "%."  # two-digit ring closures and disconnected components
)


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    listed = {entry["path"] for entry in manifest["files"]}
    missing_code = [name for name in REMOTE_CODE_FILES if name not in listed]
    if missing_code:
        raise ValueError(
            f"manifest does not list the remote-code files {missing_code}; refusing to proceed, because "
            "the loader executes them and their digests must be verified first"
        )
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest.hexdigest()} != manifest {entry['sha256']}")
    return manifest


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the snapshot against its DIMER manifest, including the two executable Python files."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    return _verify_manifest(root, MODEL_ID, MODEL_REVISION)


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at the pinned revision straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest entries that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights and the upstream Python). `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


INPUT_SCHEMA: dict[str, Any] = {
    "input": "1..MAX_MOLECULES_PER_CALL molecules as SMILES strings",
    "molecules": [1, MAX_MOLECULES_PER_CALL],
    "tokens_per_molecule": [1, MAX_TOKENS],
    "validation": (
        "syntactic only: character set, balanced parentheses and brackets, paired ring-closure digits, "
        "and the tokenized length ceiling. This package ships no cheminformatics toolkit and does not "
        "check valence, aromaticity or chemical plausibility -- use RDKit for that"
    ),
    "preprocessing": (
        "the pinned tokenizer splits SMILES into atom and symbol tokens (two-letter atoms such as Cl and "
        "Br and bracket atoms such as [C@H] stay single tokens), wraps them in <bos> ... <eos>, and pads "
        "to the longest molecule in the batch; embeddings are the mean of the last hidden state over "
        "non-padding tokens"
    ),
}


def check_smiles_syntax(smiles: str) -> list[str]:
    """Return a list of syntactic findings for one SMILES string; empty means it passed the checks.

    This is a string-level check, not a chemistry check: it catches typos and truncation, not
    impossible molecules.
    """
    findings: list[str] = []
    bad = sorted({ch for ch in smiles if ch not in SMILES_CHARS})
    if bad:
        findings.append(f"characters outside the accepted SMILES set: {bad!r}")
    for opener, closer, label in (("(", ")", "parenthesis"), ("[", "]", "bracket")):
        depth = 0
        for ch in smiles:
            depth += ch == opener
            depth -= ch == closer
            if depth < 0:
                findings.append(f"closing {label} before an opening one")
                break
        if depth > 0:
            findings.append(f"{depth} unclosed {label}(s)")
    digits: dict[str, int] = {}
    in_bracket = False
    for ch in smiles:
        if ch == "[":
            in_bracket = True
        elif ch == "]":
            in_bracket = False
        elif ch.isdigit() and not in_bracket:
            digits[ch] = digits.get(ch, 0) + 1
    odd = sorted(d for d, count in digits.items() if count % 2)
    if odd:
        findings.append(f"unpaired ring-closure digit(s): {odd}")
    return findings


def _check_molecules(molecules: Any, names: Any = None) -> tuple[list[str], list[str]]:
    """Raise TypeError/ValueError naming the first violated rule; return (smiles, ids).

    ``embed``, ``classify`` and ``validate_inputs`` all route through this function so their
    acceptance criteria cannot diverge.
    """
    if isinstance(molecules, str | bytes) or not isinstance(molecules, Sequence):
        raise TypeError("molecules must be a list of SMILES strings")
    if not 1 <= len(molecules) <= MAX_MOLECULES_PER_CALL:
        raise ValueError(f"molecules must hold 1..{MAX_MOLECULES_PER_CALL} items, got {len(molecules)}")
    checked: list[str] = []
    for i, smiles in enumerate(molecules):
        if not isinstance(smiles, str):
            raise TypeError(f"molecules[{i}] must be str, got {type(smiles).__name__}")
        if not smiles.strip():
            raise ValueError(f"molecules[{i}] is empty")
        if smiles != smiles.strip():
            raise ValueError(f"molecules[{i}] has leading or trailing whitespace; strip it first")
        findings = check_smiles_syntax(smiles)
        if findings:
            raise ValueError(
                f"molecules[{i}] ({smiles!r}) failed SMILES syntax checks: {'; '.join(findings)}"
            )
        checked.append(smiles)
    if names is None:
        ids = [f"mol-{i}" for i in range(len(checked))]
    else:
        if isinstance(names, str | bytes) or not isinstance(names, Sequence) or len(names) != len(checked):
            raise ValueError("names must be a list with exactly one id per molecule")
        ids = [str(n) for n in names]
        if len(set(ids)) != len(ids):
            raise ValueError("names must be unique")
    return checked, ids


def _softmax(logits: Sequence[float]) -> list[float]:
    import math

    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


@dataclass
class MolformerPipeline:
    """MoLFormer-XL pipeline: `embed` always; `classify` after `adapt` or `from_artifact`."""

    _embedder: Callable[[list[str]], list[list[float]]]
    device: str
    load_warnings: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    _classifier: Callable[[list[str]], list[list[float]]] | None = None
    _token_counter: Callable[[str], int] | None = None
    model: Any = None
    tokenizer: Any = None
    classifier_model: Any = None
    weights_dir: Path | None = None
    adaptation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> MolformerPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError(f"no snapshot manifest at {root} and allow_download={allow_download}")
        # Stage and verify — including the two Python files — before importing model libraries and
        # before any remote code is executed (RTM-001, MODEL_ASSET_SPEC RC3).
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        import torch
        from transformers import AutoModel, AutoTokenizer

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # The tokenizer is a native PreTrainedTokenizerFast: no remote code for this half.
            tokenizer = AutoTokenizer.from_pretrained(
                str(root), local_files_only=True, trust_remote_code=False
            )
            model = AutoModel.from_pretrained(
                str(root),
                local_files_only=True,
                trust_remote_code=True,
                deterministic_eval=DETERMINISTIC_EVAL,
            )
        model = model.to(resolved_device).eval()
        messages = [f"{w.category.__name__}: {w.message}" for w in caught]
        pipe = cls(cls._make_embedder(model, tokenizer, resolved_device), resolved_device, messages)
        pipe.model, pipe.tokenizer, pipe.weights_dir = model, tokenizer, root
        pipe._token_counter = lambda smiles: len(tokenizer(smiles)["input_ids"])
        return pipe

    # -- backends ---------------------------------------------------------------------------------

    @staticmethod
    def _encode(tokenizer: Any, molecules: list[str], device: str) -> dict[str, Any]:
        batch = tokenizer(molecules, return_tensors="pt", padding=True)
        return {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}

    @classmethod
    def _make_embedder(
        cls, model: Any, tokenizer: Any, device: str
    ) -> Callable[[list[str]], list[list[float]]]:
        import torch

        def embedder(molecules: list[str]) -> list[list[float]]:
            batch = cls._encode(tokenizer, molecules, device)
            # no_grad, not inference_mode: tensors produced here must stay usable by a later
            # training epoch that shares this module.
            with torch.no_grad():
                hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return pooled.float().cpu().tolist()

        return embedder

    @classmethod
    def _make_classifier(
        cls, model: Any, tokenizer: Any, device: str
    ) -> Callable[[list[str]], list[list[float]]]:
        import torch

        def classifier(molecules: list[str]) -> list[list[float]]:
            batch = cls._encode(tokenizer, molecules, device)
            with torch.no_grad():
                logits = model(**batch).logits
            return logits.float().cpu().tolist()

        return classifier

    def token_count(self, smiles: str) -> int:
        """Tokenized length including `<bos>`/`<eos>`; the ceiling applies to this, not to characters."""
        if self._token_counter is None:
            raise RuntimeError("token_count requires a pipeline built by from_pretrained")
        return self._token_counter(smiles)

    def _check_lengths(self, molecules: Sequence[str]) -> list[int]:
        counts = []
        for i, smiles in enumerate(molecules):
            n_tokens = self.token_count(smiles) - 2  # exclude <bos>/<eos>
            if n_tokens > MAX_TOKENS:
                raise ValueError(
                    f"molecules[{i}] tokenizes to {n_tokens} tokens; the ceiling is {MAX_TOKENS} "
                    f"(max_position_embeddings {MAX_POSITION_EMBEDDINGS} minus <bos> and <eos>). "
                    "A truncated SMILES is a different molecule, so this is refused rather than cut."
                )
            counts.append(n_tokens)
        return counts

    # -- public stages ----------------------------------------------------------------------------

    def embed(self, molecules: Sequence[str], *, names: Sequence[str] | None = None) -> dict[str, Any]:
        """Mean-pooled last-hidden-state representation per molecule (HIDDEN_SIZE floats each)."""
        checked, ids = _check_molecules(molecules, names)
        token_counts = self._check_lengths(checked)
        vectors = self._embedder(checked)
        if len(vectors) != len(checked) or any(len(v) != HIDDEN_SIZE for v in vectors):
            raise RuntimeError("backend returned embeddings of the wrong shape")
        return {
            "ids": ids,
            "embeddings": [[float(x) for x in v] for v in vectors],
            "dimension": HIDDEN_SIZE,
            "pooling": "mean of the last hidden state over non-padding tokens",
            "unit": "one vector per molecule; representations, not property predictions",
            "tokens": token_counts,
            "n_molecules": len(checked),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def classify(self, molecules: Sequence[str], *, names: Sequence[str] | None = None) -> dict[str, Any]:
        """Class scores and argmax label per molecule; requires a prior `adapt` or `from_artifact`."""
        if self._classifier is None or not self.classes:
            raise RuntimeError(
                "classify requires an adapted head: call adapt(...) or load from_artifact(...) first"
            )
        checked, ids = _check_molecules(molecules, names)
        token_counts = self._check_lengths(checked)
        logits = self._classifier(checked)
        predictions = []
        for mid, smiles, n_tokens, row in zip(ids, checked, token_counts, logits, strict=True):
            if len(row) != len(self.classes):
                raise RuntimeError("backend returned a logits row that does not match the class list")
            scores = _softmax(row)
            best = max(range(len(scores)), key=scores.__getitem__)
            predictions.append(
                {
                    "id": mid,
                    "smiles": smiles,
                    "tokens": n_tokens,
                    "label": self.classes[best],
                    "score": scores[best],
                    "scores": dict(zip(self.classes, scores, strict=True)),
                }
            )
        return {
            "predictions": predictions,
            "classes": list(self.classes),
            "decision_rule": (
                "argmax over softmax(logits); scores are softmax outputs, not calibrated probabilities"
            ),
            "n_molecules": len(checked),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "adaptation": dict(self.adaptation),
        }

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Held-out property-classification metrics (see metrics.classification_metrics)."""
        from .metrics import classification_metrics
        from .samples import validate_dataset

        validate_dataset(records, classes=self.classes)
        predicted: list[str] = []
        scores: list[list[float]] = []
        for start in range(0, len(records), MAX_MOLECULES_PER_CALL):
            chunk = records[start : start + MAX_MOLECULES_PER_CALL]
            result = self.classify([r["smiles"] for r in chunk], names=[r["id"] for r in chunk])
            for p in result["predictions"]:
                predicted.append(p["label"])
                scores.append([p["scores"][c] for c in self.classes])
        return classification_metrics([r["label"] for r in records], predicted, scores, self.classes)

    def adapt(
        self,
        train_records: Sequence[Mapping[str, Any]],
        val_records: Sequence[Mapping[str, Any]] | None = None,
        *,
        classes: Sequence[str] | None = None,
        epochs: int = 4,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        trainable_layers: int = 2,
        weight_decay: float = 0.01,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Bounded gradient fine-tuning of a property-classification head on the verified base.

        Builds `MolformerForSequenceClassification` from the pinned checkpoint (the head is newly
        initialised), freezes every parameter except the head and the last `trainable_layers`
        encoder layers, and runs AdamW for `epochs` passes. Validation records are monitored per
        epoch only; the final epoch's weights are kept (no selection).
        """
        if self.model is None or self.tokenizer is None or self.weights_dir is None:
            raise RuntimeError("adapt requires a pipeline built by from_pretrained (no loaded base model)")
        from .samples import validate_dataset

        if not 1 <= int(epochs) <= 50:
            raise ValueError("epochs must be in 1..50 (tutorial-scale adaptation)")
        if not 1 <= int(batch_size) <= MAX_MOLECULES_PER_CALL:
            raise ValueError(f"batch_size must be in 1..{MAX_MOLECULES_PER_CALL}")
        if not 0 <= int(trainable_layers) <= 12:
            raise ValueError("trainable_layers must be in 0..12 (the checkpoint has 12 encoder layers)")
        train_manifest = validate_dataset(train_records, classes=classes)
        class_list = list(train_manifest["classes"])
        if val_records is not None:
            validate_dataset(val_records, classes=class_list)

        import random

        import torch
        from transformers import AutoModelForSequenceClassification

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        clf = AutoModelForSequenceClassification.from_pretrained(
            str(self.weights_dir),
            local_files_only=True,
            trust_remote_code=True,
            num_labels=len(class_list),
            deterministic_eval=DETERMINISTIC_EVAL,
        ).to(self.device)
        for p in clf.parameters():
            p.requires_grad = False
        layers = clf.molformer.encoder.layer
        for layer in layers[len(layers) - int(trainable_layers) :] if trainable_layers else []:
            for p in layer.parameters():
                p.requires_grad = True
        for p in clf.classifier.parameters():
            p.requires_grad = True
        trainable = [n for n, p in clf.named_parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in clf.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in clf.parameters())
        optimizer = torch.optim.AdamW(
            [p for p in clf.parameters() if p.requires_grad], lr=learning_rate, weight_decay=weight_decay
        )
        label_index = {c: i for i, c in enumerate(class_list)}
        examples = [(r["smiles"], label_index[r["label"]]) for r in train_records]
        self.classes = class_list
        self.classifier_model = clf
        self._classifier = self._make_classifier(clf, self.tokenizer, self.device)

        history: list[dict[str, Any]] = []
        for epoch in range(1, int(epochs) + 1):
            clf.train()
            order = list(range(len(examples)))
            random.shuffle(order)
            total_loss, n_batches = 0.0, 0
            for start in range(0, len(order), int(batch_size)):
                rows = [examples[i] for i in order[start : start + int(batch_size)]]
                batch = self._encode(self.tokenizer, [s for s, _ in rows], self.device)
                labels = torch.tensor([y for _, y in rows], device=self.device)
                optimizer.zero_grad()
                out = clf(**batch, labels=labels)
                out.loss.backward()
                optimizer.step()
                total_loss += float(out.loss.item())
                n_batches += 1
            clf.eval()
            entry: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": round(total_loss / max(1, n_batches), 6),
                "n_batches": n_batches,
            }
            if val_records:
                val = self.evaluate(val_records)
                entry["val_accuracy"] = val["accuracy"]
                entry["val_macro_f1"] = val["macro_f1"]
            history.append(entry)
        clf.eval()
        self.adaptation = {
            "method": "gradient fine-tuning (AdamW) of the classification head"
            + (f" and the last {int(trainable_layers)} encoder layer(s)" if trainable_layers else ""),
            "classes": class_list,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "weight_decay": float(weight_decay),
            "trainable_layers": int(trainable_layers),
            "seed": int(seed),
            "precision": "float32",
            "trainable_parameters": int(n_trainable),
            "total_parameters": int(n_total),
            "trainable_parameter_names": trainable,
            "train_records": len(train_records),
            "val_records": len(val_records) if val_records else 0,
            "selection": "final epoch kept; validation metrics are monitoring only",
            "history": history,
        }
        return dict(self.adaptation)

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Export the trainable tensors as safetensors plus a JSON manifest binding them to the base."""
        if self.classifier_model is None or not self.classes:
            raise RuntimeError("save_artifact requires an adapted head (call adapt first)")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adaptation.get("trainable_parameter_names", []))
        state = self.classifier_model.state_dict()
        # Trained parameters plus the linear-attention feature buffers: training redraws those
        # buffers, and inference depends on them, so an adapter without them does not reproduce
        # the adapted model (ART3).
        serving_state = sorted(k for k in state if k.endswith(FEATURE_MAP_SUFFIX))
        tensors = {
            k: v.detach().cpu().contiguous() for k, v in state.items() if k in names or k in serving_state
        }
        if not tensors:
            raise RuntimeError("no trainable tensors recorded; nothing to export")
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path))
        digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "license": MODEL_LICENSE},
            "requires_remote_code": True,
            "classes": list(self.classes),
            "files": [
                {"path": ARTIFACT_WEIGHTS_NAME, "bytes": weights_path.stat().st_size, "sha256": digest}
            ],
            "tensors": sorted(tensors),
            "serving_state_tensors": serving_state,
            "deterministic_eval": DETERMINISTIC_EVAL,
            "adaptation": {k: v for k, v in self.adaptation.items() if k != "trainable_parameter_names"},
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Rebuild the classification head from an exported artifact (manifest verified before loading)."""
        if self.model is None or self.weights_dir is None:
            raise RuntimeError("load_artifact requires a pipeline built by from_pretrained")
        art = Path(artifact_dir)
        manifest_path = art / ARTIFACT_MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        base = manifest.get("base_model", {})
        if (base.get("model_id"), base.get("model_revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError(f"artifact was trained on {base}, this package pins {MODEL_ID}@{MODEL_REVISION}")
        classes = [str(c) for c in manifest.get("classes", [])]
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("artifact manifest must list at least two unique classes")
        for entry in manifest["files"]:
            fp = art / entry["path"]
            if not fp.is_file():
                raise FileNotFoundError(f"artifact file missing: {fp}")
            if fp.stat().st_size != entry["bytes"]:
                raise ValueError(f"{entry['path']}: size {fp.stat().st_size} != manifest {entry['bytes']}")
            if hashlib.sha256(fp.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"{entry['path']}: sha256 mismatch against the artifact manifest")
        from safetensors.torch import load_file
        from transformers import AutoModelForSequenceClassification

        clf = AutoModelForSequenceClassification.from_pretrained(
            str(self.weights_dir),
            local_files_only=True,
            trust_remote_code=True,
            num_labels=len(classes),
            deterministic_eval=DETERMINISTIC_EVAL,
        )
        tensors = load_file(str(art / ARTIFACT_WEIGHTS_NAME))
        if set(tensors) != set(manifest.get("tensors", [])):
            raise ValueError("artifact tensors do not match the names listed in its manifest")
        _missing, unexpected = clf.load_state_dict(tensors, strict=False)
        if unexpected:
            raise ValueError(
                f"artifact carries tensors the base architecture does not have: {sorted(unexpected)[:5]}"
            )
        clf = clf.to(self.device).eval()
        self.classes = classes
        self.classifier_model = clf
        self._classifier = self._make_classifier(clf, self.tokenizer, self.device)
        self.adaptation = {**manifest.get("adaptation", {}), "loaded_from_artifact": str(art)}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> MolformerPipeline:
        """Verified base snapshot + exported adapter, ready for `classify`."""
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe


def validate_inputs(
    molecules: Sequence[str],
    *,
    names: Sequence[str] | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, verdict).

    `token_counter` is the pipeline's `token_count`; without it the manifest reports character
    lengths and says that the token ceiling was not checked, rather than guessing at it.
    """
    checked, ids = _check_molecules(molecules, names)
    rows = []
    for mid, smiles in zip(ids, checked, strict=True):
        row: dict[str, Any] = {"id": mid, "smiles": smiles, "characters": len(smiles)}
        if token_counter is not None:
            n_tokens = token_counter(smiles) - 2
            if n_tokens > MAX_TOKENS:
                raise ValueError(
                    f"{mid} tokenizes to {n_tokens} tokens; the ceiling is {MAX_TOKENS} "
                    f"(max_position_embeddings {MAX_POSITION_EMBEDDINGS} minus <bos> and <eos>)"
                )
            row["tokens"] = n_tokens
        rows.append(row)
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": rows,
        "n_molecules": len(checked),
        "token_ceiling_checked": token_counter is not None,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "requires_remote_code": True,
    }
