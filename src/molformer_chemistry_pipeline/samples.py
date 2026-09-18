"""Deterministic in-code sample molecules and the labelled-dataset contract for property classification.

The tutorial task is synthetic but chemically literal: each pair of molecules is a pair of
**constitutional isomers** — the same molecular formula, differing only in connectivity. One member
is an unbranched primary alcohol (`CCCCCCCO`); the other carries the same atoms with one methyl
branch moved onto the chain (`CCCC(C)CCO`). Molecular formula therefore carries no signal by
construction, so a composition baseline sits at chance and any separation the model achieves comes
from reading structure. This is sanity evidence for the adaptation contract, not a chemistry
benchmark, and the classes are a generator rule rather than a measured property (NOTEBOOK_SPEC 2.0
DAT8).

One honest caveat, stated here and in the tutorial: a branched SMILES contains `(` and `)`, so a
*character-level* baseline that counted punctuation could separate these classes trivially. The
baseline this package ships counts heavy atoms — the molecular formula — because that is the
chemically meaningful confounder to rule out.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import MAX_TOKENS, check_smiles_syntax

DATASET_REPRESENTATION = "io.github.kurtvalcorza.dataset.chemistry.smiles-labels.v1"
SAMPLE_CLASSES: tuple[str, ...] = ("branched", "linear")
SAMPLE_SEED = 20260918
SAMPLE_SIZE = 64  # 32 pairs
MIN_CARBONS = 6
MAX_CARBONS = 14
# Terminal groups, so that pairs differ from one another rather than only in chain length. Both
# members of a pair carry the same terminus, so the terminus can never separate the two classes.
TERMINI: tuple[str, ...] = ("O", "N", "S", "F", "Cl", "Br", "I")
MIN_RECORDS = 8
MAX_RECORDS = 5_000
MAX_CLASSES = 20
MIN_RECORDS_PER_CLASS = 3
MAX_ID_CHARS = 64
MAX_LABEL_CHARS = 64
REQUIRED_COLUMNS = ("id", "smiles", "label")
_ATOM = re.compile(r"Cl|Br|[BCNOPSFI]|\[[^\]]*\]|[bcnops]")


def molecular_formula(smiles: str) -> dict[str, int]:
    """Heavy-atom counts for the organic subset this sample uses (no implicit hydrogens).

    This is a string-level count, not a cheminformatics parse: it recognises the two-letter atoms
    `Cl` and `Br`, single-letter organic-subset atoms, aromatic lowercase atoms, and bracket atoms
    as one atom each. It exists to give the tutorial a composition baseline, not to replace RDKit.
    """
    counts: dict[str, int] = {}
    for token in _ATOM.findall(smiles):
        if token.startswith("["):
            inner = token[1:-1]
            match = re.search(r"[A-Z][a-z]?|[bcnops]", inner)
            symbol = match.group(0) if match else inner
        else:
            symbol = token
        symbol = symbol.capitalize() if len(symbol) > 1 else symbol.upper()
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def formula_string(smiles: str) -> str:
    """`molecular_formula` rendered in a stable order, e.g. `C7O1`."""
    counts = molecular_formula(smiles)
    return "".join(f"{symbol}{counts[symbol]}" for symbol in sorted(counts))


def branch_positions(n_carbons: int) -> list[int]:
    """Positions at which a methyl branch can sit without recreating the linear chain."""
    return list(range(2, n_carbons - 2))


def make_isomer_pair(n_carbons: int, branch_at: int, terminus: str = "O") -> tuple[str, str]:
    """A linear molecule and a branched constitutional isomer of it: same formula, different skeleton.

    `CCCCCCCO` and `CCCC(C)CCO` both have `n_carbons` carbons and one oxygen: the branched member
    spends one of its carbons as a methyl substituent instead of extending the chain. `terminus` is
    the group at the end of the chain and is identical for both members, so it cannot carry the label.
    """
    if not MIN_CARBONS <= n_carbons <= MAX_CARBONS:
        raise ValueError(f"n_carbons must be in {MIN_CARBONS}..{MAX_CARBONS}, got {n_carbons}")
    if branch_at not in branch_positions(n_carbons):
        raise ValueError(f"branch_at must be one of {branch_positions(n_carbons)} for {n_carbons} carbons")
    if terminus not in TERMINI:
        raise ValueError(f"terminus must be one of {list(TERMINI)}, got {terminus!r}")
    linear = "C" * n_carbons + terminus
    chain = n_carbons - 1  # one carbon becomes the branch
    branched = "C" * branch_at + "(C)" + "C" * (chain - branch_at) + terminus
    return linear, branched


def generate_sample_dataset(seed: int = SAMPLE_SEED, size: int = SAMPLE_SIZE) -> list[dict[str, Any]]:
    """`size` labelled molecules (half `linear`, half `branched`), deterministic for a given seed.

    Every branched molecule is a constitutional isomer of its linear partner: identical formula,
    different connectivity.
    """
    if size < 2 or size % 2:
        raise ValueError("size must be an even number >= 2 (one linear molecule per branched isomer)")
    skeletons = [(n, terminus) for terminus in TERMINI for n in range(MIN_CARBONS, MAX_CARBONS + 1)]
    if size // 2 > len(skeletons):
        raise ValueError(
            f"size {size} needs {size // 2} distinct (chain length, terminus) pairs but only "
            f"{len(skeletons)} exist; widen MIN_CARBONS..MAX_CARBONS or TERMINI"
        )
    rng = random.Random(seed)
    rng.shuffle(skeletons)
    records: list[dict[str, Any]] = []
    for i, (n_carbons, terminus) in enumerate(skeletons[: size // 2]):
        branch_at = rng.choice(branch_positions(n_carbons))
        linear, branched = make_isomer_pair(n_carbons, branch_at, terminus)
        records.append({"id": f"lin-{i:03d}", "smiles": linear, "label": "linear"})
        records.append({"id": f"bra-{i:03d}", "smiles": branched, "label": "branched"})
    return records


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """SHA-256 over the canonical (id, smiles, label) rows; recorded in provenance (OUT9)."""
    canon = json.dumps([[r["id"], r["smiles"], r["label"]] for r in records], separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    classes: Sequence[str] | None = None,
    min_records: int = MIN_RECORDS,
    min_per_class: int = MIN_RECORDS_PER_CLASS,
) -> dict[str, Any]:
    """Check a labelled molecule dataset against the contract; return its manifest.

    Every error names the record and the violated rule (VAL4/DAT19). SMILES are checked
    syntactically only — character set, bracket balance, ring-digit pairing — because this package
    ships no cheminformatics toolkit; the manifest says so rather than implying chemical validation.
    """
    if isinstance(records, str | bytes | Mapping) or not isinstance(records, Sequence):
        raise TypeError("records must be a list of {'id', 'smiles', 'label'} mappings")
    if len(records) < min_records:
        raise ValueError(f"dataset has {len(records)} records; at least {min_records} are required")
    if len(records) > MAX_RECORDS:
        raise ValueError(f"dataset has {len(records)} records; ceiling is {MAX_RECORDS}")
    seen_ids: set[str] = set()
    seen_smiles: dict[str, str] = {}
    counts_by_class: dict[str, int] = {}
    lengths: list[int] = []
    formulas: dict[str, set[str]] = {}
    for i, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise TypeError(f"record[{i}] must be a mapping, got {type(rec).__name__}")
        missing = [c for c in REQUIRED_COLUMNS if c not in rec]
        if missing:
            raise ValueError(
                f"record[{i}] is missing required column(s) {missing}; required: {list(REQUIRED_COLUMNS)}"
            )
        rid = str(rec["id"]).strip()
        if not rid or len(rid) > MAX_ID_CHARS:
            raise ValueError(f"record[{i}] id must be 1..{MAX_ID_CHARS} characters")
        if rid in seen_ids:
            raise ValueError(f"record[{i}] duplicates id {rid!r}")
        seen_ids.add(rid)
        smiles = rec["smiles"]
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(f"record[{i}] ({rid}) smiles must be a non-empty string")
        if smiles != smiles.strip():
            raise ValueError(f"record[{i}] ({rid}) smiles has leading or trailing whitespace")
        findings = check_smiles_syntax(smiles)
        if findings:
            raise ValueError(f"record[{i}] ({rid}) failed SMILES syntax checks: {'; '.join(findings)}")
        if smiles in seen_smiles:
            raise ValueError(f"record[{i}] ({rid}) duplicates the SMILES of {seen_smiles[smiles]!r}")
        seen_smiles[smiles] = rid
        lengths.append(len(smiles))
        label = rec["label"]
        if not isinstance(label, str) or not label.strip() or len(label) > MAX_LABEL_CHARS:
            raise ValueError(
                f"record[{i}] ({rid}) label must be a non-empty string of at most {MAX_LABEL_CHARS} chars"
            )
        counts_by_class[label] = counts_by_class.get(label, 0) + 1
        formulas.setdefault(formula_string(smiles), set()).add(label)
    if classes is None:
        class_list = sorted(counts_by_class)
    else:
        class_list = [str(c) for c in classes]
        unknown = sorted(set(counts_by_class) - set(class_list))
        if unknown:
            raise ValueError(f"labels {unknown} are not in the class list {class_list}")
    if len(class_list) < 2:
        raise ValueError(f"classification needs at least 2 classes, found {class_list}")
    if len(class_list) > MAX_CLASSES:
        raise ValueError(f"{len(class_list)} classes exceeds the ceiling of {MAX_CLASSES}")
    thin = [c for c in class_list if counts_by_class.get(c, 0) < min_per_class]
    if thin:
        raise ValueError(f"classes {thin} have fewer than {min_per_class} records each (class coverage rule)")
    shared = sorted(f for f, labels in formulas.items() if len(labels) > 1)
    return {
        "verdict": "accepted",
        "representation": DATASET_REPRESENTATION,
        "validation": "syntactic SMILES checks only; no cheminformatics toolkit is used",
        "n_records": len(records),
        "classes": class_list,
        "class_counts": {c: counts_by_class.get(c, 0) for c in class_list},
        "smiles_characters": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 1),
        },
        "distinct_formulas": len(formulas),
        "formulas_shared_across_classes": len(shared),
        "ceilings": {
            "max_tokens": MAX_TOKENS,
            "max_records": MAX_RECORDS,
            "max_classes": MAX_CLASSES,
            "min_records": min_records,
            "min_records_per_class": min_per_class,
        },
        "digest": dataset_digest(records),
        "findings": [],
    }


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.25,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Stratified random train/validation/test split (assumes independent molecules, SPL3).

    Real chemical datasets are rarely independent: analogues from one series share a scaffold, and a
    random split lets the model memorise it. A scaffold or cluster split is the right tool there;
    the tutorial's molecules are generated independently, which is why a random split is honest here.
    """
    if not (0.0 < val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("val_fraction and test_fraction must be in (0, 1) and sum to less than 1")
    manifest = validate_dataset(records)
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, Any]]] = {c: [] for c in manifest["classes"]}
    for rec in records:
        by_class[rec["label"]].append(dict(rec))
    out: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for cls in manifest["classes"]:
        rows = by_class[cls]
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * val_fraction))
        n_test = max(1, round(len(rows) * test_fraction))
        if len(rows) - n_val - n_test < 1:
            raise ValueError(f"class {cls!r} has {len(rows)} records; too few to leave one per split")
        out["validation"].extend(rows[:n_val])
        out["test"].extend(rows[n_val : n_val + n_test])
        out["train"].extend(rows[n_val + n_test :])
    for part in out.values():
        rng.shuffle(part)
    return out


def load_byod_dataset(source: str | Path) -> list[dict[str, Any]]:
    """Read a user-supplied dataset (CSV with `id,smiles,label`, JSON array, or JSONL).

    SMILES are stripped of surrounding whitespace only; nothing else is rewritten (VAL7). The
    records are then validated with `validate_dataset`, whose errors name the offending row.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"BYOD dataset file not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"BYOD dataset file is empty: {path}")
    suffix = path.suffix.lower()
    records: list[dict[str, Any]] = []
    if suffix == ".csv":
        reader = csv.DictReader(text.splitlines())
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(f"CSV header {header} is missing required column(s) {missing}")
        for row in reader:
            records.append({c: (row.get(c) or "").strip() for c in REQUIRED_COLUMNS})
    elif suffix == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no} is not valid JSON: {exc}") from exc
    elif suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"file is not valid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise TypeError("JSON dataset must be a top-level array of objects")
        records = data
    else:
        raise ValueError(f"unsupported BYOD file type {suffix!r}; use .csv, .json or .jsonl")
    for rec in records:
        if isinstance(rec, Mapping) and isinstance(rec.get("smiles"), str):
            rec["smiles"] = rec["smiles"].strip()
    validate_dataset(records)
    return [dict(r) for r in records]


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write records as the BYOD CSV shape (`id,smiles,label`), so users have a template."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(REQUIRED_COLUMNS)
        for r in records:
            writer.writerow([r["id"], r["smiles"], r["label"]])
    return out
