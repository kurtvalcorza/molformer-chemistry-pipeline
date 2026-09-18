"""Classification metrics and trivial baselines for MoLFormer property-classification adaptation.

Pure Python (no scikit-learn): accuracy, macro-F1, per-class precision/recall/F1/support, and AUROC
(binary: positive class = the last entry of `classes`; multiclass: macro one-vs-rest), computed by
the Mann-Whitney rank statistic with average ranks for ties.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .samples import formula_string


def _prf(hits: int, n_pred: int, n_true: int) -> dict[str, float]:
    precision = hits / n_pred if n_pred else 0.0
    recall = hits / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def auroc(y_true: Sequence[int], scores: Sequence[float]) -> float | None:
    """Area under the ROC curve for binary 0/1 labels; None when only one class is present."""
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-based average rank of the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, y_true, strict=True) if y == 1)
    return round((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg), 4)


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    scores: Sequence[Sequence[float]] | None,
    classes: Sequence[str],
) -> dict[str, Any]:
    """Discrete and ranking metrics over one evaluation split (labels are class names).

    `scores[i][k]` is the score of class `classes[k]` for record i (softmax outputs from the
    pipeline; any monotone score works for AUROC). Class order is preserved exactly as given.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} labels vs {len(y_pred)} predictions")
    class_list = list(classes)
    unknown = sorted((set(y_true) | set(y_pred)) - set(class_list))
    if unknown:
        raise ValueError(f"labels outside the class list {class_list}: {unknown}")
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    per_class: dict[str, dict[str, Any]] = {}
    f1s: list[float] = []
    for c in class_list:
        hits = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == c and p == c)
        n_pred = sum(1 for p in y_pred if p == c)
        n_true = sum(1 for t in y_true if t == c)
        prf = _prf(hits, n_pred, n_true)
        per_class[c] = {**prf, "support": n_true, "predicted": n_pred}
        if n_true:
            f1s.append(prf["f1"])
    result: dict[str, Any] = {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "per_class": per_class,
        "classes": class_list,
        "decision_rule": "argmax over class scores",
        "auroc": None,
        "auroc_definition": None,
    }
    if scores is not None and n:
        if len(scores) != n or any(len(row) != len(class_list) for row in scores):
            raise ValueError("scores must be one row per record with one column per class")
        if len(class_list) == 2:
            pos = class_list[-1]
            result["auroc"] = auroc([1 if t == pos else 0 for t in y_true], [row[-1] for row in scores])
            result["auroc_definition"] = f"binary AUROC with positive class {pos!r} (last class in the list)"
        else:
            values = []
            for k, c in enumerate(class_list):
                a = auroc([1 if t == c else 0 for t in y_true], [row[k] for row in scores])
                if a is not None:
                    values.append(a)
            result["auroc"] = round(sum(values) / len(values), 4) if values else None
            result["auroc_definition"] = "macro-averaged one-vs-rest AUROC over classes present in the split"
    return result


def majority_baseline(
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    classes: Sequence[str],
) -> dict[str, Any]:
    """Predict the most frequent training class for every evaluation record (EVAL11)."""
    counts: dict[str, int] = {}
    for r in train_records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    majority = max(sorted(counts), key=counts.__getitem__)
    metrics = classification_metrics(
        [r["label"] for r in eval_records], [majority] * len(eval_records), None, classes
    )
    return {"baseline": "majority-class", "predicted_label": majority, **metrics}


def formula_baseline(
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    classes: Sequence[str],
) -> dict[str, Any]:
    """Predict from the molecular formula alone, learned on the training split only (SPL8).

    Molecular formula is the confounder worth ruling out in any molecular-property task: if the
    formula already separates the classes, the model has not had to learn chemistry. The rule is a
    lookup -- for each formula seen in training, predict its majority class; formulas never seen in
    training fall back to the overall training majority. On the tutorial sample every formula appears
    in both classes by construction (the pairs are constitutional isomers), so this baseline is
    expected to sit at chance.
    """
    class_list = list(classes)
    by_formula: dict[str, dict[str, int]] = {}
    overall: dict[str, int] = {}
    for record in train_records:
        formula = formula_string(record["smiles"])
        by_formula.setdefault(formula, {})
        by_formula[formula][record["label"]] = by_formula[formula].get(record["label"], 0) + 1
        overall[record["label"]] = overall.get(record["label"], 0) + 1
    fallback = max(sorted(overall), key=overall.__getitem__)
    lookup = {formula: max(sorted(counts), key=counts.__getitem__) for formula, counts in by_formula.items()}
    predicted, unseen = [], 0
    for record in eval_records:
        formula = formula_string(record["smiles"])
        if formula in lookup:
            predicted.append(lookup[formula])
        else:
            predicted.append(fallback)
            unseen += 1
    metrics = classification_metrics([r["label"] for r in eval_records], predicted, None, class_list)
    return {
        "baseline": "molecular-formula lookup",
        "distinct_train_formulas": len(lookup),
        "ambiguous_train_formulas": sum(1 for counts in by_formula.values() if len(counts) > 1),
        "eval_formulas_unseen_in_train": unseen,
        "fallback_label": fallback,
        **metrics,
    }
