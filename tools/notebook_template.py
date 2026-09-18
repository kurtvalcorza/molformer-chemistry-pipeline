"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E molecular-property workflow: the pinned MoLFormer-XL snapshot is
digest-verified (weights *and* the two Python files the loader executes), SMILES are validated and
tokenized, a synthetic constitutional-isomer dataset is split, trivial baselines are measured, a
bounded AdamW fine-tuning runs in the kernel, and the adapter — including the linear-attention
feature buffers — is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "molformer-chemistry-pipeline"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/molformer_chemistry_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ibm--research%2FMoLFormer--XL--both--10pct-ffcc4d?style=flat",
        "https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-IBM%2Fmolformer-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/IBM/molformer",
    ),
    ("arXiv", "https://img.shields.io/badge/arXiv-2106.09553-b31b1b.svg", "https://arxiv.org/abs/2106.09553"),
]

TEMPLATE = {
    "package": "molformer_chemistry_pipeline",
    "repo_name": REPO,
    "stem": "molformer_chemistry",
    "notebook_name": "molformer_chemistry_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned MoLFormer-XL snapshot (7 files, ~187 MB, including the two Python files the loader executes), loads the "
        "tokenizer without remote code and the model with it, generates a deterministic 64-molecule constitutional-isomer "
        "dataset in code (no download), validates the SMILES and the dataset contract, splits it into stratified "
        "train/validation/test sets, shows how SMILES are tokenized, computes mean-pooled molecule embeddings, measures a "
        "majority-class and a molecular-formula baseline on the test split, runs a bounded AdamW fine-tuning of the "
        "property-classification head and the last two encoder layers, evaluates accuracy, macro-F1 and AUROC on the "
        "held-out test split, classifies six freshly generated molecules, exports the adapter as safetensors with a "
        "manifest, and reloads that artifact into a fresh pipeline to verify prediction parity. The default path needs no "
        "repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes well under a minute of model time."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled molecules as a CSV (`id,smiles,label`), a JSON array or a JSONL file. They pass through the same SMILES "
        "validation, stratified split, baselines, adaptation, held-out evaluation, inference, artifact export and "
        "reload-parity cells as the synthetic sample. The expected schema, the accepted SMILES characters and the token "
        "ceiling are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is "
        "optional and never part of the default path."
    ),
    "pipeline_class": "MolformerPipeline",
    "weights_key": "molformer-xl-both-10pct",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers", "safetensors"],
    "title": "MoLFormer-XL — DIMER E2E molecular-property classification tutorial (standalone)",
    "badges": BADGES,
    "capability": "SMILES embeddings and bounded molecular-property classification fine-tuning",
    "intro": (
        "MoLFormer-XL is a chemical language model: it reads molecules written as **SMILES strings** and was pretrained with a "
        "masked-token objective on a very large corpus of them (the upstream card describes 1.1 billion molecules from PubChem "
        "and ZINC; this checkpoint is the variant trained on a 10 % sample of both). Its notable architectural choice is "
        "**linear attention with random feature maps**, which is why it scales to that corpus — and why it behaves differently "
        "from an ordinary transformer in one respect that matters for reproducibility, explained in Section 3.\n\n"
        "This tutorial adapts it to a molecular-property task. The tutorial dataset is synthetic but chemically literal: every "
        "molecule is paired with a **constitutional isomer** of itself — the same molecular formula, the same atoms, a "
        "different skeleton. One member is an unbranched chain (`CCCCCCCO`); the other carries one of those carbons as a "
        "methyl branch (`CCCC(C)CCO`). A classifier that only knows the molecular formula therefore cannot do better than "
        "chance, by construction, which is what makes the fine-tuned model's result worth reading."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, dataset and metrics modules; stage and digest-verify an "
        "immutable snapshot **including the Python the loader executes**; understand what `trust_remote_code=True` buys and "
        "costs here, and why this package overrides `deterministic_eval`; validate SMILES syntactically and split a labelled "
        "dataset without leakage; read how SMILES become tokens; extract mean-pooled molecule embeddings; measure "
        "majority-class and molecular-formula baselines; run a bounded fine-tuning with explicit hyperparameters; evaluate "
        "accuracy, macro-F1 and AUROC on an independent test split; classify new molecules; and export a safetensors adapter "
        "that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "regression on continuous properties, the published MoleculeNet benchmarks, molecule generation or optimisation, "
        "3D conformers or descriptors, chemical validity checking (no RDKit is installed), and the other MoLFormer "
        "checkpoints. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). CPU is enough — the default fine-tuning is a couple of seconds — and CUDA is used automatically when present.",
        "- **Knowledge:** how a molecule is written as SMILES, what a constitutional isomer is, and how accuracy, macro-F1 and AUROC differ.",
        "- **Remote code:** the default path executes the checkpoint's own `configuration_molformer.py` and `modeling_molformer.py` after verifying their SHA-256 against the inline manifest. Section 3 explains why that is unavoidable for this checkpoint.",
        "- **Runtime pin:** this repository pins `transformers==5.17.0`, not the fleet's 4.57.6, because the pinned upstream code calls an API that exists only in Transformers 5.5.0 and later.",
        "- **Data contract:** records are `{{id, smiles, label}}`; SMILES are non-empty strings over the accepted character set, at most 200 tokens (`MAX_TOKENS`, the checkpoint's 202 position embeddings minus `<bos>`/`<eos>`), with unique ids and unique SMILES; at least 8 records and 3 per class, 2..20 classes. BYOD accepts CSV, JSON array or JSONL.",
        "- **Validation is syntactic, not chemical:** this repository ships no cheminformatics toolkit, so it checks characters, bracket balance and ring-digit pairing — not valence or chemical plausibility. Use RDKit before trusting a SMILES set.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — an unpublished or third-party proprietary structure is exactly that. The default path uploads nothing.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample molecules, validation and split\n\n"
                "The default dataset is generated in code with a fixed seed: 32 pairs, each a linear molecule and a branched "
                "**constitutional isomer** of it. Both members have the same molecular formula and the same terminal group; they "
                "differ only in where one carbon sits. `validate_dataset` checks the schema, the SMILES syntax, duplicate ids and "
                "SMILES, and class coverage before any model runs, and reports how many distinct formulas there are and how many "
                "of them appear in **both** classes — on the sample, all of them, which is the property the whole comparison rests "
                "on. `split_dataset` shuffles within each class and cuts 20 % validation / 25 % test.\n\n"
                "Look for: 64 molecules, classes `['branched', 'linear']`, 32 distinct formulas all shared across classes, splits "
                "36/12/16, and a written `outputs/{stem}_sample_dataset.csv` — the file shape BYOD expects. One honest caveat "
                "printed with them: a branched SMILES contains `(` and `)`, so a *character-level* baseline could separate these "
                "classes trivially; the baseline this notebook ships counts atoms, because molecular formula is the chemically "
                "meaningful confounder to rule out."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "VAL_FRACTION = 0.2  # @param {{type:\"number\"}}\n"
                "TEST_FRACTION = 0.25  # @param {{type:\"number\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    records = generate_sample_dataset()\n"
                "    data_source = f'synthetic constitutional-isomer dataset (seed {{SAMPLE_SEED}}, {{SAMPLE_SIZE}} molecules)'\n\n"
                "dataset_manifest = validate_dataset(records)\n"
                "CLASSES = dataset_manifest['classes']\n"
                "splits = split_dataset(records, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, seed=SEED)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_csv(records, 'outputs/{stem}_sample_dataset.csv')\n\n"
                "print({{'data_source': data_source, 'n_records': dataset_manifest['n_records'], 'classes': CLASSES, 'class_counts': dataset_manifest['class_counts']}})\n"
                "print({{'distinct_formulas': dataset_manifest['distinct_formulas'], 'formulas_shared_across_classes': dataset_manifest['formulas_shared_across_classes'], 'validation': dataset_manifest['validation']}})\n"
                "print({{'smiles_characters': dataset_manifest['smiles_characters'], 'ceilings': dataset_manifest['ceilings'], 'digest': dataset_manifest['digest'][:16] + '...'}})\n"
                "print({{'train': len(train_records), 'validation': len(val_records), 'test': len(test_records)}})\n\n"
                "if not USE_BYOD:\n"
                "    example_linear = next(r for r in records if r['label'] == 'linear')\n"
                "    example_branched = next(r for r in records if r['id'].endswith(example_linear['id'][-3:]) and r['label'] == 'branched')\n"
                "    print({{'pair': (example_linear['smiles'], example_branched['smiles']), 'formula': (formula_string(example_linear['smiles']), formula_string(example_branched['smiles']))}})\n"
                "    assert formula_string(example_linear['smiles']) == formula_string(example_branched['smiles'])\n"
                "    print('caveat: the branched SMILES contains ( and ), so a character-level baseline would separate these classes trivially; the formula baseline below deliberately counts atoms instead')"
            ),
        },
        {
            "md": (
                "## 5. How a molecule becomes tokens\n\n"
                "The tokenizer is a native `PreTrainedTokenizerFast` reading the checkpoint's own `tokenizer.json` — no remote "
                "code for this half. It splits SMILES into chemical tokens rather than characters: two-letter atoms such as `Cl` "
                "and `Br` stay one token, bracket atoms such as `[C@H]` stay one token, and branch parentheses and bond symbols "
                "are tokens of their own.\n\n"
                "The ceiling that follows from the checkpoint is `MAX_TOKENS = 200` (its 202 position embeddings minus `<bos>` and "
                "`<eos>`). A molecule past that is **refused rather than truncated** — a truncated SMILES is a different molecule, "
                "not a shorter one — and the cell demonstrates that refusal along with three syntax rejections."
            ),
            "code": (
                "for smiles in ['CCCCCCCO', 'CCCC(C)CCO', 'ClCCBr', 'C[C@H](N)C(=O)O', 'c1ccccc1O']:\n"
                "    print({{'smiles': smiles, 'tokens': pipe.token_count(smiles), 'formula': formula_string(smiles)}})\n\n"
                "print({{'max_tokens': MAX_TOKENS, 'max_position_embeddings': MAX_POSITION_EMBEDDINGS, 'max_molecules_per_call': MAX_MOLECULES_PER_CALL}})\n\n"
                "for probe in ['CCCC(CCO', 'CC*C', 'c1ccccO', ' CCO']:\n"
                "    try:\n"
                "        validate_inputs([probe])\n"
                "        print({{'probe': probe, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': probe, 'rejected': str(exc)[:100]}})\n"
                "try:\n"
                "    pipe.embed(['C' * (MAX_TOKENS + 1)])\n"
                "except ValueError as exc:\n"
                "    print({{'too_long': str(exc)[:130]}})\n\n"
                "input_manifest = validate_inputs([r['smiles'] for r in test_records[:4]], names=[r['id'] for r in test_records[:4]], token_counter=pipe.token_count)\n"
                "print({{'verdict': input_manifest['verdict'], 'n_molecules': input_manifest['n_molecules'], 'token_ceiling_checked': input_manifest['token_ceiling_checked'], 'requires_remote_code': input_manifest['requires_remote_code']}})"
            ),
        },
        {
            "md": (
                "## 6. Molecule embeddings (representation, not prediction)\n\n"
                "`pipe.embed` runs the verified encoder and returns one 768-dimensional vector per molecule: the mean of the last "
                "hidden state over non-padding tokens. Embeddings are representations — they carry no label and no metric of their "
                "own; a downstream labelled task is what gives them meaning (EVAL9). The cell embeds eight validation molecules, "
                "writes them with their ids to `outputs/{stem}_embeddings.csv` (OUT4), and prints the mean cosine similarity "
                "within and between classes as an inspection, not an evaluation.\n\n"
                "Because this package loads with `deterministic_eval=True`, running this cell twice gives exactly the same "
                "vectors. Under the upstream default it would not — see Section 3."
            ),
            "code": (
                "import csv\n"
                "import math\n\n"
                "embed_records = val_records[:8]\n"
                "embedding_result = pipe.embed([r['smiles'] for r in embed_records], names=[r['id'] for r in embed_records])\n"
                "vectors = embedding_result['embeddings']\n"
                "print({{'n_molecules': embedding_result['n_molecules'], 'dimension': embedding_result['dimension'], 'tokens': embedding_result['tokens'], 'pooling': embedding_result['pooling']}})\n\n"
                "repeat = pipe.embed([embed_records[0]['smiles']])['embeddings'][0]\n"
                "print({{'deterministic_eval': DETERMINISTIC_EVAL, 'same_vector_on_a_second_call': repeat == vectors[0]}})\n\n"
                "def cosine(a, b):\n"
                "    dot = sum(x * y for x, y in zip(a, b))\n"
                "    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))\n\n"
                "within, between = [], []\n"
                "for i in range(len(embed_records)):\n"
                "    for j in range(i + 1, len(embed_records)):\n"
                "        sim = cosine(vectors[i], vectors[j])\n"
                "        (within if embed_records[i]['label'] == embed_records[j]['label'] else between).append(sim)\n"
                "print({{'mean_cosine_within_class': round(sum(within) / len(within), 4) if within else None, 'mean_cosine_between_classes': round(sum(between) / len(between), 4) if between else None, 'note': 'inspection only; embeddings are unlabelled representations'}})\n\n"
                "with open('outputs/{stem}_embeddings.csv', 'w', encoding='utf-8', newline='') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['id', 'smiles', 'label'] + [f'dim_{{k}}' for k in range(embedding_result['dimension'])])\n"
                "    for r, vec in zip(embed_records, vectors):\n"
                "        writer.writerow([r['id'], r['smiles'], r['label']] + [f'{{x:.6f}}' for x in vec])\n"
                "print('wrote outputs/{stem}_embeddings.csv')"
            ),
        },
        {
            "md": (
                "## 7. Baselines on the test split\n\n"
                "Two trivial predictors set the floor before any training (EVAL10/EVAL11). `majority_baseline` predicts the most "
                "frequent training class — 0.5 on a balanced split. `formula_baseline` looks up each test molecule's **molecular "
                "formula** among the training formulas and predicts that formula's majority training class, falling back to the "
                "overall training majority for formulas it has not seen (SPL8: the lookup is built on the training split only).\n\n"
                "On this sample the formula baseline is structurally helpless, and the printout says why: every formula occurs "
                "exactly twice — once as a linear molecule, once as its branched isomer — so a formula seen in training is a coin "
                "flip, and a formula that was split across train and test is unseen and falls back. That is the control working as "
                "intended, not a bug. On your own data, read this baseline first: if the formula already predicts your label, the "
                "model does not have to learn any chemistry to score well."
            ),
            "code": (
                "baseline_majority = majority_baseline(train_records, test_records, CLASSES)\n"
                "print({{k: baseline_majority[k] for k in ('baseline', 'predicted_label', 'accuracy', 'macro_f1')}})\n"
                "baseline_formula = formula_baseline(train_records, test_records, CLASSES)\n"
                "print({{k: baseline_formula[k] for k in ('baseline', 'distinct_train_formulas', 'ambiguous_train_formulas', 'eval_formulas_unseen_in_train', 'accuracy', 'macro_f1')}})"
            ),
        },
        {
            "md": (
                "## 8. Bounded fine-tuning\n\n"
                "`pipe.adapt` builds `MolformerForSequenceClassification` from the verified checkpoint (the head is newly "
                "initialised — the load report says so), freezes every parameter except the head and the last `TRAINABLE_LAYERS` "
                "encoder layers, and runs AdamW with the hyperparameters below (FT4/FT6): tutorial values chosen for a few seconds "
                "of CPU, not production settings. Validation metrics are computed after each epoch for **monitoring only**; the "
                "final epoch's weights are kept, so no selection happens on the validation split (EVAL14). Training loss going "
                "down is optimisation evidence, not task-quality evidence (FT7) — Section 9 is where quality is measured.\n\n"
                "One MoLFormer-specific detail: during training the model is in `train()` mode, so its linear-attention random "
                "features **are** redrawn each step regardless of `deterministic_eval`. That is upstream's intended stochastic "
                "regularisation; it also means the feature buffers at the end of training differ from the checkpoint's, which is "
                "why Section 10 exports them with the adapter."
            ),
            "code": (
                "import time\n\n"
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_LAYERS = 2  # @param {{type:\"integer\"}}\n\n"
                "started = time.perf_counter()\n"
                "adapt_result = pipe.adapt(\n"
                "    train_records,\n"
                "    val_records,\n"
                "    classes=CLASSES,\n"
                "    epochs=EPOCHS,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    trainable_layers=TRAINABLE_LAYERS,\n"
                "    seed=SEED,\n"
                ")\n"
                "adapt_seconds = round(time.perf_counter() - started, 2)\n"
                "print({{'method': adapt_result['method'], 'trainable_parameters': adapt_result['trainable_parameters'], 'total_parameters': adapt_result['total_parameters'], 'precision': adapt_result['precision'], 'device': pipe.device, 'seconds': adapt_seconds}})\n"
                "for step in adapt_result['history']:\n"
                "    print(step)"
            ),
        },
        {
            "md": (
                "## 9. Held-out evaluation\n\n"
                "`pipe.evaluate` classifies every molecule of a split and reports `accuracy`, `macro_f1` (the unweighted mean of "
                "per-class F1, which exposes a model that ignores a class), per-class precision/recall/F1 with support, and "
                "`auroc` (ranking quality of the positive-class score, independent of the argmax threshold). The **test split** "
                "was never used for training or monitoring, so its numbers are the independent evidence (SPL6/SPL7). These are "
                "tutorial metrics on a synthetic 16-molecule split (EVAL6): one holdout, no dispersion estimate. The report, with "
                "both baselines and the deltas against them, is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "val_metrics = pipe.evaluate(val_records)\n"
                "test_metrics = pipe.evaluate(test_records)\n"
                "print({{'split': 'validation', **{{k: val_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}}}})\n"
                "print({{'split': 'test', **{{k: test_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}}}})\n"
                "for cls_name, row in test_metrics['per_class'].items():\n"
                "    print({{'class': cls_name, **row}})\n\n"
                "evaluation_report = {{\n"
                "    'task': 'molecular-property classification (bounded fine-tuning of MoLFormer-XL)',\n"
                "    'evidence': 'tutorial sample-sanity metrics on one stratified holdout; not a chemistry benchmark',\n"
                "    'estimation': 'single train/validation/test split, seed ' + str(SEED) + ', no dispersion estimate',\n"
                "    'data_source': data_source,\n"
                "    'dataset_digest': dataset_manifest['digest'],\n"
                "    'classes': CLASSES,\n"
                "    'splits': {{'train': len(train_records), 'validation': len(val_records), 'test': len(test_records)}},\n"
                "    'baselines': {{'majority': baseline_majority, 'formula': baseline_formula}},\n"
                "    'validation_metrics': val_metrics,\n"
                "    'test_metrics': test_metrics,\n"
                "    'delta_vs_majority': {{k: round(test_metrics[k] - baseline_majority[k], 4) for k in ('accuracy', 'macro_f1')}},\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k != 'trainable_parameter_names'}},\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "print({{'delta_vs_majority': evaluation_report['delta_vs_majority'], 'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 10. Inference on new molecules, artifact export and fresh reload\n\n"
                "`pipe.classify` returns, per molecule, the argmax `label`, its `score` and the full `scores` dictionary in class "
                "order. The scores are softmax outputs of a head trained on a few dozen molecules — **not calibrated "
                "probabilities** (UNC2); the only decision rule is argmax (UNC3).\n\n"
                "`pipe.save_artifact` then writes the trained tensors **plus the 12 linear-attention feature buffers** as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and revision, the "
                "class order, the tensor names, which of them are serving state, the file size and SHA-256, and the adaptation "
                "configuration (OUT8/ART3). Those buffers are not an afterthought: training redraws them, inference depends on "
                "them, and an adapter without them does not reproduce the adapted model. `MolformerPipeline.from_artifact` "
                "re-verifies the base snapshot, checks the artifact manifest and digests **before** deserialising, rebuilds the "
                "classifier and overlays the tensors — a fresh object from files, not the in-memory model (VER2). The cell asserts "
                "identical labels and scores within `1e-5` (VER4)."
            ),
            "code": (
                "if USE_BYOD:\n"
                "    new_records = test_records[:6]\n"
                "    new_source = 'first six BYOD test-split molecules'\n"
                "else:\n"
                "    new_records = generate_sample_dataset(seed=7, size=6)\n"
                "    new_source = 'freshly generated isomer pairs (seed 7)'\n"
                "inference_result = pipe.classify([r['smiles'] for r in new_records], names=[r['id'] for r in new_records])\n"
                "predictions = inference_result['predictions']\n"
                "print({{'new_source': new_source, 'decision_rule': inference_result['decision_rule']}})\n"
                "n_match = 0\n"
                "for p, r in zip(predictions, new_records):\n"
                "    n_match += p['label'] == r['label']\n"
                "    print({{'id': p['id'], 'smiles': p['smiles'], 'predicted': p['label'], 'score': round(p['score'], 4), 'true_label': r['label']}})\n"
                "print({{'matches': n_match, 'of': len(new_records), 'note': 'sanity check on generated labels, not an evaluation'}})\n\n"
                "with open('outputs/{stem}_predictions.csv', 'w', encoding='utf-8', newline='') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['id', 'smiles', 'tokens', 'predicted_label', 'score'] + [f'score_{{c}}' for c in CLASSES])\n"
                "    for p in predictions:\n"
                "        writer.writerow([p['id'], p['smiles'], p['tokens'], p['label'], f\"{{p['score']:.6f}}\"] + [f\"{{p['scores'][c]:.6f}}\" for c in CLASSES])\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "pipe.save_artifact(artifact_dir, metadata={{'data_source': data_source, 'dataset_digest': dataset_manifest['digest'], 'test_metrics': {{k: test_metrics[k] for k in ('n', 'accuracy', 'macro_f1', 'auroc')}}}})\n"
                "with open(artifact_dir / ARTIFACT_MANIFEST_NAME, encoding='utf-8') as f:\n"
                "    artifact_manifest = json.load(f)\n"
                "print({{'format': artifact_manifest['format'], 'base_model': artifact_manifest['base_model'], 'requires_remote_code': artifact_manifest['requires_remote_code'], 'deterministic_eval': artifact_manifest['deterministic_eval'], 'n_tensors': len(artifact_manifest['tensors']), 'serving_state_tensors': len(artifact_manifest['serving_state_tensors']), 'files': artifact_manifest['files']}})\n\n"
                "reloaded_pipe = MolformerPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR)\n"
                "reloaded_result = reloaded_pipe.classify([r['smiles'] for r in new_records], names=[r['id'] for r in new_records])\n"
                "max_score_diff = 0.0\n"
                "for before, after in zip(predictions, reloaded_result['predictions']):\n"
                "    assert before['id'] == after['id'] and before['label'] == after['label'], f'reload parity failure on {{before[\"id\"]}}'\n"
                "    max_score_diff = max(max_score_diff, abs(before['score'] - after['score']))\n"
                "assert max_score_diff < 1e-5, f'reload score drift {{max_score_diff}}'\n"
                "print({{'reload_parity': 'PASS', 'labels_equal': True, 'max_abs_score_diff': max_score_diff}})"
            ),
        },
        {
            "md": (
                "## 11. Result export and provenance\n\n"
                "The last output, `outputs/{stem}_result.json`, gathers what a reader needs to interpret the files above: the "
                "notebook source revision, the model id, immutable revision and licence, **the fact that remote code was executed "
                "and which files were verified first**, the `deterministic_eval` override, the dataset source and digest, the "
                "adaptation configuration, baseline and held-out metrics, the new-molecule predictions, the artifact manifest, the "
                "reload-parity result, and the runtime versions and device (OUT6/OUT7). No credential is involved anywhere in this "
                "notebook, so none can leak into it (OUT10)."
            ),
            "code": (
                "import platform\n\n"
                "result_payload = {{\n"
                "    'task': 'molecular-property classification adaptation (MoLFormer-XL)',\n"
                "    'pipeline_class': 'MolformerPipeline',\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'remote_code_executed': True,\n"
                "    'remote_code_files': list(REMOTE_CODE_FILES),\n"
                "    'deterministic_eval': DETERMINISTIC_EVAL,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'data_source': data_source,\n"
                "    'dataset_manifest': dataset_manifest,\n"
                "    'embedding_summary': {{'n_molecules': embedding_result['n_molecules'], 'dimension': embedding_result['dimension'], 'pooling': embedding_result['pooling']}},\n"
                "    'evaluation_report': evaluation_report,\n"
                "    'inference': {{'new_source': new_source, 'decision_rule': inference_result['decision_rule'], 'predictions': predictions}},\n"
                "    'artifact_format': ARTIFACT_FORMAT,\n"
                "    'artifact_format_version': ARTIFACT_FORMAT_VERSION,\n"
                "    'artifact_manifest': artifact_manifest,\n"
                "    'reload_parity': {{'labels_equal': True, 'max_abs_score_diff': max_score_diff}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'safetensors': safetensors.__version__,\n"
                "        'device': pipe.device,\n"
                "        'precision': 'float32',\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The fine-tuned head separates molecules that share a molecular formula and differ only in where one carbon sits — which "
        "the formula baseline cannot do, by construction. That is the claim: the model's representation carries structure, and a "
        "bounded adaptation can read it. The test split has 16 synthetic molecules, the metrics come from one seeded holdout with "
        "no dispersion estimate, and the classes are a generator rule rather than a measured property. So a perfect score says the "
        "adaptation contract works, not that MoLFormer predicts solubility, toxicity, binding affinity or any real endpoint.\n\n"
        "Three things to carry to real data. **Splits:** analogues from one chemical series share a scaffold, so a random split "
        "lets the model memorise it — use a scaffold or cluster split and expect lower, truer numbers. **Validation:** this "
        "repository checks SMILES *syntax*, not chemistry; run RDKit over your set before you trust it, because a syntactically "
        "fine string can be a chemically impossible molecule and the pipeline will happily embed it. **Applicability domain:** "
        "the pretraining corpus is drug-like PubChem and ZINC chemistry, so inorganics, organometallics, polymers and very large "
        "molecules are out of distribution with no error to warn you — and the 200-token ceiling refuses them rather than "
        "truncating.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model **and the Python it executes**, validate the demonstrated "
        "dataset contract, execute bounded fine-tuning, evaluate against trivial baselines on an independent split, and emit the "
        "shown machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark "
        "superiority, production fitness, or chemical validity.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_LAYERS = 0` to train the head alone and "
        "compare; lower `EPOCHS` to 1 to see an under-trained head where AUROC may be high while accuracy sits near 0.5; or bring "
        "your own labelled set through BYOD and read the formula baseline first — if it already separates your classes, your "
        "labels may be predictable from composition alone.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/molformer-chemistry-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/molformer-chemistry-pipeline/blob/main/MODEL_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/IBM/molformer\n"
        "- Ross, J., Belgodere, B., Chenthamarakshan, V., Padhi, I., Mroueh, Y., Das, P. (2021). Large-Scale Chemical Language "
        "Representations Capture Molecular Structure and Properties. arXiv:2106.09553. https://arxiv.org/abs/2106.09553"
    ),
}
