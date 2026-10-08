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
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (generator /2.2): managed CPython, a
    # size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
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
        "After the tutorial workflow completes, set `USE_BYOD = True` and `BYOD_PATH` (a file already in the runtime; on Colab an empty path opens an upload dialog) in Section 4 and re-run from that cell to supply your own "
        "labelled molecules as a CSV (`id,smiles,label`), a JSON array or a JSONL file. They pass through the same SMILES "
        "validation, stratified split, baselines, adaptation, held-out evaluation, inference, artifact export and "
        "reload-parity cells as the synthetic sample. The expected schema, the accepted SMILES characters and the token "
        "ceiling are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is "
        "optional and never part of the default path."
    ),
    "pipeline_class": "MolformerPipeline",
    # MOL-M2: Section 3 must say what the load executes. The generator's default sentence ("no remote model code is
    # executed") is false for this checkpoint; validate_remote_code_boundary refuses it.
    "load_trust_note": (
        "**This load executes remote code.** `MolformerPipeline.from_pretrained` loads the tokenizer as a native fast "
        "tokenizer (`PreTrainedTokenizerFast`, which Transformers 5 prints as `TokenizersBackend`; `trust_remote_code=False`), but the model with `trust_remote_code=True`: Transformers "
        "imports and runs the checkpoint's own `configuration_molformer.py` and `modeling_molformer.py`, two of the 7 files "
        "verified above. That is unavoidable for this checkpoint: its `config.json` declares `model_type: \"molformer\"`, "
        "which Transformers has no native implementation of, so `trust_remote_code=False` refuses to load it. The SHA-256 "
        "check proves the code that runs is byte-for-byte the code at the pinned revision; it does **not** prove that code "
        "is safe or correct. Nobody has security-reviewed it here, so treat it like any third-party package you install. "
        "There is no fallback to a different download.\n\n"
        "**Why `deterministic_eval` is overridden.** MoLFormer's linear attention uses random feature maps. The upstream "
        "`config.json` ships `deterministic_eval: false`, which redraws those features on every forward pass, so the same "
        "molecule gets a slightly different vector (about 1e-3 apart) on each call. This package loads with "
        "`deterministic_eval=True`, which keeps the checkpoint's feature buffers fixed in evaluation, so repeated inference "
        "is reproducible; Section 6 checks it. Training (Section 8) still redraws them, as upstream intends.\n\n"
        "Depending on its logging settings, Transformers may also print two expected messages during this load: a note about instantiating a `molformer` model "
        "through `AutoModel`, and the checkpoint's `lm_head.*` weights reported as unused (they belong to the "
        "masked-language-model head, which embeddings and classification do not need). The cell ends with a one-line load "
        "summary."
    ),
    "load_summary": (
        "with open(WEIGHTS_DIR / 'config.json', encoding='utf-8') as handle:\n"
        "    upstream_deterministic_eval = json.load(handle).get('deterministic_eval')\n"
        "print({'remote_code_executed': True, 'remote_code_files_verified': list(REMOTE_CODE_FILES), 'tokenizer': type(pipe.tokenizer).__name__, 'upstream_deterministic_eval': upstream_deterministic_eval, 'deterministic_eval': DETERMINISTIC_EVAL, 'parameters': sum(p.numel() for p in pipe.model.parameters()), 'load_warnings': len(pipe.load_warnings)})"
    ),
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
    "guided": {"opening": [(
        "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter and has met the idea of a classifier, and wants to see how a chemical language model reads molecules written as SMILES, how to test whether it has learned structure rather than composition, and how to fine-tune it on a small labelled set and export the result. The audience is chemistry and data-science students and practitioners preparing their own molecular-property data; no prior experience with MoLFormer, transformers or fine-tuning is assumed — each term is explained where it first matters and again in the **Glossary**. CPU is enough: the default fine-tuning takes seconds.\n\n**Input → Model → Output.**\n\n| | Embeddings | Bounded fine-tuning and classification |\n|---|---|---|\n| Input | SMILES strings (at most 200 tokens each) | labelled records `{{id, smiles, label}}`: 64 generated constitutional-isomer molecules (36 train, 12 validation, 16 test) or your own CSV/JSON/JSONL |\n| Model | the MoLFormer-XL encoder (linear attention with random feature maps, loaded with deterministic evaluation) | the same encoder with a new classification head; the head and the last `TRAINABLE_LAYERS` encoder layers are trained with AdamW |\n| Output | one 768-dimensional mean-pooled vector per molecule | an argmax label and softmax scores per molecule, test accuracy / macro-F1 / AUROC beside two baselines, and a safetensors adapter that reloads with identical scores |\n\n**How to use this notebook.** Choose any runtime (CPU is enough), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed (the recorded hosted run of the previous version needed one; this version removes it). Sections 1–3 are **infrastructure** — the isolated environment, the carried modules and the verified snapshot with its remote code — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer from the recorded runs (the Kaggle T4 run of 18 September 2026 and the local CPU pre-flight, which agree). **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the isomer dataset, validation and split *(evaluation practice: a dataset built to defeat a shortcut)* → 5 how SMILES become tokens *(core concept)* → 6 embeddings *(core concept: representation, not prediction)* → 7 two baselines *(evaluation practice)* → 8 bounded fine-tuning *(core concept: what is trained)* → 9 held-out evaluation → 10 new molecules, export and fresh reload *(engineering)* → 11 result export → conclude."
    )]},
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, dataset and metrics modules; stage and digest-verify an "
        "immutable snapshot **including the Python the loader executes**; state what `trust_remote_code=True` executes and what "
        "the digest check does and does not protect against, and check that the `deterministic_eval` override makes repeated "
        "embeddings identical; validate SMILES syntactically and split a labelled "
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
        "- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with MoLFormer, transformers or fine-tuning. SMILES, tokens, embeddings, baselines, accuracy / macro-F1 / AUROC, epochs and adapters are explained where they are first used and again in the Glossary.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter and nothing is installed into it; a Windows or macOS kernel is not supported. CPU is enough — the default fine-tuning is a couple of seconds — and CUDA is used automatically when present.",
        "- **Knowledge:** how a molecule is written as SMILES, what a constitutional isomer is, and how accuracy, macro-F1 and AUROC differ.",
        "- **Remote code:** the default path executes the checkpoint's own `configuration_molformer.py` and `modeling_molformer.py` after verifying their SHA-256 against the inline manifest. Section 3 explains why that is unavoidable for this checkpoint.",
        "- **Runtime pin:** this repository pins `transformers==5.17.0`, not the fleet's 4.57.6, because the pinned upstream code calls an API that exists only in Transformers 5.5.0 and later.",
        "- **Data contract:** records are `{id, smiles, label}`; SMILES are non-empty strings over the accepted character set, at most 200 tokens (`MAX_TOKENS`, the checkpoint's 202 position embeddings minus `<bos>`/`<eos>`), with unique ids and unique SMILES, 2..20 classes. The validation and test splits are each held to at least 8 records and 3 per class, so for two balanced classes at the default split fractions you need **at least 36 records (18 per class)**; more classes or imbalance raise that, and Section 4 refuses a dataset that is too small (naming the split and the real minimum) before any model runs. BYOD accepts CSV, JSON array or JSONL, read from `BYOD_PATH` (works on Colab, Kaggle and Jupyter) or, on Colab with the path left empty, from an upload dialog.",
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
                "on. `split_dataset` shuffles within each class and cuts 20 % validation / 25 % test, then refuses the split if the "
                "validation or test part would fall below the contract (8 records, 3 per class) that Sections 8 and 9 check, so "
                "a too-small BYOD file stops here, with the minimum it needs, not after training.\n\n"
                "Look for: 64 molecules, classes `['branched', 'linear']`, 32 distinct formulas all shared across classes, splits "
                "36/12/16, and a written `outputs/{stem}_sample_dataset.csv` — the file shape BYOD expects (with BYOD on, your "
                "validated records are written to `outputs/{stem}_byod_dataset.csv` instead). One honest caveat "
                "printed with them: a branched SMILES contains `(` and `)`, so a *character-level* baseline could separate these "
                "classes trivially; the baseline this notebook ships counts atoms, because molecular formula is the chemically "
                "meaningful confounder to rule out.\n\n"
                "**Predict before running:** how many molecules, and how many distinct molecular formulas, will the sample have? "
                "If you knew only a molecule's formula, how well could you guess its class?"
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "VAL_FRACTION = 0.2  # @param {{type:\"number\"}}\n"
                "TEST_FRACTION = 0.25  # @param {{type:\"number\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "def byod_file(path, kind, suffixes=()):\n"
                "    \"\"\"BYOD path first (works on Colab, Kaggle and Jupyter); on Colab an empty path opens the upload dialog.\"\"\"\n"
                "    if str(path).strip():\n"
                "        source = Path(str(path).strip()).expanduser()\n"
                "        if not source.is_file():\n"
                "            raise FileNotFoundError(f'BYOD path {{str(source)!r}} does not exist or is not a file (relative paths start at {{os.getcwd()}}); give the path of one {{kind}}.')\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError(f'BYOD is on but its path field is empty, and the upload dialog exists only in Google Colab: copy the {{kind}} into this runtime (or attach it as a Kaggle dataset) and set the path field.') from None\n"
                "        uploaded = files.upload()\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one {{kind}} (received {{len(uploaded)}} files; a cancelled dialog sends none). Run this cell again.')\n"
                "        name, payload = next(iter(uploaded.items()))\n"
                "        source = Path('work') / Path(name).name\n"
                "        source.parent.mkdir(parents=True, exist_ok=True)\n"
                "        source.write_bytes(payload)\n"
                "    if suffixes and not source.name.lower().endswith(tuple(suffixes)):\n"
                "        raise ValueError(f'{{source.name}}: expected a {{kind}} ending in {{\" or \".join(suffixes)}}.')\n"
                "    return source\n"
                "\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    byod_path = byod_file(BYOD_PATH, 'labelled SMILES file (.csv, .json or .jsonl)', ('.csv', '.json', '.jsonl'))\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    data_source = 'BYOD (' + byod_path.name + ')'\n"
                "else:\n"
                "    records = generate_sample_dataset()\n"
                "    data_source = f'synthetic constitutional-isomer dataset (seed {{SAMPLE_SEED}}, {{SAMPLE_SIZE}} molecules)'\n\n"
                "dataset_manifest = validate_dataset(records)\n"
                "CLASSES = dataset_manifest['classes']\n"
                "splits = split_dataset(records, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION, seed=SEED)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "dataset_csv = 'outputs/{stem}_byod_dataset.csv' if USE_BYOD else 'outputs/{stem}_sample_dataset.csv'\n"
                "write_dataset_csv(records, dataset_csv)\n\n"
                "print({{'data_source': data_source, 'n_records': dataset_manifest['n_records'], 'classes': CLASSES, 'class_counts': dataset_manifest['class_counts']}})\n"
                "print({{'distinct_formulas': dataset_manifest['distinct_formulas'], 'formulas_shared_across_classes': dataset_manifest['formulas_shared_across_classes'], 'validation': dataset_manifest['validation']}})\n"
                "print({{'smiles_characters': dataset_manifest['smiles_characters'], 'ceilings': dataset_manifest['ceilings'], 'digest': dataset_manifest['digest'][:16] + '...'}})\n"
                "print({{'train': len(train_records), 'validation': len(val_records), 'test': len(test_records), 'written': dataset_csv}})\n\n"
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
                "<details><summary>Check your reasoning</summary>64 molecules in two balanced classes split 36 / 12 / 16, with all 32 formulas shared across the classes. "
                "Because every linear molecule has a branched isomer with the same formula, knowing the formula alone tells you "
                "nothing about the class — that is the whole point of the construction, and Section 7 measures it.</details>"
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
                "not a shorter one — and the cell demonstrates that refusal along with four syntax rejections (an unclosed branch, a character "
                "outside the accepted set, an unpaired ring digit and leading whitespace). The printed `tokens_incl_bos_eos` "
                "counts `<bos>` and `<eos>`; the 200-token ceiling does not.\n\n"
                "**Predict before running:** how many tokens will `ClCCBr` become — six characters, or fewer?"
            ),
            "code": (
                "for smiles in ['CCCCCCCO', 'CCCC(C)CCO', 'ClCCBr', 'C[C@H](N)C(=O)O', 'c1ccccc1O']:\n"
                "    print({{'smiles': smiles, 'tokens_incl_bos_eos': pipe.token_count(smiles), 'formula': formula_string(smiles)}})\n\n"
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
                "<details><summary>Check your reasoning</summary>In the recorded run `CCCCCCCO` became 10 tokens (eight atoms plus `<bos>` and `<eos>`), `ClCCBr` 6 tokens "
                "— `Cl` and `Br` stay single tokens — and `C[C@H](N)C(=O)O` 13 tokens, with the bracket atom `[C@H]` kept "
                "whole. A 201-token molecule is refused, not truncated: a cut SMILES is a different molecule.</details>"
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
                "Because this package loads with `deterministic_eval=True`, embedding the same molecules twice gives exactly the "
                "same vectors: the cell embeds the eight molecules a second time, prints whether every number is identical and "
                "asserts it. Under the upstream default (`deterministic_eval: false`, Section 3) the second call would differ by "
                "about 1e-3.\n\n"
                "A different, smaller effect is shown next to it: the first molecule embedded **on its own** differs from the same "
                "molecule inside the padded 8-molecule batch by around 1e-6. That is float32 rounding from padding and batch "
                "shape, not random features; the cell prints it against a stated tolerance of `1e-4` (far below the ~1e-3 "
                "random-feature drift) for inspection, not as a pass/fail check."
            ),
            "code": (
                "import csv\n"
                "import math\n\n"
                "embed_records = val_records[:8]\n"
                "embedding_result = pipe.embed([r['smiles'] for r in embed_records], names=[r['id'] for r in embed_records])\n"
                "vectors = embedding_result['embeddings']\n"
                "print({{'n_molecules': embedding_result['n_molecules'], 'dimension': embedding_result['dimension'], 'tokens': embedding_result['tokens'], 'pooling': embedding_result['pooling']}})\n\n"
                "second_call = pipe.embed([r['smiles'] for r in embed_records], names=[r['id'] for r in embed_records])['embeddings']\n"
                "identical = second_call == vectors\n"
                "print({{'deterministic_eval': DETERMINISTIC_EVAL, 'identical_vectors_on_a_second_identical_call': identical}})\n"
                "assert identical, 'deterministic_eval=True must make a repeated identical embed call return identical vectors'\n"
                "alone = pipe.embed([embed_records[0]['smiles']])['embeddings'][0]\n"
                "single_vs_batch = max(abs(a - b) for a, b in zip(alone, vectors[0]))\n"
                "print({{'single_molecule_vs_padded_batch_max_abs_diff': single_vs_batch, 'tolerance': 1e-4, 'within_tolerance': single_vs_batch <= 1e-4, 'cause': 'float32 rounding from padding and batch shape, not random features'}})\n\n"
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
                "model does not have to learn any chemistry to score well.\n\n"
                "On the default sample the cell also prints a third, deliberately crude rule that is **not** a chemistry "
                "baseline: \"a `(` in the SMILES means branched\". Section 4 warned that the branch parentheses make the "
                "classes separable at the character level; this rule measures it, so you can read Section 9's score against it.\n\n"
                "**Predict before running:** write down the test accuracy you expect from each baseline, and from the "
                "one-character rule."
            ),
            "code": (
                "baseline_majority = majority_baseline(train_records, test_records, CLASSES)\n"
                "print({{k: baseline_majority[k] for k in ('baseline', 'predicted_label', 'accuracy', 'macro_f1')}})\n"
                "baseline_formula = formula_baseline(train_records, test_records, CLASSES)\n"
                "print({{k: baseline_formula[k] for k in ('baseline', 'distinct_train_formulas', 'ambiguous_train_formulas', 'eval_formulas_unseen_in_train', 'accuracy', 'macro_f1')}})\n"
                "if not USE_BYOD:\n"
                "    rule_correct = sum(('(' in r['smiles']) == (r['label'] == 'branched') for r in test_records)\n"
                "    print({{'rule': 'one-character rule (default sample only): a ( in the SMILES means branched', 'correct': rule_correct, 'of': len(test_records), 'accuracy': round(rule_correct / len(test_records), 4)}})"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>In the recorded runs the majority baseline scored accuracy 0.5 and macro-F1 0.3333 on the balanced 16-molecule "
                "test split; the formula baseline scored 0.25 / 0.2 — below the majority, because formulas seen in training are "
                "coin flips and 9 test formulas were unseen and fell back. Neither can beat chance on this dataset by construction. "
                "The one-character rule, by contrast, scores 16 of 16: on this sample the class is visible in a single "
                "character, which matters when you read Section 9.</details>"
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
                "both baselines and the deltas against them, is written to `outputs/{stem}_evaluation_report.json`.\n\n"
                "**Predict before running:** after four epochs, will the fine-tuned model beat both baselines on the test split? "
                "By how much — and would a perfect score make you trust it more or less?"
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
                "<details><summary>Check your reasoning</summary>In the recorded runs train loss fell from 0.4516 to 0.0027 over four epochs, validation accuracy was 1.0 from "
                "epoch 1, and the test split scored accuracy, macro-F1 and AUROC all 1.0 (n = 16) against 0.5 and 0.25 for the "
                "baselines. A perfect score on 16 synthetic molecules is a saturated tutorial task: it says the adaptation "
                "contract works, not that the model predicts any real property. And the one-character rule of Section 7 also "
                "scores 16 of 16, so this sample cannot tell \"the model reads structure\" from \"the model reads the `(` "
                "token\". Lowering `EPOCHS` to 1 or training the head alone still gives 1.0 here; the first optional experiment "
                "at the end uses a change that does move the score.</details>"
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
                "identical labels and scores within `1e-5` (VER4).\n\n"
                "**Predict before running:** the reloaded pipeline is rebuilt from files on disk. Will its scores match the "
                "in-memory model exactly, approximately, or not at all?"
            ),
            "code": (
                "if USE_BYOD:\n"
                "    new_records = test_records[:6]\n"
                "    new_source = 'held-out BYOD test molecules (the first six of the test split, already scored in Section 9; not new data)'\n"
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
                "<details><summary>Check your reasoning</summary>Identical labels, and in the recorded runs a maximum absolute score difference of 0.0 for the 6 freshly "
                "generated molecules (all 6 classified correctly). Before the 12 linear-attention feature buffers were exported "
                "with the adapter, the same reload differed by 0.0023 — the buffers are serving state, not an afterthought.</details>"
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
        "the formula baseline cannot do, by construction. That rules out composition as the explanation. It does **not** show "
        "that the model reads chemical structure in any deeper sense: on this sample a one-character rule (a `(` in the SMILES "
        "means branched) scores the same 16 of 16, so the result is equally consistent with the model reading the branch token. "
        "Use BYOD with labels that no single character gives away for a real test. The test split has 16 synthetic molecules, the metrics come from one seeded holdout with "
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
        "**Optional experiments (they do not affect the default path).** For each one: write down a prediction, change the one "
        "form field in Section 8, then re-run **Sections 8, 9, 10 and 11 in order** (all four, so the evaluation report, the "
        "adapter and `result.json` describe the same run), compare with the default run, and explain the difference. Restore "
        "the default value and re-run the same four sections to return.\n\n"
        "1. **A step size too small to learn.** Set `LEARNING_RATE = 1e-6` (100 times smaller). *Predict:* will the test "
        "accuracy stay at 1.0? *What happened in our local CPU run:* train loss stayed near 0.8 for all four epochs (the "
        "default falls to about 0.003) and test accuracy dropped to 0.25, below the 0.5 majority baseline. With such small "
        "steps the newly initialised head hardly moves from its random start in 20 updates, so what it scores is that random "
        "start, not learning. Your exact numbers can differ on a GPU; the direction should not.\n"
        "2. **Less training on a saturated task.** Set `EPOCHS = 1`, or `TRAINABLE_LAYERS = 0` to train the head alone. "
        "*Predict:* does the test score fall? It does not: both still score 1.0 on this sample (validation accuracy is "
        "already 1.0 after the first default epoch). On a task this easy the score cannot rank the settings; the training "
        "loss and the trainable-parameter count are what change.\n"
        "3. **Your own data.** Bring a labelled set through BYOD and read the formula baseline first: if it already separates "
        "your classes, your labels may be predictable from composition alone.\n\n"
        "`pipe.adapt` builds a fresh classifier from the verified checkpoint on "
        "every call, so re-running Section 8 with other values never continues the previous adaptation.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"This notebook needs a Linux x86_64 runtime\"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n"
        "- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n"
        "- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n"
        "- **\"The isolated environment's Python process exited\"** — usually out of memory; restart the session and choose **Run all**.\n"
        "- **Section 3 reports a size or SHA-256 mismatch (weights or one of the two `.py` files)** — the message names the file. Delete the folder Section 3 prints as `weights_dir` and run Section 3 again; never edit the remote-code files.\n"
        "- **A SMILES is refused** — the message names the rule (character set, bracket balance, ring digits, or more than 200 tokens). Fix or drop that molecule; nothing is truncated.\n"
        "- **BYOD: \"BYOD path … does not exist\"** — the path is relative to the working directory printed in the message.\n"
        "- **BYOD: \"the upload dialog exists only in Google Colab\"** — on Kaggle or Jupyter, copy the file into the runtime and set `BYOD_PATH`.\n"
        "- **BYOD: \"Upload exactly one …\"** — the dialog was cancelled or several files were chosen; run Section 4 again.\n"
        "- **BYOD: a `validate_dataset` refusal** — it names the record and the rule (duplicate id or SMILES, fewer than 8 records or 3 per class, more than 20 classes).\n"
        "- **BYOD: \"the validation split would have … records\"** — your file passed validation but is too small to split: the validation and test splits each need 8 records and 3 per class. The message gives the per-class count to aim for (18 per class, 36 in total, for two balanced classes at the default fractions).\n\n"
        "## Glossary\n\n"
        "- **SMILES** — a line notation for a molecule: atoms as letters, branches in parentheses, ring closures as digits.\n"
        "- **Constitutional isomers** — molecules with the same formula and atoms but a different skeleton (here: one carbon moved to a methyl branch).\n"
        "- **Token** — the unit the tokenizer produces: an atom (`Cl`, `[C@H]`), a bond or a parenthesis; `MAX_TOKENS = 200`.\n"
        "- **Linear attention / random features** — MoLFormer's attention approximation; the random feature buffers are fixed for evaluation (`deterministic_eval`) and exported with the adapter.\n"
        "- **Remote code (`trust_remote_code`)** — Python shipped with the checkpoint that the loader executes; here both files are digest-verified first.\n"
        "- **Embedding** — a 768-dimensional vector per molecule (mean of the last hidden state); a representation, not a prediction.\n"
        "- **Majority / formula baseline** — always the most frequent training class; the training-majority class of the molecule's formula.\n"
        "- **Accuracy / macro-F1 / AUROC** — share correct; unweighted mean of per-class F1; ranking quality of the positive-class score, independent of the threshold.\n"
        "- **Epoch / learning rate / trainable layers** — one pass over the training molecules; the AdamW step size; how many final encoder layers are trained with the head.\n"
        "- **Validation vs test split** — monitored during training vs never touched until the final evaluation.\n"
        "- **Saturated task** — one where the model scores at the ceiling, so the score no longer distinguishes better from worse settings.\n"
        "- **Adapter / reload parity** — the trained tensors plus serving buffers, overlaid on the pinned base; the reloaded pipeline gives the same scores.\n"
        "- **BYOD** — bring your own data: your labelled SMILES through the same cells.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional — fill in from **your** run:\n\n"
        "- The baselines scored ___ (majority) and ___ (formula); the fine-tuned model scored ___ accuracy, ___ macro-F1, ___ AUROC on ___ test molecules.\n"
        "- I would / would not trust this number for a real property because ___ (for example the split, the dataset size, or saturation).\n"
        "- Reload parity: max score difference ___.\n"
        "- One change I would make before using my own data: ___.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/molformer-chemistry-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/molformer-chemistry-pipeline/blob/main/MODEL_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/IBM/molformer\n"
        "- Ross, J., Belgodere, B., Chenthamarakshan, V., Padhi, I., Mroueh, Y., Das, P. (2021). Large-Scale Chemical Language "
        "Representations Capture Molecular Structure and Properties. arXiv:2106.09553. https://arxiv.org/abs/2106.09553"
    ),
}
