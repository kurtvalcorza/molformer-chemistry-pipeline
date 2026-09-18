# Weight provenance and DIMER hosting

This repository pins **one** snapshot with its own `dimer-base-manifest.json`.

## MoLFormer-XL both-10pct weights

- Upstream: `ibm-research/MoLFormer-XL-both-10pct`
- Immutable revision: `361063d0ad524ef77cf39b08469f6be770dc550f`
- Weight format: SafeTensors (`model.safetensors`, 187,248,784 bytes)
- Upstream weight license: Apache-2.0 (`license: apache-2.0` in the pinned upstream README front matter and in the Hub repository metadata)
- Local layout: `weights/molformer-xl-both-10pct/` holds the 7 manifest entries (`config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `configuration_molformer.py`, `modeling_molformer.py`, upstream `README.md`; 187,355,205 bytes total) with byte size and SHA-256 for each. `verify_snapshot()` in `src/molformer_chemistry_pipeline/pipeline.py` checks all of them before any load; `stage_missing_files(allow_download=True)` fetches only absent entries at the pinned revision. Neither `.safetensors` nor `.py` files under `weights/` are tracked by Git; the repository vendors neither the checkpoint nor the executable upstream code.
- Cross-check: the manifest's `model.safetensors` digest `0795977fe7192c4acdaf052f0e8464af57bc4bb59211271c5e61aaba2637b9c6` equals the `oid sha256` of the Hub LFS pointer at the pinned revision (`https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct/raw/361063d0ad524ef77cf39b08469f6be770dc550f/model.safetensors`), and is identical at the earlier `7b12d946` revision — the weights have not changed since the SafeTensors variant was added.

## The two Python files are manifest entries on purpose

`configuration_molformer.py` (7,101 bytes, SHA-256 `b88ea8d4b7b5e54f4f186cc7a230eff308928020a030f153a038fd12c05e3bed`) and `modeling_molformer.py` (36,884 bytes, SHA-256 `6f1ef72022de2c69e95661899422a7bb39a40a2cc5a6cb6216f14e9b7d84559c`) are **executed** by the loader, so they are treated as part of the snapshot rather than as incidental repository files. `verify_snapshot()` raises if a manifest omits either of them (`tests/test_pipeline.py::test_verify_snapshot_refuses_a_manifest_without_the_remote_code_files`), so the code is digest-verified before it is imported and an upstream change fails loudly instead of silently executing.

This is a narrowed trust boundary, not a safety claim. Digest verification proves the file is the one that was pinned; it says nothing about what that file does. No security review of the upstream code has been performed beyond reading its architecture. MODEL_ASSET_SPEC §12 RC4 (explicit code review before production qualification) is **not** satisfied.

## Files deliberately not staged

The upstream repository at the pinned revision carries three further files that the manifest deliberately omits: `convert_molformer_original_checkpoint_to_pytorch.py` (3,545 bytes, a one-off conversion script the loader never imports), `pipeline.jpeg` (241,085 bytes, a card illustration) and `.gitattributes` (1,519 bytes). There is **no** `pytorch_model.bin` at this revision — `model.safetensors` is the only weight file, so no pickle is ever a candidate for loading; the loader passes `use_safetensors=True` regardless. A DIMER profile upload for this model should use `model.safetensors` together with the two tokenizer files and the two Python files the loader executes.

## DIMER hosting

- Apache-2.0 permits use, modification, redistribution and commercial use subject to preservation of the licence and notices. DIMER may mirror the pinned snapshot in its model store under those terms; the weights would be redistributed unmodified.
- **Open gate — remote code.** `transformers` cannot build this checkpoint natively: `config.json` declares `model_type: "molformer"`, for which there is no native implementation, and the linear-attention architecture exists only in the upstream Python. `AutoModel.from_pretrained(..., trust_remote_code=False)` refuses it with a `ValueError` naming the custom code — verified, not assumed. Under MODEL_ASSET_SPEC §12 that makes the model not ordinarily qualified (RC6) until a hosting decision is made; it is **not** the RC2 failure mode of enabling remote code merely because loading failed, because the requirement is architectural. The same decision is pending for `nucleotide-transformer-genomics-pipeline`; rule on the two together.
- **Runtime gate.** A DIMER runtime for this model needs `transformers==5.17.0`, not the fleet's 4.57.6: the pinned upstream code calls `transformers.masking_utils.create_bidirectional_mask(inputs_embeds=...)`, whose current signature exists from Transformers 5.5.0.
- **Determinism.** The pinned `config.json` ships `deterministic_eval: false`, under which the linear-attention random feature maps are redrawn on every forward pass and repeated inference on the same molecule returns different embeddings. Any hosted profile for this model **must** load with `deterministic_eval=True`, as this pipeline does, and any exported adapter must carry the 12 `.feature_map.weight` buffers as serving state.
- Line endings: `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's newlines and break its recorded digest.
