# `proteinfoundation/inference/` — generated output (gitignored)

This directory holds generated output from Proteina inference/evaluation runs. Its contents are **not tracked in git** (see the parent `.gitignore`, where `proteinfoundation/inference/` is ignored and only this `README.md` is excepted).

## Why it exists

Upstream Proteina inference scripts (in `proteinfoundation/inference/*.py`) default to writing per-config run subdirectories and results CSVs here when invoked without an override for `ckpt_path`/output directory. Early April 2026 smoke / FID sweep runs populated this dir with:

- `inference_fid_60m_{baseline,repa,repa_layer0,repa_layer9,smoke}/` — per-config run dirs (typically `samples_fid/*.pdb` and other artefacts)
- `results_inference_fid_60m_*_fid.csv` — one-row CSVs of aggregate FID / fJSD / fS metrics per run

These have since been superseded by the lite-eval sweep under `evaluation/proteina/` (in the main repo), which writes to `evaluation/proteina/results/pdb/fid/` with richer per-step coverage. The artefacts here are kept on disk for provenance but are not canonical.

## Maintenance

- Safe to delete this entire directory (except `README.md`) at any time — nothing in the active pipeline reads from it.
- New inference runs should redirect output to `evaluation/proteina/results/...` rather than writing here.
- If the directory gets noisy, delete it; re-running an inference config locally will recreate the subdirs as needed.

## Future cleanup

Once we're confident no downstream analysis references these CSVs, the whole directory can be removed outright. Leaving this note so future-us doesn't re-investigate the same "what is this, can I delete it?" question.
