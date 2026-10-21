# AGENTS.md

Train our own weights for Apple's SHARP (single image to 3D Gaussians, [paper](docs/paper/sharp-2512.10685.pdf), arXiv 2512.10685). This repo is a fork of [apple/ml-sharp](https://github.com/apple/ml-sharp). `upstream` is Apple. Upstream code lives in `src/sharp` and stays as close to Apple as possible. Training code lives in `src/sharp_train`.

**Skills are the source of truth. This file points, it does not copy.** Never edit a shared skill. Project rules stay in this file.

| Task | Skill |
| --- | --- |
| any Python: naming, design, a minimal diff | [clean-code](skills/clean-code/SKILL.md) |
| implementing a plan | [swarm](skills/swarm/SKILL.md) |
| stage 1 verification on Modal | [verify-ml-sharp](skills/verify-ml-sharp/SKILL.md) |

`skills/` is installed by `harness sync` from DGFX/skills at the commit in `skills.lock`. `verify-ml-sharp` is the only project-owned skill. `.agent`, `.agents`, `.claude`, `.cursor`, `.gemini`, and `.opencode` each link `skills` to `../skills`.

## Paper

`docs/paper/` has the PDF, a `pdftotext -layout` dump, and a markdown copy. The losses, loss weights, activations, learning-rate schedule and view-frustum mask are in Section 3.4 and Appendix A. [docs/training-spec.md](docs/training-spec.md) lists them with what this repo implements so far.

## Stage plan

These are our stages, not the paper's two-stage curriculum.

| Stage | Goal | Status |
| --- | --- | --- |
| 1 | Prove the training loop: view-pair loader, forward, gsplat render in both views, L1 on color and disparity. Gradient check, Apple-weights check, overfit check. | `skills/verify-ml-sharp` |
| 2 | Smoke run: Hypersim pairs, Gaussian grid 768 to 192, visibility mask on the novel view, short run, LPIPS and DISTS on held-out pairs against Apple weights, camera-move video. | `modal run modal_app.py --task stage2-checks`, then `--task stage2-train`, then `--task stage2-eval` |
| 3 | Full losses: perceptual (feature + Gram) and alpha BCE on the full novel view, regularizers, depth-correction losses. Finetune the stage 3 run 2 checkpoint at grid 384 (`A100-40GB`) or 768 (`H100`). Override the GPU with `--gpu` or `STAGE3_GPU`. | `modal run modal_app.py --task stage3 --mode check`, then `--mode train`, then `--mode eval` |

## Where things run

| Machine | Does |
| --- | --- |
| Modal | All GPU work: training, evaluation, renders. App `ml-sharp-train`, volume `ml-sharp-data` (`/vol/hypersim`, `/vol/checkpoints`, `/vol/runs`). |
| ai-vps (Oracle, no GPU) | Hypersim download and prep (`scripts/prepare_hypersim.py`), `modal volume put`, checkpoint backups, results preview. |
| Laptop or cloud agent | Code. Never run training locally. |

## Budget rules

- Modal GPU spend is capped by Dominik per task. Stage 1 and 2 run on `A100-40GB` ($2.10/h). Stage 3 stays on `A100-40GB` at grid 192 and 384. Grid 768 defaults to `H100`.
- Every Modal function sets `timeout`. Detached runs (`modal run --detach`) only with a timeout and a `max_minutes` budget inside the loop.
- Never leave an app running. After a run, `modal app list` must show nothing running for `ml-sharp-train`.
- Check spend with `modal billing report` before and after a run.
- Never print, commit or pass Modal tokens on a command line. Auth comes from `~/.modal.toml` on the machine that runs `modal`.

## Licenses

- Code in `src/sharp` and the repo: Apple sample-code license in [LICENSE](LICENSE). Changes are allowed; keep Apple's notice.
- Apple's released weights (`sharp_2572gikvuh.pt`): [LICENSE_MODEL](LICENSE_MODEL), research use only, no commercial use. Weights we train starting from Apple's weights, including the monodepth backbone we freeze, inherit that restriction.
- Hypersim: dataset under CC BY-SA 3.0, code under Apple's sample-code license. Credit Hypersim (Roberts et al., ICCV 2021) wherever renders are shown. Details in [docs/training-spec.md](docs/training-spec.md#data-and-license). Data is never committed; it lives on the Modal volume.
- Plan for more Hypersim scenes, own Blender data, bigger runs, and the open license questions on releasing weights: [docs/data-scaling.md](docs/data-scaling.md).

## Commands

| Command | Does |
| --- | --- |
| `uv pip install -e ".[train]"` | Python deps for prep and Modal (no private repos needed) |
| `uvx ruff@0.17.1 check src scripts modal_app.py` | Lint |
| `uvx ruff@0.17.1 format src/sharp_train scripts modal_app.py` | Format our code. Leave Apple's files alone. |
| `npm install` | Optional: installs the DGFX harness and syncs skills. Needs GitHub access and skips itself without it. |
| `npm run skills:check` | Vendored skills match `skills.lock` |
| `python scripts/prepare_hypersim.py --out DIR --scenes ai_001_001 ...` | Download a Hypersim subset by HTTP range requests and convert it |
| `modal run modal_app.py --task stage1` | Stage 1 checks |
| `python scripts/pick_eval_pairs.py --pairs DIR/pairs.json` | Regenerate the fixed eval list `src/sharp_train/eval_pairs.json` and the validation list `src/sharp_train/val_pairs.json` (held-out train scene `ai_011_001`) |
| `python scripts/compare_eval.py --base A/report.json --new B/report.json` | Paired eval comparison and the stage 3 pass gate (exit 1 on fail) |
| `PYTHONPATH=src python -m sharp_train.stage3 --data DIR` | Stage 3 CPU checks on Apple's weights and real pairs. `DIR` holds `ai_005_001`, `ai_007_001`, `ai_011_001`. |
| `PYTHONPATH=src python -m sharp_train.predict --checkpoint final.pt --image photo.jpg --out photo.ply` | One image to a `.ply` splat with our checkpoint. On Modal: `modal run modal_app.py --task predict --image photo.jpg --checkpoint runs/<run>/checkpoints/final.pt --out photo.ply`. |
| `PYTHONPATH=src python -m sharp_train.predict --check --image data/teaser.jpg` | CPU check: the `.ply` loads back and matches the model's own Gaussians (exit 1 on fail) |

## Always on

- No code comments in our code. No mock-style unit tests. Behavioral checks that run the real model on real data (`src/sharp_train/checks.py`) are required.
- Latest library versions, pinned in `modal_app.py`.
- Upstream inference code was written for inference: `apply_transform` and `decompose_covariance_matrices` in `src/sharp/utils/gaussians.py` detach and run SVD on the CPU, which cuts gradients to scales and rotations. Training must go through `sharp_train.model.to_metric`, which keeps covariances differentiable and passes them to gsplat directly.
- Merges into this repo's `main` only after `verify-ml-sharp` passes.
