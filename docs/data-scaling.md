# Data scaling plan

Plan only. Nothing here has been run on Modal. Every number is tagged `measured` (a file, a header, a page or a script I ran, named in the text), `project` (given in the task or in this repo's docs), `estimate` (arithmetic shown) or `guess` (to calibrate). Fetched on 2026-10-21 against `origin/main` at `391a8ae`.

Goal: match Apple 768 on LPIPS, DISTS, holes and GT PSNR. Today the best checkpoint is E2 (`runs/stage3-g384-391a8ae-lr4e5-dc1/checkpoints/final.pt`).

| Inference setting, 26 eval pairs | LPIPS | DISTS | holes | frustum holes | GT PSNR |
| --- | --- | --- | --- | --- | --- |
| E2, grid 384 | 0.373 | 0.189 | 0.049 | 0.028 | 21.36 |
| Apple decoder at 384 | 0.378 | 0.204 | 0.079 | 0.008 | 23.00 |
| Apple 768 | 0.311 | 0.162 | 0.062 | 0.004 | 23.49 |

E2 already beats the Apple decoder at 384 on LPIPS, DISTS and holes. It loses on frustum holes and GT PSNR, and it is far from Apple 768 on LPIPS and DISTS (0.062 and 0.027).

The E2 numbers are for E2's decoder with Apple's monodepth head.4, not the head E2 trained. Every checkpoint saved before #10 dropped that trained 1x1 conv (66 parameters, the two disparity channels), so loading E2 `final.pt`, for eval or as an init, puts Apple's conv back. The trained conv is lost. How much this changes the numbers is not measured. Step A0 in section 5 measures it and gives E2 a matching head before stage A.

## Decisions needed from Dominik

1. A lawyer's answer on weight release (section 1) before any weights or renders leave the lab. Training on Modal can go ahead without it.
2. A spend cap per task. Section 5 gives the totals.
3. 150 or 300 scenes first. The sets are nested, so 150 can grow to 300 later.
4. Whether to keep scenes from the same Hypersim volumes as the eval and validation scenes out of training (section 3).

## 1. Hypersim license

This is not legal advice. It is a reading of the texts below by an engineer.

### What the sources say

Hypersim README, `https://raw.githubusercontent.com/apple/ml-hypersim/main/README.md` (fetched 2026-10-21):

> The Hypersim Dataset is licensed under the [Creative Commons Attribution-ShareAlike 3.0 Unported License](http://creativecommons.org/licenses/by-sa/3.0/).

> Note that our dataset is roughly 1.9TB. We have partitioned the dataset into a few hundred separate ZIP files, where each ZIP file is between 1GB and 20GB. Our [download script](code/python/tools/dataset_download_images.py) contains the URLs for each ZIP file. [Thomas Germer](https://github.com/99991) has generously contributed an [alternative download script](contrib/99991) that can be used to download subsets of files from within each ZIP archive.

> Note also that we manually excluded images containing people and prominent logos from our public release, and therefore our public release contains 74,619 images, rather than 77,400 images.

> To obtain the ground truth triangle meshes for each scene, you must purchase the asset files [here](https://www.turbosquid.com/Search/3D-Models?include_artist=evermotion).

> Our dataset: (1) relies exclusively on publicly available 3D assets;

> In order to generate the full Hypersim Dataset, we use Evermotion Archinteriors Volumes 1-55 excluding 20,25,40,49. All the Evermotion Archinteriors volumes are available for purchase [here](https://www.turbosquid.com/Search/3D-Models?include_artist=evermotion).

> In the Hypersim dataset, there is a small amount of asset reuse across scenes. This asset reuse is difficult to detect by analyzing metadata in the original scene assets, but it is evident when manually browsing through our rendered images. We do not attempt to address this issue when generating our split. Therefore, individual objects in our training images are occasionally also present in our validation and test images.

The README says nothing about training models on the images, about model weights, or about derivative works. The only dataset term it states is the CC BY-SA 3.0 pointer. It does not say what license the Evermotion assets carry, and it does not say that Evermotion agreed to the CC BY-SA 3.0 release of renders made from those assets.

Hypersim code license, `https://raw.githubusercontent.com/apple/ml-hypersim/main/LICENSE.txt`:

> Apple grants you a personal, non-exclusive license, under Apple's copyrights in this original Apple software (the "Apple Software"), to use, reproduce, modify and redistribute the Apple Software, with or without modifications, in source and/or binary forms; provided that if you redistribute the Apple Software in its entirety and without modifications, you must retain this notice and the following text and disclaimers in all such redistributions of the Apple Software.

That file covers the toolkit code ("SOFTWARE DISTRIBUTED IN THIS REPOSITORY"). Third-party parts are listed in `ACKNOWLEDGMENTS.txt`. It does not mention the images.

CC BY-SA 3.0 legal code, `https://creativecommons.org/licenses/by-sa/3.0/legalcode.txt`:

> "Adaptation" means a work based upon the Work, or upon the Work and other pre-existing works, such as a translation, adaptation, derivative work, arrangement of music or other alterations of a literary or artistic work, or phonogram or performance and includes cinematographic adaptations or any other form in which the Work may be recast, transformed, or adapted including in any form recognizably derived from the original, except that a work that constitutes a Collection will not be considered an Adaptation for the purpose of this License.

> 3(b) to create and Reproduce Adaptations provided that any such Adaptation, including any translation in any medium, takes reasonable steps to clearly label, demarcate or otherwise identify that changes were made to the original Work.

> 4(b) You may Distribute or Publicly Perform an Adaptation only under the terms of: (i) this License; (ii) a later version of this License with the same License Elements as this License; (iii) a Creative Commons jurisdiction license (either this or a later license version) that contains the same License Elements as this License (e.g., Attribution-ShareAlike 3.0 US)); (iv) a Creative Commons Compatible License.

> 4(c) ... keep intact all copyright notices for the Work and provide, reasonable to the medium or means You are utilizing: (i) the name of the Original Author ... (ii) the title of the Work if supplied; (iii) to the extent reasonably practicable, the URI, if any, that Licensor specifies to be associated with the Work ... and (iv) , consistent with Ssection 3(b), in the case of an Adaptation, a credit identifying the use of the Work in the Adaptation

> 2. Fair Dealing Rights. Nothing in this License is intended to reduce, limit, or restrict any uses free from copyright or rights arising from limitations or exceptions that are provided for in connection with the copyright protection under copyright law or other applicable laws.

Apple model license, `LICENSE_MODEL` in this repo:

> ... to use, copy, modify, distribute, and create Model Derivatives (defined below) of the Apple Machine Learning Research Model exclusively for Research Purposes.

> "Research Purposes" does not include any commercial exploitation, product development or use in any commercial product or service.

> "Model Derivatives" means any models or any other artifacts created by modifications, improvements, adaptations, alterations to the architecture, algorithm or training processes of the Apple Machine Learning Research Model, or by any retraining, fine-tuning of the Apple Machine Learning Research Model.

> If you choose to redistribute Apple Machine Learning Research Model or its Model Derivatives, you must provide a copy of this Agreement to such third party, and ensure that the following attribution notice be provided: "Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement." Additionally, all Model Derivatives must clearly be identified as such, including disclosure of modifications and changes made to the Apple Machine Learning Research Model.

### What this means

| Use | Reading | Confidence |
| --- | --- | --- |
| Download Hypersim and copy it to a Modal volume | CC BY-SA 3.0 §3(a) grants the right to "Reproduce the Work". No extra terms found. | Reasonable |
| Convert it (tone map, planar depth, PNG, npy) and keep the result on the volume | These are Adaptations under §1(a). §3(b) allows them if the change is labelled. We should label the change: write a `NOTICE` file on the volume that names Hypersim, the paper, the URL, and lists what `prepare_hypersim.py` does. | Reasonable |
| Train a model on the images | The README is silent. The CC text has no clause on training. Whether training makes the weights an Adaptation is not settled in any text I found, and I cannot settle it. Fair-dealing or text-and-data-mining exceptions (§2) depend on the jurisdiction. | Open question |
| Release weights | Two separate limits. (a) Apple's `LICENSE_MODEL` applies to anything fine-tuned or retrained from Apple's checkpoint: research only, no commercial use, copy of the agreement, the attribution notice, and marked as a Model Derivative. (b) If weights count as an Adaptation of Hypersim, §4(b) requires CC BY-SA 3.0 (or a compatible license) and says you "may not offer or impose any terms on the Adaptation that restrict" the recipient's rights under it. A research-only, no-commercial license restricts those rights. The two would conflict. | Blocking until a lawyer answers |
| Release renders made from Hypersim input images | A render recognizably derived from a Hypersim image is an Adaptation. Release it under CC BY-SA 3.0 with the §4(c) credit (Roberts et al., ICCV 2021, the dataset URL, "adapted"). `AGENTS.md` already says this. | Reasonable |
| Release renders made from other images | The Hypersim terms do not apply. `LICENSE_MODEL` still governs the model. | Reasonable |

Our weights inherit `LICENSE_MODEL`. The decoder and prediction head of E2 were reset and retrained (`model.reset_gaussian_decoder`), but the frozen monodepth backbone, the initializer and the depth alignment come from Apple's checkpoint, and the training starts from the stage 3 run 2 checkpoint. I did not check whether the monodepth backbone has a separate license of its own.

### What blocks or may block a release

1. **Commercial use is blocked by `LICENSE_MODEL`**, whatever data we use. More Hypersim scenes or our own Blender scenes do not change it.
2. **ShareAlike against research-only.** If a court treats the weights as an Adaptation of Hypersim, a research-only release violates §4(b). If it does not, only attribution is left. This is the main open question.
3. **Evermotion.** The images are renders of purchased Evermotion Archinteriors scenes. The README says the assets are "publicly available" and that meshes must be bought. It does not say Evermotion's terms allow CC BY-SA redistribution of renders. I could not verify this from the repo. If the chain is broken upstream, no downstream license can repair it.
4. **Our own data.** Blender scenes that Dominik owns carry none of these limits. They help with items 2 and 3 but not item 1.

### Questions for a lawyer

1. Is a model trained on CC BY-SA 3.0 images an "Adaptation" under §1(a), in the jurisdictions we release in?
2. If yes, can weights be released under a research-only license, or must they be CC BY-SA 3.0 too? Does that clash with `LICENSE_MODEL`, which we cannot relicense?
3. Did Apple have the right to release renders of Evermotion assets under CC BY-SA 3.0? Do Evermotion's terms restrict what we do with them?
4. Does `LICENSE_MODEL` reach weights whose decoder was re-initialized and retrained, when the frozen backbone is Apple's?
5. What attribution text and notice files must ship with weights, renders and videos?

## 2. Where to prep: Modal CPU containers

### Why not ai-vps

The disk is 84 % full (160G of 192G) and the machine is loaded (load average 18 on 4 cores when I checked). 150 scenes is about 23 GB of download and 15 GB on disk. Prep runs on Modal CPU containers and writes into the `ml-sharp-data` volume. Nothing is downloaded to ai-vps.

### Where the data is

`https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip`. A `HEAD` request on 2026-10-21 returned `200`, `Accept-Ranges: bytes`, `Server: AmazonS3`, served through Apple's CDN, `Last-Modified: Tue, 29 Dec 2020`. The download script lists 457 scene zips. `prepare_hypersim.py` reads each zip with `remotezip` (HTTP range requests), so it fetches only the central directory and the files it needs.

### What the script fetches per scene

Per frame it reads two entries from the zip: `images/scene_cam_00_final_hdf5/frame.NNNN.color.hdf5` and `images/scene_cam_00_geometry_hdf5/frame.NNNN.depth_meters.hdf5`. Per scene it reads two small pose files (`camera_keyframe_positions.hdf5`, `camera_keyframe_orientations.hdf5`). Only `cam_00` and only frames listed as `included_in_public_release` are used.

I read the zip central directories with range requests and summed the entry sizes for frames 0 to 39 (`measured`):

| Scene | Zip size | Color, 40 frames | Depth, 40 frames | Fetched | Share of zip |
| --- | --- | --- | --- | --- | --- |
| ai_005_001 (eval) | 3,047,314,871 B | 150,539,947 B | 14,011,963 B | 164.6 MB | 5.4 % |
| ai_007_001 (eval) | 4,612,374,491 B | 133,675,708 B | 9,027,041 B | 142.7 MB | 3.1 % |
| ai_011_001 (val) | 2,156,333,113 B | 139,073,712 B | 9,823,178 B | 148.9 MB | 6.9 % |
| ai_013_001 | 3,719,332,262 B | 143,415,727 B | 9,240,549 B | 152.7 MB | 4.1 % |

The mean is 152.2 MB per 40-frame scene. The entries are stored without compression (`compress_size == file_size`). The central directory costs 0.5 to 1.1 MB per scene. The script default is `--frames 30`, but the three prepared scenes have 40 frames, so use `--frames 40`.

### Output size

Prepared size on disk per 40-frame scene, `measured` in `/home/ubuntu/data/hypersim-cpu/`: 106M, 93M, 96M. Mean 98.3 MB. The parts are 40 PNG files (about 47 MB) plus 40 float16 `.npy` depth files (1024 x 768 x 2 B = 1.57 MB each, 63 MB).

| Scenes (new, on top of the 51 train scenes) | Download | On volume (new) | Total volume (all scenes) | Inodes (about 83 per scene) |
| --- | --- | --- | --- | --- |
| 52 today | - | - | 5.1 GB | about 4.3k |
| 150 total (99 new) | 99 x 152.2 MB = 15.1 GB | 99 x 98.3 MB = 9.7 GB | 150 x 98.3 MB = 14.7 GB | about 12.5k |
| 300 total (249 new) | 249 x 152.2 MB = 37.9 GB | 249 x 98.3 MB = 24.5 GB | 300 x 98.3 MB = 29.5 GB | about 24.9k |

Modal's volume page (`https://modal.com/docs/guide/volumes`, fetched 2026-10-21) says: "Volumes work best when they contain less than 50,000 files and directories" and "There is currently a hard limit of 500,000 inodes". 300 scenes stays under both. I cannot run `modal volume` here, so I could not check whether `ml-sharp-data` is a v1 or v2 volume or how full it is.

### Modal prices

From `https://modal.com/pricing`, fetched 2026-10-21:

| Item | Price | Per hour |
| --- | --- | --- |
| CPU | $0.0000131 per physical core per second (2 vCPU), minimum 0.125 core per container | $0.0472 per core |
| Memory | $0.00000222 per GiB per second | $0.0080 per GiB |
| Volumes | $0.09 per GiB per month, 1 TiB per month free | - |
| Network egress | $0.04 per GiB | - |
| A100 40 GB | $0.000583 per second | $2.099 |
| H100 SXM5 | $0.001097 per second | $3.949 |

The page lists Egress but no ingress price. I assume downloads into Modal cost nothing and the first dry run will show it on the bill. Functions run at these (preemptible) rates. The 3x non-preemptible rate applies to Sandboxes, which we do not use. Prep is idempotent per scene, so a preemption only costs a retry.

The project figure of $2.23 per hour for the A100 job is 6.3 % above the $2.099 list GPU rate (2.23 / 2.099 = 1.063). I take the gap to be CPU and memory of the training container and did not verify it. I apply the same 1.063 factor to the H100 (3.949 x 1.063 = $4.20 per hour, `estimate`).

### Cost of prep (estimate)

CPU time per scene was not measured: the VPS is too loaded for a fair timing and has no h5py in the project venv. Assumption per frame: 0.5 s of single-core work (hdf5 decode, tone map percentile over 786k values, planar depth, PNG encode of 1024 x 768 RGB). 40 frames = 20 s. Pair selection loads up to 8 candidate depth maps per frame, about 3 s. Zip open and range reads add a few seconds. Round to 60 s per scene, which is a 2x margin.

Memory: the script keeps all frames in float32 before writing. One frame is 1024 x 768 x 3 x 4 B = 9.4 MB color plus 3.1 MB depth. 40 frames = 503 MB. Request 4 GiB.

Rate with 1 core and 4 GiB: 0.0000131 + 4 x 0.00000222 = $0.00002198 per second.

| Case | Per scene | 99 new scenes | 249 new scenes |
| --- | --- | --- | --- |
| Base: 60 s | $0.0013 | $0.13 | $0.33 |
| Worst: 300 s | $0.0066 | $0.65 | $1.64 |
| Dry run of the whole candidate pool: 427 scenes x about 10 s, about 1.3 MB each (`measured` on 30 scenes: 39,977,145 B / 30) | $0.0002 | - | $0.09 for 427 |

Prep cost is negligible. The real limits are download speed from Apple's CDN and wall time. At 5 containers and 60 s per scene, 249 scenes take 249 x 60 / 5 = 50 min.

### Function shape to add to `modal_app.py` later

A separate light image, not the CUDA image. The existing image is large and has no hdf5 or remotezip.

```python
prep_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("numpy", "pillow", "h5py", "remotezip")
    .add_local_dir("scripts", "/root/repo/scripts")
)


@app.function(
    image=prep_image,
    cpu=1.0,
    memory=4096,
    timeout=30 * 60,
    retries=2,
    max_containers=5,
    volumes={VOLUME_ROOT: volume},
)
def prep_scene(scene: str, frames: int = 40, dry_run: bool = False) -> dict:
    volume.reload()
    scene_dir = DATA_ROOT / scene
    if (scene_dir / "pairs_scene.json").exists() and not dry_run:
        return {"scene": scene, "status": "exists"}
    work = Path("/tmp/prep")
    report = prepare_and_filter(scene, frames, work, dry_run)
    if report["status"] == "ok" and not dry_run:
        shutil.copytree(work / scene, scene_dir, dirs_exist_ok=True)
        volume.commit()
    return report
```

Notes on the shape:

- `max_containers=5` follows the volume page: "If you are committing small changes, avoid doing more than 5 concurrent commits". Check the argument name against the pinned `modal` version when implementing.
- `prepare_and_filter` wraps `prepare_scene` and `select_pairs` (import them from `scripts/prepare_hypersim.py` or move them to `src/sharp_train/prep.py`) and the filters of section 3.
- The scene is written to container disk first. `pairs_scene.json` is written last and doubles as the done marker, so a retry after a partial copy redoes the scene.
- Memory for 40 frames is 0.5 GB, but billing is `max(request, use)`, so 4 GiB costs only the margin.
- Do not call the script's `main()` for new scenes. It ends with `(args.out / "pairs.json").write_text(...)`, which would overwrite the volume's `pairs.json` with only the scenes of that run.

Entry points (new `main` parameters, not present today):

```
modal run modal_app.py --task prep --prep-scenes scenes_candidates.txt --dry-run 1
modal run modal_app.py --task prep --prep-scenes scenes_150.txt --dry-run 0
modal run modal_app.py --task merge-pairs
```

`prep` runs `prep_scene.map(scenes, order_outputs=False, return_exceptions=True)` and writes `prep_report.json` to the volume. `merge-pairs` is one small CPU function. It copies the current `pairs.json` to `pairs.52.json` once (never overwritten), then writes `pairs.json` from all `pairs_scene.json` files plus the existing entries, and commits.

### Steps

1. Dry run over the candidate pool (about 400 scenes). It reads the zip directory, the pose files and 4 depth frames per scene, applies the scene filters of section 3, and writes nothing to the dataset. It also lists the scenes already on the volume. Cost under $0.10.
2. Dominik or the orchestrator reads `prep_report.json`, fixes the 150 and 300 lists, and checks the excluded scenes by eye.
3. Real prep of the 99 new scenes. Then `merge-pairs`.
4. Run the pair and file checks of section 3 on the merged `pairs.json` (CPU).
5. `--task stage3 --mode check` on the new pool (section 5).

## 3. Scene selection, held-out sets, pairs, quality filters

### Fixed sets

| Set | Scenes | Pairs | Source |
| --- | --- | --- | --- |
| Eval, forever | ai_005_001, ai_007_001 | 26 (13 each) | `src/sharp_train/eval_pairs.json` |
| Validation | ai_011_001 | 8 | `src/sharp_train/val_pairs.json` |
| Train | everything else in `pairs.json`, or the list passed with `--train-scenes` | 6015 over 51 scenes today (`project`) | `stage2.held_out_split` |

`held_out_split` removes the eval and validation scenes from whatever is in `pairs.json`. Adding scenes to `pairs.json` therefore adds them to the default train set. Pass `--train-scenes` to pin the set. The comma list for 300 scenes is about 3.4 kB, which fits on a command line.

Do not regenerate `eval_pairs.json` (`docs/training-spec.md` explains why).

### Candidate pool (measured from the Hypersim CSVs)

From `metadata_camera_parameters.csv`, `metadata_images_split_scene_v1.csv` and `metadata_camera_trajectories.csv` in `apple/ml-hypersim` (fetched 2026-10-21):

| Step | Scenes left |
| --- | --- |
| In the camera table | 482 |
| Pinhole (`|M_proj_02|` and `|M_proj_12|` below 1e-6, as the script checks) | 460 |
| Pinhole and with public `cam_00` frames | 427 |
| And at least 20 public frames among the first 40 | 416 |
| Minus the 3 held-out scenes | 413 |
| Minus the other scenes of volumes 005, 007, 011 (20 scenes) | 393, in 48 volumes |
| Of those, in Hypersim's own `train` partition | 314 |

All 482 scenes are 1024 x 768. 394 of the 449 scenes with public `cam_00` frames have all of the first 40 frames. Frames from trajectories marked `OUTSIDE VIEWING AREA` or `CONTENT FLAGGED FOR REMOVAL` have `included_in_public_release = False`, and the script already skips them (the CSV shows no row with an exclude reason and `True`).

The README adds: "We recommend excluding these scenes and camera trajectories in downstream learning applications". The script's `included` filter follows that already.

Volume siblings. Hypersim has about 10 scenes per Evermotion volume and the README says assets are reused across scenes. The eval scenes are ai_005_001 (Hypersim `test` partition) and ai_007_001 (`val`), the validation scene is ai_011_001 (`train`). I suggest keeping the other 20 scenes of volumes 005, 007 and 011 out of the new additions, so that eval objects and styles are not in training. I do not know whether the current 51 train scenes already include some of them. The first dry run lists the volume's scenes, so we can see. If they are in, keep them for nesting and report it.

### Selection rule

- Nested sets: S51 (today's train scenes) inside S150 inside S300. Runs on different sizes stay comparable.
- New scenes are taken round-robin over volumes (about 2 per volume for 150, 5 per volume for 300 over 48 volumes), Hypersim `train` partition first, fixed seed 0, so one volume or one room type does not dominate. Living rooms are 121 of 427 candidates (28 %).
- Every candidate passes the filters below first.

### Pair generation (reuse the existing logic)

`select_pairs` in `scripts/prepare_hypersim.py`, defaults unchanged:

- For each frame, pair it with the next 4 frames (`--max-offset 4`), in both directions (src/tgt swapped).
- Keep a pair when the camera baseline is in 0.05 to 1.5 m (`--min-baseline`, `--max-baseline`) and the frustum overlap is in 0.5 to 0.995 (`--min-overlap`, `--max-overlap`). Overlap is the share of the target view's depth samples (stride 8) that land inside the source image with 5 % margin.
- Entry format: `{"scene", "src", "tgt", "baseline", "overlap"}`. `src` and `tgt` are 4-digit strings.

On the three prepared scenes (`measured`, `/home/ubuntu/data/hypersim-cpu/pairs.json`):

| Scene | Frames | Pairs | Baseline min / median / max (m) | Overlap min / median / max | Camera step between consecutive frames, median (m) |
| --- | --- | --- | --- | --- | --- |
| ai_005_001 | 40 | 118 | 0.129 / 1.040 / 1.495 | 0.540 / 0.881 / 0.993 | 1.22 |
| ai_007_001 | 40 | 131 | 0.120 / 0.890 / 1.495 | 0.503 / 0.739 / 0.994 | 1.12 |
| ai_011_001 | 40 | 144 | 0.226 / 1.115 / 1.485 | 0.515 / 0.867 / 0.993 | 1.21 |

Pairs per scene: 118, 131, 144 (mean 131). The 51 train scenes average 6015 / 51 = 117.9 (`project`). I use 117.9 for the arithmetic below.

### Measured scale reference

I read the depth maps and poses of other scenes with range requests (about 1.3 MB per scene, nothing stored). 30 random scenes from the Hypersim `train` partition (pinhole, seed 0, not ai_005_001, ai_007_001 or ai_011_001; the sample includes two scenes of volume 005). Distance is Euclidean, in meters, from two frames per scene (0 and 20):

| Quantity | Min | p10 | Median | p90 | Max |
| --- | --- | --- | --- | --- | --- |
| Median distance per scene (m) | 3.15 | 4.2 | 5.5 | 10.2 | 44.5 (ai_026_001, art gallery) |
| p95 distance per scene (m) | 4.1 | 6.5 | 12.5 | 36.7 | 63.0 |
| Non-finite or non-positive depth fraction | 0 | 0 | 0 | 0.042 | 0.080 (ai_048_006) |
| Median camera step per scene (m) | 0.947 | 1.04 | 1.159 | 1.342 | 1.377 |
| Camera step / median distance | 0.027 | 0.133 | 0.199 | 0.301 | 0.367 |

The prepared eval and validation scenes (planar depth) have per-frame median depth of 9.2 to 24.7 m (ai_005_001, a church), 3.1 to 6.0 m (ai_007_001) and 6.4 to 12.4 m (ai_011_001).

### The known bad case: ai_013_001

The repo has no record of it. `grep -rn ai_013` over the repo and `git log -S` find nothing, so I cannot tell what went wrong when it was used. The Hypersim trajectory table lists it as `Other`, note `Colosseum (outdoor)`. I measured it with range requests on four frames (0, 10, 20, 30):

| Quantity | ai_013_001 | ai_005_001 (church, for comparison) |
| --- | --- | --- |
| Metres per asset unit (CSV) | 0.01 | 0.0254 |
| Median Euclidean distance | 390 to 400 m | 17.5 to 25.4 m |
| p05 / p95 distance | about 258 m / 1740 m | 5.4 to 6.1 m / 84 to 93 m |
| Median camera step, frames 0 to 39 | 1.131 m | 1.220 m |
| Camera step / median distance | 0.0029 | about 0.057 |

The camera moves about 1.1 m per frame, like every other scene, but the scene is 400 m away. Baseline over depth is ten times below the lowest value in the 30-scene sample. A 1 m move at 400 m gives almost no parallax, so the pair carries no signal for novel views. The depth scale itself may be correct (a Colosseum is large). I do not know whether the earlier problem was this or a unit error. Either way the scene should be filtered, and the filters below catch it.

### Filters

Thresholds marked `guess` are starting values. Calibrate them on the dry-run report of the whole candidate pool, and on the 52 scenes already on the volume, before the real prep.

| # | Name | Metric | Threshold | Action | Basis |
| --- | --- | --- | --- | --- | --- |
| 1 | Not pinhole | `|M_proj_02|` or `|M_proj_12|` at least 1e-6 | any | drop scene (the script does this today) | measured: 22 of 482 scenes |
| 2 | Hypersim exclusion | `included_in_public_release` false (covers `OUTSIDE VIEWING AREA` and flagged content) | any frame | drop frame (the script does this today) | measured |
| 3 | Too few frames | public frames among the first 40 | below 20 | drop scene | guess. 416 of 427 pass. The sample had 23, 35, 39 |
| 4 | Missing files | color or depth entry missing in the zip, or `cameras.json` missing a pose for a kept frame | any | drop frame. Drop scene if filter 3 then fails | rule, no threshold |
| 5 | Corrupt files | PNG does not open as RGB at `(width, height)`; depth `.npy` shape is not `(height, width)`; any pose with a non-finite value | any | drop frame | rule, no threshold |
| 6 | Bad poses | `|det(R) - 1|` or max `|R^T R - I|` above 1e-3; or a jump between consecutive frames above 5 m | any | drop scene | guess. Measured steps: median 0.95 to 1.38 m, largest 1.94 m in the three prepared scenes |
| 7 | Non-finite depth | share of non-finite or non-positive depth per frame | above 0.25 | drop frame | guess. Measured: prepared scenes at most 0.095, the 30-scene sample at most 0.080 |
| 8 | Non-finite depth, scene | mean of filter 7's metric over the kept frames | above 0.10 | drop scene | guess, same data |
| 9 | Depth scale, too far | scene median Euclidean distance D (m), median over frames 0, 10, 20, 30 of the per-frame median | above 60 m; review band 30 to 60 m | drop scene; list the review band for a look | measured: sample max 44.5 m, ai_013_001 395 m. Cut at 60 is a guess between them |
| 10 | Depth scale, too near | same D | below 1.0 m | drop scene | guess. Sample min 3.15 m |
| 11 | Degenerate scene baseline | R = median consecutive camera step / D | below 0.01 | drop scene | measured: sample min 0.027, ai_013_001 0.0029. The 0.01 cut is a guess between them |
| 12 | Monodepth against GT scale | median(Apple monodepth metric depth / GT planar depth) on one frame per scene | outside 0.5 to 2.0 | flag for review, do not drop | guess, not measured. Needs one GPU forward per scene. Apple's `inference` path (no GT), A100, a few minutes for the pool |
| 13 | Degenerate pair baseline | baseline / median planar depth of the target frame | below 0.01 | drop pair | measured on the 393 prepared pairs: min 0.007, 2 pairs below 0.01, 6 below 0.02, max 0.399. The 0.01 cut is a guess |
| 14 | Baseline range | baseline in m | outside 0.05 to 1.5 | drop pair (the script does this today) | existing |
| 15 | Overlap range | frustum overlap | outside 0.5 to 0.995 | drop pair (the script does this today) | existing |
| 16 | Near-duplicate pair | mean absolute difference of the two images as 64 x 48 grey, 0 to 1 | below 0.02 | drop pair | measured on the 393 prepared pairs: min 0.042, p5 0.070, median 0.105. No existing pair would be dropped, so the cut is a safety margin, a guess |
| 17 | Blank frame | mean grey below 0.02 or above 0.98 | either | drop frame | guess, not measured |
| 18 | Too few pairs | pairs left in the scene | below 40 | drop scene | guess. Measured 118, 131, 144; train mean 117.9 |
| 19 | Held-out volume | scene name starts with `ai_005_`, `ai_007_` or `ai_011_` | any | do not add. Keep scenes already in S51 and report them | decision for Dominik |

Other measured relative rotations between src and tgt in the prepared pairs: 1.2 to 37.6 degrees (median 7.0). No rotation filter is needed. The overlap filter already limits it.

Filters 1 to 3, 6 to 11 and 19 run in the dry run, because they need only the zip directory, the pose files and four depth frames. Filters 4, 5, 13 to 18 run after prep, on the written files. Filter 12 runs once on GPU. Scenes dropped at any step go into `prep_report.json` with the metric value.

Expected loss: assumption 5 % of the pool fails scene filters and 2 % of pairs fail pair filters. Pair counts below are before filters.

## 4. Dominik's own Blender data

### What the loader reads

`src/sharp_train/data.py` (`HypersimPairs`) and `src/sharp_train/geometry.py`. The layout copies the prepared Hypersim layout.

```
<root>/
  pairs.json
  <scene>/
    cameras.json
    images/0000.png ...
    depth/0000.npy ...
```

| File | Required content | Read by |
| --- | --- | --- |
| `<scene>/images/NNNN.png` | 8-bit RGB, no alpha channel, `(height, width)` equal to `cameras.json`. `np.asarray(Image.open(...), float32) / 255` then permuted to channels first, so an RGBA file would break it | `load_view` |
| `<scene>/depth/NNNN.npy` | 2-D array `(height, width)`, any float dtype (Hypersim prep writes float16, the loader casts to float32). Meters. Planar z-depth. Non-finite or non-positive values become NaN | `load_view` |
| `<scene>/cameras.json` | `width`, `height`, `focal_px`, `cam_to_world_opencv` (see below). `scene` and `tonemap_scale` are written by the Hypersim prep but nothing reads them | `camera()` |
| `pairs.json` | list of `{"scene", "src", "tgt"}`; `baseline` and `overlap` are also written (`pick_eval_pairs.py` and the filters read `baseline`) | `HypersimPairs.__init__` |

`cameras.json`:

```json
{
  "scene": "bl_loft_001",
  "width": 1024,
  "height": 768,
  "focal_px": 886.81,
  "tonemap_scale": 1.0,
  "cam_to_world_opencv": {
    "0000": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  }
}
```

Conventions, from the code:

- Frame key: 4-digit zero-padded decimal string, the same string as the file stem and as `src` and `tgt` in `pairs.json`. `pair_name` is `{scene}_{src}_{tgt}`, so scene names must be unique. Use `bl_<name>_<NNN>`, never `ai_`.
- One focal length per scene, square pixels, principal point at the image center: `intrinsics_matrix` builds `fx = fy = focal_px`, `cx = width / 2`, `cy = height / 2`. No lens distortion, no shift.
- Pixel centers are at +0.5 (`geometry.pixel_grid`).
- Pose: 4 x 4, row-major nested lists, camera to world, OpenCV axes (x right, y down, z forward), translation in meters. The loader computes `tgt_from_src = inv(world_from_tgt) @ world_from_src`, so only relative poses matter. The rotation must be proper (determinant +1).
- Depth is planar z-depth. `geometry.unproject_depth` uses `z = depth` directly: `x = (u - cx) / fx * z`. `prepare_hypersim.py` converts Hypersim's Euclidean distance to the camera center into planar depth with `euclidean_to_planar`: `z = distance * f / sqrt(u'^2 + v'^2 + f^2)`, where `u'` and `v'` are pixel offsets from the center. Blender data must arrive as planar z-depth.
- Hypersim stores OpenGL-convention camera orientations and the script does `orientations[frame] @ diag(1, -1, -1)`. That is the same flip as Blender to OpenCV below.
- Resolution: nothing in `src/sharp_train` hardcodes 1024 x 768, and training uses batch size 1. Other sizes are untested. Use 1024 x 768 first so that the focal length in pixels and the input distribution match the 52 scenes.

Colour. The PNG is read as is and divided by 255, so it must be display-referred sRGB. The Hypersim prep does: per frame `brightness = 0.3 R + 0.59 G + 0.11 B` of the linear HDR color, `scale = 0.8^2.2 / p90(brightness)`, take the median of the per-frame scales as one scene scale, then `clip((scale * rgb)^(1/2.2), 0, 1) * 255`. So the 90th percentile of display brightness sits at 0.8 and the curve is a plain power 2.2.

### Spec for the Blender export script

Not written here. The script must produce, per frame:

1. **Intrinsics.** Square pixels, `shift_x = shift_y = 0`, resolution percentage 100. With sensor fit horizontal (or auto and width at least height): `focal_px = focal_mm * width_px / sensor_width_mm`. With vertical fit use the sensor height and `height_px`. One camera, one focal length per scene (do not animate the lens). Hypersim uses `focal_px = 886.81` at 1024 px wide, a horizontal field of view of 60 degrees (`measured` from the camera CSV: `M_proj_00 * width / 2`). Using the same 60 degrees reproduces the training intrinsics exactly. Other values are untested.
2. **Pose.** After `view_layer.update()`, take `M = camera.matrix_world` (final, including parents and constraints). Blender cameras look along -Z with +Y up; OpenCV looks along +Z with +Y down. So `cam_to_world_opencv = M @ diag(1, -1, -1, 1)`. Multiply the translation by `scene.unit_settings.scale_length` so the world is in meters. The depth must use the same scale.
3. **Depth.** Enable the Z (Depth) pass in Cycles. For a perspective camera it is the distance along the camera's viewing axis, which is planar z-depth. Check it once on a flat wall that faces the camera: the pass must be constant across the wall. Convert to meters with the same scale factor. Write NaN for background and sky. Cycles puts a huge value (1e10) there; a stored 1e10 would be read as valid depth. Save as `.npy`, float16, shape `(height, width)`. Cycles filters the Z pass with the pixel filter, which blends depth at edges; set the film filter width to its minimum for the depth render, or render at 2x and take nearest-neighbour samples. This is a recommendation I did not test.
4. **Colour.** Preferred: save the combined pass as linear float (OpenEXR), then apply `tonemap_scale` and `tonemap` from `scripts/prepare_hypersim.py` with the scene-median scale, so the Blender images share the Hypersim distribution. The alternative is a PNG straight from Blender with view transform `Standard`, look `None`, sRGB output, and an exposure set so that p90 of display brightness is about 0.8. Do not use Filmic or AgX. No alpha channel.
5. **Trajectory.** About 40 frames per scene along a smooth path. Hypersim cam_00 moves 1.0 to 1.4 m between consecutive frames (`measured` above) and pairs use the next 4 frames, giving 0.05 to 1.5 m baselines. For smaller rooms scale the step so the baseline to depth ratio stays between 0.03 and 0.4 (the range of the 393 prepared pairs, 0.007 to 0.399, `measured`). Avoid poses inside geometry and views of an empty sky.
6. **Pairs.** Generate `pairs.json` entries with the same `select_pairs` rules as section 3 (baseline 0.05 to 1.5 m, overlap 0.5 to 0.995, next 4 frames, both directions). Reuse the function; do not write a second one.
7. **Names.** Scene `bl_<name>_<NNN>`; do not reuse a name.

### Acceptance check before a Blender scene is used

Run the existing visibility check on the new pairs on CPU (`checks.visibility_check` in `src/sharp_train/checks.py`). It warps the source image into the target view with the target's ground-truth depth. On the eval pairs the warp error is 0.043 inside the visibility mask and 0.143 on the occluded pixels (`docs/training-spec.md`). A wrong axis flip, a Euclidean-distance depth, a wrong unit or a mismatched focal length shows up as a large error inside the mask. Reject the scene until the visible error is clearly below the occluded one.

### Getting it onto the volume

`modal volume put ml-sharp-data <local scene dir> /hypersim/<scene>` from Dominik's machine, then `--task merge-pairs`. Merge adds the scene to the default train set. Pin the set with `--train-scenes` when you want to compare with and without Blender data.

## 5. Training runs on the bigger set

### Numbers used

| Quantity | Value | Source |
| --- | --- | --- |
| Pairs per scene | 6015 / 51 = 117.9 | project |
| Train pairs, 150 scenes | 150 x 117.9 = 17,691 | estimate, before filters |
| Train pairs, 300 scenes | 300 x 117.9 = 35,382 | estimate, before filters |
| Batch size | 1 pair per step (`train_loader`) | `modal_app.py` |
| A100-40GB, grid 384 | 0.474 s per step | project |
| A100 hourly | $2.23 | project |
| E2 | 1500 steps = 0.25 epoch of 6015 pairs = 711 s = 11.9 min = $0.44 | arithmetic |

Cost per 1000 steps at 384: 1000 x 0.474 / 3600 x 2.23 = $0.29 and 7.9 min.

### Grid 768 time per step (not measured)

Nobody has measured grid 768. Estimate:

- 768 has 4x the Gaussians of 384 (1,179,648 against 294,912) and the trainable decoder runs at 4x the pixels. The Depth Pro backbone runs at 1536 in both and does not scale. Let `f` be the backbone's share of the 384 step; I do not know it. Take `f` between 0 and 0.3.
- A100 time at 768 = 0.474 x (f + 4 x (1 - f)): f = 0 gives 1.90 s; f = 0.3 gives 0.474 x 3.1 = 1.47 s.
- Gradient checkpointing (`--grad-checkpointing 1`) may be needed to fit memory. Assume it adds 30 %: up to 1.90 x 1.3 = 2.47 s.
- H100 against A100: the Hopper page on Wikipedia (`https://en.wikipedia.org/wiki/Hopper_(microarchitecture)`, fetched 2026-10-21) says Hopper has "twice as many FP32 operations per cycle per SM" and that HBM3 memory gives 3 TB/s, against 1.6 TB/s for the A100 40 GB (A100 page, same date). The memory ratio is 3 / 1.6 = 1.9. I assume a speedup between 1.5 (conservative) and 2.0.
- H100 step time: low 1.47 / 2.0 = 0.73 s, high 2.47 / 1.5 = 1.64 s. **Band: 0.7 to 1.7 s per step** (`estimate`).
- H100 hourly, $4.20 (estimate above). Per 1000 steps: $0.85 to $1.91.

First action at 768, before any run: measure it.

```
modal run modal_app.py --task stage3 --mode check --grid 768 --run-name stage3-g768-check --init-checkpoint runs/stage3-g384-391a8ae-lr4e5-dc1/checkpoints/final.pt --depth-correction 1 --timeout-minutes 30 --max-usd 2.10
```

`--timeout-minutes 30` caps the spend. If it fails on memory, repeat with `--grad-checkpointing 1`. The check prints `seconds_per_step` and `peak_memory_gb`. Replace the band above with that number.

### Recipe family

The E2 recipe: init from a finetuned checkpoint, low peak learning rate, `--depth-correction 1`, clip 1000, kill rule, validation every 500, checkpoint every 1000. Re-warming an annealed checkpoint to 1.6e-4 hurt (run 12k). 4e-5 finetunes were safe.

### Step A0: give E2 back a trained depth head (before stage A)

Why: E2 `final.pt` pairs E2's decoder and alignment with Apple's head.4 (see the top of this doc). Stage A from E2 would re-learn the head anyway. Without A0, any stage A gain mixes the head fix with the extra data, and nobody knows what the lost head cost. A0 is a short finetune of E2 on E2's own data with code at or after #10, so its checkpoints keep the head.

Same recipe as E2, 1000 steps, default train scenes (as E2). `<sha>` is any `main` commit at or after `98aed10`.

```
modal run modal_app.py --task stage3 --mode train --grid 384 --run-name stage3-g384-<sha>-e2h-lr4e5-dc1 --init-checkpoint runs/stage3-g384-391a8ae-lr4e5-dc1/checkpoints/final.pt --depth-correction 1 --steps 1000 --lr 4e-5 --warmup 100 --grad-clip 1000 --kill-val-ratio 1.06 --checkpoint-every 1000 --val-every 500 --max-minutes 15 --timeout-minutes 25 --max-usd 0.93
```

```
modal run modal_app.py --task stage3 --mode eval --grid 384 --run-name stage3-g384-<sha>-e2h-lr4e5-dc1 --checkpoints final --max-usd 8.92
```

Measurement M1, the cost of the lost head. It needs no extra GPU. The loss, model, render and validation code are unchanged from `391a8ae` to `98aed10` (checked with `git diff`). So A0's step-0 validation (E2 weights with Apple's head) compares directly with E2's last validation (E2 with its own trained head, same 8 pairs from `ai_011_001`):

```
modal volume get ml-sharp-data runs/stage3-g384-391a8ae-lr4e5-dc1/val_history.json reports/e2-val.json
modal volume get ml-sharp-data runs/stage3-g384-<sha>-e2h-lr4e5-dc1/val_history.json reports/e2h-val.json
python -c "import json; a = json.load(open('reports/e2-val.json'))[-1]; b = json.load(open('reports/e2h-val.json'))[0]; [print(k, round(a[k], 5), round(b[k], 5), f'{(b[k] - a[k]) / abs(a[k]):+.1%}') for k in ('total', 'color_src', 'color_tgt', 'depth_src', 'depth_tgt', 'alpha', 'percep') if a[k]]"
```

If `total` is within 1 %, the lost head did not matter and the r2, E1 and E2 numbers stand as measured. If it is worse by more, those evals understate the trained models, and the frustum holes and GT PSNR gaps are partly caused by the mismatch.

Gate for A0:

```
modal volume get ml-sharp-data runs/stage3-g384-<sha>-e2h-lr4e5-dc1/eval/final/report.json reports/e2h-final.json
python scripts/compare_eval.py --base reports/e2-final.json --new reports/e2h-final.json
```

Exit 0, with no kill: E2h is the new best. Stage A then starts from `runs/stage3-g384-<sha>-e2h-lr4e5-dc1/checkpoints/final.pt` and gates against `reports/e2h-final.json`. Exit 1: stage A starts from E2 as written below.

Cost (`estimate`): the 1000 steps take 1000 x 0.474 s = 7.9 min = $0.29. Startup, three validation passes and the checkpoint save are not measured; `--timeout-minutes 25` caps the run at $0.93. The eval uses the 15 min placeholder below, $0.56. Total about $0.9 to $1.5.

### Stage order

| Stage | Grid | Data | Init | Steps | Epochs | Time | Cost | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 384 | S150 (17,691 pairs) | E2h final if A0 passes, else E2 final | 9000 | 0.51 | 9000 x 0.474 = 4266 s = 71 min | 1.185 h x 2.23 = $2.64 | A100-40GB |
| B | 384 | S300 (35,382 pairs) | best of A (or E2) | 18,000 | 0.51 | 8532 s = 142 min | 2.37 h x 2.23 = $5.29 | A100-40GB |
| C | 768 | S300 | best 384 | 6000 | 0.17 | 0.73 to 1.64 s per step = 4380 to 9840 s = 1.2 to 2.7 h | 1.22 h x 4.20 = $5.1 up to 2.73 h x 4.20 = $11.5 | H100 |

Why these steps. E2 used 0.25 epoch of 6015 pairs. Stage A uses about twice that fraction of an epoch on a pool 2.9x bigger, so each pair is seen at most once and the model sees 6x the steps. Stage B scales the steps with the pool. A full epoch would cost 17,691 x 0.474 / 3600 x 2.23 = $5.19 (150 scenes) or $10.39 (300 scenes); do that only if the checkpoints show the curves still falling at half an epoch. Stage C is limited by cost: 6000 steps is 0.17 epoch, so the 768 gain mainly comes from the 384 to 768 change and the 384 weights. If the 768 validation curve is still falling, extend with `--resume`.

Stage C starts from the best 384 checkpoint. The loader rescales the base splat scale by 1/2 for 384 to 768 (`rescale_base_scale`). Use peak lr 4e-5, warmup 200. If `--kill-val-ratio` stops the run, rerun at 2e-5. These two values are guesses.

Evals: each stage evaluates `final` and two mid checkpoints. I did not measure the duration of an eval call. Placeholder: 15 min on A100 = $0.56, 20 min on H100 = $1.40. Replace it with the duration in the E2 eval log.

Budget summary (`estimate`):

| Item | Low | High |
| --- | --- | --- |
| Prep and dry run (section 2) | $0.2 | $2.0 |
| A0 + 1 eval | $0.9 | $1.5 |
| 768 check | $0.1 | $0.5 |
| Stage A + 3 evals | $2.64 + 1.7 = $4.3 | same |
| Stage B + 3 evals | $5.29 + 1.7 = $7.0 | same |
| Stage C + 3 evals | $5.1 + 4.2 = $9.3 | $11.5 + 4.2 = $15.7 |
| Total | about $22 | about $31 |

A 1-epoch run of stage B instead of 0.5 epoch adds $5.1.

### Commands

Replace `<sha>` with the commit of `modal_app.py` you run. `scenes_150.txt` and `scenes_300.txt` hold one scene name per line.

Stage A (if A0 passed, replace the E2 `final.pt` path with E2h's and `reports/e2-final.json` with `reports/e2h-final.json`):

```
modal run modal_app.py --task stage3 --mode check --grid 384 --run-name stage3-g384-<sha>-s150-check --train-scenes "$(paste -sd, scenes_150.txt)" --init-checkpoint runs/stage3-g384-391a8ae-lr4e5-dc1/checkpoints/final.pt --depth-correction 1 --max-usd 6.69
```

```
modal run modal_app.py --task stage3 --mode train --grid 384 --run-name stage3-g384-<sha>-s150-lr4e5-dc1 --train-scenes "$(paste -sd, scenes_150.txt)" --init-checkpoint runs/stage3-g384-391a8ae-lr4e5-dc1/checkpoints/final.pt --depth-correction 1 --steps 9000 --lr 4e-5 --warmup 100 --grad-clip 1000 --kill-val-ratio 1.06 --checkpoint-every 1000 --val-every 500 --max-minutes 100 --timeout-minutes 130 --max-usd 4.83
```

```
modal run modal_app.py --task stage3 --mode eval --grid 384 --run-name stage3-g384-<sha>-s150-lr4e5-dc1 --checkpoints final,step_003000,step_006000 --max-usd 8.92
```

Then fetch the report and compare with the base (the base report path assumes E2's eval already exists):

```
modal volume get ml-sharp-data runs/stage3-g384-<sha>-s150-lr4e5-dc1/eval/final/report.json reports/s150-final.json
python scripts/compare_eval.py --base reports/e2-final.json --new reports/s150-final.json
```

Stage B: the same, with `scenes_300.txt`, `--run-name stage3-g384-<sha>-s300-lr4e5-dc1`, `--init-checkpoint runs/stage3-g384-<sha>-s150-lr4e5-dc1/checkpoints/final.pt` (or E2 if stage A fails its gate), `--steps 18000`, `--max-minutes 200 --timeout-minutes 240`.

Stage C:

```
modal run modal_app.py --task stage3 --mode train --grid 768 --gpu H100 --run-name stage3-g768-<sha>-s300-lr4e5-dc1 --train-scenes "$(paste -sd, scenes_300.txt)" --init-checkpoint runs/stage3-g384-<sha>-s300-lr4e5-dc1/checkpoints/final.pt --depth-correction 1 --steps 6000 --lr 4e-5 --warmup 200 --grad-clip 1000 --kill-val-ratio 1.06 --checkpoint-every 1000 --val-every 500 --max-minutes 240 --timeout-minutes 300 --max-usd 21.00
```

```
modal run modal_app.py --task stage3 --mode eval --grid 768 --run-name stage3-g768-<sha>-s300-lr4e5-dc1 --checkpoints final --max-usd 67.20
```

Add `--grad-checkpointing 1` if the check ran out of memory.

### Timeouts

- The Modal function timeout is between 1 second and 24 hours (`https://modal.com/docs/guide/timeouts`, fetched 2026-10-21).
- Defaults in `modal_app.py` scale with the grid squared: train 120 x (grid / 192)^2 minutes, so 480 min at 384 and **1920 min (32 h) at 768, which is above the 24 h limit**. At 768 always pass `--timeout-minutes` (I use 300). `--max-minutes` defaults to 50 x (grid / 192)^2 = 200 at 384 and 800 at 768. It is a budget inside the loop. Stage B (142 min expected) is close to the 200 min default, so set it explicitly as in the commands below.
- Keep `--max-minutes` 20 to 30 % below `--timeout-minutes`, so the final checkpoint and the validation pass finish before Modal kills the container.
- Runs longer than one invocation use `--resume runs/<RUN>/checkpoints/step_NNNNNN.pt`. It is in `main` since #10 (`98aed10`). It restores Adam, the schedule, the data position, logs, validation history and RNG, and its CPU check proves a stopped and resumed run equals an uninterrupted one bit for bit. Only checkpoints saved at or after `98aed10` hold the training state; E2 `final.pt` does not, so it can be an init but not a resume point.
- Detached runs (`modal run --detach`) need both a timeout and `--max-minutes`. After every run, `modal app list` must show nothing running for `ml-sharp-train`.

### Pass gates

Gate 1, against the previous best, for every stage:

```
python scripts/compare_eval.py --base <previous best report.json> --new <new report.json>
```

It exits 1 unless LPIPS, DISTS and hole fraction are each no worse than the base by more than `--noise-k` (default 1) paired standard errors, and LPIPS is no worse by more than 0.005 in any scene. E2 passes this gate against r2. The eval has 26 pairs from 2 scenes, and the paired LPIPS standard error is 0.002 to 0.004. Differences under 0.005 are not distinguishable. Pick the stage's checkpoint by this gate and by validation, not by repeated peeking at single eval numbers.

Gate 2, absolute targets, inference setting:

| Stage | LPIPS | DISTS | holes | frustum holes | GT PSNR |
| --- | --- | --- | --- | --- | --- |
| A and B (against Apple at 384) | at most 0.378 | at most 0.204 | at most 0.079 | at most 0.008 (E2 is 0.028) | at least 23.00 (E2 is 21.36) |
| C (against Apple 768) | at most 0.311 | at most 0.162 | at most 0.062 | at most 0.004 | at least 23.49 |

E2 already meets the first three numbers of the 384 row. The 384 stages are about closing frustum holes and GT PSNR without losing those three. The full goal is the 768 row. If stage B closes none of the PSNR gap, more Hypersim scenes are probably not the lever and the run should stop there.

Gate 3, on the data side: before stage A, `--mode check` on the new pool must finish with finite losses, `clipped_fraction` close to the E2 value, and no `spikes.json` entries from a single scene. If `spikes.json` names one scene repeatedly, drop it and rerun the check.

## 6. Risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| License: ShareAlike against research-only, Evermotion chain, `LICENSE_MODEL` | Weights or renders cannot be released as planned | Section 1. Ask a lawyer before release. Research use on Modal is not affected |
| Running the old script overwrites `pairs.json` | The 52-scene pair list is lost | Per-scene `pairs_scene.json`, a merge step, a one-time `pairs.52.json` copy |
| Eval is 26 pairs from 2 scenes, SE 0.002 to 0.004 | Small gains look like noise, and repeated choices overfit to the eval | Keep the eval fixed. Report the paired SE. Decide on the gate, not on one number |
| Same-volume leakage | Eval objects appear in training | Filter 19. Check the current 51 scenes |
| Bad scenes (depth scale, tiny baseline) | Gradient spikes, wasted steps | Filters 9 to 13, `spikes.json` after the check run |
| Apple's CDN slows or changes | Prep stalls | 5 containers, retries, idempotent scenes. The zips are dated 2020 |
| Volume contention or limits | Commits slow | At most 5 containers. 25k inodes at 300 scenes against the 50k guidance. Volume version not checked |
| 768 time and memory unknown | Cost or out-of-memory | Measure with `--mode check --grid 768` first. `--grad-checkpointing 1` |
| 768 default timeout over 24 h | Modal rejects the run | Pass `--timeout-minutes` |
| `--resume` not in `main` | Long runs cannot continue | Merge `stage3-resume` first |
| More data may not fix the gap | Money spent for nothing | Stage A is $2.64. Stop after B if PSNR does not move |
| Domain: Hypersim is indoor synthetic only | Limited generalization | Blender scenes of Dominik's own |
| Wrong Blender conventions | Silent training on bad geometry | Acceptance check in section 4 |

## Not verified

- Whether `ml-sharp-data` is a v1 or v2 volume, and which 52 scenes are on it (I may not run `modal`).
- Whether Apple's CDN has request or rate limits. The pages I fetched state none.
- CPU seconds per scene. Estimate only.
- Seconds per step at 768 and the memory it needs.
- The duration of an eval call.
- Evermotion's license terms, and whether the Depth Pro backbone has its own license.
- That `stage3-resume` restores everything the commit message claims.
- Blender's Z pass being planar z-depth for perspective cameras, and the edge filtering advice. Stated from memory of the Cycles manual; check with the wall test.
- The ingress price on Modal.

## Sources

- `https://raw.githubusercontent.com/apple/ml-hypersim/main/README.md`, `LICENSE.txt`, `ACKNOWLEDGMENTS.txt`, `contrib/mikeroberts3000/metadata_camera_parameters.csv`, `evermotion_dataset/analysis/metadata_images_split_scene_v1.csv`, `evermotion_dataset/analysis/metadata_camera_trajectories.csv`, `code/python/tools/dataset_download_images.py`
- `https://creativecommons.org/licenses/by-sa/3.0/legalcode.txt`
- `https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip` (HEAD and range requests)
- `https://modal.com/pricing`, `https://modal.com/docs/guide/volumes`, `https://modal.com/docs/guide/scale`, `https://modal.com/docs/guide/resources`, `https://modal.com/docs/guide/timeouts`
- `https://en.wikipedia.org/wiki/Hopper_(microarchitecture)`, `https://en.wikipedia.org/wiki/Ampere_(microarchitecture)`
- In this repo: `LICENSE_MODEL`, `scripts/prepare_hypersim.py`, `scripts/pick_eval_pairs.py`, `src/sharp_train/{data,geometry,stage2,checks}.py`, `modal_app.py`, `docs/training-spec.md`, `/home/ubuntu/data/hypersim-cpu/`
