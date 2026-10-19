# Apple-weights check

With Apple's released weights, our loader, cameras and losses must give low loss right away. A high loss means our cameras, axes or losses are wrong.

## Sub-features

- Losses per pair: color L1 input view, color L1 novel view (full and inside the frustum mask), disparity L1.
- Controls: inverted pose, identity baseline (input image compared to the novel target), monodepth without GT-depth alignment.

## How to get to it (user POV)

`stage1_report.json` → `apple_weights.mean`, `apple_weights.criteria`, and the `apple_<pair>.png` images.

## Driving it with modal

`--eval-scenes` picks the scenes, `--num-apple-pairs` the count (default 6, spread over the pairs list).

## Gotchas

- Hypersim `depth_meters` is distance along the ray, not z. `scripts/prepare_hypersim.py` converts it to planar depth.
- Hypersim cameras are OpenGL style (y up, looking down -z). The prep converts to OpenCV (x right, y down, z forward), which SHARP uses.
- Some Hypersim scenes use a tilt-shift projection. The prep skips any scene whose `M_proj` is not a centered pinhole.
- Windows and sky have no depth (NaN). Losses mask them; the alignment input fills them with the frame's max depth.
