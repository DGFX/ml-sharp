import argparse
import csv
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from remotezip import RemoteZip

ZIP_URL = (
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip"
)
CAMERA_CSV_URL = "https://raw.githubusercontent.com/apple/ml-hypersim/main/contrib/mikeroberts3000/metadata_camera_parameters.csv"
SPLIT_CSV_URL = "https://raw.githubusercontent.com/apple/ml-hypersim/main/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv"
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def fetch_csv(url):
    import urllib.request

    with urllib.request.urlopen(url) as response:
        return list(csv.DictReader(io.StringIO(response.read().decode())))


def read_h5(archive, name):
    with h5py.File(io.BytesIO(archive.read(name)), "r") as handle:
        return handle["dataset"][()]


def euclidean_to_planar(distance, focal_px):
    height, width = distance.shape
    u = np.arange(width) + 0.5 - width / 2
    v = np.arange(height) + 0.5 - height / 2
    uu, vv = np.meshgrid(u, v)
    return distance * focal_px / np.sqrt(uu**2 + vv**2 + focal_px**2)


def tonemap_scale(rgb):
    brightness = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
    brightness = brightness[np.isfinite(brightness)]
    current = np.percentile(brightness, 90)
    return float(np.power(0.8, 2.2) / max(current, 1e-4))


def tonemap(rgb, scale):
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0)
    return np.clip(np.power(np.maximum(scale * rgb, 0.0), 1 / 2.2), 0.0, 1.0)


def frustum_overlap(depth_tgt, focal_px, transform_src_from_tgt, stride=8):
    height, width = depth_tgt.shape
    v, u = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
    z = depth_tgt[v, u]
    valid = np.isfinite(z) & (z > 0)
    x = (u + 0.5 - width / 2) / focal_px * z
    y = (v + 0.5 - height / 2) / focal_px * z
    points = np.stack([x, y, z], -1)[valid]
    points = points @ transform_src_from_tgt[:3, :3].T + transform_src_from_tgt[:3, 3]
    in_front = points[:, 2] > 1e-3
    ndc_x = 2 * (points[:, 0] / np.maximum(points[:, 2], 1e-3) * focal_px) / width
    ndc_y = 2 * (points[:, 1] / np.maximum(points[:, 2], 1e-3) * focal_px) / height
    inside = in_front & (np.abs(ndc_x) <= 1.05) & (np.abs(ndc_y) <= 1.05)
    return float(inside.mean()) if len(inside) else 0.0


def prepare_scene(scene, camera_row, included, out_dir, num_frames):
    archive = RemoteZip(ZIP_URL.format(scene=scene))
    names = set(archive.namelist())
    width = int(float(camera_row["settings_output_img_width"]))
    height = int(float(camera_row["settings_output_img_height"]))
    focal_px = float(camera_row["M_proj_00"]) * width / 2
    meters_per_unit = float(camera_row["settings_units_info_meters_scale"])
    detail = f"{scene}/_detail/cam_00"
    positions = read_h5(archive, f"{detail}/camera_keyframe_positions.hdf5") * meters_per_unit
    orientations = read_h5(archive, f"{detail}/camera_keyframe_orientations.hdf5")
    scene_dir = out_dir / scene
    (scene_dir / "images").mkdir(parents=True, exist_ok=True)
    (scene_dir / "depth").mkdir(parents=True, exist_ok=True)
    frames = {}
    raw = {}
    for frame in range(min(num_frames, len(positions))):
        if (scene, frame) not in included:
            continue
        color_name = f"{scene}/images/scene_cam_00_final_hdf5/frame.{frame:04d}.color.hdf5"
        depth_name = (
            f"{scene}/images/scene_cam_00_geometry_hdf5/frame.{frame:04d}.depth_meters.hdf5"
        )
        if color_name not in names or depth_name not in names:
            continue
        raw[frame] = (
            read_h5(archive, color_name).astype(np.float32),
            read_h5(archive, depth_name).astype(np.float32),
        )
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = orientations[frame] @ OPENGL_TO_OPENCV
        cam_to_world[:3, 3] = positions[frame]
        frames[f"{frame:04d}"] = cam_to_world.tolist()
    scale = float(np.median([tonemap_scale(rgb) for rgb, _ in raw.values()]))
    for frame, (rgb, distance) in raw.items():
        image = (tonemap(rgb, scale) * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(image).save(scene_dir / "images" / f"{frame:04d}.png")
        depth = euclidean_to_planar(distance, focal_px).astype(np.float16)
        np.save(scene_dir / "depth" / f"{frame:04d}.npy", depth)
    meta = {
        "scene": scene,
        "width": width,
        "height": height,
        "focal_px": focal_px,
        "tonemap_scale": scale,
        "cam_to_world_opencv": frames,
    }
    (scene_dir / "cameras.json").write_text(json.dumps(meta))
    return meta


def select_pairs(scene_dir, meta, max_offset, min_baseline, max_baseline, min_overlap, max_overlap):
    ids = sorted(meta["cam_to_world_opencv"])
    poses = {i: np.array(meta["cam_to_world_opencv"][i]) for i in ids}
    pairs = []
    for index, src in enumerate(ids):
        for tgt in ids[index + 1 : index + 1 + max_offset]:
            for a, b in ((src, tgt), (tgt, src)):
                transform_src_from_tgt = np.linalg.inv(poses[a]) @ poses[b]
                baseline = float(np.linalg.norm(transform_src_from_tgt[:3, 3]))
                if not min_baseline <= baseline <= max_baseline:
                    continue
                depth_tgt = np.load(scene_dir / "depth" / f"{b}.npy").astype(np.float32)
                overlap = frustum_overlap(depth_tgt, meta["focal_px"], transform_src_from_tgt)
                if min_overlap <= overlap <= max_overlap:
                    pairs.append(
                        {
                            "scene": meta["scene"],
                            "src": a,
                            "tgt": b,
                            "baseline": baseline,
                            "overlap": overlap,
                        }
                    )
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--max-offset", type=int, default=4)
    parser.add_argument("--min-baseline", type=float, default=0.05)
    parser.add_argument("--max-baseline", type=float, default=1.5)
    parser.add_argument("--min-overlap", type=float, default=0.5)
    parser.add_argument("--max-overlap", type=float, default=0.995)
    args = parser.parse_args()
    cameras = {row["scene_name"]: row for row in fetch_csv(CAMERA_CSV_URL)}
    included = {
        (row["scene_name"], int(row["frame_id"]))
        for row in fetch_csv(SPLIT_CSV_URL)
        if row["camera_name"] == "cam_00" and row["included_in_public_release"] == "True"
    }
    args.out.mkdir(parents=True, exist_ok=True)
    all_pairs = []
    for scene in args.scenes:
        row = cameras[scene]
        is_pinhole = abs(float(row["M_proj_02"])) < 1e-6 and abs(float(row["M_proj_12"])) < 1e-6
        if not is_pinhole:
            print(f"skip {scene}: non-pinhole projection")
            continue
        meta = prepare_scene(scene, row, included, args.out, args.frames)
        pairs = select_pairs(
            args.out / scene,
            meta,
            args.max_offset,
            args.min_baseline,
            args.max_baseline,
            args.min_overlap,
            args.max_overlap,
        )
        print(f"{scene}: {len(meta['cam_to_world_opencv'])} frames, {len(pairs)} pairs")
        all_pairs.extend(pairs)
    (args.out / "pairs.json").write_text(json.dumps(all_pairs, indent=1))


if __name__ == "__main__":
    main()
