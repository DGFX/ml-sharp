import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sharp_train import geometry


def pair_name(pair):
    return f"{pair['scene']}_{pair['src']}_{pair['tgt']}"


class HypersimPairs(torch.utils.data.Dataset):
    def __init__(self, root, scenes=None, pairs=None, names=None):
        self.root = Path(root)
        all_pairs = json.loads((self.root / "pairs.json").read_text())
        if scenes is not None:
            all_pairs = [pair for pair in all_pairs if pair["scene"] in scenes]
        if names is not None:
            by_name = {pair_name(pair): pair for pair in all_pairs}
            missing = [name for name in names if name not in by_name]
            if missing:
                raise KeyError(f"pairs not in {self.root / 'pairs.json'}: {missing}")
            all_pairs = [by_name[name] for name in names]
        self.pairs = all_pairs if pairs is None else [all_pairs[index] for index in pairs]
        self.cameras = {}

    def __len__(self):
        return len(self.pairs)

    def camera(self, scene):
        if scene not in self.cameras:
            self.cameras[scene] = json.loads((self.root / scene / "cameras.json").read_text())
        return self.cameras[scene]

    def load_view(self, scene, frame):
        image = np.asarray(
            Image.open(self.root / scene / "images" / f"{frame}.png"), dtype=np.float32
        )
        depth = np.load(self.root / scene / "depth" / f"{frame}.npy").astype(np.float32)
        image = torch.from_numpy(image / 255.0).permute(2, 0, 1)
        depth = torch.from_numpy(depth)[None]
        depth[~torch.isfinite(depth) | (depth <= 0)] = float("nan")
        return image, depth

    def __getitem__(self, index):
        pair = self.pairs[index]
        camera = self.camera(pair["scene"])
        image_src, depth_src = self.load_view(pair["scene"], pair["src"])
        image_tgt, depth_tgt = self.load_view(pair["scene"], pair["tgt"])
        poses = camera["cam_to_world_opencv"]
        world_from_src = torch.tensor(poses[pair["src"]], dtype=torch.float64)
        world_from_tgt = torch.tensor(poses[pair["tgt"]], dtype=torch.float64)
        tgt_from_src = (torch.linalg.inv(world_from_tgt) @ world_from_src).float()
        intrinsics = geometry.intrinsics_matrix(
            camera["focal_px"], camera["width"], camera["height"]
        )
        frustum = geometry.frustum_mask(
            depth_tgt[None], intrinsics, torch.linalg.inv(tgt_from_src)[None]
        )[0]
        visibility = geometry.visibility_mask(
            depth_tgt[None], depth_src[None], intrinsics, torch.linalg.inv(tgt_from_src)[None]
        )[0]
        return {
            "name": pair_name(pair),
            "image_src": image_src,
            "depth_src": depth_src,
            "image_tgt": image_tgt,
            "depth_tgt": depth_tgt,
            "focal_px": torch.tensor(camera["focal_px"], dtype=torch.float32),
            "intrinsics": intrinsics[0],
            "tgt_from_src": tgt_from_src,
            "mask_frustum_tgt": frustum,
            "mask_visible_tgt": visibility,
        }
