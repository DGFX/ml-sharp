import dataclasses

import gsplat
import torch
from sharp.utils.color_space import linearRGB2sRGB

LOW_PASS_EPS = 1e-2


@dataclasses.dataclass
class Rendering:
    color: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor


def render(gaussians, viewmats, intrinsics, width, height):
    colors, depths, alphas = [], [], []
    for index in range(gaussians.means.shape[0]):
        output, alpha, _ = gsplat.rasterization(
            means=gaussians.means[index],
            quats=None,
            scales=None,
            covars=gaussians.covariances[index],
            opacities=gaussians.opacities[index],
            colors=gaussians.colors[index],
            viewmats=viewmats[index : index + 1],
            Ks=intrinsics[index : index + 1],
            width=width,
            height=height,
            render_mode="RGB+D",
            rasterize_mode="classic",
            packed=False,
            eps2d=LOW_PASS_EPS,
        )
        output = output.permute(0, 3, 1, 2)
        alpha = alpha.permute(0, 3, 1, 2)
        colors.append(linearRGB2sRGB(output[:, :3].clamp(min=0.0)))
        depths.append(output[:, 3:4] / alpha.clamp(min=1e-6))
        alphas.append(alpha)
    return Rendering(torch.cat(colors), torch.cat(depths), torch.cat(alphas))


def render_pair(gaussians, batch, width, height):
    identity = torch.eye(4, device=batch["intrinsics"].device).expand_as(batch["tgt_from_src"])
    src = render(gaussians, identity.contiguous(), batch["intrinsics"], width, height)
    tgt = render(gaussians, batch["tgt_from_src"], batch["intrinsics"], width, height)
    return src, tgt
