import math

import torch
import torch.nn.functional as F
from torch import nn

STAGE3_WEIGHTS = {
    "alpha": 1.0,
    "percep": 3.0,
    "tv": 1.0,
    "grad": 0.5,
    "delta": 1.0,
    "splat": 1.0,
    "scale": 0.1,
    "grad_scale": 5.0,
}
OFFSET_LIMIT = 400.0
FLOATER_SIGMA = 1e-2
FLOATER_EPS = 1e-2
SPLAT_MIN = 1e-1
SPLAT_MAX = 1e2
SPLAT_LOW_PASS = 1e-2
SCALE_TV_LEVELS = 6
GRAD_CLIP_NORM = 1.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAPER_PEAK_LR = 1.6e-4
PAPER_WARMUP = 10_000
PAPER_FINAL_RATIO = 0.1
PAPER_GRID = 768
REFERENCE_DEPTH = 1536
SHORT_RUN_WARMUP = 1_000
FULL_IMAGE = "full"


def clip_gradients(parameters):
    return torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP_NORM)


def cosine_lr(step, total, peak, warmup, final_ratio=PAPER_FINAL_RATIO):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def imagenet_normalize(image, mean, std):
    return (image - mean) / std


def masked_mean(values, mask):
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def valid_depth_mask(depth):
    return (torch.isfinite(depth) & (depth > 0)).float()


def color_l1(rendered, target, mask):
    return masked_mean((rendered - target).abs(), mask)


def disparity_l1(rendered_depth, target_depth, mask):
    target = 1.0 / torch.nan_to_num(target_depth, nan=1.0).clamp(min=1e-3)
    rendered = torch.where(
        rendered_depth > 1e-3,
        1.0 / rendered_depth.clamp(min=1e-3),
        torch.zeros_like(rendered_depth),
    )
    return masked_mean((rendered - target).abs(), mask)


def resolve_color_mask_key(color_mask_key, stage3):
    if color_mask_key is not None:
        return color_mask_key
    if stage3:
        return FULL_IMAGE
    return None


def pair_losses(
    rendering_src, rendering_tgt, batch, tgt_mask_key=None, depth_weight=0.2, color_mask_key=None
):
    valid_src = valid_depth_mask(batch["depth_src"])
    valid_tgt = valid_depth_mask(batch["depth_tgt"])
    color_mask_src = torch.ones_like(valid_src)
    color_mask_tgt = torch.ones_like(valid_tgt)
    if color_mask_key == FULL_IMAGE:
        color_key = None
    elif color_mask_key is not None:
        color_key = color_mask_key
    else:
        color_key = tgt_mask_key
    if color_key is not None:
        color_mask_tgt = batch[color_key]
    if tgt_mask_key is not None:
        valid_tgt = valid_tgt * batch[tgt_mask_key]
    terms = {
        "color_src": color_l1(rendering_src.color, batch["image_src"], color_mask_src),
        "color_tgt": color_l1(rendering_tgt.color, batch["image_tgt"], color_mask_tgt),
        "depth_src": disparity_l1(rendering_src.depth, batch["depth_src"], valid_src),
        "depth_tgt": disparity_l1(rendering_tgt.depth, batch["depth_tgt"], valid_tgt),
    }
    terms["total"] = (
        terms["color_src"]
        + terms["color_tgt"]
        + depth_weight * (terms["depth_src"] + terms["depth_tgt"])
    )
    return terms


def hole_fraction(alpha, threshold=0.5, mask=None):
    holes = (alpha < threshold).float()
    if mask is None:
        return holes.mean()
    return masked_mean(holes, mask)


def alpha_bce(alpha, mask=None):
    safe = alpha.clamp(1e-4, 1.0 - 1e-4)
    probability = alpha + (safe - alpha).detach()
    target = torch.ones_like(probability)
    if mask is None:
        return F.binary_cross_entropy(probability, target)
    return masked_mean(F.binary_cross_entropy(probability, target, reduction="none"), mask)


def total_variation(image, pixel_scale=1.0):
    loss = image.new_zeros(())
    dx = (image[..., :, 1:] - image[..., :, :-1]) * pixel_scale
    dy = (image[..., 1:, :] - image[..., :-1, :]) * pixel_scale
    if dx.numel():
        loss = loss + dx.abs().mean()
    if dy.numel():
        loss = loss + dy.abs().mean()
    return loss


def gradient_magnitude(image, pixel_scale=1.0):
    dx = F.pad(image[..., :, 1:] - image[..., :, :-1], (0, 1)) * pixel_scale
    dy = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1)) * pixel_scale
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


def depth_pixel_scale(depth):
    return depth.shape[-1] / REFERENCE_DEPTH


def variance_scale(grid):
    return (PAPER_GRID / grid) ** 2


def splat_limits(grid):
    scale = variance_scale(grid)
    return SPLAT_MIN * scale, SPLAT_MAX * scale


def connected_term(source, compute):
    if source.requires_grad:
        return compute()
    return source.new_zeros(())


def floater_loss(
    opacities, inverse_depth, base_x, base_y, sigma=FLOATER_SIGMA, eps=FLOATER_EPS, pixel_scale=1.0
):
    grad = gradient_magnitude(inverse_depth[:, :1], pixel_scale)
    batch, _, layers, height, width = base_x.shape
    grad_in = grad.repeat_interleave(layers, dim=0)
    grid = torch.stack(
        [
            base_x[:, 0].reshape(batch * layers, height, width),
            base_y[:, 0].reshape(batch * layers, height, width),
        ],
        dim=-1,
    )
    sampled = F.grid_sample(grad_in, grid, align_corners=False, padding_mode="border")
    sampled = sampled.view(batch, layers, height, width)
    penalty = 1.0 - torch.exp(-torch.relu(sampled - eps) / sigma)
    return (opacities * penalty).mean()


def offset_hinge(delta_xy, limit=OFFSET_LIMIT):
    return torch.relu(delta_xy.abs() - limit).sum(dim=1).mean()


def projected_variance(means, covariances, intrinsics, eps2d=SPLAT_LOW_PASS):
    x, y, z = means.unbind(dim=-1)
    z = z.clamp(min=1e-4)
    fx = intrinsics[:, 0, 0].unsqueeze(1)
    fy = intrinsics[:, 1, 1].unsqueeze(1)
    zero = torch.zeros_like(x)
    row_x = torch.stack([fx / z, zero, -fx * x / z.square()], dim=-1)
    row_y = torch.stack([zero, fy / z, -fy * y / z.square()], dim=-1)
    jacobian = torch.stack([row_x, row_y], dim=-2)
    covariance_2d = jacobian @ covariances @ jacobian.transpose(-1, -2)
    return covariance_2d.diagonal(dim1=-2, dim2=-1).mean(dim=-1) + eps2d


def splat_hinge(variance, low=SPLAT_MIN, high=SPLAT_MAX):
    return (torch.relu(variance - high) + torch.relu(low - variance)).mean()


def scale_mae(scale):
    return (scale - 1).abs().mean()


def multiscale_tv(scale, levels=SCALE_TV_LEVELS, pixel_scale=1.0):
    total = scale.new_zeros(())
    current = scale
    for _ in range(levels):
        if min(current.shape[-2:]) < 2:
            break
        current = F.avg_pool2d(current, 2)
        total = total + total_variation(current, pixel_scale)
    return total


def gram_matrix(features):
    batch, channels, height, width = features.shape
    matrix = features.flatten(2)
    return matrix @ matrix.transpose(1, 2) / (height * width)


def feature_and_gram(rendered, target):
    batch = rendered.shape[0]
    channels, height, width = rendered.shape[1:]
    feature = (rendered - target).square().sum() / (channels * height * width * batch)
    gram = (gram_matrix(rendered) - gram_matrix(target)).square().sum()
    return feature, 10.0 * gram / (channels * channels * batch)


def feature_gram_loss(rendered, target):
    feature, gram = feature_and_gram(rendered, target)
    return feature + gram


def in_frustum(rendered, target, mask):
    return rendered * mask + target.detach() * (1.0 - mask)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        self.network = network
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.eval()

    def features(self, image):
        self.eval()
        value = imagenet_normalize(image.clamp(0, 1), self.mean, self.std)
        value = self.network.relu(self.network.bn1(self.network.conv1(value)))
        value = self.network.maxpool(value)
        layers = []
        for layer in (
            self.network.layer1,
            self.network.layer2,
            self.network.layer3,
            self.network.layer4,
        ):
            value = layer(value)
            layers.append(value)
        return layers

    def value(self, rendered, target):
        with torch.no_grad():
            target_layers = self.features(target)
        rendered_layers = self.features(rendered)
        total = rendered.new_zeros(())
        for rendered_layer, target_layer in zip(rendered_layers, target_layers, strict=True):
            total = total + feature_gram_loss(rendered_layer, target_layer)
        return total

    def forward(self, rendered, target):
        if not torch.is_grad_enabled():
            return self.value(rendered, target)
        leaf = rendered.detach().requires_grad_(True)
        value = self.value(leaf, target)
        (grad,) = torch.autograd.grad(value, leaf)
        surrogate = (rendered * grad).sum()
        return value.detach() + surrogate - surrogate.detach()


def stage3_terms(rendering_src, rendering_tgt, batch, gaussians, state, perceptual):
    inverse_depth = 1.0 / state.aligned_depth.clamp(min=1e-4)
    layers, height, width = state.base_x.shape[2:]
    opacities = gaussians.opacities.view(-1, layers, height, width)
    pixel_scale = depth_pixel_scale(state.aligned_depth)
    low, high = splat_limits(width)
    return {
        "alpha": alpha_bce(rendering_src.alpha) + alpha_bce(rendering_tgt.alpha),
        "percep": perceptual(rendering_tgt.color, batch["image_tgt"]),
        "tv": connected_term(
            state.aligned_depth, lambda: total_variation(inverse_depth[:, 1:2], pixel_scale)
        ),
        "grad": floater_loss(
            opacities, inverse_depth[:, :1], state.base_x, state.base_y, pixel_scale=pixel_scale
        ),
        "delta": offset_hinge(state.delta_xy),
        "splat": splat_hinge(
            projected_variance(gaussians.means, gaussians.covariances, batch["intrinsics"]),
            low,
            high,
        ),
        "scale": connected_term(state.scale_map, lambda: scale_mae(state.scale_map)),
        "grad_scale": connected_term(
            state.scale_map, lambda: multiscale_tv(state.scale_map, pixel_scale=pixel_scale)
        ),
    }


def add_stage3(terms, extra, weights=None):
    weights = STAGE3_WEIGHTS if weights is None else weights
    terms.update(extra)
    terms["total"] = terms["total"] + sum(weights[key] * extra[key] for key in weights)
    return terms
