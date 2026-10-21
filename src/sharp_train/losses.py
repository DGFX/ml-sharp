import torch
import torch.nn.functional as F
from torch import nn

PAIR_WEIGHTS = {"color_src": 1.0, "color_tgt": 1.0, "depth_src": 0.2, "depth_tgt": 0.2}
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
GRAD_CLIP_NORM = 1000.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def pair_losses(rendering_src, rendering_tgt, batch, tgt_mask_key=None):
    valid_src = valid_depth_mask(batch["depth_src"])
    valid_tgt = valid_depth_mask(batch["depth_tgt"])
    color_mask_src = torch.ones_like(valid_src)
    color_mask_tgt = torch.ones_like(valid_tgt)
    if tgt_mask_key is not None:
        color_mask_tgt = batch[tgt_mask_key]
        valid_tgt = valid_tgt * batch[tgt_mask_key]
    terms = {
        "color_src": color_l1(rendering_src.color, batch["image_src"], color_mask_src),
        "color_tgt": color_l1(rendering_tgt.color, batch["image_tgt"], color_mask_tgt),
        "depth_src": disparity_l1(rendering_src.depth, batch["depth_src"], valid_src),
        "depth_tgt": disparity_l1(rendering_tgt.depth, batch["depth_tgt"], valid_tgt),
    }
    terms["total"] = sum(PAIR_WEIGHTS[key] * terms[key] for key in PAIR_WEIGHTS)
    return terms


def hole_fraction(alpha, threshold=0.5, mask=None):
    holes = (alpha < threshold).float()
    if mask is None:
        return holes.mean()
    return masked_mean(holes, mask)


def alpha_bce(alpha):
    probability = alpha.clamp(1e-4, 1.0 - 1e-4)
    return F.binary_cross_entropy(probability, torch.ones_like(probability))


def total_variation(image):
    loss = image.new_zeros(())
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    if dx.numel():
        loss = loss + dx.abs().mean()
    if dy.numel():
        loss = loss + dy.abs().mean()
    return loss


def gradient_magnitude(image):
    dx = F.pad(image[..., :, 1:] - image[..., :, :-1], (0, 1))
    dy = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


def floater_loss(opacities, inverse_depth, base_x, base_y, sigma=FLOATER_SIGMA, eps=FLOATER_EPS):
    grad = gradient_magnitude(inverse_depth[:, :1])
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


def multiscale_tv(scale, levels=SCALE_TV_LEVELS):
    total = scale.new_zeros(())
    current = scale
    for _ in range(levels):
        if min(current.shape[-2:]) < 2:
            break
        current = F.avg_pool2d(current, 2)
        total = total + total_variation(current)
    return total


def gram_matrix(features):
    batch, channels, height, width = features.shape
    matrix = features.flatten(2)
    return matrix @ matrix.transpose(1, 2) / (height * width)


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
        value = (image.clamp(0, 1) - self.mean) / self.std
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
        batch = rendered.shape[0]
        for rendered_layer, target_layer in zip(rendered_layers, target_layers, strict=True):
            channels, height, width = rendered_layer.shape[1:]
            feature = (rendered_layer - target_layer).square().sum() / (
                channels * height * width * batch
            )
            gram = (gram_matrix(rendered_layer) - gram_matrix(target_layer)).square().sum()
            gram = gram / (channels * channels * batch)
            total = total + feature + 10.0 * gram
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
    return {
        "alpha": alpha_bce(rendering_src.alpha) + alpha_bce(rendering_tgt.alpha),
        "percep": perceptual(rendering_tgt.color, batch["image_tgt"]),
        **regularizer_terms(batch, gaussians, state),
    }


def regularizer_terms(batch, gaussians, state):
    inverse_depth = 1.0 / state.aligned_depth.clamp(min=1e-4)
    layers, height, width = state.base_x.shape[2:]
    opacities = gaussians.opacities.view(-1, layers, height, width)
    return {
        "tv": total_variation(inverse_depth[:, 1:2]),
        "grad": floater_loss(opacities, inverse_depth[:, :1], state.base_x, state.base_y),
        "delta": offset_hinge(state.delta_xy),
        "splat": splat_hinge(
            projected_variance(gaussians.means, gaussians.covariances, batch["intrinsics"])
        ),
        "scale": scale_mae(state.scale_map),
        "grad_scale": multiscale_tv(state.scale_map),
    }


def add_stage3(terms, extra):
    terms.update(extra)
    terms["total"] = terms["total"] + sum(
        STAGE3_WEIGHTS[key] * extra[key] for key in STAGE3_WEIGHTS
    )
    return terms


def weighted_terms(terms):
    weights = {**PAIR_WEIGHTS, **STAGE3_WEIGHTS}
    return {key: weights[key] * value for key, value in terms.items() if key in weights}
