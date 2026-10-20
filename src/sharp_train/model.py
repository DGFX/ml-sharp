import dataclasses

import torch
import torch.nn.functional as F
from sharp.models import PredictorParams, create_predictor
from sharp.utils.gaussians import Gaussians3D, compose_covariance_matrices

APPLE_CHECKPOINT_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
INTERNAL_RESOLUTION = 1536


@dataclasses.dataclass
class MetricGaussians:
    means: torch.Tensor
    covariances: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor


def build_predictor(checkpoint=None, grad_checkpointing=False):
    params = PredictorParams()
    params.gaussian_decoder.grad_checkpointing = grad_checkpointing
    predictor = create_predictor(params)
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, weights_only=True, map_location="cpu")
        predictor.load_state_dict(state_dict)
    return predictor


def reset_gaussian_decoder(predictor):
    fresh = create_predictor(PredictorParams())
    predictor.feature_model.load_state_dict(fresh.feature_model.state_dict())
    predictor.prediction_head.load_state_dict(fresh.prediction_head.state_dict())
    return predictor


def trainable_parameters(predictor):
    return [parameter for parameter in predictor.parameters() if parameter.requires_grad]


def ndc_to_camera_matrix(focal_px, width, height):
    batch = focal_px.shape[0]
    scale_x = INTERNAL_RESOLUTION / width
    scale_y = INTERNAL_RESOLUTION / height
    intrinsics = torch.zeros(batch, 4, 4, device=focal_px.device)
    intrinsics[:, 0, 0] = focal_px * scale_x
    intrinsics[:, 1, 1] = focal_px * scale_y
    intrinsics[:, 0, 2] = INTERNAL_RESOLUTION / 2
    intrinsics[:, 1, 2] = INTERNAL_RESOLUTION / 2
    intrinsics[:, 2, 2] = 1.0
    intrinsics[:, 3, 3] = 1.0
    ndc = torch.tensor(
        [
            [2.0 / INTERNAL_RESOLUTION, 0.0, -1.0, 0.0],
            [0.0, 2.0 / INTERNAL_RESOLUTION, -1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=focal_px.device,
    )
    return torch.linalg.inv(ndc[None] @ intrinsics)


def to_metric(gaussians_ndc: Gaussians3D, focal_px, width, height):
    transform = ndc_to_camera_matrix(focal_px, width, height)
    linear = transform[:, None, :3, :3]
    offset = transform[:, None, :3, 3]
    means = (linear @ gaussians_ndc.mean_vectors[..., None])[..., 0] + offset
    covariances = compose_covariance_matrices(
        gaussians_ndc.quaternions, gaussians_ndc.singular_values
    )
    covariances = linear @ covariances @ linear.transpose(-1, -2)
    return MetricGaussians(means, covariances, gaussians_ndc.colors, gaussians_ndc.opacities)


def fill_invalid_depth(depth):
    valid = torch.isfinite(depth) & (depth > 0)
    fallback = torch.where(valid, depth, torch.zeros_like(depth)).flatten(1).amax(dim=1)
    return torch.where(valid, depth, fallback.view(-1, 1, 1, 1))


def predict(predictor, image, focal_px, depth=None, delta_hook=None):
    _, _, height, width = image.shape
    resized = F.interpolate(
        image, size=(INTERNAL_RESOLUTION, INTERNAL_RESOLUTION), mode="bilinear", align_corners=True
    )
    aligned_depth = None
    if depth is not None:
        aligned_depth = F.interpolate(
            fill_invalid_depth(depth),
            size=(INTERNAL_RESOLUTION, INTERNAL_RESOLUTION),
            mode="nearest",
        )
    handle = None
    if delta_hook is not None:
        handle = predictor.prediction_head.register_forward_hook(
            lambda _, __, output: delta_hook(output)
        )
    gaussians_ndc = predictor(resized, focal_px / width, aligned_depth)
    if handle is not None:
        handle.remove()
    return to_metric(gaussians_ndc, focal_px, width, height)
