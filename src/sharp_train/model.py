import dataclasses

import torch
import torch.nn.functional as F
from sharp.models import PredictorParams, create_predictor
from sharp.utils.gaussians import Gaussians3D, compose_covariance_matrices
from torch import nn

APPLE_CHECKPOINT_NAME = "sharp_2572gikvuh.pt"
APPLE_CHECKPOINT_URL = f"https://ml-site.cdn-apple.com/models/sharp/{APPLE_CHECKPOINT_NAME}"
INTERNAL_RESOLUTION = 1536
FULL_GRID = 768


@dataclasses.dataclass
class MetricGaussians:
    means: torch.Tensor
    covariances: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor


@dataclasses.dataclass
class TrainingState:
    aligned_depth: torch.Tensor
    scale_map: torch.Tensor
    delta_xy: torch.Tensor
    base_x: torch.Tensor
    base_y: torch.Tensor


def align_depth(alignment, monodepth, gt_depth, features, enabled):
    corrected, scale = alignment(monodepth, gt_depth if enabled else None, features)
    return corrected, scale[:, :1]


def build_predictor(checkpoint, grad_checkpointing=False):
    params = PredictorParams()
    params.gaussian_decoder.grad_checkpointing = grad_checkpointing
    predictor = create_predictor(params)
    state_dict = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    predictor.load_state_dict(state_dict)
    del state_dict
    return predictor


def cast_floating(value):
    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: cast_floating(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cast_floating(item) for item in value]
    if isinstance(value, tuple):
        casted = [cast_floating(item) for item in value]
        if hasattr(value, "_fields"):
            return type(value)(*casted)
        return tuple(casted)
    return value


def apply_backbone(module, fn, *args):
    if not getattr(module, "backbone_bf16", False):
        return fn(*args)
    device_type = args[0].device.type if torch.is_tensor(args[0]) else "cpu"
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        output = fn(*args)
    return cast_floating(output)


class GridPredictor(nn.Module):
    def __init__(self, predictor, grid, backbone_bf16=False):
        super().__init__()
        if grid > FULL_GRID or FULL_GRID % grid:
            raise ValueError(f"grid {grid} must divide {FULL_GRID}")
        self.predictor = predictor
        self.grid = grid
        self.factor = FULL_GRID // grid
        self.input_resolution = INTERNAL_RESOLUTION // self.factor
        self.backbone_bf16 = backbone_bf16
        self.source_grid = grid
        self.scale_prior_ratio = 1.0

    @property
    def prediction_head(self):
        return self.predictor.prediction_head

    def downsample(self, tensor):
        return tensor if self.factor == 1 else F.avg_pool2d(tensor, self.factor)

    def backbone(self, image):
        return apply_backbone(self, self.predictor.monodepth_model, image)

    def forward(self, image, disparity_factor, depth=None, depth_correction=True):
        predictor = self.predictor
        monodepth_output = self.backbone(image)
        disparity = self.downsample(monodepth_output.disparity)
        monodepth = disparity_factor[:, None, None, None] / disparity.clamp(min=1e-4, max=1e4)
        monodepth, scale_map = align_depth(
            predictor.depth_alignment,
            monodepth,
            depth,
            self.downsample(monodepth_output.decoder_features),
            depth_correction,
        )
        init_output = predictor.init_model(self.downsample(image), monodepth)
        image_features = predictor.feature_model(
            init_output.feature_input,
            encodings=[self.downsample(feature) for feature in monodepth_output.output_features],
        )
        delta_values = predictor.prediction_head(image_features)
        gaussians = predictor.gaussian_composer(
            delta=delta_values,
            base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale,
        )
        base = init_output.gaussian_base_values
        self.training_state = TrainingState(
            aligned_depth=monodepth,
            scale_map=scale_map,
            delta_xy=delta_values[:, :2],
            base_x=base.mean_x_ndc,
            base_y=base.mean_y_ndc,
        )
        return gaussians


def resolution_for_grid(grid):
    return INTERNAL_RESOLUTION * grid // FULL_GRID


def median_base_scale(initializer, resolution):
    image = torch.zeros(1, 3, resolution, resolution)
    depth = torch.ones(1, 2, resolution, resolution)
    with torch.no_grad():
        scales = initializer(image, depth).gaussian_base_values.scales
    return scales.median()


def rescale_base_scale(grid_predictor, source_grid):
    target_grid = grid_predictor.grid
    ratio = source_grid / float(target_grid)
    grid_predictor.source_grid = source_grid
    grid_predictor.scale_prior_ratio = ratio
    if source_grid == target_grid:
        return
    initializer = grid_predictor.predictor.init_model
    source_scale = median_base_scale(initializer, resolution_for_grid(source_grid))
    target_scale = median_base_scale(initializer, grid_predictor.input_resolution)
    measured = float(target_scale / source_scale)
    correction = ratio / measured
    if abs(correction - 1.0) > 1e-3:
        initializer.scale_factor = float(initializer.scale_factor) * correction


def input_resolution(predictor):
    if isinstance(predictor, GridPredictor):
        return predictor.input_resolution
    return INTERNAL_RESOLUTION


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


def predict(predictor, image, focal_px, depth=None, delta_hook=None, depth_correction=True):
    _, _, height, width = image.shape
    resized = F.interpolate(
        image, size=(INTERNAL_RESOLUTION, INTERNAL_RESOLUTION), mode="bilinear", align_corners=True
    )
    aligned_depth = None
    if depth is not None:
        aligned_depth = F.interpolate(
            fill_invalid_depth(depth),
            size=(input_resolution(predictor),) * 2,
            mode="nearest",
        )
    handle = None
    if delta_hook is not None:
        handle = predictor.prediction_head.register_forward_hook(
            lambda _, __, output: delta_hook(output)
        )
    if isinstance(predictor, GridPredictor):
        gaussians_ndc = predictor(
            resized, focal_px / width, aligned_depth, depth_correction=depth_correction
        )
    else:
        gaussians_ndc = predictor(
            resized, focal_px / width, aligned_depth if depth_correction else None
        )
    if handle is not None:
        handle.remove()
    return to_metric(gaussians_ndc, focal_px, width, height)
