import torch
import torch.nn.functional as F


def intrinsics_matrix(focal_px, width, height):
    focal = torch.as_tensor(focal_px, dtype=torch.float32).reshape(-1)
    matrix = torch.zeros(focal.shape[0], 3, 3, dtype=torch.float32, device=focal.device)
    matrix[:, 0, 0] = focal
    matrix[:, 1, 1] = focal
    matrix[:, 0, 2] = width / 2
    matrix[:, 1, 2] = height / 2
    matrix[:, 2, 2] = 1.0
    return matrix


def pixel_grid(height, width, device):
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32) + 0.5,
        torch.arange(width, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    return u, v


def unproject_depth(depth, intrinsics):
    batch, _, height, width = depth.shape
    u, v = pixel_grid(height, width, depth.device)
    fx = intrinsics[:, 0, 0].view(batch, 1, 1)
    fy = intrinsics[:, 1, 1].view(batch, 1, 1)
    cx = intrinsics[:, 0, 2].view(batch, 1, 1)
    cy = intrinsics[:, 1, 2].view(batch, 1, 1)
    z = depth[:, 0]
    return torch.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], dim=-1)


def transform_points(points, transform):
    rotation = transform[:, None, None, :3, :3]
    translation = transform[:, None, None, :3, 3]
    return (rotation @ points[..., None])[..., 0] + translation


def project_points(points, intrinsics):
    batch = points.shape[0]
    z = points[..., 2].clamp(min=1e-6)
    fx = intrinsics[:, 0, 0].view(batch, 1, 1)
    fy = intrinsics[:, 1, 1].view(batch, 1, 1)
    cx = intrinsics[:, 0, 2].view(batch, 1, 1)
    cy = intrinsics[:, 1, 2].view(batch, 1, 1)
    return points[..., 0] / z * fx + cx, points[..., 1] / z * fy + cy


def reproject(depth_tgt, intrinsics, transform_src_from_tgt):
    points_src = transform_points(unproject_depth(depth_tgt, intrinsics), transform_src_from_tgt)
    u_src, v_src = project_points(points_src, intrinsics)
    return points_src, u_src, v_src


def frustum_mask(depth_tgt, intrinsics, transform_src_from_tgt, margin=1.05):
    _, _, height, width = depth_tgt.shape
    points_src, u_src, v_src = reproject(depth_tgt, intrinsics, transform_src_from_tgt)
    ndc_x = 2 * u_src / width - 1
    ndc_y = 2 * v_src / height - 1
    valid = torch.isfinite(depth_tgt[:, 0]) & (depth_tgt[:, 0] > 0)
    inside = (ndc_x.abs() <= margin) & (ndc_y.abs() <= margin) & (points_src[..., 2] > 1e-3)
    return (valid & inside)[:, None].float()


def sample_at(image, u, v):
    _, _, height, width = image.shape
    grid = torch.stack([2 * u / width - 1, 2 * v / height - 1], dim=-1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)


def visibility_mask(depth_tgt, depth_src, intrinsics, transform_src_from_tgt, tolerance=0.05):
    points_src, u_src, v_src = reproject(depth_tgt, intrinsics, transform_src_from_tgt)
    frustum = frustum_mask(depth_tgt, intrinsics, transform_src_from_tgt, margin=1.0)
    depth_src_filled = torch.nan_to_num(depth_src, nan=0.0)
    sampled = sample_at(depth_src_filled, u_src, v_src)[:, 0]
    consistent = (points_src[..., 2] - sampled).abs() <= tolerance * points_src[..., 2]
    return frustum * consistent[:, None].float()


def warp_src_to_tgt(image_src, depth_tgt, intrinsics, transform_src_from_tgt):
    _, u_src, v_src = reproject(
        torch.nan_to_num(depth_tgt, nan=1.0), intrinsics, transform_src_from_tgt
    )
    return sample_at(image_src, u_src, v_src)
