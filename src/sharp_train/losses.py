import torch


def masked_mean(values, mask):
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def valid_depth_mask(depth):
    return (torch.isfinite(depth) & (depth > 0)).float()


def color_l1(rendered, target, mask):
    return masked_mean((rendered - target).abs(), mask)


def disparity_l1(rendered_depth, target_depth, mask):
    target = torch.nan_to_num(target_depth, nan=1.0).clamp(min=1e-3)
    rendered = rendered_depth.clamp(min=1e-3)
    return masked_mean((1.0 / rendered - 1.0 / target).abs(), mask)


def pair_losses(rendering_src, rendering_tgt, batch, tgt_mask_key=None, depth_weight=0.2):
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
    terms["total"] = (
        terms["color_src"]
        + terms["color_tgt"]
        + depth_weight * (terms["depth_src"] + terms["depth_tgt"])
    )
    return terms
