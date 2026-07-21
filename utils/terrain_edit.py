import math
from typing import Optional, Tuple

import numpy as np


def inverse_rotate_point_rc(
    point_rc: Tuple[float, float],
    base_shape_rc: Tuple[int, int],
    rotated_shape_rc: Tuple[int, int],
    angle_deg: float,
) -> Optional[Tuple[float, float]]:
    """Map a point from a PIL-expanded rotated image back into the base image."""

    point_r, point_c = point_rc
    base_h, base_w = int(base_shape_rc[0]), int(base_shape_rc[1])
    rot_h, rot_w = int(rotated_shape_rc[0]), int(rotated_shape_rc[1])
    if base_h <= 0 or base_w <= 0 or rot_h <= 0 or rot_w <= 0:
        return None
    if abs(angle_deg) < 1e-6:
        if 0.0 <= point_r < base_h and 0.0 <= point_c < base_w:
            return float(point_r), float(point_c)
        return None

    base_center_r = (base_h - 1) * 0.5
    base_center_c = (base_w - 1) * 0.5
    rot_center_r = (rot_h - 1) * 0.5
    rot_center_c = (rot_w - 1) * 0.5

    # Convert from image-space rows/cols into a Cartesian frame with +Y pointing upward,
    # then apply the inverse of PIL's CCW image rotation.
    x_rot = float(point_c) - rot_center_c
    y_rot = rot_center_r - float(point_r)
    theta = math.radians(float(angle_deg))

    src_x = x_rot * math.cos(theta) + y_rot * math.sin(theta)
    src_y = -x_rot * math.sin(theta) + y_rot * math.cos(theta)

    src_r = base_center_r - src_y
    src_c = base_center_c + src_x
    eps = 1e-6
    if src_r < -eps or src_r > (base_h - 1) + eps or src_c < -eps or src_c > (base_w - 1) + eps:
        return None
    src_r = min(max(src_r, 0.0), float(base_h - 1))
    src_c = min(max(src_c, 0.0), float(base_w - 1))
    return float(src_r), float(src_c)


def map_scene_pos_to_source_rc(
    scene_xy_px: Tuple[float, float],
    *,
    cell_size_px: float,
    placement_origin_rc: Tuple[int, int],
    rotated_canvas_shape_rc: Tuple[int, int],
    rotated_source_shape_rc: Tuple[int, int],
    base_source_shape_rc: Tuple[int, int],
    rotation_deg: float,
) -> Optional[Tuple[float, float]]:
    """Map a TerraMapMaker scene position to the native source array row/col."""

    if cell_size_px <= 0.0:
        return None

    scene_x_px, scene_y_px = scene_xy_px
    scene_r = float(scene_y_px) / float(cell_size_px)
    scene_c = float(scene_x_px) / float(cell_size_px)

    canvas_r = float(placement_origin_rc[0]) + scene_r
    canvas_c = float(placement_origin_rc[1]) + scene_c
    canvas_h, canvas_w = int(rotated_canvas_shape_rc[0]), int(rotated_canvas_shape_rc[1])
    if canvas_r < 0.0 or canvas_r >= canvas_h or canvas_c < 0.0 or canvas_c >= canvas_w:
        return None

    src_h, src_w = int(rotated_source_shape_rc[0]), int(rotated_source_shape_rc[1])
    if src_h <= 0 or src_w <= 0 or canvas_h <= 0 or canvas_w <= 0:
        return None

    ratio_r = float(src_h) / float(canvas_h)
    ratio_c = float(src_w) / float(canvas_w)
    rotated_source_r = canvas_r * ratio_r
    rotated_source_c = canvas_c * ratio_c

    return inverse_rotate_point_rc(
        (rotated_source_r, rotated_source_c),
        base_shape_rc=base_source_shape_rc,
        rotated_shape_rc=rotated_source_shape_rc,
        angle_deg=rotation_deg,
    )


def circular_brush_mask(
    shape_rc: Tuple[int, int],
    center_rc: Tuple[float, float],
    radius_cells: float,
) -> np.ndarray:
    rows, cols = int(shape_rc[0]), int(shape_rc[1])
    if rows <= 0 or cols <= 0 or radius_cells <= 0.0:
        return np.zeros((rows, cols), dtype=bool)

    rr, cc = np.indices((rows, cols), dtype=np.float32)
    dist = np.sqrt((rr - float(center_rc[0])) ** 2 + (cc - float(center_rc[1])) ** 2)
    return dist <= float(radius_cells)


def circular_brush_falloff(
    shape_rc: Tuple[int, int],
    center_rc: Tuple[float, float],
    radius_cells: float,
) -> np.ndarray:
    rows, cols = int(shape_rc[0]), int(shape_rc[1])
    if rows <= 0 or cols <= 0 or radius_cells <= 0.0:
        return np.zeros((rows, cols), dtype=np.float32)

    rr, cc = np.indices((rows, cols), dtype=np.float32)
    dist = np.sqrt((rr - float(center_rc[0])) ** 2 + (cc - float(center_rc[1])) ** 2)
    weights = 1.0 - (dist / float(radius_cells))
    weights = np.clip(weights, 0.0, 1.0)
    return weights.astype(np.float32)


def apply_sculpt_brush(
    surface: np.ndarray,
    *,
    center_rc: Tuple[float, float],
    radius_cells: float,
    delta_m: float,
) -> np.ndarray:
    updated = np.asarray(surface, dtype=np.float32).copy()
    if updated.ndim != 2 or abs(float(delta_m)) <= 1e-9 or radius_cells <= 0.0:
        return updated

    weights = circular_brush_falloff(updated.shape, center_rc, radius_cells)
    updated += weights * float(delta_m)
    return updated


def apply_flatten_brush(
    surface: np.ndarray,
    *,
    center_rc: Tuple[float, float],
    radius_cells: float,
    max_step_m: float,
) -> np.ndarray:
    updated = np.asarray(surface, dtype=np.float32).copy()
    if updated.ndim != 2 or abs(float(max_step_m)) <= 1e-9 or radius_cells <= 0.0:
        return updated

    mask = circular_brush_mask(updated.shape, center_rc, radius_cells)
    if not np.any(mask):
        return updated

    target_height = float(np.mean(updated[mask]))
    weights = circular_brush_falloff(updated.shape, center_rc, radius_cells)
    step_limit = abs(float(max_step_m))
    delta = target_height - updated
    limited = np.clip(delta, -step_limit, step_limit)
    updated += limited * weights
    return updated
