import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


ALLOWED_WORKSPACE_TYPES = {"excavate", "collect_dumped_soil"}


def normalize_workspace_type(workspace_type: Any) -> str:
    if workspace_type is None:
        raise ValueError("workspace_type is required")
    normalized = str(workspace_type).strip()
    if not normalized:
        raise ValueError("workspace_type is required")
    if normalized not in ALLOWED_WORKSPACE_TYPES:
        raise ValueError(
            f"workspace_type must be one of {sorted(ALLOWED_WORKSPACE_TYPES)}, got {workspace_type!r}"
        )
    return normalized


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def build_plan_alignment(
    *,
    meters_per_tile: float,
    origin_map_xy_m: Sequence[float],
    yaw_map_from_plan_rad: float,
    source_map_frame_id: str = "map",
) -> dict[str, Any]:
    if meters_per_tile <= 0.0:
        raise ValueError("meters_per_tile must be > 0")
    if len(origin_map_xy_m) != 2:
        raise ValueError("origin_map_xy_m must contain exactly two values")
    if not source_map_frame_id:
        raise ValueError("source_map_frame_id must be non-empty")

    return {
        "source_map_frame_id": source_map_frame_id,
        "alignment": {
            "meters_per_tile": float(meters_per_tile),
            "origin_map_xy_m": [float(origin_map_xy_m[0]), float(origin_map_xy_m[1])],
            "yaw_map_from_plan_rad": float(yaw_map_from_plan_rad),
        },
    }


def build_plan_alignment_from_placed_canvas(
    *,
    meters_per_tile: float,
    map_center_xy_m: Sequence[float],
    rotated_canvas_shape_rc: Sequence[float],
    placement_origin_rc: Sequence[float],
    display_rotation_deg: float,
    source_map_frame_id: str = "map",
) -> dict[str, Any]:
    """Build the plan->map rigid transform for a plan placed on the rotated TerraMapMaker canvas.

    TerraMapMaker first rotates the source map canvas for display, then the user places the Terra plan on that
    rotated canvas. Export maps masks back into the source GridMap by inverse-rotating around the canvas center.
    Runtime still wants a simple rigid transform:

        p_map = origin_map_xy_m + R(yaw_map_from_plan_rad) * p_plan

    where `p_plan` uses the plan-grid axes (`row`, `col`) scaled by `meters_per_tile`. The correct runtime yaw is the
    inverse of the display rotation, and the translation must absorb the rotated-canvas center offset.
    """

    if len(map_center_xy_m) != 2:
        raise ValueError("map_center_xy_m must contain exactly two values")
    if len(rotated_canvas_shape_rc) != 2:
        raise ValueError("rotated_canvas_shape_rc must contain exactly two values")
    if len(placement_origin_rc) != 2:
        raise ValueError("placement_origin_rc must contain exactly two values")

    yaw_map_from_plan_rad = -math.radians(float(display_rotation_deg))
    center_row = float(rotated_canvas_shape_rc[0]) / 2.0
    center_col = float(rotated_canvas_shape_rc[1]) / 2.0
    rel_row = float(placement_origin_rc[0]) - center_row
    rel_col = float(placement_origin_rc[1]) - center_col

    c = math.cos(yaw_map_from_plan_rad)
    s = math.sin(yaw_map_from_plan_rad)
    origin_map_x = float(map_center_xy_m[0]) + ((c * rel_row) - (s * rel_col)) * float(meters_per_tile)
    origin_map_y = float(map_center_xy_m[1]) + ((s * rel_row) + (c * rel_col)) * float(meters_per_tile)

    return build_plan_alignment(
        meters_per_tile=meters_per_tile,
        origin_map_xy_m=[origin_map_x, origin_map_y],
        yaw_map_from_plan_rad=yaw_map_from_plan_rad,
        source_map_frame_id=source_map_frame_id,
    )


def load_alignment_from_terra_metadata(
    metadata_path: str | Path, *, source_map_frame_id: str = "map"
) -> dict[str, Any]:
    metadata_file = Path(metadata_path)
    with metadata_file.open("r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    origin_map_xy_m = metadata.get("terra_origin_map_m")
    if not isinstance(origin_map_xy_m, list) or len(origin_map_xy_m) != 2:
        raise ValueError("terra_metadata.yaml must contain terra_origin_map_m with two entries")

    return build_plan_alignment(
        meters_per_tile=float(metadata["meters_per_tile"]),
        origin_map_xy_m=origin_map_xy_m,
        yaw_map_from_plan_rad=math.radians(float(metadata.get("rotation_deg", 0.0))),
        source_map_frame_id=source_map_frame_id,
    )


def _normalize_mask(mask: Any) -> list[list[Any]]:
    if not isinstance(mask, list):
        raise ValueError("mask field must be a 2D list")
    normalized_rows: list[list[Any]] = []
    for row in mask:
        if not isinstance(row, list):
            raise ValueError("mask row must be a list")
        normalized_rows.append(list(row))
    return normalized_rows


def _normalize_pos_base(pos_base: Sequence[float]) -> list[float]:
    if len(pos_base) != 2:
        raise ValueError("agent_state.pos_base must contain exactly two values")
    return [float(pos_base[0]), float(pos_base[1])]


def _normalize_agent_state(agent_state: Mapping[str, Any]) -> dict[str, Any]:
    pos_base = _normalize_pos_base(agent_state["pos_base"])
    angle_base = float(agent_state.get("angle_base", 0.0))
    angle_cabin = float(agent_state.get("angle_cabin", 0.0))
    wheel_angle = float(agent_state.get("wheel_angle", 0.0))

    return {
        "pos_base": pos_base,
        "angle_base_rad": normalize_angle(angle_base),
        "angle_cabin_rad": normalize_angle(angle_cabin),
        "wheel_angle_rad": normalize_angle(wheel_angle),
    }


def _normalize_workspace_geometry(workspace_geometry: Any) -> dict[str, Any]:
    if not isinstance(workspace_geometry, Mapping):
        raise ValueError("workspace_geometry must be a mapping")
    geometry_type = str(workspace_geometry["type"])
    if geometry_type == "fan":
        return {
            "type": "fan",
            "heading_rad": normalize_angle(float(workspace_geometry["heading_rad"])),
            "min_radius_m": float(workspace_geometry["min_radius_m"]),
            "max_radius_m": float(workspace_geometry["max_radius_m"]),
            "aperture_rad": float(workspace_geometry["aperture_rad"]),
        }
    if geometry_type == "circle":
        return {
            "type": "circle",
            "center_row": float(workspace_geometry["center_row"]),
            "center_col": float(workspace_geometry["center_col"]),
            "radius_tiles": float(workspace_geometry["radius_tiles"]),
            "radius_m": float(workspace_geometry["radius_m"]),
        }
    raise ValueError(f"unknown workspace_geometry type: {geometry_type!r}")


def package_waypoints_schema_v2(
    waypoints: Iterable[Mapping[str, Any]],
    alignment_fields: Mapping[str, Any],
) -> dict[str, Any]:
    if "source_map_frame_id" not in alignment_fields or "alignment" not in alignment_fields:
        raise ValueError("alignment_fields must come from build_plan_alignment/load_alignment_from_terra_metadata")

    packaged_waypoints: list[dict[str, Any]] = []
    for index, waypoint in enumerate(waypoints):
        if "agent_state" not in waypoint:
            raise ValueError(f"waypoint[{index}] is missing agent_state")

        agent_state = _normalize_agent_state(waypoint["agent_state"])

        loaded = False
        loaded_change = waypoint.get("loaded_state_change")
        if isinstance(loaded_change, Mapping) and "after" in loaded_change:
            loaded = bool(loaded_change["after"])
        elif "loaded" in waypoint["agent_state"]:
            loaded = bool(waypoint["agent_state"]["loaded"])

        packaged_waypoint = {
            "step": int(waypoint.get("step", index)),
            "workspace_type": normalize_workspace_type(waypoint.get("workspace_type")),
            "traversability_mask": _normalize_mask(waypoint.get("traversability_mask", [])),
            "terrain_modification_mask": _normalize_mask(waypoint.get("terrain_modification_mask", [])),
            "dug_mask": _normalize_mask(waypoint.get("dug_mask", [])),
            "dump_mask": _normalize_mask(waypoint.get("dump_mask", [])),
            "agent_type": int(waypoint.get("agent_type", 0)),
            "agent_index": int(waypoint.get("agent_index", 0)),
            "agent_state": {**agent_state, "loaded": loaded},
        }
        if "workspace_geometry" in waypoint and waypoint["workspace_geometry"] is not None:
            packaged_waypoint["workspace_geometry"] = _normalize_workspace_geometry(waypoint["workspace_geometry"])
        packaged_waypoints.append(packaged_waypoint)

    for pair_start in range(0, len(packaged_waypoints) - 1, 2):
        first = packaged_waypoints[pair_start]["workspace_type"]
        second = packaged_waypoints[pair_start + 1]["workspace_type"]
        if first != second:
            raise ValueError(
                f"waypoint pair {pair_start}/{pair_start + 1} has mismatched workspace_type: "
                f"{first!r} != {second!r}"
            )

    return {
        "schema_version": 2,
        "source_map_frame_id": alignment_fields["source_map_frame_id"],
        "alignment": alignment_fields["alignment"],
        "waypoints": packaged_waypoints,
    }


def dump_plan_json(plan_document: Mapping[str, Any], output_path: str | Path) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(plan_document, f, indent=2)
        f.write("\n")
