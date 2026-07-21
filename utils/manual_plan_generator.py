import argparse
import json
import math
import os
import pickle
from dataclasses import dataclass, asdict
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

import numpy as np

try:
    from utils.plan_packager import (
        dump_plan_json,
        load_alignment_from_terra_metadata,
        normalize_workspace_type,
        package_waypoints_schema_v2,
    )
except ImportError:
    from plan_packager import (
        dump_plan_json,
        load_alignment_from_terra_metadata,
        normalize_workspace_type,
        package_waypoints_schema_v2,
    )

try:
    # Qt is available in the main application; importing here allows an optional UI dialog.
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QDoubleSpinBox,
        QVBoxLayout,
        QPlainTextEdit,
    )
except Exception:  # pragma: no cover - allows using this module as a pure CLI tool
    QDialog = None  # type: ignore


@dataclass
class AgentState:
    pos_base: Tuple[float, float]
    angle_base: float
    angle_cabin: float
    wheel_angle: float


@dataclass
class LoadedStateChange:
    before: bool
    after: bool


@dataclass
class WorkspaceFanGeometry:
    type: str
    heading_rad: float
    min_radius_m: float
    max_radius_m: float
    aperture_rad: float


@dataclass
class WorkspaceCircleGeometry:
    type: str
    center_row: float
    center_col: float
    radius_tiles: float
    radius_m: float


WorkspaceGeometry = Union[WorkspaceFanGeometry, WorkspaceCircleGeometry]


@dataclass
class PlanEntry:
    step: int
    workspace_type: str
    traversability_mask: np.ndarray
    terrain_modification_mask: np.ndarray
    dug_mask: np.ndarray
    dump_mask: np.ndarray
    agent_state: AgentState
    loaded_state_change: LoadedStateChange
    agent_type: int
    agent_index: int
    workspace_geometry: Optional[WorkspaceGeometry] = None

    def to_serializable(self) -> dict:
        """Convert masks to lists so JSON / pickle match extract_map-style structure."""
        serialized = {
            "step": self.step,
            "workspace_type": normalize_workspace_type(self.workspace_type),
            "traversability_mask": self.traversability_mask.astype(bool).tolist(),
            "terrain_modification_mask": self.terrain_modification_mask.astype(bool).tolist(),
            "dug_mask": self.dug_mask.astype(bool).tolist(),
            "dump_mask": self.dump_mask.astype(bool).tolist(),
            "agent_state": asdict(self.agent_state),
            "loaded_state_change": asdict(self.loaded_state_change),
            "agent_type": self.agent_type,
            "agent_index": self.agent_index,
        }
        if self.workspace_geometry is not None:
            serialized["workspace_geometry"] = asdict(self.workspace_geometry)
        return serialized


def _load_action_map_shape(map_root: str) -> Tuple[int, int]:
    """
    Infer action map shape from Terra map export (map/actions/img_1.npy).
    Falls back to occupancy if needed.
    """
    actions_npy = os.path.join(map_root, "actions", "img_1.npy")
    if os.path.exists(actions_npy):
        arr = np.load(actions_npy)
        if arr.ndim == 2:
            return int(arr.shape[0]), int(arr.shape[1])

    # Fallback: occupancy
    occ_npy = os.path.join(map_root, "occupancy", "img_1.npy")
    if os.path.exists(occ_npy):
        arr = np.load(occ_npy)
        if arr.ndim == 2:
            return int(arr.shape[0]), int(arr.shape[1])

    raise FileNotFoundError(
        f"Could not determine action_map shape – expected 'actions/img_1.npy' "
        f"or 'occupancy/img_1.npy' under: {map_root}"
    )


def _cone_mask(
    shape: Tuple[int, int],
    agent_row: float,
    agent_col: float,
    tile_size: float,
    fan_min_radius_m: float,
    fan_max_radius_m: float,
    angles_cabin: int,
    base_yaw: float = 0.0,
) -> np.ndarray:
    """
    Boolean cone mask in grid coordinates (row, col), centered at the agent base.

    This implements the fan workspace:
      r_min = fan_min_radius_m
      r_max = fan_max_radius_m
      theta_max = 2*pi / angles_cabin
      theta_min = -theta_max
      mask = (r in [r_min, r_max]) & (theta in [theta_min, theta_max])
    where r, theta are cylindrical coordinates around the agent base.
    """
    r_min_m, r_max_m = _validated_fan_radii(fan_min_radius_m, fan_max_radius_m)
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    # Work in tile units, convert radial bounds from meters → tiles.
    dx_tiles = xx - agent_col
    dy_tiles = yy - agent_row
    r_tiles = np.sqrt(dx_tiles * dx_tiles + dy_tiles * dy_tiles)

    # Angle relative to agent heading (scaling cancels out)
    theta = np.arctan2(dy_tiles, dx_tiles) - base_yaw
    # Wrap to [-pi, pi]
    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    # Convert meters to tiles
    r_min = r_min_m / max(tile_size, 1e-6)
    r_max = r_max_m / max(tile_size, 1e-6)

    theta_max = 2 * np.pi / max(1, int(angles_cabin))
    theta_min = -theta_max

    dig_mask_r = np.logical_and(r_tiles >= r_min, r_tiles <= r_max)
    dig_mask_theta = np.logical_and(theta >= theta_min, theta <= theta_max)

    return np.logical_and(dig_mask_r, dig_mask_theta)


def _ring_mask(
    shape: Tuple[int, int],
    agent_row: float,
    agent_col: float,
    tile_size: float,
    fan_min_radius_m: float,
    fan_max_radius_m: float,
) -> np.ndarray:
    """Radial ring of valid cone origins around the agent (no theta restriction)."""
    r_min_m, r_max_m = _validated_fan_radii(fan_min_radius_m, fan_max_radius_m)
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    dx_tiles = xx - agent_col
    dy_tiles = yy - agent_row
    r_tiles = np.sqrt(dx_tiles * dx_tiles + dy_tiles * dy_tiles)

    r_min = r_min_m / max(tile_size, 1e-6)
    r_max = r_max_m / max(tile_size, 1e-6)
    return np.logical_and(r_tiles >= r_min, r_tiles <= r_max)


def _circle_mask(shape: Tuple[int, int], center_row: float, center_col: float, radius_tiles: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    dx = xx - float(center_col)
    dy = yy - float(center_row)
    return (dx * dx + dy * dy) <= float(radius_tiles) * float(radius_tiles)


def _normalize_angle(angle_rad: float) -> float:
    return float(math.atan2(math.sin(angle_rad), math.cos(angle_rad)))


def _validated_fan_radii(fan_min_radius_m: float, fan_max_radius_m: float) -> Tuple[float, float]:
    min_radius = float(fan_min_radius_m)
    max_radius = float(fan_max_radius_m)
    if min_radius < 0.0:
        raise ValueError("fan_min_radius_m must be non-negative")
    if max_radius < min_radius:
        raise ValueError("fan_max_radius_m must be greater than or equal to fan_min_radius_m")
    return min_radius, max_radius


def _fan_geometry(
    *,
    heading_rad: float,
    fan_min_radius_m: float,
    fan_max_radius_m: float,
    angles_cabin: int,
) -> WorkspaceFanGeometry:
    min_radius_m, max_radius_m = _validated_fan_radii(fan_min_radius_m, fan_max_radius_m)
    aperture_rad = 4.0 * math.pi / max(1, int(angles_cabin))
    return WorkspaceFanGeometry(
        type="fan",
        heading_rad=_normalize_angle(float(heading_rad)),
        min_radius_m=min_radius_m,
        max_radius_m=max_radius_m,
        aperture_rad=aperture_rad,
    )


def _circle_geometry(
    *, center_row: float, center_col: float, radius_tiles: float, tile_size: float
) -> WorkspaceCircleGeometry:
    return WorkspaceCircleGeometry(
        type="circle",
        center_row=float(center_row),
        center_col=float(center_col),
        radius_tiles=float(radius_tiles),
        radius_m=float(radius_tiles) * float(tile_size),
    )


def _ui_yaw_to_plan_yaw(ui_yaw_rad: float) -> float:
    """Convert TerraMapMaker canvas yaw into terra_planner's plan-frame yaw.

    The GUI arrow is authored in image/grid coordinates:
      - 0 rad points to increasing grid column (right on screen)
      - +pi/2 points to increasing grid row (down on screen)

    The runtime schema-v2 contract uses plan axes:
      - plan_x = row
      - plan_y = col
      - 0 rad points along +plan_x (increasing row)

    That is a 90 degree axis swap, so UI yaw and plan yaw differ by:

      yaw_plan = pi/2 - yaw_ui
    """

    return _normalize_angle((math.pi * 0.5) - float(ui_yaw_rad))


def _ui_relative_angle_to_plan_relative(ui_relative_angle_rad: float) -> float:
    """Convert a UI-authored relative angle into the runtime plan-frame convention.

    Relative angles such as cabin offsets are authored against the same UI frame as the base yaw, so the axis swap
    above flips their sign when converted into the plan frame.
    """

    return _normalize_angle(-float(ui_relative_angle_rad))


def build_manual_plan(
    map_root: str,
    dig_center: Tuple[float, float],
    dig_radius: float,
    dump_center: Tuple[float, float],
    dump_radius: float,
    traversability_from_occupancy: bool = True,
    agent_pos: Optional[Tuple[float, float]] = None,
    agent_yaw: float = 0.0,
    agent_cabin_angle: float = 0.0,
    agent_wheel_angle: float = 0.0,
    *,
    shape_override: Optional[Tuple[int, int]] = None,
    traversability_override: Optional[np.ndarray] = None,
    tile_size: float = 0.1,
    fan_min_radius_m: float = 4.0,
    fan_max_radius_m: float = 6.0,
    angles_cabin: int = 8,
    dig_angles_cabin: Optional[int] = None,
    dump_angles_cabin: Optional[int] = None,
    dump_workspace_shape: str = "fan",
    dig_mask_override: Optional[np.ndarray] = None,
    dump_mask_override: Optional[np.ndarray] = None,
    dig_cabin_override: Optional[float] = None,
    dump_cabin_override: Optional[float] = None,
    dig_limit_mask: Optional[np.ndarray] = None,
    dump_limit_mask: Optional[np.ndarray] = None,
    workspace_type: str = "excavate",
) -> List[PlanEntry]:
    """
    Build a two-step manual plan (dig, then dump) compatible with extract_map.py output.

    The masks are created in the same shape as the simulator action_map
    (derived from Terra export under `map_root`, or from shape_override if given).
    """
    if shape_override is not None:
        h, w = int(shape_override[0]), int(shape_override[1])
    else:
        h, w = _load_action_map_shape(map_root)
    workspace_type = normalize_workspace_type(workspace_type)
    fan_min_radius_m, fan_max_radius_m = _validated_fan_radii(fan_min_radius_m, fan_max_radius_m)
    dig_angles_cabin = int(angles_cabin if dig_angles_cabin is None else dig_angles_cabin)
    dump_angles_cabin = int(angles_cabin if dump_angles_cabin is None else dump_angles_cabin)
    dump_workspace_shape = str(dump_workspace_shape or "fan").strip()
    if dump_workspace_shape not in ("fan", "circle"):
        raise ValueError("dump_workspace_shape must be 'fan' or 'circle'")

    # Traversability: either from occupancy or fully traversable
    if traversability_override is not None:
        traversability = traversability_override.astype(bool)
        if traversability.shape != (h, w):
            raise ValueError(f"traversability_override shape {traversability.shape} != {(h, w)}")
    elif traversability_from_occupancy:
        occ_npy = os.path.join(map_root, "occupancy", "img_1.npy")
        if not os.path.exists(occ_npy):
            raise FileNotFoundError(f"Requested traversability_from_occupancy=True but file not found: {occ_npy}")
        occ = np.load(occ_npy).astype(bool)
        if occ.shape != (h, w):
            raise ValueError(f"Occupancy mask shape {occ.shape} does not match action map shape {(h, w)}")
        traversability = ~occ  # traversable where no obstacle
    else:
        traversability = np.ones((h, w), dtype=bool)

    # Agent pose (grid coordinates, not meters – leave translation to consumer)
    if agent_pos is None:
        agent_pos = (float(dig_center[1]), float(dig_center[0]))  # (x, y) ~ (col, row)

    base_pos = [float(agent_pos[1]), float(agent_pos[0])]  # [row, col]

    dig_agent_state = AgentState(
        pos_base=list(base_pos),
        angle_base=float(agent_yaw),
        angle_cabin=float(dig_cabin_override if dig_cabin_override is not None else agent_cabin_angle),
        wheel_angle=float(agent_wheel_angle),
    )
    dump_agent_state = AgentState(
        pos_base=list(base_pos),
        angle_base=float(agent_yaw),
        angle_cabin=float(dump_cabin_override if dump_cabin_override is not None else agent_cabin_angle),
        wheel_angle=float(agent_wheel_angle),
    )
    dig_heading = float(agent_yaw) + float(dig_agent_state.angle_cabin)
    dump_heading = float(agent_yaw) + float(dump_agent_state.angle_cabin)

    # Dig step: cone in front of the agent (r/theta-based reachability)
    if dig_mask_override is not None:
        dug_mask_dig = dig_mask_override.astype(bool)
    else:
        dug_mask_dig = _cone_mask(
            (h, w),
            agent_row=agent_pos[1],
            agent_col=agent_pos[0],
            tile_size=tile_size,
            fan_min_radius_m=fan_min_radius_m,
            fan_max_radius_m=fan_max_radius_m,
            angles_cabin=dig_angles_cabin,
            base_yaw=dig_heading,
        )

    # Optionally restrict digging to a subset of tiles (e.g. foundation layer)
    if workspace_type == "excavate" and dig_limit_mask is not None:
        limit = np.asarray(dig_limit_mask, dtype=bool)
        if limit.shape != (h, w):
            raise ValueError(f"dig_limit_mask shape {limit.shape} != {(h, w)}")
        dug_mask_dig = np.logical_and(dug_mask_dig, limit)
    dump_mask_dig = np.zeros((h, w), dtype=bool)
    terrain_mod_dig = dug_mask_dig | dump_mask_dig

    dig_entry = PlanEntry(
        step=0,
        workspace_type=workspace_type,
        traversability_mask=traversability,
        terrain_modification_mask=terrain_mod_dig,
        dug_mask=dug_mask_dig,
        dump_mask=dump_mask_dig,
        agent_state=dig_agent_state,
        loaded_state_change=LoadedStateChange(before=False, after=True),
        agent_type=0,
        agent_index=0,
        workspace_geometry=_fan_geometry(
            heading_rad=dig_heading,
            fan_min_radius_m=fan_min_radius_m,
            fan_max_radius_m=fan_max_radius_m,
            angles_cabin=dig_angles_cabin,
        ),
    )

    # Dump step: reuse or override cone as a dump mask
    dug_mask_dump = np.zeros((h, w), dtype=bool)
    if dump_mask_override is not None:
        dump_mask_dump = dump_mask_override.astype(bool)
    elif dump_workspace_shape == "circle":
        dump_mask_dump = _circle_mask((h, w), dump_center[0], dump_center[1], dump_radius)
    else:
        dump_mask_dump = _cone_mask(
            (h, w),
            agent_row=agent_pos[1],
            agent_col=agent_pos[0],
            tile_size=tile_size,
            fan_min_radius_m=fan_min_radius_m,
            fan_max_radius_m=fan_max_radius_m,
            angles_cabin=dump_angles_cabin,
            base_yaw=dump_heading,
        )

    # Optionally restrict dumping to allowed tiles (e.g. exclude obstacle/foundation/nodump)
    if dump_limit_mask is not None:
        limit_d = np.asarray(dump_limit_mask, dtype=bool)
        if limit_d.shape != (h, w):
            raise ValueError(f"dump_limit_mask shape {limit_d.shape} != {(h, w)}")
        dump_mask_dump = np.logical_and(dump_mask_dump, limit_d)
    terrain_mod_dump = dug_mask_dump | dump_mask_dump

    dump_entry = PlanEntry(
        step=1,
        workspace_type=workspace_type,
        traversability_mask=traversability,
        terrain_modification_mask=terrain_mod_dump,
        dug_mask=dug_mask_dump,
        dump_mask=dump_mask_dump,
        agent_state=dump_agent_state,
        loaded_state_change=LoadedStateChange(before=True, after=False),
        agent_type=0,
        agent_index=0,
        workspace_geometry=(
            _circle_geometry(
                center_row=dump_center[0],
                center_col=dump_center[1],
                radius_tiles=dump_radius,
                tile_size=tile_size,
            )
            if dump_workspace_shape == "circle"
            else _fan_geometry(
                heading_rad=dump_heading,
                fan_min_radius_m=fan_min_radius_m,
                fan_max_radius_m=fan_max_radius_m,
                angles_cabin=dump_angles_cabin,
            )
        ),
    )

    return [dig_entry, dump_entry]


class ManualPlanDialog(QDialog):  # type: ignore[misc]
    """
    Floating UI for creating a manual dig/dump plan while the main canvas stays interactive.

    It simply wraps build_manual_plan() and writes JSON / PKL in the same format
    as extract_map.py.
    """

    def __init__(
        self,
        parent=None,
        grid_size: int = 256,
        default_map_root: Optional[str] = None,
        scene=None,
        tile_size: float = 0.1,
        plan_alignment_provider: Optional[Callable[[], Optional[Mapping[str, Any]]]] = None,
    ) -> None:
        if QDialog is None:
            raise RuntimeError("Qt is not available; ManualPlanDialog cannot be used.")
        super().__init__(parent)
        self.setWindowTitle("Manual Dig/Dump Plan")
        self.grid_size = int(grid_size)
        self.scene = scene
        self.tile_size = float(tile_size)
        self.plan_alignment_provider = plan_alignment_provider
        self.plan_entries: List[PlanEntry] = []
        # Cached cone / picks for current waypoint
        self._dig_cone_mask: Optional[np.ndarray] = None
        self._dump_cone_mask: Optional[np.ndarray] = None
        self._dig_target_pos: Optional[Tuple[int, int]] = None
        self._dump_target_pos: Optional[Tuple[int, int]] = None
        self._dig_cabin_angle: Optional[float] = None
        self._dump_cabin_angle: Optional[float] = None
        self._default_agent_cabin_angle = 0.0
        self._default_agent_wheel_angle = 0.0
        self._agent_base_selected: bool = False
        self._agent_type_catalog: List[Tuple[str, int]] = [("Excavator", 0), ("Truck", 1)]
        self._agent_counts: dict[int, int] = {0: 1, 1: 0}
        self._workspace_type_options: List[Tuple[str, str]] = [
            ("Excavate", "excavate"),
            ("Collect dumped soil", "collect_dumped_soil"),
        ]
        self._dump_workspace_shape_options: List[Tuple[str, str]] = [
            ("Fan", "fan"),
            ("Circle", "circle"),
        ]
        # Per-waypoint visuals (permanent cones on the scene)
        self._waypoint_visuals: List[dict] = []
        self._dump_coverage_items: List[Any] = []
        self._build_ui()

    def set_grid_context(self, *, grid_size: int, scene=None, tile_size: float) -> None:
        """Refresh the canvas context used for preview and exported workspace geometry."""
        self.grid_size = int(grid_size)
        self.scene = scene
        self.tile_size = float(tile_size)
        if hasattr(self, "agent_x_spin"):
            self.agent_x_spin.setRange(0.0, float(self.grid_size))
        if hasattr(self, "agent_y_spin"):
            self.agent_y_spin.setRange(0.0, float(self.grid_size))
        if self._agent_base_selected:
            self._update_agent_marker()
            self._refresh_cone_previews_from_yaw()

    def _get_plan_alignment(self) -> Mapping[str, Any]:
        if self.plan_alignment_provider is None:
            raise RuntimeError("No plan alignment provider is configured for ManualPlanDialog.")
        alignment = self.plan_alignment_provider()
        if alignment is None:
            raise RuntimeError(
                "Plan alignment is unavailable. Load/place a source GridMap first so the manual plan can be packaged."
            )
        return alignment

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)

        # Section: Reachability
        reach_label = QLabel("Reachability")
        reach_label.setStyleSheet("QLabel { font-weight:600; color:#444; }")
        form.addRow(reach_label)

        self.fan_min_radius_spin = QDoubleSpinBox()
        self.fan_min_radius_spin.setRange(0.0, 100.0)
        self.fan_min_radius_spin.setDecimals(3)
        self.fan_min_radius_spin.setValue(4.0)
        self.fan_min_radius_spin.valueChanged.connect(self._on_fan_min_radius_changed)
        form.addRow("Fan min radius (m):", self.fan_min_radius_spin)

        self.fan_max_radius_spin = QDoubleSpinBox()
        self.fan_max_radius_spin.setRange(4.0, 100.0)
        self.fan_max_radius_spin.setDecimals(3)
        self.fan_max_radius_spin.setValue(6.0)
        self.fan_max_radius_spin.valueChanged.connect(self._on_workspace_params_changed)
        form.addRow("Fan max radius (m):", self.fan_max_radius_spin)

        self.dig_angles_cabin_spin = QSpinBox()
        self.dig_angles_cabin_spin.setRange(1, 360)
        self.dig_angles_cabin_spin.setValue(12)
        self.dig_angles_cabin_spin.valueChanged.connect(self._on_workspace_params_changed)
        form.addRow("Dig cabin angle steps:", self.dig_angles_cabin_spin)

        self.dump_angles_cabin_spin = QSpinBox()
        self.dump_angles_cabin_spin.setRange(1, 360)
        self.dump_angles_cabin_spin.setValue(12)
        self.dump_angles_cabin_spin.valueChanged.connect(self._on_workspace_params_changed)
        form.addRow("Dump cabin angle steps:", self.dump_angles_cabin_spin)

        self.dump_workspace_shape_combo = QComboBox()
        for label, value in self._dump_workspace_shape_options:
            self.dump_workspace_shape_combo.addItem(label, value)
        self.dump_workspace_shape_combo.currentIndexChanged.connect(self._on_workspace_params_changed)
        form.addRow("Dump workspace shape:", self.dump_workspace_shape_combo)

        self.dump_circle_radius_spin = QDoubleSpinBox()
        self.dump_circle_radius_spin.setRange(1.0, 200.0)
        self.dump_circle_radius_spin.setDecimals(1)
        self.dump_circle_radius_spin.setValue(5.0)
        self.dump_circle_radius_spin.valueChanged.connect(self._on_workspace_params_changed)
        form.addRow("Dump circle radius (tiles):", self.dump_circle_radius_spin)

        # Spacer between sections
        form.addRow(QLabel(""))

        # Section: Operation
        operation_label = QLabel("Operation")
        operation_label.setStyleSheet("QLabel { font-weight:600; color:#444; }")
        form.addRow(operation_label)

        self.workspace_type_combo = QComboBox()
        for label, value in self._workspace_type_options:
            self.workspace_type_combo.addItem(label, value)
        form.addRow("Workspace type:", self.workspace_type_combo)

        form.addRow(QLabel(""))

        # Section: Agent pose & cones
        pose_label = QLabel("Agent & Cones")
        pose_label.setStyleSheet("QLabel { font-weight:600; color:#444; }")
        form.addRow(pose_label)

        self.excavator_count_spin = QSpinBox()
        self.excavator_count_spin.setRange(0, 128)
        self.excavator_count_spin.setValue(int(self._agent_counts.get(0, 1)))
        self.excavator_count_spin.valueChanged.connect(lambda v: self._on_agent_count_changed(0, v))
        form.addRow("Excavator count:", self.excavator_count_spin)

        self.truck_count_spin = QSpinBox()
        self.truck_count_spin.setRange(0, 128)
        self.truck_count_spin.setValue(int(self._agent_counts.get(1, 0)))
        self.truck_count_spin.valueChanged.connect(lambda v: self._on_agent_count_changed(1, v))
        self.truck_count_spin.setEnabled(False)
        self.truck_count_spin.setToolTip("Truck assignment is planned but currently disabled.")
        form.addRow("Truck count:", self.truck_count_spin)

        self.waypoint_agent_combo = QComboBox()
        self._rebuild_waypoint_agent_combo()
        form.addRow("Waypoint agent:", self.waypoint_agent_combo)

        self.agent_x_spin = QDoubleSpinBox()
        self.agent_x_spin.setRange(0.0, float(self.grid_size))
        self.agent_y_spin = QDoubleSpinBox()
        self.agent_y_spin.setRange(0.0, float(self.grid_size))
        self.agent_x_spin.valueChanged.connect(self._on_agent_pose_changed)
        self.agent_y_spin.valueChanged.connect(self._on_agent_pose_changed)
        agent_row = QHBoxLayout()
        agent_row.setSpacing(6)
        agent_row.addWidget(QLabel("x"))
        agent_row.addWidget(self.agent_x_spin)
        agent_row.addWidget(QLabel("y"))
        agent_row.addWidget(self.agent_y_spin)
        pick_agent_btn = QPushButton("Pick pose on grid")
        pick_agent_btn.clicked.connect(self._on_pick_agent)
        agent_row.addWidget(pick_agent_btn)
        form.addRow("Agent base:", agent_row)

        # Dig / dump cone selection (picked on grid inside valid area)
        dig_row = QHBoxLayout()
        dig_row.setSpacing(6)
        pick_dig_btn = QPushButton("Pick dig workspace on grid")
        pick_dig_btn.clicked.connect(self._on_pick_dig)
        dig_row.addWidget(pick_dig_btn)
        form.addRow("Dig workspace:", dig_row)

        dump_row = QHBoxLayout()
        dump_row.setSpacing(6)
        pick_dump_btn = QPushButton("Pick dump workspace on grid")
        pick_dump_btn.clicked.connect(self._on_pick_dump)
        dump_row.addWidget(pick_dump_btn)
        form.addRow("Dump workspace:", dump_row)

        self.block_dump_foundation_checkbox = QCheckBox("Block dump inside foundation")
        self.block_dump_foundation_checkbox.setChecked(False)
        form.addRow("", self.block_dump_foundation_checkbox)

        self.agent_yaw_spin = QDoubleSpinBox()
        self.agent_yaw_spin.setRange(-2.0 * math.pi, 2.0 * math.pi)
        self.agent_yaw_spin.setDecimals(2)
        self.agent_yaw_spin.setSingleStep(0.1)
        self.agent_yaw_spin.valueChanged.connect(self._on_agent_pose_changed)
        form.addRow("Base yaw [rad]:", self.agent_yaw_spin)

        # Spacer between sections
        form.addRow(QLabel(""))

        # Section: Output
        out_label = QLabel("Output")
        out_label.setStyleSheet("QLabel { font-weight:600; color:#444; }")
        form.addRow(out_label)

        self.json_path_edit = QLineEdit(os.path.join(os.getcwd(), "manual_plan.json"))
        json_row = QHBoxLayout()
        json_row.setSpacing(6)
        json_row.addWidget(self.json_path_edit)
        json_browse = QPushButton("Browse…")
        json_browse.clicked.connect(self._on_browse_json)
        json_row.addWidget(json_browse)
        form.addRow("JSON file:", json_row)

        self.pkl_path_edit = QLineEdit("")
        pkl_row = QHBoxLayout()
        pkl_row.setSpacing(6)
        pkl_row.addWidget(self.pkl_path_edit)
        pkl_browse = QPushButton("Browse…")
        pkl_browse.clicked.connect(self._on_browse_pkl)
        pkl_row.addWidget(pkl_browse)
        form.addRow("PKL file:", pkl_row)

        layout.addLayout(form)

        # Waypoints preview
        self.waypoints_label = QLabel("Waypoints")
        self.waypoints_label.setStyleSheet("QLabel { font-weight:600; margin-top:4px; }")
        layout.addWidget(self.waypoints_label)
        self.waypoints_list = QPlainTextEdit()
        self.waypoints_list.setReadOnly(True)
        self.waypoints_list.setMaximumHeight(140)
        layout.addWidget(self.waypoints_list)

        # Buttons (close on the left)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)
        btn_row.setSpacing(8)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        add_btn = QPushButton("Add waypoint")
        add_btn.clicked.connect(self._on_add_waypoint)
        remove_btn = QPushButton("Remove last")
        remove_btn.clicked.connect(self._on_remove_last_waypoint)
        generate_btn = QPushButton("Generate plan")
        generate_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(close_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addWidget(generate_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    def _rebuild_waypoint_agent_combo(self) -> None:
        previous = 0
        if hasattr(self, "waypoint_agent_combo") and self.waypoint_agent_combo is not None:
            previous = max(0, self.waypoint_agent_combo.currentIndex())
            self.waypoint_agent_combo.blockSignals(True)
            self.waypoint_agent_combo.clear()
        for type_name, type_id in self._agent_type_catalog:
            count = int(self._agent_counts.get(type_id, 0))
            for idx in range(count):
                self.waypoint_agent_combo.addItem(f"{type_name} {idx + 1}", (type_id, idx))
        if hasattr(self, "waypoint_agent_combo") and self.waypoint_agent_combo is not None:
            has_any_agents = self.waypoint_agent_combo.count() > 0
            if has_any_agents:
                self.waypoint_agent_combo.setCurrentIndex(min(previous, self.waypoint_agent_combo.count() - 1))
            self.waypoint_agent_combo.blockSignals(False)
            self.waypoint_agent_combo.setEnabled(has_any_agents)

    def _on_agent_count_changed(self, agent_type: int, value: int) -> None:
        self._agent_counts[int(agent_type)] = max(0, int(value))
        self._rebuild_waypoint_agent_combo()

    def _set_waypoint_visuals_visible(self, visible: bool) -> None:
        """Show/hide all permanent waypoint cones."""
        if self.scene is None:
            return
        # Toggle planning flag so painting is disabled while any visuals are shown
        try:
            if hasattr(self.scene, "planning_mode_active"):
                self.scene.planning_mode_active = bool(visible)
        except Exception:
            pass
        for visuals in self._waypoint_visuals:
            for it in visuals.get("dig_items", []) + visuals.get("dump_items", []):
                try:
                    it.setVisible(visible)
                except Exception:
                    pass
        for it in self._dump_coverage_items:
            try:
                it.setVisible(visible)
            except Exception:
                pass

    def _refresh_dump_coverage_visual(self) -> None:
        if self.scene is None or not hasattr(self.scene, "add_manual_workspace_cone"):
            return
        for it in self._dump_coverage_items:
            try:
                self.scene.removeItem(it)
            except Exception:
                pass
        self._dump_coverage_items = []
        if not self.plan_entries:
            return
        dumped = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for dig_entry, dump_entry in zip(self.plan_entries[0::2], self.plan_entries[1::2]):
            workspace_type = normalize_workspace_type(dig_entry.workspace_type)
            dig_mask = dig_entry.dug_mask.astype(bool)
            dump_mask = dump_entry.dump_mask.astype(bool)
            if workspace_type == "collect_dumped_soil":
                dumped &= ~dig_mask
            dumped |= dump_mask
        self._dump_coverage_items = self.scene.add_manual_workspace_cone("dump", dumped, style="covered_dump")

    def _clear_waypoint_visuals(self) -> None:
        if self.scene is not None:
            for visuals in self._waypoint_visuals:
                for it in visuals.get("dig_items", []) + visuals.get("dump_items", []):
                    try:
                        self.scene.removeItem(it)
                    except Exception:
                        pass
            for it in self._dump_coverage_items:
                try:
                    self.scene.removeItem(it)
                except Exception:
                    pass
        self._waypoint_visuals = []
        self._dump_coverage_items = []

    def _workspace_geometry_from_dict(self, geometry: Optional[Mapping[str, Any]]) -> Optional[WorkspaceGeometry]:
        if geometry is None:
            return None
        geometry_type = str(geometry["type"])
        if geometry_type == "fan":
            return WorkspaceFanGeometry(
                type="fan",
                heading_rad=float(geometry["heading_rad"]),
                min_radius_m=float(geometry["min_radius_m"]),
                max_radius_m=float(geometry["max_radius_m"]),
                aperture_rad=float(geometry["aperture_rad"]),
            )
        if geometry_type == "circle":
            return WorkspaceCircleGeometry(
                type="circle",
                center_row=float(geometry["center_row"]),
                center_col=float(geometry["center_col"]),
                radius_tiles=float(geometry["radius_tiles"]),
                radius_m=float(geometry["radius_m"]),
            )
        raise ValueError(f"unknown workspace_geometry type: {geometry_type!r}")

    def _plan_entry_from_schema_v2_waypoint(self, waypoint: Mapping[str, Any], index: int) -> PlanEntry:
        def mask(name: str) -> np.ndarray:
            arr = np.asarray(waypoint.get(name, []), dtype=bool)
            if arr.shape != (self.grid_size, self.grid_size):
                raise ValueError(f"waypoint[{index}].{name} shape {arr.shape} != {(self.grid_size, self.grid_size)}")
            return arr

        agent = waypoint.get("agent_state", {})
        pos_base = agent.get("pos_base", [0.0, 0.0])
        loaded_change = waypoint.get("loaded_state_change", {})
        loaded_after = bool(agent.get("loaded", loaded_change.get("after", False)))
        loaded_before = bool(loaded_change.get("before", not loaded_after))
        return PlanEntry(
            step=int(waypoint.get("step", index)),
            workspace_type=normalize_workspace_type(waypoint.get("workspace_type")),
            traversability_mask=mask("traversability_mask"),
            terrain_modification_mask=mask("terrain_modification_mask"),
            dug_mask=mask("dug_mask"),
            dump_mask=mask("dump_mask"),
            agent_state=AgentState(
                pos_base=[float(pos_base[0]), float(pos_base[1])],
                angle_base=float(agent.get("angle_base_rad", agent.get("angle_base", 0.0))),
                angle_cabin=float(agent.get("angle_cabin_rad", agent.get("angle_cabin", 0.0))),
                wheel_angle=float(agent.get("wheel_angle_rad", agent.get("wheel_angle", 0.0))),
            ),
            loaded_state_change=LoadedStateChange(before=loaded_before, after=loaded_after),
            agent_type=int(waypoint.get("agent_type", 0)),
            agent_index=int(waypoint.get("agent_index", 0)),
            workspace_geometry=self._workspace_geometry_from_dict(waypoint.get("workspace_geometry")),
        )

    def _add_waypoint_visuals(self, dig_entry: PlanEntry, dump_entry: PlanEntry, wp_index: int) -> None:
        visuals = {"dig_items": [], "dump_items": []}
        if self.scene is not None and hasattr(self.scene, "add_manual_workspace_cone"):
            workspace_type = normalize_workspace_type(dig_entry.workspace_type)
            dug_mask = dig_entry.dug_mask.astype(bool)
            dump_mask = dump_entry.dump_mask.astype(bool)
            dig_style = "small_marks" if workspace_type == "collect_dumped_soil" else "covered_dig"
            dump_style = "small_marks"
            if workspace_type == "excavate" and hasattr(self.scene, "foundation_mask"):
                foundation_mask = np.asarray(self.scene.foundation_mask, dtype=bool)
                if foundation_mask.shape == dug_mask.shape:
                    dug_mask = dug_mask & foundation_mask
            visuals["dig_items"] = self.scene.add_manual_workspace_cone(
                "dig", dug_mask, label=str(wp_index), style=dig_style
            )
            visuals["dump_items"] = self.scene.add_manual_workspace_cone(
                "dump", dump_mask, label=str(wp_index), style=dump_style
            )
        self._waypoint_visuals.append(visuals)

    def load_schema_v2_plan(self, plan_path: str) -> None:
        with open(plan_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        waypoints = doc.get("waypoints") if isinstance(doc, Mapping) else doc
        if not isinstance(waypoints, list):
            raise ValueError("plan JSON must contain a waypoints list or be a waypoint list")
        if len(waypoints) % 2 != 0:
            raise ValueError("manual plan import expects paired dig/dump waypoints")

        self._clear_waypoint_visuals()
        self.plan_entries = [self._plan_entry_from_schema_v2_waypoint(wp, idx) for idx, wp in enumerate(waypoints)]
        for idx, entry in enumerate(self.plan_entries):
            entry.step = idx
        for pair_index, (dig_entry, dump_entry) in enumerate(zip(self.plan_entries[0::2], self.plan_entries[1::2])):
            if normalize_workspace_type(dig_entry.workspace_type) != normalize_workspace_type(
                dump_entry.workspace_type
            ):
                raise ValueError(f"waypoint pair {pair_index} has mismatched workspace_type")
            self._add_waypoint_visuals(dig_entry, dump_entry, pair_index + 1)
        self._refresh_dump_coverage_visual()
        if hasattr(self, "json_path_edit"):
            self.json_path_edit.setText(plan_path)
        self._refresh_preview()

    def _clear_previews_and_agent(self) -> None:
        """Remove transient previews (rings/cones) and agent marker from the scene."""
        if self.scene is None:
            return
        try:
            if hasattr(self.scene, "manual_pick_active"):
                self.scene.manual_pick_active = False
                self.scene.manual_pick_callback = None
                self.scene.manual_pick_press_callback = None
                self.scene.manual_pick_move_callback = None
                self.scene.manual_pick_release_callback = None
                self.scene.manual_pick_dragging = False
            # Clear agent marker
            if getattr(self.scene, "manual_agent_item", None) is not None:
                try:
                    self.scene.removeItem(self.scene.manual_agent_item)
                except Exception:
                    pass
                self.scene.manual_agent_item = None
            # Clear preview cells
            for attr in ("manual_preview_dig_cells", "manual_preview_dump_cells"):
                items = getattr(self.scene, attr, [])
                for it in items:
                    try:
                        self.scene.removeItem(it)
                    except Exception:
                        pass
                setattr(self.scene, attr, [])
        except Exception:
            pass

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Show permanent waypoint cones when dialog is visible
        self._set_waypoint_visuals_visible(True)
        self._update_agent_marker()
        self._refresh_cone_previews_from_yaw()

    def closeEvent(self, event) -> None:
        # Hide permanent cones and clear transient previews & agent marker
        self._set_waypoint_visuals_visible(False)
        self._clear_previews_and_agent()
        # Keep dialog instance alive; just hide it
        self.hide()
        event.ignore()

    def _on_browse_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Select JSON output", self.json_path_edit.text(), "JSON (*.json)")
        if path:
            self.json_path_edit.setText(path)

    def _on_browse_pkl(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Select PKL output", self.pkl_path_edit.text(), "Pickle (*.pkl)")
        if path:
            self.pkl_path_edit.setText(path)

    # ----- Grid picking helpers -----
    def _current_agent_pos(self) -> Optional[Tuple[float, float]]:
        if not self._agent_base_selected:
            return None
        return (self.agent_x_spin.value(), self.agent_y_spin.value())

    def _update_agent_marker(self) -> None:
        agent_pos = self._current_agent_pos()
        if agent_pos is None or self.scene is None or not hasattr(self.scene, "set_manual_agent_marker"):
            return
        self.scene.set_manual_agent_marker(
            int(round(agent_pos[0])),
            int(round(agent_pos[1])),
            float(self.agent_yaw_spin.value()),
        )

    def _build_cone_from_current_yaw(self) -> np.ndarray:
        agent_pos = self._current_agent_pos()
        if agent_pos is None:
            raise RuntimeError("Pick an agent base on the grid before previewing a cone.")
        return _cone_mask(
            (self.grid_size, self.grid_size),
            agent_row=agent_pos[1],
            agent_col=agent_pos[0],
            tile_size=self.tile_size,
            fan_min_radius_m=self.fan_min_radius_spin.value(),
            fan_max_radius_m=self.fan_max_radius_spin.value(),
            angles_cabin=self.dig_angles_cabin_spin.value(),
            base_yaw=float(self.agent_yaw_spin.value()),
        )

    def _build_ring_mask(self) -> np.ndarray:
        agent_pos = self._current_agent_pos()
        if agent_pos is None:
            raise RuntimeError("Pick an agent base on the grid before selecting a cone.")
        return _ring_mask(
            (self.grid_size, self.grid_size),
            agent_row=agent_pos[1],
            agent_col=agent_pos[0],
            tile_size=self.tile_size,
            fan_min_radius_m=self.fan_min_radius_spin.value(),
            fan_max_radius_m=self.fan_max_radius_spin.value(),
        )

    def _cabin_angle_from_target(self, target_pos: Tuple[int, int]) -> float:
        agent_pos = self._current_agent_pos()
        if agent_pos is None:
            raise RuntimeError("Pick an agent base on the grid before selecting a cone.")
        dx = float(target_pos[0]) - float(agent_pos[0])
        dy = float(target_pos[1]) - float(agent_pos[1])
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return 0.0
        heading = float(np.arctan2(dy, dx))
        return _normalize_angle(heading - float(self.agent_yaw_spin.value()))

    def _dump_workspace_shape(self) -> str:
        return str(self.dump_workspace_shape_combo.currentData() or "fan")

    def _build_cone_from_target(self, target_pos: Tuple[int, int], *, kind: str) -> tuple[np.ndarray, float]:
        cabin_angle = self._cabin_angle_from_target(target_pos)
        heading = float(self.agent_yaw_spin.value()) + cabin_angle
        agent_pos = self._current_agent_pos()
        if agent_pos is None:
            raise RuntimeError("Pick an agent base on the grid before selecting a cone.")
        angles_cabin = self.dump_angles_cabin_spin.value() if kind == "dump" else self.dig_angles_cabin_spin.value()
        cone = _cone_mask(
            (self.grid_size, self.grid_size),
            agent_row=agent_pos[1],
            agent_col=agent_pos[0],
            tile_size=self.tile_size,
            fan_min_radius_m=self.fan_min_radius_spin.value(),
            fan_max_radius_m=self.fan_max_radius_spin.value(),
            angles_cabin=angles_cabin,
            base_yaw=heading,
        )
        return cone, cabin_angle

    def _build_dump_workspace_from_target(self, target_pos: Tuple[int, int]) -> tuple[np.ndarray, float]:
        cabin_angle = self._cabin_angle_from_target(target_pos)
        if self._dump_workspace_shape() == "circle":
            return (
                _circle_mask(
                    (self.grid_size, self.grid_size),
                    center_row=target_pos[1],
                    center_col=target_pos[0],
                    radius_tiles=self.dump_circle_radius_spin.value(),
                ),
                cabin_angle,
            )
        return self._build_cone_from_target(target_pos, kind="dump")

    def _refresh_cone_previews_from_yaw(self) -> None:
        if self._dig_target_pos is not None:
            self._dig_cone_mask, self._dig_cabin_angle = self._build_cone_from_target(self._dig_target_pos, kind="dig")
            if self.scene is not None and hasattr(self.scene, "set_manual_workspace_cone"):
                self.scene.set_manual_workspace_cone("dig", self._dig_cone_mask.astype(bool))
        elif self._dig_cone_mask is not None:
            self._dig_cone_mask = self._build_cone_from_current_yaw()
            self._dig_cabin_angle = 0.0
            if self.scene is not None and hasattr(self.scene, "set_manual_workspace_cone"):
                self.scene.set_manual_workspace_cone("dig", self._dig_cone_mask.astype(bool))
        if self._dump_target_pos is not None:
            self._dump_cone_mask, self._dump_cabin_angle = self._build_dump_workspace_from_target(self._dump_target_pos)
            if self.scene is not None and hasattr(self.scene, "set_manual_workspace_cone"):
                self.scene.set_manual_workspace_cone("dump", self._dump_cone_mask.astype(bool))
        elif self._dump_cone_mask is not None:
            if self._dump_workspace_shape() == "circle":
                self._dump_cone_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
            else:
                agent_pos = self._current_agent_pos()
                if agent_pos is None:
                    return
                self._dump_cone_mask = _cone_mask(
                    (self.grid_size, self.grid_size),
                    agent_row=agent_pos[1],
                    agent_col=agent_pos[0],
                    tile_size=self.tile_size,
                    fan_min_radius_m=self.fan_min_radius_spin.value(),
                    fan_max_radius_m=self.fan_max_radius_spin.value(),
                    angles_cabin=self.dump_angles_cabin_spin.value(),
                    base_yaw=float(self.agent_yaw_spin.value()),
                )
            self._dump_cabin_angle = 0.0
            if self.scene is not None and hasattr(self.scene, "set_manual_workspace_cone"):
                self.scene.set_manual_workspace_cone("dump", self._dump_cone_mask.astype(bool))

    def _on_agent_pose_changed(self, *_args) -> None:
        if not self._agent_base_selected:
            return
        self._update_agent_marker()
        self._refresh_cone_previews_from_yaw()

    def _on_workspace_params_changed(self, *_args) -> None:
        if not self._agent_base_selected:
            return
        self._refresh_cone_previews_from_yaw()

    def _on_fan_min_radius_changed(self, value: float) -> None:
        self.fan_max_radius_spin.setMinimum(float(value))
        self._on_workspace_params_changed()

    def _set_agent_yaw_from_drag(self, x: int, y: int) -> None:
        agent_pos = self._current_agent_pos()
        if agent_pos is None:
            return
        dx = float(x) - float(agent_pos[0])
        dy = float(y) - float(agent_pos[1])
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return
        yaw = float(np.arctan2(dy, dx))
        self.agent_yaw_spin.setValue(yaw)

    def _start_pick(self, kind: str) -> None:
        if self.scene is None or not hasattr(self.scene, "manual_pick_active"):
            QMessageBox.warning(self, "Grid Picking", "Scene is not available for picking.")
            return

        # Agent pick: simple, no cone validation
        if kind == "agent":

            def cb_agent_press(x: int, y: int) -> None:
                self._agent_base_selected = True
                self.agent_x_spin.setValue(float(x))
                self.agent_y_spin.setValue(float(y))
                self._update_agent_marker()

            def cb_agent_move(x: int, y: int) -> None:
                self._set_agent_yaw_from_drag(x, y)

            def cb_agent_release(x: int, y: int) -> None:
                self._set_agent_yaw_from_drag(x, y)

            self.scene.manual_pick_callback = None
            self.scene.manual_pick_press_callback = cb_agent_press
            self.scene.manual_pick_move_callback = cb_agent_move
            self.scene.manual_pick_release_callback = cb_agent_release
            self.scene.manual_pick_dragging = False
            self.scene.manual_pick_active = True
            return

        if not self._agent_base_selected:
            QMessageBox.warning(
                self, "Agent Base Required", "Pick an agent base on the grid before selecting dig/dump cones."
            )
            return
        try:
            ring = self._build_ring_mask()
        except Exception as exc:
            QMessageBox.critical(self, "Cone Error", str(exc))
            return
        try:
            if self.scene is not None and hasattr(self.scene, "set_manual_workspace_cone"):
                self.scene.set_manual_workspace_cone(kind, ring.astype(bool))
        except Exception:
            pass
        h, w = ring.shape

        def cb_cone(x: int, y: int) -> None:
            if not (0 <= y < h and 0 <= x < w) or not bool(ring[int(y), int(x)]):
                QMessageBox.warning(
                    self,
                    "Invalid Location",
                    "Selected cell is outside the reachable ring. Please click inside the highlighted area.",
                )
                self.scene.manual_pick_callback = cb_cone
                self.scene.manual_pick_active = True
                return
            target_pos = (x, y)
            if kind == "dig":
                cone, cabin_angle = self._build_cone_from_target(target_pos, kind="dig")
                self._dig_target_pos = target_pos
                self._dig_cone_mask = cone
                self._dig_cabin_angle = cabin_angle
            else:
                cone, cabin_angle = self._build_dump_workspace_from_target(target_pos)
                self._dump_target_pos = target_pos
                self._dump_cone_mask = cone
                self._dump_cabin_angle = cabin_angle
            if self.scene is not None and hasattr(self.scene, "set_manual_workspace_cone"):
                self.scene.set_manual_workspace_cone(kind, cone.astype(bool))

        self.scene.manual_pick_callback = cb_cone
        self.scene.manual_pick_press_callback = None
        self.scene.manual_pick_move_callback = None
        self.scene.manual_pick_release_callback = None
        self.scene.manual_pick_dragging = False
        self.scene.manual_pick_active = True

    def _on_pick_dig(self) -> None:
        self._start_pick("dig")

    def _on_pick_dump(self) -> None:
        self._start_pick("dump")

    def _on_pick_agent(self) -> None:
        self._start_pick("agent")

    def _on_add_waypoint(self) -> None:
        try:
            agent_pos = None
            if self._agent_base_selected:
                agent_pos = (self.agent_x_spin.value(), self.agent_y_spin.value())
            else:
                QMessageBox.warning(
                    self, "Agent Base Required", "Please pick an agent base position from the grid first."
                )
                return

            if self._dig_cone_mask is None or self._dump_cone_mask is None:
                QMessageBox.warning(
                    self,
                    "Dig/Dump Required",
                    "Please select both the dig workspace and the dump workspace on the grid before adding a waypoint.",
                )
                return

            selected_agent_data = self.waypoint_agent_combo.currentData()
            if selected_agent_data is None:
                QMessageBox.warning(
                    self,
                    "No Agent Available",
                    "Configure at least one agent (excavator or truck) before adding a waypoint.",
                )
                return
            selected_agent_type, selected_agent_index = selected_agent_data
            selected_agent_type = int(selected_agent_type)
            selected_agent_index = int(selected_agent_index)

            # For UI mode we use current grid size and a fully traversable mask by default
            shape = (self.grid_size, self.grid_size)
            traversability = np.ones(shape, dtype=bool)
            workspace_type = normalize_workspace_type(self.workspace_type_combo.currentData())

            # Limit digging to foundation layer if available from the scene
            dig_limit = None
            dump_limit = None
            try:
                if self.scene is not None:
                    if workspace_type == "excavate" and hasattr(self.scene, "foundation_mask"):
                        fm = np.asarray(self.scene.foundation_mask, dtype=bool)
                        if fm.shape == shape:
                            dig_limit = fm
                    # For dumping, always exclude obstacles and nodump tiles. Foundation is optional because some plans
                    # intentionally dump into the future dig area.
                    obstacle = np.zeros(shape, dtype=bool)
                    nodump = np.zeros(shape, dtype=bool)
                    if hasattr(self.scene, "obstacle_mask"):
                        om = np.asarray(self.scene.obstacle_mask, dtype=bool)
                        if om.shape == shape:
                            obstacle = om
                    if hasattr(self.scene, "nodump_mask"):
                        nm = np.asarray(self.scene.nodump_mask, dtype=bool)
                        if nm.shape == shape:
                            nodump = nm
                    if self.block_dump_foundation_checkbox.isChecked() and hasattr(self.scene, "foundation_mask"):
                        fm2 = np.asarray(self.scene.foundation_mask, dtype=bool)
                        if fm2.shape == shape:
                            pass
                        else:
                            fm2 = np.zeros(shape, dtype=bool)
                    else:
                        fm2 = np.zeros(shape, dtype=bool)
                    blocked = obstacle | nodump | fm2
                    dump_limit = ~blocked
            except Exception:
                dig_limit = None
                dump_limit = None

            # Build a local 2-step (dig + dump) plan for this waypoint
            agent_yaw_plan = _ui_yaw_to_plan_yaw(self.agent_yaw_spin.value())
            agent_cabin_plan = _ui_relative_angle_to_plan_relative(self._default_agent_cabin_angle)
            agent_wheel_plan = _ui_relative_angle_to_plan_relative(self._default_agent_wheel_angle)
            dig_cabin_plan = (
                _ui_relative_angle_to_plan_relative(self._dig_cabin_angle)
                if self._dig_cabin_angle is not None
                else None
            )
            dump_cabin_plan = (
                _ui_relative_angle_to_plan_relative(self._dump_cabin_angle)
                if self._dump_cabin_angle is not None
                else None
            )
            dump_center = (0.0, 0.0)
            dump_radius = 0.0
            if self._dump_workspace_shape() == "circle":
                assert self._dump_target_pos is not None
                dump_center = (float(self._dump_target_pos[1]), float(self._dump_target_pos[0]))
                dump_radius = float(self.dump_circle_radius_spin.value())
            plan_entries = build_manual_plan(
                map_root="",  # unused in UI path due to shape_override
                dig_center=(0.0, 0.0),
                dig_radius=0.0,
                dump_center=dump_center,
                dump_radius=dump_radius,
                traversability_from_occupancy=False,
                agent_pos=agent_pos,
                agent_yaw=agent_yaw_plan,
                agent_cabin_angle=agent_cabin_plan,
                agent_wheel_angle=agent_wheel_plan,
                shape_override=shape,
                traversability_override=traversability,
                tile_size=self.tile_size,
                fan_min_radius_m=self.fan_min_radius_spin.value(),
                fan_max_radius_m=self.fan_max_radius_spin.value(),
                dig_angles_cabin=self.dig_angles_cabin_spin.value(),
                dump_angles_cabin=self.dump_angles_cabin_spin.value(),
                dump_workspace_shape=self._dump_workspace_shape(),
                dig_mask_override=self._dig_cone_mask,
                dump_mask_override=self._dump_cone_mask,
                dig_cabin_override=dig_cabin_plan,
                dump_cabin_override=dump_cabin_plan,
                dig_limit_mask=dig_limit,
                dump_limit_mask=dump_limit,
                workspace_type=workspace_type,
            )

            # Append to global plan list with continuous step indices
            base_step = len(self.plan_entries)
            for i, entry in enumerate(plan_entries):
                entry.step = base_step + i
                entry.agent_type = selected_agent_type
                entry.agent_index = selected_agent_index
                self.plan_entries.append(entry)

            # Visualize cone masks on grid for this waypoint if scene is available (permanent)
            try:
                visuals = {"dig_items": [], "dump_items": []}
                if self.scene is not None and hasattr(self.scene, "add_manual_workspace_cone"):
                    dug_mask = plan_entries[0].dug_mask.astype(bool)
                    dump_mask = plan_entries[1].dump_mask.astype(bool)
                    dig_style = "small_marks" if workspace_type == "collect_dumped_soil" else "covered_dig"
                    dump_style = "small_marks"
                    if workspace_type == "excavate" and hasattr(self.scene, "foundation_mask"):
                        foundation_mask = np.asarray(self.scene.foundation_mask, dtype=bool)
                        if foundation_mask.shape == dug_mask.shape:
                            dug_mask = dug_mask & foundation_mask
                    # Waypoint index (1-based) for labels
                    wp_index = (base_step // 2) + 1
                    visuals["dig_items"] = self.scene.add_manual_workspace_cone(
                        "dig", dug_mask, label=str(wp_index), style=dig_style
                    )
                    visuals["dump_items"] = self.scene.add_manual_workspace_cone(
                        "dump", dump_mask, label=str(wp_index), style=dump_style
                    )
                self._waypoint_visuals.append(visuals)
                self._refresh_dump_coverage_visual()
            except Exception:
                pass

            # Reset dig/dump picks and masks for next waypoint (agent base is kept)
            self._dig_cone_mask = None
            self._dump_cone_mask = None
            self._dig_target_pos = None
            self._dump_target_pos = None
            self._dig_cabin_angle = None
            self._dump_cabin_angle = None

            # Refresh waypoint preview
            self._refresh_preview()

        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to add waypoint:\n{exc}")

    def _on_remove_last_waypoint(self) -> None:
        """Remove the last added waypoint (dig+dump pair) and its visuals."""
        if not self.plan_entries or not self._waypoint_visuals:
            QMessageBox.warning(self, "No Waypoints", "There is no waypoint to remove.")
            return
        try:
            # Remove last two steps from plan (dig and dump)
            if len(self.plan_entries) >= 2:
                self.plan_entries = self.plan_entries[:-2]
            else:
                self.plan_entries = []

            # Remove last visuals from scene
            visuals = self._waypoint_visuals.pop()
            if self.scene is not None:
                for it in visuals.get("dig_items", []) + visuals.get("dump_items", []):
                    try:
                        self.scene.removeItem(it)
                    except Exception:
                        pass
            self._refresh_dump_coverage_visual()

            # Renumber steps to keep them contiguous
            for idx, entry in enumerate(self.plan_entries):
                entry.step = idx

            self._refresh_preview()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to remove waypoint:\n{exc}")

    def _refresh_preview(self) -> None:
        """Update the text preview of all waypoints in the current plan."""
        serializable_waypoints = [entry.to_serializable() for entry in self.plan_entries]
        preview_lines = []
        for idx, wp in enumerate(serializable_waypoints):
            step = wp.get("step", idx)
            dug_count = sum(sum(1 for v in row if v) for row in wp.get("dug_mask", []))
            dump_count = sum(sum(1 for v in row if v) for row in wp.get("dump_mask", []))
            loaded = wp.get("loaded_state_change", {})
            workspace_type = normalize_workspace_type(wp.get("workspace_type"))
            agent_type = int(wp.get("agent_type", 0))
            agent_index = int(wp.get("agent_index", 0))
            if agent_type == 0:
                agent_label = f"excavator-{agent_index + 1}"
            elif agent_type == 1:
                agent_label = f"truck-{agent_index + 1}"
            else:
                agent_label = f"type{agent_type}-{agent_index + 1}"
            preview_lines.append(
                f"Step {step}: {workspace_type}, {agent_label}, dug={dug_count} cells, dump={dump_count} cells, loaded {loaded.get('before')}→{loaded.get('after')}"
            )
        self.waypoints_list.setPlainText("\n".join(preview_lines))

    def _on_generate(self) -> None:
        if not self.plan_entries:
            QMessageBox.warning(self, "No Waypoints", "Add at least one waypoint before generating a plan.")
            return

        try:
            serializable_waypoints = [entry.to_serializable() for entry in self.plan_entries]
            out_obj = package_waypoints_schema_v2(
                serializable_waypoints,
                self._get_plan_alignment(),
            )

            json_path = self.json_path_edit.text().strip()
            if json_path:
                dump_plan_json(out_obj, json_path)

            pkl_path = self.pkl_path_edit.text().strip()
            if pkl_path:
                os.makedirs(os.path.dirname(os.path.abspath(pkl_path)), exist_ok=True)
                with open(pkl_path, "wb") as f:
                    pickle.dump(serializable_waypoints, f)

            QMessageBox.information(self, "Manual Plan", "Plan exported successfully.")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to generate manual plan:\n{exc}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate a manual dig/dump plan compatible with extract_map.py.\n"
            "This uses a simple cylindrical (disk) workspace around user-specified centers.\n"
            "You can extend this later with sector (theta) parameters without changing the output format."
        )
    )
    p.add_argument(
        "--map-root",
        type=str,
        required=True,
        help="Path to Terra map root (folder containing 'actions/', 'occupancy/', etc.).",
    )
    p.add_argument("--dig-row", type=float, required=True, help="Dig center row index (0-based).")
    p.add_argument("--dig-col", type=float, required=True, help="Dig center col index (0-based).")
    p.add_argument("--dig-radius", type=float, required=True, help="Dig radius in cells.")
    p.add_argument("--dump-row", type=float, required=True, help="Dump center row index (0-based).")
    p.add_argument("--dump-col", type=float, required=True, help="Dump center col index (0-based).")
    p.add_argument("--dump-radius", type=float, required=True, help="Dump radius in cells.")

    p.add_argument(
        "--agent-x",
        type=float,
        default=None,
        help="Agent base x (col) in grid coordinates; defaults to dump center col.",
    )
    p.add_argument(
        "--agent-y",
        type=float,
        default=None,
        help="Agent base y (row) in grid coordinates; defaults to dump center row.",
    )
    p.add_argument("--agent-yaw", type=float, default=0.0, help="Base yaw angle in radians.")
    p.add_argument("--agent-cabin-angle", type=float, default=0.0, help="Cabin angle in radians.")
    p.add_argument("--cabin-angle-steps", type=int, default=8, help="Legacy shared cabin angle steps.")
    p.add_argument("--fan-min-radius-m", type=float, default=4.0, help="Inner fan workspace radius in meters.")
    p.add_argument("--fan-max-radius-m", type=float, default=6.0, help="Outer fan workspace radius in meters.")
    p.add_argument("--dig-cabin-angle-steps", type=int, default=None, help="Cabin angle steps for the dig workspace.")
    p.add_argument(
        "--dump-cabin-angle-steps",
        type=int,
        default=None,
        help="Cabin angle steps for the dump fan workspace.",
    )
    p.add_argument(
        "--dump-workspace-shape",
        type=str,
        default="fan",
        choices=["fan", "circle"],
        help="Dump workspace mask shape.",
    )
    p.add_argument(
        "--agent-wheel-angle",
        type=float,
        default=0.0,
        help="Wheel angle in radians.",
    )

    p.add_argument(
        "--no-traversability-from-occupancy",
        action="store_true",
        help="If set, traversability mask will be all True instead of derived from occupancy.",
    )

    p.add_argument(
        "--serialize-json",
        type=str,
        help="Output JSON file path. If omitted, prints JSON to stdout.",
    )
    p.add_argument(
        "--serialize-pkl",
        type=str,
        help="Optional .pkl output path containing a list of per-waypoint dicts.",
    )
    p.add_argument(
        "--terra-metadata-yaml",
        type=str,
        help="terra_metadata.yaml used to package the manual plan into schema-v2 map coordinates.",
    )
    p.add_argument(
        "--workspace-type",
        type=str,
        default="excavate",
        choices=["excavate", "collect_dumped_soil"],
        help="Workspace operation for the generated dig/dump pair.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    map_root = args.map_root
    if not os.path.isdir(map_root):
        raise SystemExit(f"map_root is not a directory: {map_root}")

    agent_pos: Optional[Tuple[float, float]] = None
    if args.agent_x is not None and args.agent_y is not None:
        agent_pos = (args.agent_x, args.agent_y)

    plan_entries = build_manual_plan(
        map_root=map_root,
        dig_center=(args.dig_row, args.dig_col),
        dig_radius=args.dig_radius,
        dump_center=(args.dump_row, args.dump_col),
        dump_radius=args.dump_radius,
        traversability_from_occupancy=not args.no_traversability_from_occupancy,
        agent_pos=agent_pos,
        agent_yaw=args.agent_yaw,
        agent_cabin_angle=args.agent_cabin_angle,
        agent_wheel_angle=args.agent_wheel_angle,
        fan_min_radius_m=args.fan_min_radius_m,
        fan_max_radius_m=args.fan_max_radius_m,
        angles_cabin=args.cabin_angle_steps,
        dig_angles_cabin=args.dig_cabin_angle_steps,
        dump_angles_cabin=args.dump_cabin_angle_steps,
        dump_workspace_shape=args.dump_workspace_shape,
        workspace_type=args.workspace_type,
    )

    serializable_waypoints = [entry.to_serializable() for entry in plan_entries]
    out_obj = None
    if args.serialize_json or not args.serialize_pkl:
        if not args.terra_metadata_yaml:
            raise SystemExit("manual_plan_generator now emits schema-v2 JSON and requires --terra-metadata-yaml")
        alignment = load_alignment_from_terra_metadata(args.terra_metadata_yaml)
        out_obj = package_waypoints_schema_v2(
            serializable_waypoints,
            alignment,
        )

    if args.serialize_json:
        assert out_obj is not None
        dump_plan_json(out_obj, args.serialize_json)
        print(f"Wrote manual plan JSON to {args.serialize_json}")
    elif out_obj is not None:
        print(json.dumps(out_obj, indent=2))

    if args.serialize_pkl:
        os.makedirs(os.path.dirname(os.path.abspath(args.serialize_pkl)), exist_ok=True)
        with open(args.serialize_pkl, "wb") as f:
            # Downstream expects a list of per-waypoint dicts (same as extract_map.py)
            pickle.dump(serializable_waypoints, f)
        print(f"Wrote manual plan PKL to {args.serialize_pkl}")


if __name__ == "__main__":
    main()
