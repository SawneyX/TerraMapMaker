import argparse
import json
import os
import pickle
import re
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np

try:
    # Qt is available in the main application; importing here allows an optional UI dialog.
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QCheckBox,
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
class PlanEntry:
    step: int
    traversability_mask: np.ndarray
    terrain_modification_mask: np.ndarray
    dug_mask: np.ndarray
    dump_mask: np.ndarray
    agent_state: AgentState
    loaded_state_change: LoadedStateChange
    agent_type: int
    agent_index: int

    def to_serializable(self) -> dict:
        """Convert masks to lists so JSON / pickle match extract_map-style structure."""
        return {
            "step": self.step,
            "traversability_mask": self.traversability_mask.astype(bool).tolist(),
            "terrain_modification_mask": self.terrain_modification_mask.astype(bool).tolist(),
            "dug_mask": self.dug_mask.astype(bool).tolist(),
            "dump_mask": self.dump_mask.astype(bool).tolist(),
            "agent_state": asdict(self.agent_state),
            "loaded_state_change": asdict(self.loaded_state_change),
            "agent_type": self.agent_type,
            "agent_index": self.agent_index,
        }


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
    fixed_extension: float,
    dig_portion_radius: float,
    min_distance_from_agent: float,
    angles_cabin: int,
    base_yaw: float = 0.0,
) -> np.ndarray:
    """
    Boolean cone mask in grid coordinates (row, col), centered at the agent base.

    This implements the same semantics as the provided reference:
      r_min = fixed_extension + min_distance_from_agent
      r_max = fixed_extension + min_distance_from_agent + dig_portion_radius * tile_size
      theta_max = 2*pi / angles_cabin
      theta_min = -theta_max
      mask = (r in [r_min, r_max]) & (theta in [theta_min, theta_max])
    where r, theta are cylindrical coordinates around the agent base.
    """
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    # Work in tile units, convert radial bounds from meters → tiles.
    dx_tiles = (xx - agent_col)
    dy_tiles = (yy - agent_row)
    r_tiles = np.sqrt(dx_tiles * dx_tiles + dy_tiles * dy_tiles)

    # Angle relative to agent heading (scaling cancels out)
    theta = np.arctan2(dy_tiles, dx_tiles) - base_yaw
    # Wrap to [-pi, pi]
    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    r_min_m = fixed_extension + min_distance_from_agent
    r_max_m = fixed_extension + min_distance_from_agent + dig_portion_radius * tile_size
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
    fixed_extension: float,
    dig_portion_radius: float,
    min_distance_from_agent: float,
) -> np.ndarray:
    """Radial ring of valid cone origins around the agent (no theta restriction)."""
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    dx_tiles = (xx - agent_col)
    dy_tiles = (yy - agent_row)
    r_tiles = np.sqrt(dx_tiles * dx_tiles + dy_tiles * dy_tiles)

    r_min_m = fixed_extension + min_distance_from_agent
    r_max_m = fixed_extension + min_distance_from_agent + dig_portion_radius * tile_size
    r_min = r_min_m / max(tile_size, 1e-6)
    r_max = r_max_m / max(tile_size, 1e-6)
    return np.logical_and(r_tiles >= r_min, r_tiles <= r_max)


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
    tile_size: float = 1.0,
    fixed_extension: float = 0.5,
    dig_portion_radius: float = 1.0,
    min_distance_from_agent: float = 0.0,
    angles_cabin: int = 8,
    dig_mask_override: Optional[np.ndarray] = None,
    dump_mask_override: Optional[np.ndarray] = None,
    dig_yaw_override: Optional[float] = None,
    dump_yaw_override: Optional[float] = None,
    dig_limit_mask: Optional[np.ndarray] = None,
    dump_limit_mask: Optional[np.ndarray] = None,
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

    # Traversability: either from occupancy or fully traversable
    if traversability_override is not None:
        traversability = traversability_override.astype(bool)
        if traversability.shape != (h, w):
            raise ValueError(f"traversability_override shape {traversability.shape} != {(h, w)}")
    elif traversability_from_occupancy:
        occ_npy = os.path.join(map_root, "occupancy", "img_1.npy")
        if not os.path.exists(occ_npy):
            raise FileNotFoundError(
                f"Requested traversability_from_occupancy=True but file not found: {occ_npy}"
            )
        occ = np.load(occ_npy).astype(bool)
        if occ.shape != (h, w):
            raise ValueError(
                f"Occupancy mask shape {occ.shape} does not match action map shape {(h, w)}"
            )
        traversability = ~occ  # traversable where no obstacle
    else:
        traversability = np.ones((h, w), dtype=bool)

    # Agent pose (grid coordinates, not meters – leave translation to consumer)
    if agent_pos is None:
        agent_pos = (float(dig_center[1]), float(dig_center[0]))  # (x, y) ~ (col, row)

    base_agent_state = AgentState(
        pos_base=[float(agent_pos[0]), float(agent_pos[1])],  # x, y
        angle_base=float(agent_yaw),
        angle_cabin=float(agent_cabin_angle),
        wheel_angle=float(agent_wheel_angle),
    )

    # Dig step: cone in front of the agent (r/theta-based reachability)
    if dig_mask_override is not None:
        dug_mask_dig = dig_mask_override.astype(bool)
    else:
        dug_mask_dig = _cone_mask(
            (h, w),
            agent_row=agent_pos[1],
            agent_col=agent_pos[0],
            tile_size=tile_size,
            fixed_extension=fixed_extension,
            dig_portion_radius=dig_portion_radius,
            min_distance_from_agent=min_distance_from_agent,
            angles_cabin=angles_cabin,
            base_yaw=agent_yaw,
        )

    # Optionally restrict digging to a subset of tiles (e.g. foundation layer)
    if dig_limit_mask is not None:
        limit = np.asarray(dig_limit_mask, dtype=bool)
        if limit.shape != (h, w):
            raise ValueError(f"dig_limit_mask shape {limit.shape} != {(h, w)}")
        dug_mask_dig = np.logical_and(dug_mask_dig, limit)
    dump_mask_dig = np.zeros((h, w), dtype=bool)
    terrain_mod_dig = dug_mask_dig | dump_mask_dig

    dig_entry = PlanEntry(
        step=0,
        traversability_mask=traversability,
        terrain_modification_mask=terrain_mod_dig,
        dug_mask=dug_mask_dig,
        dump_mask=dump_mask_dig,
        agent_state=base_agent_state,
        loaded_state_change=LoadedStateChange(before=False, after=True),
        agent_type=0,
        agent_index=0,
    )

    # Dump step: reuse or override cone as a dump mask
    dug_mask_dump = np.zeros((h, w), dtype=bool)
    if dump_mask_override is not None:
        dump_mask_dump = dump_mask_override.astype(bool)
    else:
        dump_mask_dump = dug_mask_dig.copy()

    # Optionally restrict dumping to allowed tiles (e.g. exclude obstacle/foundation/nodump)
    if dump_limit_mask is not None:
        limit_d = np.asarray(dump_limit_mask, dtype=bool)
        if limit_d.shape != (h, w):
            raise ValueError(f"dump_limit_mask shape {limit_d.shape} != {(h, w)}")
        dump_mask_dump = np.logical_and(dump_mask_dump, limit_d)
    terrain_mod_dump = dug_mask_dump | dump_mask_dump

    dump_entry = PlanEntry(
        step=1,
        traversability_mask=traversability,
        terrain_modification_mask=terrain_mod_dump,
        dug_mask=dug_mask_dump,
        dump_mask=dump_mask_dump,
        agent_state=base_agent_state,
        loaded_state_change=LoadedStateChange(before=True, after=False),
        agent_type=0,
        agent_index=0,
    )

    # Allow per-step yaw overrides
    if dig_yaw_override is not None:
        dig_entry.agent_state.angle_base = float(dig_yaw_override)
    if dump_yaw_override is not None:
        dump_entry.agent_state.angle_base = float(dump_yaw_override)

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
        tile_size: float = 1.0,
    ) -> None:
        if QDialog is None:
            raise RuntimeError("Qt is not available; ManualPlanDialog cannot be used.")
        super().__init__(parent)
        self.setWindowTitle("Manual Dig/Dump Plan")
        self.grid_size = int(grid_size)
        self.scene = scene
        self.tile_size = float(tile_size)
        self.plan_entries: List[PlanEntry] = []
        # Cached cone / picks for current waypoint
        self._current_ring_mask: Optional[np.ndarray] = None
        self._dig_cone_mask: Optional[np.ndarray] = None
        self._dump_cone_mask: Optional[np.ndarray] = None
        self._dig_yaw: Optional[float] = None
        self._dump_yaw: Optional[float] = None
        self._dig_pick_pos: Optional[Tuple[int, int]] = None
        self._dump_pick_pos: Optional[Tuple[int, int]] = None
        # Per-waypoint visuals (permanent cones on the scene)
        self._waypoint_visuals: List[dict] = []
        # Track which tiles have already been dug by previous waypoints
        self._cumulative_dug_mask: Optional[np.ndarray] = None
        self._build_ui()

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

        self.fixed_extension_spin = QDoubleSpinBox()
        self.fixed_extension_spin.setRange(0.0, 10.0)
        self.fixed_extension_spin.setDecimals(3)
        self.fixed_extension_spin.setValue(0.5)
        form.addRow("Fixed extension (m):", self.fixed_extension_spin)

        self.min_dist_spin = QDoubleSpinBox()
        self.min_dist_spin.setRange(0.0, 50.0)
        self.min_dist_spin.setDecimals(3)
        self.min_dist_spin.setValue(3.0)
        form.addRow("Min distance from agent (m):", self.min_dist_spin)

        self.dig_portion_radius_spin = QDoubleSpinBox()
        self.dig_portion_radius_spin.setRange(0.0, 50.0)
        self.dig_portion_radius_spin.setDecimals(3)
        self.dig_portion_radius_spin.setValue(6.0)
        form.addRow("Dig portion radius (tiles):", self.dig_portion_radius_spin)

        self.angles_cabin_spin = QSpinBox()
        self.angles_cabin_spin.setRange(1, 360)
        self.angles_cabin_spin.setValue(12)
        form.addRow("Cabin angle steps:", self.angles_cabin_spin)

        # Spacer between sections
        form.addRow(QLabel(""))

        # Section: Agent pose & cones
        pose_label = QLabel("Agent & Cones")
        pose_label.setStyleSheet("QLabel { font-weight:600; color:#444; }")
        form.addRow(pose_label)

        self.agent_x_spin = QDoubleSpinBox()
        self.agent_x_spin.setRange(0.0, float(self.grid_size))
        self.agent_y_spin = QDoubleSpinBox()
        self.agent_y_spin.setRange(0.0, float(self.grid_size))
        agent_row = QHBoxLayout()
        agent_row.setSpacing(6)
        agent_row.addWidget(QLabel("x"))
        agent_row.addWidget(self.agent_x_spin)
        agent_row.addWidget(QLabel("y"))
        agent_row.addWidget(self.agent_y_spin)
        pick_agent_btn = QPushButton("Pick from grid")
        pick_agent_btn.clicked.connect(self._on_pick_agent)
        agent_row.addWidget(pick_agent_btn)
        form.addRow("Agent base:", agent_row)

        # Dig / dump cone selection (picked on grid inside valid area)
        dig_row = QHBoxLayout()
        dig_row.setSpacing(6)
        pick_dig_btn = QPushButton("Pick dig cone on grid")
        pick_dig_btn.clicked.connect(self._on_pick_dig)
        dig_row.addWidget(pick_dig_btn)
        form.addRow("Dig cone:", dig_row)

        dump_row = QHBoxLayout()
        dump_row.setSpacing(6)
        pick_dump_btn = QPushButton("Pick dump cone on grid")
        pick_dump_btn.clicked.connect(self._on_pick_dump)
        dump_row.addWidget(pick_dump_btn)
        form.addRow("Dump cone:", dump_row)

        self.agent_yaw_spin = QDoubleSpinBox()
        self.agent_yaw_spin.setRange(-360.0, 360.0)
        self.agent_yaw_spin.setDecimals(2)
        form.addRow("Base yaw:", self.agent_yaw_spin)

        self.agent_cabin_spin = QDoubleSpinBox()
        self.agent_cabin_spin.setRange(-360.0, 360.0)
        self.agent_cabin_spin.setDecimals(2)
        form.addRow("Cabin angle:", self.agent_cabin_spin)

        self.agent_wheel_spin = QDoubleSpinBox()
        self.agent_wheel_spin.setRange(-360.0, 360.0)
        self.agent_wheel_spin.setDecimals(2)
        form.addRow("Wheel angle:", self.agent_wheel_spin)

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

    def _clear_previews_and_agent(self) -> None:
        """Remove transient previews (rings/cones) and agent marker from the scene."""
        if self.scene is None:
            return
        try:
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
    def _ensure_ring_mask(self, agent_pos: Tuple[float, float], kind: str) -> Optional[np.ndarray]:
        """Compute and preview the reachable ring where cones can be placed."""
        shape = (self.grid_size, self.grid_size)
        mask = _ring_mask(
            shape,
            agent_row=agent_pos[1],
            agent_col=agent_pos[0],
            tile_size=self.tile_size,
            fixed_extension=self.fixed_extension_spin.value(),
            dig_portion_radius=self.dig_portion_radius_spin.value(),
            min_distance_from_agent=self.min_dist_spin.value(),
        )
        self._current_ring_mask = mask
        # Preview valid region on grid (use dig / dump color depending on kind)
        try:
            if self.scene is not None and hasattr(self.scene, "set_manual_workspace_cone"):
                role = "dump" if kind == "dump" else "dig"
                self.scene.set_manual_workspace_cone(role, mask.astype(bool))
        except Exception:
            pass
        return mask

    def _start_pick(self, kind: str) -> None:
        if self.scene is None or not hasattr(self.scene, "manual_pick_active"):
            QMessageBox.warning(self, "Grid Picking", "Scene is not available for picking.")
            return

        # Agent pick: simple, no cone validation
        if kind == "agent":
            def cb_agent(x: int, y: int) -> None:
                self.agent_x_spin.setValue(float(x))
                self.agent_y_spin.setValue(float(y))
                try:
                    self.scene.set_manual_agent_marker(x, y)
                except Exception:
                    pass
            self.scene.manual_pick_callback = cb_agent
            self.scene.manual_pick_active = True
            return

        # Dig / dump pick: must lie inside current cone
        if self.agent_x_spin.value() == 0.0 and self.agent_y_spin.value() == 0.0:
            QMessageBox.warning(self, "Agent Base Required", "Pick an agent base on the grid before selecting dig/dump cones.")
            return
        agent_pos = (self.agent_x_spin.value(), self.agent_y_spin.value())
        ring = self._ensure_ring_mask(agent_pos, kind)
        if ring is None:
            QMessageBox.critical(self, "Cone Error", "Failed to compute reachability cone.")
            return
        h, w = ring.shape

        def cb_cone(x: int, y: int) -> None:
            if not (0 <= y < h and 0 <= x < w) or not bool(ring[int(y), int(x)]):
                QMessageBox.warning(self, "Invalid Location", "Selected cell is outside the reachable ring. Please click inside the highlighted area.")
                # keep picking active
                return
            # Direction from agent to picked cell (tile units are enough for angle)
            dx = (x - agent_pos[0])
            dy = (y - agent_pos[1])
            theta_dir = float(np.arctan2(dy, dx))
            # Build cone mask oriented along this direction
            cone = _cone_mask(
                (self.grid_size, self.grid_size),
                agent_row=agent_pos[1],
                agent_col=agent_pos[0],
                tile_size=self.tile_size,
                fixed_extension=self.fixed_extension_spin.value(),
                dig_portion_radius=self.dig_portion_radius_spin.value(),
                min_distance_from_agent=self.min_dist_spin.value(),
                angles_cabin=self.angles_cabin_spin.value(),
                base_yaw=theta_dir,
            )
            if kind == "dig":
                self._dig_pick_pos = (x, y)
                self._dig_cone_mask = cone
                self._dig_yaw = theta_dir
                try:
                    if self.scene is not None and hasattr(self.scene, "set_manual_workspace_cone"):
                        self.scene.set_manual_workspace_cone("dig", cone.astype(bool))
                except Exception:
                    pass
            else:
                self._dump_pick_pos = (x, y)
                self._dump_cone_mask = cone
                self._dump_yaw = theta_dir
                try:
                    if self.scene is not None and hasattr(self.scene, "set_manual_workspace_cone"):
                        self.scene.set_manual_workspace_cone("dump", cone.astype(bool))
                except Exception:
                    pass
            # stop picking after a valid selection
            self.scene.manual_pick_active = False
            self.scene.manual_pick_callback = None

        self.scene.manual_pick_callback = cb_cone
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
            if self.agent_x_spin.value() != 0.0 or self.agent_y_spin.value() != 0.0:
                agent_pos = (self.agent_x_spin.value(), self.agent_y_spin.value())
            else:
                QMessageBox.warning(self, "Agent Base Required", "Please pick an agent base position from the grid first.")
                return

            if self._dig_cone_mask is None or self._dump_cone_mask is None:
                QMessageBox.warning(self, "Dig/Dump Required", "Please select both a dig cone and a dump cone location inside the reachable area before adding a waypoint.")
                return

            # For UI mode we use current grid size and a fully traversable mask by default
            shape = (self.grid_size, self.grid_size)
            traversability = np.ones(shape, dtype=bool)

            # Limit digging to foundation layer if available from the scene
            dig_limit = None
            dump_limit = None
            try:
                if self.scene is not None:
                    if hasattr(self.scene, "foundation_mask"):
                        fm = np.asarray(self.scene.foundation_mask, dtype=bool)
                        if fm.shape == shape:
                            dig_limit = fm
                    # For dumping, exclude obstacles, foundation, and nodump tiles
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
                    if hasattr(self.scene, "foundation_mask"):
                        fm2 = np.asarray(self.scene.foundation_mask, dtype=bool)
                        if fm2.shape == shape:
                            # reuse foundation mask
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
            plan_entries = build_manual_plan(
                map_root="",  # unused in UI path due to shape_override
                dig_center=(0.0, 0.0),
                dig_radius=0.0,
                dump_center=(0.0, 0.0),
                dump_radius=0.0,
                traversability_from_occupancy=False,
                agent_pos=agent_pos,
                agent_yaw=self.agent_yaw_spin.value(),
                agent_cabin_angle=self.agent_cabin_spin.value(),
                agent_wheel_angle=self.agent_wheel_spin.value(),
                shape_override=shape,
                traversability_override=traversability,
                tile_size=self.tile_size,
                fixed_extension=self.fixed_extension_spin.value(),
                dig_portion_radius=self.dig_portion_radius_spin.value(),
                min_distance_from_agent=self.min_dist_spin.value(),
                angles_cabin=self.angles_cabin_spin.value(),
                dig_mask_override=self._dig_cone_mask,
                dump_mask_override=self._dump_cone_mask,
                dig_yaw_override=self._dig_yaw,
                dump_yaw_override=self._dump_yaw,
                dig_limit_mask=dig_limit,
                dump_limit_mask=dump_limit,
            )

            # Ensure we only dig tiles that have not already been dug by previous waypoints
            dug_mask_new = plan_entries[0].dug_mask.astype(bool)
            if self._cumulative_dug_mask is None:
                self._cumulative_dug_mask = np.zeros_like(dug_mask_new, dtype=bool)
            if self._cumulative_dug_mask.shape != dug_mask_new.shape:
                self._cumulative_dug_mask = np.zeros_like(dug_mask_new, dtype=bool)
            dug_mask_new = np.logical_and(dug_mask_new, ~self._cumulative_dug_mask)
            if not np.any(dug_mask_new):
                QMessageBox.warning(
                    self,
                    "No New Dig Cells",
                    "The selected dig cone only covers tiles that have already been dug by previous waypoints.",
                )
                return
            # Update dig entry masks with filtered dig region
            plan_entries[0].dug_mask = dug_mask_new
            plan_entries[0].terrain_modification_mask = np.logical_or(
                dug_mask_new, plan_entries[0].dump_mask.astype(bool)
            )
            # Update cumulative dug mask
            self._cumulative_dug_mask |= dug_mask_new

            # Append to global plan list with continuous step indices
            base_step = len(self.plan_entries)
            for i, entry in enumerate(plan_entries):
                entry.step = base_step + i
                self.plan_entries.append(entry)

            # Visualize cone masks on grid for this waypoint if scene is available (permanent)
            try:
                visuals = {"dig_items": [], "dump_items": []}
                if self.scene is not None and hasattr(self.scene, "add_manual_workspace_cone"):
                    dug_mask = plan_entries[0].dug_mask.astype(bool)
                    dump_mask = plan_entries[1].dump_mask.astype(bool)
                    # Waypoint index (1-based) for labels
                    wp_index = (base_step // 2) + 1
                    visuals["dig_items"] = self.scene.add_manual_workspace_cone("dig", dug_mask, label=str(wp_index))
                    visuals["dump_items"] = self.scene.add_manual_workspace_cone("dump", dump_mask, label=str(wp_index))
                self._waypoint_visuals.append(visuals)
            except Exception:
                pass

            # Reset dig/dump picks and masks for next waypoint (agent base is kept)
            self._dig_cone_mask = None
            self._dump_cone_mask = None
            self._dig_pick_pos = None
            self._dump_pick_pos = None

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

            # Renumber steps to keep them contiguous
            for idx, entry in enumerate(self.plan_entries):
                entry.step = idx

            # Rebuild cumulative dug mask from remaining waypoints
            if self.plan_entries:
                cum = None
                for entry in self.plan_entries:
                    dug = entry.dug_mask.astype(bool)
                    if cum is None:
                        cum = np.zeros_like(dug, dtype=bool)
                    cum |= dug
                self._cumulative_dug_mask = cum
            else:
                self._cumulative_dug_mask = None

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
            preview_lines.append(
                f"Step {step}: dug={dug_count} cells, dump={dump_count} cells, loaded {loaded.get('before')}→{loaded.get('after')}"
            )
        self.waypoints_list.setPlainText("\n".join(preview_lines))

    def _on_generate(self) -> None:
        if not self.plan_entries:
            QMessageBox.warning(self, "No Waypoints", "Add at least one waypoint before generating a plan.")
            return

        try:
            serializable_waypoints = [entry.to_serializable() for entry in self.plan_entries]
            out_obj = {"waypoints": serializable_waypoints}

            json_path = self.json_path_edit.text().strip()
            if json_path:
                os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
                # Dump with indentation then post-process to compact arrays and format keys
                json_str = json.dumps(out_obj, indent=2)

                def compact_arrays(text: str) -> str:
                    """Compact multi-line arrays to single-line, matching extractor style."""
                    result = []
                    i = 0
                    while i < len(text):
                        if text[i] == '[':
                            depth = 1
                            j = i + 1
                            while j < len(text) and depth > 0:
                                if text[j] == '[':
                                    depth += 1
                                elif text[j] == ']':
                                    depth -= 1
                                j += 1
                            array_content = text[i + 1 : j - 1]
                            if '\n' in array_content:
                                compacted = re.sub(r'\s+', '', array_content)
                                result.append('[' + compacted + ']')
                            else:
                                result.append(text[i:j])
                            i = j
                        else:
                            result.append(text[i])
                            i += 1
                    return ''.join(result)

                def fix_dict_keys(text: str) -> str:
                    """Ensure dictionary keys are on separate lines, similar to foundation-plan.json."""
                    text = re.sub(r'\},\s*"', r'},\n      "', text)
                    text = re.sub(r',\s*"([^"]+)":\s*', r',\n      "\1": ', text)
                    return text

                json_str = compact_arrays(json_str)
                json_str = fix_dict_keys(json_str)

                with open(json_path, "w", encoding="utf-8") as f:
                    f.write(json_str)

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
    p.add_argument(
        "--dump-row", type=float, required=True, help="Dump center row index (0-based)."
    )
    p.add_argument(
        "--dump-col", type=float, required=True, help="Dump center col index (0-based)."
    )
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
    p.add_argument("--agent-yaw", type=float, default=0.0, help="Base yaw angle (radians or deg).")
    p.add_argument(
        "--agent-cabin-angle", type=float, default=0.0, help="Cabin angle (same units as yaw)."
    )
    p.add_argument(
        "--agent-wheel-angle",
        type=float,
        default=0.0,
        help="Wheel angle (same units as yaw).",
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
    )

    serializable_waypoints = [entry.to_serializable() for entry in plan_entries]
    out_obj = {"waypoints": serializable_waypoints}

    if args.serialize_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.serialize_json)), exist_ok=True)
        with open(args.serialize_json, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2)
        print(f"Wrote manual plan JSON to {args.serialize_json}")
    else:
        print(json.dumps(out_obj, indent=2))

    if args.serialize_pkl:
        os.makedirs(os.path.dirname(os.path.abspath(args.serialize_pkl)), exist_ok=True)
        with open(args.serialize_pkl, "wb") as f:
            # Downstream expects a list of per-waypoint dicts (same as extract_map.py)
            pickle.dump(serializable_waypoints, f)
        print(f"Wrote manual plan PKL to {args.serialize_pkl}")


if __name__ == "__main__":
    main()


