## 🗺️ TerraMapMaker GUI

Simple 2D/3D PyQt5 GUI to load excavation-map `GridMap` artifacts, paint on a grid, and plan waypoints.

### Setup

Should be able to work within the moleworks_ros docker. To run it as a standalone tool see [Setup to use standalone](#setup-to-use-standalone).

### Setup to use standalone

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

### Coordinate convention (RViz-like top-down)

The 2D canvas is displayed with a flipped view transform to mimic the common RViz top-down “map” look:

- The **origin (0,0) is at the bottom-right** of the canvas.
- **+X points up** on the screen.
- **+Y points left** on the screen.

This is a visualization convention only (a view transform); the underlying Qt scene still uses the default pixel coordinates.

# TerraMapMaker

## Exported Map Alignment Metadata

When TerraMapMaker exports Terra map artifacts onto a source GridMap, it writes `terra_metadata.yaml`.

That file stores the rigid transform needed to align the local Terra canvas back into the source ROS `map` frame:

- `meters_per_tile`
- `terra_origin_map_m`
- `rotation_deg`

Interpretation:

- `terra_origin_map_m` is the position of Terra `(0,0)` expressed in the source GridMap `map` frame, in meters.
- `rotation_deg` is the Terra yaw relative to that `map` frame.

Schema-v2 packaging consumes this metadata and embeds the same rigid transform directly into the final plan JSON.
`terra_planner` runtime now executes directly in `map`.

## Current Artifact Ownership

Today TerraMapMaker is the only part of the toolchain that knows both:

- where that canvas was placed on the source ROS `map`
- how a manual plan drawn in the GUI should be aligned back into that `map`

That is why the current schema-v2 path is:

- TerraMapMaker exports `terra_metadata.yaml`
- `utils/manual_plan_generator.py` packages manual plans directly into schema-v2 using that alignment
- runtime accepts only schema-v2

For policy-generated plans, the same rule applies: the policy may output plan-local waypoint contents, but the final
schema-v2 runtime artifact is packaged by the offline source-tree tool
`high_level_planning/terra_planner/scripts/package_policy_plan.py` once the plan-to-map alignment from
`terra_metadata.yaml` is available.

No legacy runtime conversion path is kept.

## Canonical Input Map Artifacts

TerraMapMaker should be pointed at saved excavation-map artifacts, not a standalone elevation-map export.

- Survey/current-surface input: `mole_maps/maps/<map_name>/<map_name>_surface`
- Existing runtime-design input: `mole_maps/maps/<map_name>/<map_name>_design`
- Expected core layer: `elevation`
- Optional authored layers that are preserved on load: `desired_elevation`, `dig_zone`, `dump_zone`, other excavation layers
- `ros2 launch mole_mapping save_map.launch.py artifact_stage:=surface` saves the survey/current-surface artifact

Typical flow:

1. survey and save `<map_name>_surface`
2. load that artifact in TerraMapMaker
3. optionally correct the surveyed `elevation` with `Brush` mode
4. author target geometry
5. export either `<map_name>_design` or a corrected surface artifact

When `desired_elevation` already exists in the loaded artifact, TerraMapMaker now uses it as the editable target
surface instead of overwriting it from `elevation`.

## Survey Surface Editing

Use `Brush` mode to edit the original survey surface before design authoring.

- `Sculpt`: raises or lowers the surveyed `elevation` with a signed brush strength
- `Flatten`: blends the touched area toward the local mean height to smooth or level a patch
- `Radius (m)`: controls brush footprint size
- `Reset Surface`: restores the surface that was loaded at the start of the session

Brush edits are applied to the native survey-resolution `elevation` array and then resampled back into the 2D canvas.

## 3D Preview

The 3D tab renders manually to avoid expensive reloads during painting, surface brushing, and foundation selection.

1. Choose `Surface elevation` or `Desired elevation` from `3D layer`.
2. Click `Render 3D`.
3. After edits, click `Render 3D` again to refresh.

`Surface elevation` is the corrected survey surface. `Desired elevation` is the authored design surface.

## Export Modes

The export bar now separates artifact type from bag format:

- `Terra`: exports the Terra folder structure and alignment metadata
- `Design GridMap`: exports the authored design artifact, including `desired_elevation` plus excavation layers such as `dig_zone`, `dump_zone`, `obstacles`, and `grid_region_mask`
- `Surface GridMap`: exports a survey/current-surface artifact with only the `elevation` layer and `basic_layers: [elevation]`

For `Design GridMap`, `elevation` is the corrected survey surface after any Brush edits, while `desired_elevation`
is the authored target surface used by the excavation plan.

`Surface GridMap` is intended for corrected surveys and synthetic surfaces such as maps sculpted from `flat`.
For bag-backed sources it preserves the source GridMap geometry and frame metadata. For flat or `.npy` sources, TerraMapMaker synthesizes a minimal ROS2/MCAP GridMap for both design and surface exports; ROS1 GridMap export is only available when the session started from a bag-backed source.

## Importing An Exported Session

Use `Import Exported Map` to reopen a previously exported design for small edits. The dialog asks for:

- the exported `Design GridMap` bag, MCAP, ROS2 bag folder, or `.npy`;
- the exported Terra folder, either the export root or its `map/` subfolder;
- an optional schema-v2 plan JSON.

The importer loads the design GridMap, restores Terra placement from `map/metadata/terra_metadata.yaml`, imports the
Terra editable layers, and reloads the manual plan previews when a plan JSON is provided.

## Manual Plan Packaging Contract

`utils/manual_plan_generator.py` now writes schema-v2 plans with explicit normalized conventions:

- `agent_state.pos_base == [row, col]`
- `angle_base_rad`, `angle_cabin_rad`, `wheel_angle_rad`
- `workspace_geometry` is optional descriptive metadata for the authored workspace shape
- embedded `alignment` and `source_map_frame_id`

The runtime consumes the exported masks. `workspace_geometry` is preserved for inspection and future tooling:

- `{"type": "fan", "heading_rad": ..., "min_radius_m": ..., "max_radius_m": ..., "aperture_rad": ...}`
- `{"type": "circle", "center_row": ..., "center_col": ..., "radius_tiles": ..., "radius_m": ...}`

If TerraMapMaker cannot provide alignment, packaging fails fast.
