## TerraMapMaker GUI (prototype)

Simple 2D PyQt5 GUI to load an elevation map, downsample to 64x64, and paint on a grid.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

### Features (current)
- Load elevation map from `.npy` or image (PNG/JPG)
- Downsample to 64x64 for fast interaction
- Display heatmap-like background
- Paint cells on top (toggle paint/erase)
- Multiple paint layers (obstacle/dump)
- Foundation placement and dig depth
- Export to Terra or heightmap



# TerraMapMaker
