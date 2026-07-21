import numpy as np

try:
    from scipy import ndimage as _ndi

    HAS_SCIPY = True
except Exception:
    _ndi = None
    HAS_SCIPY = False

try:
    from matplotlib import cm as _cm

    HAS_MPL = True
except Exception:
    _cm = None
    HAS_MPL = False


def compute_distance_map_taxicab(target_map: np.ndarray) -> np.ndarray:
    """
    Compute normalized Manhattan distance to nearest dump zone (target_map > 0).

    Mirrors the standalone relocation-distance script:
    - SciPy cityblock distance transform when available
    - Two-pass DP fallback otherwise
    - Normalizes with a realistic max distance tuned for 64x64 Terra maps
    """
    dump = target_map > 0
    if HAS_SCIPY:
        dist = _ndi.distance_transform_cdt(~dump, metric="taxicab").astype(np.float32)
    else:
        h, w = target_map.shape
        big = np.int32(10**6)
        dist = np.where(dump, 0, big).astype(np.int32)
        # forward pass
        for i in range(h):
            for j in range(w):
                if dist[i, j] == 0:
                    continue
                left = dist[i - 1, j] + 1 if i > 0 else big
                up = dist[i, j - 1] + 1 if j > 0 else big
                dist[i, j] = min(dist[i, j], left, up)
        # backward pass
        for i in range(h - 1, -1, -1):
            for j in range(w - 1, -1, -1):
                current = dist[i, j]
                right = dist[i + 1, j] + 1 if i < h - 1 else big
                down = dist[i, j + 1] + 1 if j < w - 1 else big
                dist[i, j] = min(current, right, down)
        dist = dist.astype(np.float32)

    # BETTER NORMALIZATION: Use realistic max distance for optimal resolution (Same as in Terra)
    h, w = target_map.shape
    # For 64x64 map with center dump zones, max realistic distance is ~24 tiles
    # This gives perfect resolution: 2 tiles = 0.083, 4 tiles = 0.167, 8 tiles = 0.333
    realistic_max_distance = 24  # Optimal for 64x64 maps with center dump zones
    norm = float(realistic_max_distance) if realistic_max_distance > 0 else 1.0
    return dist / norm


def distance_map_to_rgb_viridis(distance_map: np.ndarray) -> np.ndarray:
    """
    Convert a distance map to an RGB image using the same 'viridis' colormap
    as the analysis script. Matplotlib's imshow automatically normalizes data
    to [0,1] for the colormap, so we do the same here.

    If matplotlib is not available, falls back to a simple green→blue ramp.
    """
    dist = distance_map.astype(np.float32)
    h, w = dist.shape

    if HAS_MPL:
        # Matplotlib's viridis colormap, identical to the analysis script
        # Normalize to [0,1] for colormap (like imshow does automatically)
        dist_min = dist.min()
        dist_max = dist.max()
        if dist_max > dist_min:
            dist_normalized = (dist - dist_min) / (dist_max - dist_min)
        else:
            dist_normalized = np.zeros_like(dist)

        cmap = _cm.get_cmap("viridis")
        rgba = cmap(dist_normalized)  # shape (H, W, 4), floats in [0,1]
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        return rgb

    # Fallback: simple green-blue gradient (close=green, far=blue)
    # Normalize for fallback too
    dist_min = dist.min()
    dist_max = dist.max()
    if dist_max > dist_min:
        dist_normalized = (dist - dist_min) / (dist_max - dist_min)
    else:
        dist_normalized = np.zeros_like(dist)

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 1] = ((1.0 - dist_normalized) * 255.0).astype(np.uint8)  # green channel
    rgb[:, :, 2] = (dist_normalized * 255.0).astype(np.uint8)  # blue channel
    return rgb
