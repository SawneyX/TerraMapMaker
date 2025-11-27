import sys
import os
from typing import Optional, Tuple, List

import json
import numpy as np
from PIL import Image
try:
    import yaml
except ImportError:
    yaml = None
try:
    from rosbags.highlevel import AnyReader
    HAS_ROSBAGS = True
except ImportError:
    HAS_ROSBAGS = False
try:
    from stl import mesh
    HAS_STL = True
except ImportError:
    HAS_STL = False
from PyQt5.QtCore import Qt, QRectF, QPointF, QSize
from PyQt5.QtGui import QBrush, QColor, QImage, QPixmap, QPen, QIcon, QPainter, QFont, QPolygonF, QTransform
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsRectItem,
    QGraphicsPixmapItem,
    QGraphicsSimpleTextItem,
    QGraphicsPolygonItem,
    QGraphicsLineItem,
    QMessageBox,
    QToolBar,
    QAction,
    QActionGroup,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QLineEdit,
    QWidgetAction,
    QSlider,
    QTabWidget,
    QCheckBox,
    QSplitter,
    QSizePolicy,
)

# NumPy compatibility shim for pyqtgraph on NumPy>=2.0
try:
    if not hasattr(np, "product"):
        np.product = np.prod  # type: ignore[attr-defined]
except Exception:
    pass

# Try to import pyqtgraph for 3D view
try:
    import pyqtgraph as pg  # noqa: F401
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import MeshData
    HAS_GL = True
except Exception:
    gl = None
    MeshData = None
    HAS_GL = False


GRID_SIZE = 64
CELL_SIZE = 12  # pixels for on-screen cell size (scales with view)
DEFAULT_METERS_PER_TILE = 0.6875

# Colors per type
COLOR_DUMP = QColor(0, 200, 0, 130)        # green
COLOR_FOUNDATION = QColor(150, 40, 220, 130)  # purple
COLOR_OBSTACLE = QColor(0, 0, 0, 160)      # black
COLOR_NODUMP = QColor(120, 120, 120, 140)  # grey

# Solid colors (opaque) for borders/highlights
SOLID_DUMP = QColor(0, 200, 0)
SOLID_FOUNDATION = QColor(150, 40, 220)
SOLID_OBSTACLE = QColor(0, 0, 0)
SOLID_NODUMP = QColor(120, 120, 120)
ACCENT_BLUE = QColor(0, 120, 255)


def numpy_to_qimage_grayscale(arr: np.ndarray) -> QImage:
    clipped = np.clip(arr, 0.0, 1.0)
    img_8bit = (clipped * 255).astype(np.uint8)
    h, w = img_8bit.shape
    # QImage constructor: (data, width, height, bytesPerLine, format)
    # numpy array: arr[row, col] where row=0 is top, col=0 is left
    # Ensure array is contiguous and correct byte order
    if not img_8bit.flags['C_CONTIGUOUS']:
        img_8bit = np.ascontiguousarray(img_8bit)
    # QImage expects row-major data: arr[0,0] = top-left pixel
    qimg = QImage(img_8bit.data, w, h, w, QImage.Format_Grayscale8)
    return qimg.copy()


class GridScene(QGraphicsScene):
    def __init__(self, grid_size: int = GRID_SIZE, parent=None) -> None:
        super().__init__(parent)
        self.grid_size = grid_size
        self.meters_per_tile: float = DEFAULT_METERS_PER_TILE
        self.cell_items: list[list[QGraphicsRectItem]] = []
        self.dump_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.foundation_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.foundation_depth_map = None  # Optional: per-cell depth map (meters below surface)
        self.foundation_original_elevation = None  # Store original elevation when foundation is drawn, for restoration
        self.foundation_groups: list[dict] = []  # List of imported foundation groups with cells and outline
        self.selected_foundation_group: Optional[dict] = None  # Currently selected foundation group
        self.obstacle_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.nodump_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.background_item: Optional[QGraphicsPixmapItem] = None
        self.transpose_background: bool = True  # swap axes for georef-aligned maps by default
        # Overlay image (satellite) support
        self.overlay_item: Optional[QGraphicsPixmapItem] = None
        self.overlay_scale: float = 1.0
        self.overlay_rotation_deg: float = 0.0
        # Callback on background drag commit
        self.on_background_moved = None  # type: Optional[callable]
        self._bg_drag_start_pos: Optional[QPointF] = None
        self._select_drag_start_pos: Optional[QPointF] = None
        self._select_drag_target: Optional[str] = None  # 'overlay' or 'background'
        self.tool_mode: str = "rect"  # "select", "cell", "rect", "polygon", or "ruler"
        self.current_type: str = "dump"  # dump | foundation | obstacle | nodump | eraser
        self._drag_start_cell: Optional[Tuple[int, int]] = None
        self._is_painting: bool = False  # Track if we're currently painting (mouse down)
        self._rubber_item: Optional[QGraphicsRectItem] = None
        self._label_w: Optional[QGraphicsSimpleTextItem] = None
        self._label_h: Optional[QGraphicsSimpleTextItem] = None
        # Polygon mode state
        self._polygon_points: list[QPointF] = []
        self._polygon_item: Optional[QGraphicsPolygonItem] = None
        self._polygon_lines: list[QGraphicsLineItem] = []
        self._polygon_label: Optional[QGraphicsSimpleTextItem] = None
        # Ruler tool state
        self._ruler_start: Optional[QPointF] = None
        self._ruler_line: Optional[QGraphicsLineItem] = None
        self._ruler_label: Optional[QGraphicsSimpleTextItem] = None
        # Foundation group dragging state
        self._dragged_group: Optional[dict] = None
        self._drag_group_start_pos: Optional[Tuple[int, int]] = None
        self.setSceneRect(0, 0, grid_size * CELL_SIZE, grid_size * CELL_SIZE)
        self._init_grid()
        self.on_rectangle_committed = None  # type: Optional[callable]
        self.on_mask_changed = None  # type: Optional[callable]
        self.get_current_depth = None  # type: Optional[callable]  # Callback to get current depth value
        self.get_elevation = None  # type: Optional[callable]  # Callback to get elevation array
        self.on_group_selected = None  # type: Optional[callable]  # Callback when group is selected

    def _init_grid(self) -> None:
        pen = QPen(QColor(200, 200, 200, 255))
        pen.setWidth(0)
        self.cell_items = []
        for y in range(self.grid_size):
            row_items = []
            for x in range(self.grid_size):
                rect = QRectF(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                item = QGraphicsRectItem(rect)
                item.setPen(pen)
                item.setBrush(QBrush(Qt.NoBrush))
                item.setZValue(1)
                item.setFlag(QGraphicsRectItem.ItemIsSelectable, False)
                item.setFlag(QGraphicsRectItem.ItemIsMovable, False)
                # Ensure grid cells do not intercept mouse events; allows overlay below to receive drags
                item.setAcceptedMouseButtons(Qt.NoButton)
                self.addItem(item)
                row_items.append(item)
            self.cell_items.append(row_items)

    def set_background_from_array(self, arr: np.ndarray) -> None:
        # Ensure array is exactly grid_size x grid_size
        if arr.shape[0] != self.grid_size or arr.shape[1] != self.grid_size:
            # Crop or pad to fit
            if arr.shape[0] > self.grid_size or arr.shape[1] > self.grid_size:
                arr = arr[:self.grid_size, :self.grid_size]
            else:
                # Pad if smaller
                pad_h = self.grid_size - arr.shape[0]
                pad_w = self.grid_size - arr.shape[1]
                arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=arr.min())
        # Optionally transpose to fix diagonal mirroring when GNSS alignment is active
        arr_original = arr.copy()  # Keep original for debug
        should_transpose = getattr(self, "transpose_background", True)
        arr = arr.T if should_transpose else arr.copy()
        qimg = numpy_to_qimage_grayscale(arr)
        # Debug: verify first pixel matches first array element
        if qimg.width() > 0 and qimg.height() > 0:
            first_pixel_val = qimg.pixel(0, 0)  # Returns grayscale value
            first_arr_val = arr_original[0, 0]
            print(f"set_background_from_array: original arr[0,0]={first_arr_val}, transposed arr[0,0]={arr[0,0]}, QImage pixel(0,0)={first_pixel_val}")
        # Create pixmap at exact size (one pixel per grid cell, then scale)
        pix = QPixmap.fromImage(qimg)
        # Scale to physical size: grid_size * CELL_SIZE pixels
        pix = pix.scaled(
            self.grid_size * CELL_SIZE,
            self.grid_size * CELL_SIZE,
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation,
        )
        if self.background_item is None:
            self.background_item = QGraphicsPixmapItem(pix)
            self.background_item.setZValue(0)
            self.background_item.setFlag(QGraphicsPixmapItem.ItemIsMovable, False)
            self.background_item.setFlag(QGraphicsPixmapItem.ItemIsSelectable, False)
            # QGraphicsPixmapItem's pos() is the top-left corner of the pixmap
            # Set to scene origin to align with grid cell (0,0)
            self.background_item.setPos(0.0, 0.0)
            # Ensure no offset: pixmap origin should be at (0,0)
            self.background_item.setOffset(0.0, 0.0)
            self.addItem(self.background_item)
        else:
            self.background_item.setPixmap(pix)
            # Force position and offset reset
            self.background_item.setPos(0.0, 0.0)
            self.background_item.setOffset(0.0, 0.0)
        # Verify actual position
        actual_pos = self.background_item.pos()
        br = self.background_item.boundingRect()
        print(f"Background: array {arr.shape} -> pixmap {pix.width()}x{pix.height()}")
        print(f"  Item pos: ({actual_pos.x()}, {actual_pos.y()}), boundingRect: ({br.x()}, {br.y()}, {br.width()}, {br.height()})")
        print(f"  Grid cell 0,0 at scene pos: ({0 * CELL_SIZE}, {0 * CELL_SIZE})")
        print(f"  Grid cell 0,1 at scene pos: ({1 * CELL_SIZE}, {0 * CELL_SIZE})")

    def set_background_visible(self, visible: bool) -> None:
        if self.background_item is not None:
            self.background_item.setVisible(visible)

    def load_overlay_image(self, img_path: str) -> None:
        try:
            img = Image.open(img_path).convert("RGBA")
            qimg = QImage(img.tobytes(), img.width, img.height, img.width * 4, QImage.Format_RGBA8888)
            pix = QPixmap.fromImage(qimg)
        except Exception:
            return
        if self.overlay_item is None:
            self.overlay_item = QGraphicsPixmapItem(pix)
            # Place overlay above background but below grid lines
            self.overlay_item.setZValue(0.5)
            self.overlay_item.setOpacity(0.7)
            self.overlay_item.setFlag(QGraphicsPixmapItem.ItemIsMovable, True)
            self.overlay_item.setFlag(QGraphicsPixmapItem.ItemIsSelectable, True)
            # Set rotation/scale origin to center of the image
            self.overlay_item.setTransformOriginPoint(pix.rect().center())
            # Start at origin; user can drag to position
            self.overlay_item.setPos(0, 0)
            self.addItem(self.overlay_item)
        else:
            self.overlay_item.setPixmap(pix)
            # Update transform origin if image size changed
            self.overlay_item.setTransformOriginPoint(pix.rect().center())
        # Apply current transform
        self.apply_overlay_transform()

    def apply_overlay_transform(self) -> None:
        if self.overlay_item is None:
            return
        self.overlay_item.setScale(self.overlay_scale)
        self.overlay_item.setRotation(self.overlay_rotation_deg)

    def set_overlay_visible(self, visible: bool) -> None:
        if self.overlay_item is not None:
            self.overlay_item.setVisible(visible)

    def clear_paint(self) -> None:
        self.dump_mask[:, :] = 0
        self.foundation_mask[:, :] = 0
        self.obstacle_mask[:, :] = 0
        self.nodump_mask[:, :] = 0
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                self.cell_items[y][x].setBrush(QBrush(Qt.NoBrush))

    def _cell_from_pos(self, pos: QPointF) -> Optional[Tuple[int, int]]:
        x = int(pos.x() // CELL_SIZE)
        y = int(pos.y() // CELL_SIZE)
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return x, y
        return None

    def _set_cell_type(self, x: int, y: int, type_key: Optional[str]) -> None:
        # exclusive
        self.dump_mask[y, x] = 0
        self.foundation_mask[y, x] = 0
        self.obstacle_mask[y, x] = 0
        self.nodump_mask[y, x] = 0
        
        # Get current depth if setting foundation cells
        current_depth = None
        if type_key == "foundation" and callable(self.get_current_depth):
            try:
                current_depth = float(self.get_current_depth())
            except:
                current_depth = None
        
        # Initialize depth map and original elevation backup if needed
        if type_key == "foundation" and current_depth is not None:
            if self.foundation_depth_map is None:
                self.foundation_depth_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            # Store original elevation when foundation is first drawn
            if self.foundation_original_elevation is None and callable(self.get_elevation):
                try:
                    elev = self.get_elevation()
                    if elev is not None and elev.shape == (self.grid_size, self.grid_size):
                        self.foundation_original_elevation = elev.copy()
                except:
                    pass
        
        if type_key == "dump":
            self.dump_mask[y, x] = 1
        elif type_key == "foundation":
            self.foundation_mask[y, x] = 1
            # Store the current depth value for this cell
            if current_depth is not None:
                self.foundation_depth_map[y, x] = current_depth
            elif self.foundation_depth_map is not None:
                self.foundation_depth_map[y, x] = 0.0
            # Store original elevation for this cell if not already stored
            if self.foundation_original_elevation is None and callable(self.get_elevation):
                try:
                    elev = self.get_elevation()
                    if elev is not None and elev.shape == (self.grid_size, self.grid_size):
                        self.foundation_original_elevation = elev.copy()
                except:
                    pass
        elif type_key == "obstacle":
            self.obstacle_mask[y, x] = 1
        elif type_key == "nodump":
            self.nodump_mask[y, x] = 1
        elif type_key is None:  # Eraser
            # Clear depth map when erasing
            if self.foundation_depth_map is not None:
                self.foundation_depth_map[y, x] = 0.0
            # Restore original elevation when foundation is removed
            if self.foundation_original_elevation is not None and callable(self.get_elevation):
                try:
                    elev = self.get_elevation()
                    if elev is not None and elev.shape == self.foundation_original_elevation.shape:
                        # Restore original elevation for this cell
                        elev[y, x] = self.foundation_original_elevation[y, x]
                except:
                    pass
            
            # Update imported foundation groups when erasing
            self._update_imported_groups_after_erase([(x, y)])
        
        self._update_cell_brush(x, y)

    def _find_connected_foundation_groups(self) -> List[List[Tuple[int, int]]]:
        """Find all connected groups of foundation cells using flood-fill."""
        if self.foundation_mask is None:
            return []
        
        groups = []
        visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        # Flood-fill to find connected components
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.foundation_mask[y, x] == 1 and not visited[y, x]:
                    # Found a new group, flood-fill it
                    group_cells = []
                    stack = [(x, y)]
                    
                    while stack:
                        cx, cy = stack.pop()
                        if visited[cy, cx]:
                            continue
                        
                        visited[cy, cx] = True
                        group_cells.append((cx, cy))
                        
                        # Check neighbors (4-connected)
                        neighbors = [
                            (cx-1, cy), (cx+1, cy),
                            (cx, cy-1), (cx, cy+1)
                        ]
                        for nx, ny in neighbors:
                            if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and
                                self.foundation_mask[ny, nx] == 1 and not visited[ny, nx]):
                                stack.append((nx, ny))
                    
                    if group_cells:
                        groups.append(group_cells)
        
        return groups
    
    def _create_group_outline(self, cells: List[Tuple[int, int]]) -> QPolygonF:
        """Create a polygon outline around a group of cells."""
        if not cells:
            return QPolygonF()
        
        # Find boundary cells (cells that are on the edge)
        cell_set = set(cells)
        boundary_cells = []
        
        # Check each cell's neighbors
        for x, y in cells:
            # Check if any neighbor is not in the group
            neighbors = [
                (x-1, y), (x+1, y),
                (x, y-1), (x, y+1)
            ]
            is_boundary = False
            for nx, ny in neighbors:
                if (nx, ny) not in cell_set:
                    is_boundary = True
                    break
            if is_boundary:
                boundary_cells.append((x, y))
        
        if not boundary_cells:
            # All cells are together, use bounding box
            xs = [x for x, y in cells]
            ys = [y for x, y in cells]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Create rectangle outline
            poly = QPolygonF([
                QPointF(min_x * CELL_SIZE, min_y * CELL_SIZE),
                QPointF((max_x + 1) * CELL_SIZE, min_y * CELL_SIZE),
                QPointF((max_x + 1) * CELL_SIZE, (max_y + 1) * CELL_SIZE),
                QPointF(min_x * CELL_SIZE, (max_y + 1) * CELL_SIZE),
            ])
            return poly
        
        # Create outline by finding the convex hull or boundary walk
        # Simple approach: use bounding box with padding
        xs = [x for x, y in boundary_cells]
        ys = [y for x, y in boundary_cells]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Create rectangle outline around all cells
        poly = QPolygonF([
            QPointF(min_x * CELL_SIZE, min_y * CELL_SIZE),
            QPointF((max_x + 1) * CELL_SIZE, min_y * CELL_SIZE),
            QPointF((max_x + 1) * CELL_SIZE, (max_y + 1) * CELL_SIZE),
            QPointF(min_x * CELL_SIZE, (max_y + 1) * CELL_SIZE),
        ])
        return poly
    
    def _update_foundation_groups(self) -> None:
        """Update foundation groups by finding connected components and creating groups."""
        # Remember which group was selected (if any)
        selected_group_cells = None
        if self.selected_foundation_group is not None:
            selected_group_cells = set(self.selected_foundation_group.get('cells', []))
        
        # Clear existing drawn foundation groups (keep imported ones)
        groups_to_remove = []
        for i, group in enumerate(self.foundation_groups):
            # Only remove drawn groups (not imported ones)
            if not group.get('is_imported', False):
                # Remove outline items
                if group.get('outline_item'):
                    try:
                        self.removeItem(group['outline_item'])
                    except:
                        pass
                groups_to_remove.append(i)
        
        # Remove groups (reverse order to maintain indices)
        for i in reversed(groups_to_remove):
            self.foundation_groups.pop(i)
        
        # Clear selection temporarily
        self.selected_foundation_group = None
        
        # Find all connected groups of foundation cells
        connected_groups = self._find_connected_foundation_groups()
        
        # Create group objects for each connected component
        new_selected_group = None
        for group_cells in connected_groups:
            if not group_cells:
                continue
            
            # Create outline polygon
            outline_polygon = self._create_group_outline(group_cells)
            
            # Store depth map for this group
            group_depth_map = None
            if self.foundation_depth_map is not None:
                group_depth_map = {}
                for x, y in group_cells:
                    depth_value = self.foundation_depth_map[y, x]
                    if abs(depth_value) > 1e-6:
                        group_depth_map[(x, y)] = depth_value
            
            # Create group object
            group = {
                'cells': group_cells,
                'outline': outline_polygon,
                'outline_item': None,  # Will be created
                'depth_map': group_depth_map,
                'id': len(self.foundation_groups),  # Unique ID
                'is_imported': False  # Mark as drawn (not imported)
            }
            
            # Add outline to scene
            group['outline_item'] = self._draw_group_outline(group)
            
            # Add to groups list
            self.foundation_groups.append(group)
            
            # Check if this is the previously selected group (by matching cells)
            if selected_group_cells is not None:
                group_cell_set = set(group_cells)
                if group_cell_set == selected_group_cells:
                    new_selected_group = group
        
        # Restore selection if group still exists
        if new_selected_group is not None:
            self._select_foundation_group(new_selected_group)
        else:
            self._select_foundation_group(None)
    
    def _update_imported_groups_after_erase(self, erased_cells: List[Tuple[int, int]]) -> None:
        """Update imported foundation groups after erasing cells. Remove empty groups."""
        if not erased_cells:
            return
        
        erased_set = set(erased_cells)
        groups_to_remove = []
        was_selected_group_removed = False
        
        # Check each imported group
        for i, group in enumerate(self.foundation_groups):
            if not group.get('is_imported', False):
                continue  # Skip drawn groups (handled by _update_foundation_groups)
            
            group_cells = group.get('cells', [])
            if not group_cells:
                continue
            
            # Find cells that were erased from this group
            cells_to_remove = [cell for cell in group_cells if cell in erased_set]
            if not cells_to_remove:
                continue
            
            # Remove erased cells from group
            new_group_cells = [cell for cell in group_cells if cell not in erased_set]
            group['cells'] = new_group_cells
            
            # Remove from depth map
            if group.get('depth_map') is not None:
                for cell in cells_to_remove:
                    group['depth_map'].pop(cell, None)
            
            # Check if group is now empty
            if not new_group_cells:
                # Mark for removal
                groups_to_remove.append(i)
                # Check if this was the selected group
                if group == self.selected_foundation_group:
                    was_selected_group_removed = True
            else:
                # Update outline for remaining cells
                new_outline = self._create_group_outline(new_group_cells)
                group['outline'] = new_outline
                
                # Update outline item
                old_outline_item = group.get('outline_item')
                if old_outline_item:
                    try:
                        self.removeItem(old_outline_item)
                    except:
                        pass
                
                # Create new outline
                group['outline_item'] = self._draw_group_outline(group)
        
        # Remove empty groups (reverse order to maintain indices)
        for i in reversed(groups_to_remove):
            group = self.foundation_groups[i]
            # Remove outline item
            if group.get('outline_item'):
                try:
                    self.removeItem(group['outline_item'])
                except:
                    pass
            self.foundation_groups.pop(i)
        
        # Deselect if selected group was removed, or update selection if group still exists but was modified
        if was_selected_group_removed:
            self._select_foundation_group(None)
        elif self.selected_foundation_group is not None:
            # Selected group still exists but may have been modified - refresh selection
            # This ensures the UI (e.g., depth textbox) updates if the group changed
            selected_group = self.selected_foundation_group
            self._select_foundation_group(selected_group)  # Refresh selection to update UI
    
    def _draw_group_outline(self, group: dict) -> QGraphicsPolygonItem:
        """Draw an outline polygon around a foundation group."""
        outline = group['outline']
        if outline.isEmpty():
            return None
        
        # Create polygon item
        outline_item = QGraphicsPolygonItem(outline)
        outline_item.setPen(QPen(QColor(0, 200, 255), 3))  # Cyan outline, 3px wide
        outline_item.setBrush(QBrush(Qt.NoBrush))  # No fill
        outline_item.setZValue(5)  # Above cells but below other UI
        outline_item.setFlag(QGraphicsPolygonItem.ItemIsSelectable, True)
        # Only show outline if this group is selected
        outline_item.setVisible(group is self.selected_foundation_group)
        self.addItem(outline_item)
        return outline_item
    
    def _select_foundation_group(self, group: Optional[dict]) -> None:
        """Select a foundation group and update outline visibility."""
        # Deselect previous group
        if self.selected_foundation_group is not None:
            old_outline = self.selected_foundation_group.get('outline_item')
            if old_outline:
                old_outline.setVisible(False)
        
        # Select new group
        self.selected_foundation_group = group
        
        # Show outline of selected group
        if group is not None:
            outline = group.get('outline_item')
            if outline:
                outline.setVisible(True)
        
        # Notify callback to update UI (e.g., show depth textbox)
        if callable(self.on_group_selected):
            self.on_group_selected(group)
    
    def _move_foundation_group(self, group: dict, dx_cells: int, dy_cells: int) -> None:
        """Move a foundation group by offset in cells."""
        if not group or not group['cells']:
            return
        
        # Calculate new positions
        new_cells = []
        old_cells = group['cells']
        
        # Calculate new positions first (before clearing)
        for x, y in old_cells:
            new_x = x + dx_cells
            new_y = y + dy_cells
            # Clamp to grid bounds
            new_x = max(0, min(self.grid_size - 1, new_x))
            new_y = max(0, min(self.grid_size - 1, new_y))
            new_cells.append((new_x, new_y))
        
        # Find cells that are being moved away from (not in new positions)
        old_cell_set = set(old_cells)
        new_cell_set = set(new_cells)
        cells_to_clear = old_cell_set - new_cell_set
        
        # Clear old cells that are not in new positions
        # Restore terrain elevation for cleared cells
        for x, y in cells_to_clear:
            if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                self.foundation_mask[y, x] = 0
                if self.foundation_depth_map is not None:
                    self.foundation_depth_map[y, x] = 0
                # Restore original elevation when foundation is removed
                if self.foundation_original_elevation is not None and callable(self.get_elevation):
                    try:
                        elev = self.get_elevation()
                        if elev is not None and elev.shape == self.foundation_original_elevation.shape:
                            # Restore original elevation for this cell
                            elev[y, x] = self.foundation_original_elevation[y, x]
                    except:
                        pass
                self._update_cell_brush(x, y)
        
        # Update group
        group['cells'] = new_cells
        
        # Update masks at new positions
        for x, y in new_cells:
            if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                self.foundation_mask[y, x] = 1
                # Copy depth map if available
                if group.get('depth_map') is not None:
                    # Find corresponding old cell (by index)
                    idx = new_cells.index((x, y))
                    if idx < len(old_cells):
                        old_cell = old_cells[idx]
                        if old_cell in group['depth_map']:
                            if self.foundation_depth_map is None:
                                self.foundation_depth_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
                            self.foundation_depth_map[y, x] = group['depth_map'][old_cell]
                self._update_cell_brush(x, y)
        
        # Update outline
        outline = self._create_group_outline(new_cells)
        group['outline'] = outline
        if group['outline_item']:
            group['outline_item'].setPolygon(outline)
        
        # Update depth map if it exists
        if group.get('depth_map') is not None:
            # Update depth map with new positions
            new_depth_map = {}
            for i, (x, y) in enumerate(new_cells):
                if i < len(old_cells):
                    old_cell = old_cells[i]
                    if old_cell in group['depth_map']:
                        new_depth_map[(x, y)] = group['depth_map'][old_cell]
            group['depth_map'] = new_depth_map
    
    def _batch_set_cell_type(self, cells: List[Tuple[int, int]], type_key: Optional[str]) -> None:
        """Batch update multiple cells at once for better performance"""
        if not cells:
            return
        
        # Get current depth if setting foundation cells and callback is available
        current_depth = None
        if type_key == "foundation" and callable(self.get_current_depth):
            try:
                current_depth = float(self.get_current_depth())
            except:
                current_depth = None
        
        # Initialize depth map and original elevation backup if needed
        if type_key == "foundation" and current_depth is not None:
            if self.foundation_depth_map is None:
                self.foundation_depth_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            # Store original elevation when foundation is first drawn
            if self.foundation_original_elevation is None and callable(self.get_elevation):
                try:
                    elev = self.get_elevation()
                    if elev is not None and elev.shape == (self.grid_size, self.grid_size):
                        self.foundation_original_elevation = elev.copy()
                except:
                    pass
        
        # Update all masks at once
        for x, y in cells:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.dump_mask[y, x] = 0
                self.foundation_mask[y, x] = 0
                self.obstacle_mask[y, x] = 0
                self.nodump_mask[y, x] = 0
                if type_key == "dump":
                    self.dump_mask[y, x] = 1
                elif type_key == "foundation":
                    self.foundation_mask[y, x] = 1
                    # Store the current depth value for this cell
                    if current_depth is not None:
                        self.foundation_depth_map[y, x] = current_depth
                    elif self.foundation_depth_map is not None:
                        # Clear depth if erasing (type_key is None) or if depth not available
                        self.foundation_depth_map[y, x] = 0.0
                    # Store original elevation for this cell if not already stored
                    if self.foundation_original_elevation is None and callable(self.get_elevation):
                        try:
                            elev = self.get_elevation()
                            if elev is not None and elev.shape == (self.grid_size, self.grid_size):
                                self.foundation_original_elevation = elev.copy()
                        except:
                            pass
                elif type_key == "obstacle":
                    self.obstacle_mask[y, x] = 1
                elif type_key == "nodump":
                    self.nodump_mask[y, x] = 1
                elif type_key is None:  # Eraser
                    # Clear depth map when erasing
                    if self.foundation_depth_map is not None:
                        self.foundation_depth_map[y, x] = 0.0
                    # Restore original elevation when foundation is removed
                    if self.foundation_original_elevation is not None and callable(self.get_elevation):
                        try:
                            elev = self.get_elevation()
                            if elev is not None and elev.shape == self.foundation_original_elevation.shape:
                                # Restore original elevation for this cell
                                elev[y, x] = self.foundation_original_elevation[y, x]
                        except:
                            pass
            
            # Update imported foundation groups when erasing (batch mode)
            if type_key is None:
                self._update_imported_groups_after_erase(cells)
        
        # Update all cell brushes at once
        for x, y in cells:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self._update_cell_brush(x, y)

    def _update_cell_brush(self, x: int, y: int) -> None:
        if self.dump_mask[y, x] == 1:
            self.cell_items[y][x].setBrush(QBrush(COLOR_DUMP))
        elif self.foundation_mask[y, x] == 1:
            self.cell_items[y][x].setBrush(QBrush(COLOR_FOUNDATION))
        elif self.obstacle_mask[y, x] == 1:
            self.cell_items[y][x].setBrush(QBrush(COLOR_OBSTACLE))
        elif self.nodump_mask[y, x] == 1:
            self.cell_items[y][x].setBrush(QBrush(COLOR_NODUMP))
        else:
            self.cell_items[y][x].setBrush(QBrush(Qt.NoBrush))

    def _apply_current(self, x: int, y: int) -> None:
        if self.current_type == "eraser":
            self._set_cell_type(x, y, None)
        else:
            self._set_cell_type(x, y, self.current_type)
        
        # Update foundation groups if foundation was drawn or erased
        if self.current_type == "foundation" or self.current_type == "eraser":
            self._update_foundation_groups()
        
        # Only update 3D view if not currently painting (defer until mouse release)
        if not self._is_painting and callable(self.on_mask_changed):
            self.on_mask_changed()

    def _start_rect_drag(self, start_cell: Tuple[int, int]) -> None:
        self._drag_start_cell = start_cell
        if self._rubber_item is None:
            self._rubber_item = QGraphicsRectItem()
            self._rubber_item.setZValue(2)
            self._rubber_item.setPen(QPen(ACCENT_BLUE))
            self._rubber_item.setBrush(QBrush(QColor(0, 120, 255, 60)))
            self.addItem(self._rubber_item)
        if self._label_w is None:
            self._label_w = QGraphicsSimpleTextItem("")
            self._label_w.setZValue(3)
            f = QFont()
            f.setPointSize(9)
            self._label_w.setFont(f)
            self._label_w.setBrush(QBrush(QColor(20, 20, 20)))
            self.addItem(self._label_w)
        if self._label_h is None:
            self._label_h = QGraphicsSimpleTextItem("")
            self._label_h.setZValue(3)
            f2 = QFont()
            f2.setPointSize(9)
            self._label_h.setFont(f2)
            self._label_h.setBrush(QBrush(QColor(20, 20, 20)))
            self.addItem(self._label_h)

    def _update_rect_drag(self, current_cell: Tuple[int, int]) -> None:
        if self._drag_start_cell is None or self._rubber_item is None:
            return
        x0, y0 = self._drag_start_cell
        x1, y1 = current_cell
        left_cell = min(x0, x1)
        top_cell = min(y0, y1)
        right_cell = max(x0, x1)
        bottom_cell = max(y0, y1)
        left = left_cell * CELL_SIZE
        top = top_cell * CELL_SIZE
        right = (right_cell + 1) * CELL_SIZE
        bottom = (bottom_cell + 1) * CELL_SIZE
        self._rubber_item.setRect(QRectF(left, top, right - left, bottom - top))
        # Compute dimensions in tiles and meters
        tiles_w = (right_cell - left_cell + 1)
        tiles_h = (bottom_cell - top_cell + 1)
        meters_w = tiles_w * self.meters_per_tile
        meters_h = tiles_h * self.meters_per_tile
        # Update width label at top edge center
        if self._label_w is not None:
            self._label_w.setText(f"{tiles_w} tiles | {meters_w:.2f} m")
            text_rect = self._label_w.boundingRect()
            cx = (left + right) / 2 - text_rect.width() / 2
            cy = top - text_rect.height() - 4
            if cy < 0:
                cy = top + 4
            self._label_w.setPos(cx, cy)
        # Update height label at left edge center (vertical)
        if self._label_h is not None:
            # Render as horizontal text but position at left center
            self._label_h.setText(f"{tiles_h} tiles | {meters_h:.2f} m")
            text_rect_h = self._label_h.boundingRect()
            cx_h = left - text_rect_h.width() - 6
            if cx_h < 0:
                cx_h = left + 6
            cy_h = (top + bottom) / 2 - text_rect_h.height() / 2
            self._label_h.setPos(cx_h, cy_h)

    def _finish_rect_drag(self, end_cell: Tuple[int, int]) -> None:
        if self._drag_start_cell is None:
            return
        x0, y0 = self._drag_start_cell
        x1, y1 = end_cell
        xmin, xmax = sorted((x0, x1))
        ymin, ymax = sorted((y0, y1))
        
        # Batch update all cells at once
        if self.current_type == "eraser":
            cells = [(xx, yy) for yy in range(ymin, ymax + 1) for xx in range(xmin, xmax + 1)]
            self._batch_set_cell_type(cells, None)
        else:
            cells = [(xx, yy) for yy in range(ymin, ymax + 1) for xx in range(xmin, xmax + 1)]
            self._batch_set_cell_type(cells, self.current_type)
        
        # Notify rectangle committed if in foundation mode
        if callable(self.on_rectangle_committed) and self.current_type == "foundation":
            self.on_rectangle_committed((xmin, ymin, xmax, ymax))
        
        # Update foundation groups if foundation was drawn or erased
        if self.current_type == "foundation" or self.current_type == "eraser":
            self._update_foundation_groups()
        
        if callable(self.on_mask_changed):
            self.on_mask_changed()
        if self._rubber_item is not None:
            self.removeItem(self._rubber_item)
            self._rubber_item = None
        if self._label_w is not None:
            self.removeItem(self._label_w)
            self._label_w = None
        if self._label_h is not None:
            self.removeItem(self._label_h)
            self._label_h = None
        self._drag_start_cell = None

    def _point_in_polygon(self, pt: QPointF, polygon: list[QPointF]) -> bool:
        """Ray casting algorithm for point-in-polygon test"""
        if len(polygon) < 3:
            return False
        x, y = pt.x(), pt.y()
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0].x(), polygon[0].y()
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n].x(), polygon[i % n].y()
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _start_polygon(self, first_point: QPointF) -> None:
        self._polygon_points = [first_point]
        self._update_polygon_preview()

    def _add_polygon_point(self, point: QPointF) -> None:
        # Check if clicking near first point (close polygon)
        if len(self._polygon_points) >= 3:
            first_pt = self._polygon_points[0]
            dist = ((point.x() - first_pt.x()) ** 2 + (point.y() - first_pt.y()) ** 2) ** 0.5
            if dist < CELL_SIZE * 1.5:  # Close enough to snap to first point
                self._finish_polygon()
                return
        self._polygon_points.append(point)
        self._update_polygon_preview()

    def _update_polygon_preview(self) -> None:
        # Remove old preview items
        if self._polygon_item is not None:
            self.removeItem(self._polygon_item)
            self._polygon_item = None
        for line in self._polygon_lines:
            self.removeItem(line)
        self._polygon_lines.clear()
        if self._polygon_label is not None:
            self.removeItem(self._polygon_label)
            self._polygon_label = None

        if len(self._polygon_points) < 2:
            return

        # Draw polygon fill preview
        if len(self._polygon_points) >= 3:
            poly = QPolygonF(self._polygon_points)
            self._polygon_item = QGraphicsPolygonItem(poly)
            self._polygon_item.setZValue(2)
            self._polygon_item.setPen(QPen(ACCENT_BLUE, 2))
            preview_color = QColor(0, 120, 255, 60)
            if self.current_type == "eraser":
                preview_color = QColor(255, 0, 0, 60)
            elif self.current_type == "foundation":
                preview_color = QColor(150, 40, 220, 60)
            elif self.current_type == "dump":
                preview_color = QColor(0, 200, 0, 60)
            elif self.current_type == "obstacle":
                preview_color = QColor(0, 0, 0, 60)
            elif self.current_type == "nodump":
                preview_color = QColor(120, 120, 120, 60)
            self._polygon_item.setBrush(QBrush(preview_color))
            self.addItem(self._polygon_item)

        # Draw lines connecting points
        for i in range(len(self._polygon_points) - 1):
            line = QGraphicsLineItem(
                self._polygon_points[i].x(), self._polygon_points[i].y(),
                self._polygon_points[i + 1].x(), self._polygon_points[i + 1].y()
            )
            line.setPen(QPen(ACCENT_BLUE, 2))
            line.setZValue(3)
            self._polygon_lines.append(line)
            self.addItem(line)

        # Calculate and display perimeter and area
        if len(self._polygon_points) >= 2:
            # Calculate perimeter: sum of distances between consecutive points
            perimeter_px = 0.0
            for i in range(len(self._polygon_points)):
                p1 = self._polygon_points[i]
                p2 = self._polygon_points[(i + 1) % len(self._polygon_points)]
                dx = p2.x() - p1.x()
                dy = p2.y() - p1.y()
                perimeter_px += (dx * dx + dy * dy) ** 0.5
            # Convert to tiles and meters
            perimeter_tiles = perimeter_px / CELL_SIZE
            perimeter_meters = perimeter_tiles * self.meters_per_tile
            
            # Calculate area using shoelace formula (if at least 3 points)
            area_sq_meters = 0.0
            if len(self._polygon_points) >= 3:
                # Shoelace formula for polygon area
                area_px_sq = 0.0
                n = len(self._polygon_points)
                for i in range(n):
                    j = (i + 1) % n
                    area_px_sq += self._polygon_points[i].x() * self._polygon_points[j].y()
                    area_px_sq -= self._polygon_points[j].x() * self._polygon_points[i].y()
                area_px_sq = abs(area_px_sq) / 2.0
                # Convert from pixels² to tiles² to meters²
                area_tiles_sq = area_px_sq / (CELL_SIZE * CELL_SIZE)
                area_sq_meters = area_tiles_sq * (self.meters_per_tile * self.meters_per_tile)
            
            # Create/update label
            if self._polygon_label is None:
                self._polygon_label = QGraphicsSimpleTextItem("")
                self._polygon_label.setZValue(3)
                f = QFont()
                f.setPointSize(9)
                self._polygon_label.setFont(f)
                self._polygon_label.setBrush(QBrush(QColor(20, 20, 20)))
                self.addItem(self._polygon_label)
            
            if len(self._polygon_points) >= 3:
                self._polygon_label.setText(f"Perimeter: {perimeter_tiles:.1f} tiles | {perimeter_meters:.2f} m | Area: {area_sq_meters:.2f} m²")
            else:
                self._polygon_label.setText(f"Perimeter: {perimeter_tiles:.1f} tiles | {perimeter_meters:.2f} m")
            
            # Position label near the top-left of polygon bounding box
            if self._polygon_points:
                xs = [pt.x() for pt in self._polygon_points]
                ys = [pt.y() for pt in self._polygon_points]
                min_x, min_y = min(xs), min(ys)
                text_rect = self._polygon_label.boundingRect()
                label_x = min_x
                label_y = min_y - text_rect.height() - 4
                if label_y < 0:
                    label_y = min_y + 4
                self._polygon_label.setPos(label_x, label_y)

    def _finish_polygon(self) -> None:
        if len(self._polygon_points) < 3:
            self._clear_polygon()
            return

        # Close polygon by adding first point at end if needed
        poly_points = self._polygon_points.copy()
        if poly_points[-1] != poly_points[0]:
            poly_points.append(poly_points[0])

        # Fill all cells inside polygon - batch update
        cells_to_update = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                # Test center of cell
                cell_center = QPointF(x * CELL_SIZE + CELL_SIZE / 2, y * CELL_SIZE + CELL_SIZE / 2)
                if self._point_in_polygon(cell_center, poly_points):
                    cells_to_update.append((x, y))
        
        # Batch update all cells at once
        if self.current_type == "eraser":
            self._batch_set_cell_type(cells_to_update, None)
        else:
            self._batch_set_cell_type(cells_to_update, self.current_type)
        
        cells_filled = len(cells_to_update)

        # Notify if foundation polygon
        if callable(self.on_rectangle_committed) and self.current_type == "foundation" and cells_filled > 0:
            # Find bounding box for notification
            xs = [pt.x() // CELL_SIZE for pt in self._polygon_points]
            ys = [pt.y() // CELL_SIZE for pt in self._polygon_points]
            xmin, xmax = int(min(xs)), int(max(xs))
            ymin, ymax = int(min(ys)), int(max(ys))
            self.on_rectangle_committed((xmin, ymin, xmax, ymax))

        # Update foundation groups if foundation was drawn or erased
        if self.current_type == "foundation" or self.current_type == "eraser":
            self._update_foundation_groups()

        if callable(self.on_mask_changed):
            self.on_mask_changed()

        self._clear_polygon()

    def _clear_polygon(self) -> None:
        if self._polygon_item is not None:
            self.removeItem(self._polygon_item)
            self._polygon_item = None
        for line in self._polygon_lines:
            self.removeItem(line)
        self._polygon_lines.clear()
        if self._polygon_label is not None:
            self.removeItem(self._polygon_label)
            self._polygon_label = None
        self._polygon_points = []

    def _clear_ruler(self) -> None:
        if self._ruler_line is not None:
            self.removeItem(self._ruler_line)
            self._ruler_line = None
        if self._ruler_label is not None:
            self.removeItem(self._ruler_label)
            self._ruler_label = None
        self._ruler_start = None

    def mousePressEvent(self, event) -> None:
        if self.tool_mode == "select":
            # Check if clicking on a foundation group outline or cells
            pos = event.scenePos()
            cell = self._cell_from_pos(pos)
            
            # First check if clicking on any group's cells
            clicked_group = None
            if cell:
                x, y = cell
                for group in self.foundation_groups:
                    if (x, y) in group.get('cells', []):
                        clicked_group = group
                        break
            
            # Also check if clicking on outline (even if not visible, check bounds)
            if not clicked_group:
                for group in self.foundation_groups:
                    if group.get('outline_item'):
                        outline_item = group['outline_item']
                        if outline_item.contains(pos) or outline_item.boundingRect().contains(pos):
                            clicked_group = group
                            break
            
            if clicked_group:
                # Select the group (or start dragging if already selected)
                if clicked_group == self.selected_foundation_group:
                    # Same group - start dragging
                    self._dragged_group = clicked_group
                    if cell:
                        self._drag_group_start_pos = cell
                else:
                    # Different group - select it
                    self._select_foundation_group(clicked_group)
                event.accept()
                return
            else:
                # Clicked outside groups - deselect
                self._select_foundation_group(None)
            
            # Track drag start; prefer overlay if click within its bounds
            self._select_drag_start_pos = pos
            if self.overlay_item is not None and self.overlay_item.isVisible():
                local = self.overlay_item.mapFromScene(pos)
                if self.overlay_item.pixmap().rect().contains(local.toPoint()):
                    self._select_drag_target = 'overlay'
                    super().mousePressEvent(event)
                    return
            # Background dragging disabled - only use offset textboxes
            # Nothing to drag
            self._select_drag_target = None
            return
        if event.buttons() & Qt.LeftButton:
            pos = event.scenePos()
            cell = self._cell_from_pos(pos)
            if cell is not None:
                if self.tool_mode == "cell":
                    # Set painting flag to defer 3D updates
                    self._is_painting = True
                    x, y = cell
                    self._apply_current(x, y)
                elif self.tool_mode == "rect":
                    self._start_rect_drag(cell)
                elif self.tool_mode == "polygon":
                    if len(self._polygon_points) == 0:
                        self._start_polygon(pos)
                    else:
                        self._add_polygon_point(pos)
                elif self.tool_mode == "ruler":
                    # Start ruler at exact scene pos (not snapped to cell center)
                    self._clear_ruler()
                    self._ruler_start = pos
                    self._ruler_line = QGraphicsLineItem()
                    self._ruler_line.setZValue(4)
                    self._ruler_line.setPen(QPen(ACCENT_BLUE, 2))
                    self.addItem(self._ruler_line)
                    self._ruler_label = QGraphicsSimpleTextItem("")
                    self._ruler_label.setZValue(4)
                    f = QFont()
                    f.setPointSize(9)
                    self._ruler_label.setFont(f)
                    self._ruler_label.setBrush(QBrush(QColor(20, 20, 20)))
                    self.addItem(self._ruler_label)
        elif event.buttons() & Qt.RightButton:
            if self.tool_mode == "polygon" and len(self._polygon_points) >= 3:
                self._finish_polygon()
        event.accept()
        return

    def mouseMoveEvent(self, event) -> None:
        # Handle foundation group dragging
        if self._dragged_group is not None and self._drag_group_start_pos is not None:
            pos = event.scenePos()
            current_cell = self._cell_from_pos(pos)
            if current_cell:
                dx = current_cell[0] - self._drag_group_start_pos[0]
                dy = current_cell[1] - self._drag_group_start_pos[1]
                if dx != 0 or dy != 0:
                    # Move the group
                    self._move_foundation_group(self._dragged_group, dx, dy)
                    self._drag_group_start_pos = current_cell
                    # Don't trigger 3D update during drag - too expensive
                    # Only update 2D view (cell brushes)
            event.accept()
            return
        
        if self.tool_mode == "select":
            if self._select_drag_target == 'overlay':
                super().mouseMoveEvent(event)
                return
            # Background dragging disabled - only use offset textboxes
            return
        if self.tool_mode == "ruler" and (event.buttons() & Qt.LeftButton) and self._ruler_start is not None and self._ruler_line is not None:
            pos = event.scenePos()
            self._ruler_line.setLine(self._ruler_start.x(), self._ruler_start.y(), pos.x(), pos.y())
            dx = pos.x() - self._ruler_start.x()
            dy = pos.y() - self._ruler_start.y()
            dist_px = (dx*dx + dy*dy) ** 0.5
            tiles = dist_px / CELL_SIZE
            meters = tiles * self.meters_per_tile
            if self._ruler_label is not None:
                self._ruler_label.setText(f"{tiles:.1f} tiles | {meters:.2f} m")
                mid_x = (self._ruler_start.x() + pos.x()) / 2.0
                mid_y = (self._ruler_start.y() + pos.y()) / 2.0
                text_rect = self._ruler_label.boundingRect()
                self._ruler_label.setPos(mid_x - text_rect.width()/2.0, mid_y - text_rect.height() - 6)
            event.accept()
            return
        if event.buttons() & Qt.LeftButton:
            pos = event.scenePos()
            cell = self._cell_from_pos(pos)
            if cell is not None:
                if self.tool_mode == "cell":
                    x, y = cell
                    self._apply_current(x, y)
                elif self.tool_mode == "rect":
                    self._update_rect_drag(cell)
        event.accept()
        return

    def mouseReleaseEvent(self, event) -> None:
        # End foundation group dragging
        if self._dragged_group is not None:
            # Restore terrain elevation for any cells that were cleared during the move
            # This ensures old foundation areas have their terrain restored
            if callable(self.get_elevation) and self.foundation_original_elevation is not None:
                try:
                    elev = self.get_elevation()
                    if elev is not None and elev.shape == self.foundation_original_elevation.shape:
                        # Restore elevation for all non-foundation cells
                        for y in range(self.grid_size):
                            for x in range(self.grid_size):
                                if self.foundation_mask[y, x] == 0:
                                    elev[y, x] = self.foundation_original_elevation[y, x]
                except:
                    pass
            # Update 3D view only once after dragging is complete
            if self.on_mask_changed:
                self.on_mask_changed()
            self._dragged_group = None
            self._drag_group_start_pos = None
            event.accept()
            return
        
        # End painting - update 3D view once
        if self._is_painting:
            self._is_painting = False
            if callable(self.on_mask_changed):
                self.on_mask_changed()
        if self.tool_mode == "select":
            # Background dragging disabled - only use offset textboxes
            # Cleanup
            self._select_drag_start_pos = None
            self._select_drag_target = None
            super().mouseReleaseEvent(event)
            return
        if self.tool_mode == "ruler":
            # Finalize current measurement: keep line/label, stop updating until a new start
            self._ruler_start = None
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.tool_mode == "rect":
            pos = event.scenePos()
            cell = self._cell_from_pos(pos)
            if cell is not None:
                self._finish_rect_drag(cell)
        event.accept()
        return

    def keyPressEvent(self, event) -> None:
        if self.tool_mode == "polygon":
            if event.key() == Qt.Key_Escape:
                if len(self._polygon_points) > 0:
                    self._clear_polygon()
                    event.accept()
                    return
            elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if len(self._polygon_points) >= 3:
                    self._finish_polygon()
                    event.accept()
                    return
        if self.tool_mode == "ruler":
            if event.key() == Qt.Key_Escape:
                self._clear_ruler()
                event.accept()
                return
        super().keyPressEvent(event)


# ---------- Point cloud helpers ----------

def load_pointcloud(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        pts = np.load(path)
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(".npy must be Nx3 or Nx>=3 array")
        return pts[:, :3]
    if ext in (".csv", ".txt"):
        pts = np.loadtxt(path, delimiter=",")
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("CSV must have at least 3 columns: x,y,z")
        return pts[:, :3]
    raise ValueError("Unsupported point cloud format. Use .npy (Nx3) or .csv")


def rasterize_pointcloud(points_xyz: np.ndarray, meters_per_tile: float) -> tuple[np.ndarray, tuple[float, float]]:
    if points_xyz.size == 0:
        raise ValueError("Empty point cloud")
    xs = points_xyz[:, 0]
    ys = points_xyz[:, 1]
    zs = points_xyz[:, 2]
    minx, miny = float(xs.min()), float(ys.min())
    xi = np.floor((xs - minx) / meters_per_tile).astype(int)
    yi = np.floor((ys - miny) / meters_per_tile).astype(int)
    width = int(xi.max()) + 1
    height = int(yi.max()) + 1
    canvas = np.full((height, width), np.nan, dtype=np.float32)
    for x, y, z in zip(xi, yi, zs):
            canvas[y, x] = np.nanmean([canvas[y, x], z]) if not np.isnan(canvas[y, x]) else z
    return canvas, (minx, miny)


def apply_placement(canvas: np.ndarray, grid_size: int, mode: str, offset_x: int, offset_y: int, fill_value: float) -> np.ndarray:
    h, w = canvas.shape
    # Replace NaNs with fill_value before placement
    base = np.where(np.isnan(canvas), fill_value, canvas)
    # Determine base start by mode
    if mode == "topleft":
        # Align centers: floor(w/2 - grid/2) ensures symmetric crop/overflow, then apply offset
        start_x = int(np.floor(w / 2.0 - grid_size / 2.0) + offset_x)
        start_y = int(np.floor(h / 2.0 - grid_size / 2.0) + offset_y)
    elif mode == "center":
        # Top-left: start at (0,0) then apply offset, plus constant offset of half heightmap size
        start_x = int(offset_x + w / 2)
        start_y = int(offset_y + h / 2)
    else:
        start_x = int(offset_x)
        start_y = int(offset_y)
    # Build placed output with symmetric overflow handling (no edge replication)
    out = np.full((grid_size, grid_size), fill_value, dtype=np.float32)
    # For each output grid cell (i, j), extract from source
    # numpy arrays: base[row, col] where row=0 is top, col=0 is left
    # grid cell (i, j) = (row, col) in output array
    for i in range(grid_size):
        for j in range(grid_size):
            src_row = start_y + i  # Source row (y)
            src_col = start_x + j  # Source column (x)
            if 0 <= src_row < h and 0 <= src_col < w:
                out[i, j] = base[src_row, src_col]
    # Debug: verify extraction region
    if mode == "topleft" and offset_x == 0 and offset_y == 0:
        print(f"Placement debug: mode={mode}, start=({start_x},{start_y}), extracting src[{start_y}:{start_y+grid_size}, {start_x}:{start_x+grid_size}] from {h}x{w}")
        # Verify actual extracted values - check multiple points
        corner_val = out[0, 0] if out.size > 0 else None
        edge_val = out[0, grid_size-1] if out.size > 0 else None
        center_val = out[grid_size//2, grid_size//2] if out.size > grid_size*grid_size//2 else None
        src_corner_val = base[start_y, start_x] if start_y < h and start_x < w else None
        src_edge_val = base[start_y, start_x+grid_size-1] if start_y < h and start_x+grid_size-1 < w else None
        src_center_val = base[h//2, w//2] if h > 0 and w > 0 else None
        print(f"  Extracted: out[0,0]={corner_val}, out[0,{grid_size-1}]={edge_val}, out[{grid_size//2},{grid_size//2}]={center_val}")
        print(f"  Source: base[{start_y},{start_x}]={src_corner_val}, base[{start_y},{start_x+grid_size-1}]={src_edge_val}, base[{h//2},{w//2}]={src_center_val}")
        print(f"  Match check: corner={corner_val==src_corner_val}, edge={edge_val==src_edge_val}")
    return out


def _make_color_icon(color: QColor, size: int = 20) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    painter_img = QImage(size, size, QImage.Format_ARGB32)
    painter_img.fill(Qt.transparent)
    pm2 = QPixmap.fromImage(painter_img)
    from PyQt5.QtGui import QPainter
    p = QPainter(pm2)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QBrush(color))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(2, 2, size - 4, size - 4, 5, 5)
    p.end()
    return QIcon(pm2)


def _make_symbol_icon(symbol: str, fg: QColor = QColor(40, 40, 40), size: int = 20) -> QIcon:
    from PyQt5.QtGui import QPainter, QFont
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setPen(fg)
    font = QFont()
    font.setPointSize(int(size * 0.8))
    p.setFont(font)
    p.drawText(0, 0, size, size, Qt.AlignCenter, symbol)
    p.end()
    return QIcon(pm)


def _rgba(c: QColor, alpha: float) -> str:
    return f"rgba({c.red()},{c.green()},{c.blue()},{alpha})"

def _make_filled_rect_icon(size: int = 20, fg: QColor = QColor(40, 40, 40)) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    from PyQt5.QtGui import QPainter
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(fg)
    p.setBrush(QBrush(fg))
    margin = int(size * 0.18)
    radius = int(size * 0.12)
    p.drawRoundedRect(margin, margin, size - 2 * margin, size - 2 * margin, radius, radius)
    p.end()
    return QIcon(pm)

def _make_cross_icon(size: int = 20, color: QColor = QColor(200, 0, 0), thickness: int = 3) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    from PyQt5.QtGui import QPainter
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(color)
    pen.setWidth(thickness)
    p.setPen(pen)
    margin = int(size * 0.25)
    p.drawLine(margin, margin, size - margin, size - margin)
    p.drawLine(size - margin, margin, margin, size - margin)
    p.end()
    return QIcon(pm)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TerraMapMaker - Prototype")
        self.grid_size = GRID_SIZE
        self.meters_per_tile = DEFAULT_METERS_PER_TILE
        self.scene = GridScene(self.grid_size)
        self.scene.meters_per_tile = self.meters_per_tile
        self.scene.on_background_moved = self._on_background_dragged
        self._configure_gnss_usage(True)
        # Tile offsets for PCL placement (drag background to adjust)
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.last_elevation_array: Optional[np.ndarray] = None
        self.last_pcl_canvas: Optional[np.ndarray] = None
        self.last_placed_elevation: Optional[np.ndarray] = None  # meters grid after placement
        self.desired_elevation_canvas: Optional[np.ndarray] = None  # Resized desired_elevation for display
        self.original_desired_elevation_array: Optional[np.ndarray] = None  # Original desired_elevation before resizing
        self.previous_desired_elevation_canvas: Optional[np.ndarray] = None  # Previous desired_elevation from loaded bag (before overwriting)
        self.original_previous_desired_elevation_array: Optional[np.ndarray] = None  # Original previous desired_elevation before resizing
        self.rotation_deg: float = 0.0
        # Unrotated base arrays (used to reapply rotation)
        self._base_canvas: Optional[np.ndarray] = None
        self._base_desired_canvas: Optional[np.ndarray] = None
        self._base_previous_desired_canvas: Optional[np.ndarray] = None
        self._base_original_elevation_array: Optional[np.ndarray] = None
        self._base_original_desired_array: Optional[np.ndarray] = None
        self._base_original_previous_desired_array: Optional[np.ndarray] = None
        # Store original bag file info for exporting
        self.original_bag_path: Optional[str] = None
        self.original_gridmap_msg = None
        self.original_gridmap_resolution: Optional[float] = None
        self.original_gridmap_conn_info = None  # Store connection info (msgdef, rihs01, msgtype) for writing
        self.foundation_rect: Optional[Tuple[int, int, int, int]] = None  # xmin,ymin,xmax,ymax in cells
        # Marker items for GridMap center (info.pose.position)
        self._map_center_marker_items: list[QGraphicsItem] = []
        self._current_canvas_shape: Optional[Tuple[int, int]] = None
        # Attach callbacks
        self.scene.on_rectangle_committed = self.on_rectangle_committed
        self.scene.on_mask_changed = self._on_mask_changed
        self.scene.get_current_depth = lambda: self.depth_spin.value()
        self.scene.get_elevation = lambda: self.last_placed_elevation
        self.scene.on_group_selected = self.on_group_selected
        
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(self.view.renderHints())
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.view.setBackgroundBrush(QBrush(Qt.white))
        self.view.setMouseTracking(True)
        self.view.viewport().setMouseTracking(True)
        self.view.setFocusPolicy(Qt.StrongFocus)  # Allow keyboard focus for ESC key
        # Enable intuitive zooming behavior
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.canvas_zoom = 1.0

        # Toolbar and painting controls (styles omitted for brevity in this diff)
        self.toolbar = QToolBar("Tools")
        self.addToolBar(Qt.LeftToolBarArea, self.toolbar)
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QSize(26, 26))
        self.toolbar.setStyleSheet(
            """
            QToolBar { background: #ffffff; border-right: 1px solid #e5e5e5; spacing: 6px; padding: 6px; }
            QToolBar QToolButton { border-radius: 8px; padding: 6px; margin: 2px; background: #f7f7f9; border: 1px solid #e3e3e7; }
            QToolBar QToolButton:hover { background: #f0f0f3; }
            QToolBar QToolButton:checked { background: #e6f0ff; border: 1px solid #99c3ff; }
            QLabel.sectionLabel { color: #6b6b6b; font-weight: bold; letter-spacing: 0.5px; padding: 2px 4px; }
            """
        )
        mode_label = QLabel("Mode:")
        mode_label.setProperty("class", "sectionLabel")
        mode_label.setStyleSheet("QLabel { color:#6b6b6b; font-weight:600; }")
        mode_label_act = QWidgetAction(self)
        mode_label_act.setDefaultWidget(mode_label)
        self.action_tool_select = QAction(_make_symbol_icon("↖"), "Select", self)
        self.action_tool_cell = QAction(_make_symbol_icon("●"), "Cell", self)
        self.action_tool_rect = QAction(_make_filled_rect_icon(), "Rect", self)
        self.action_tool_polygon = QAction(_make_symbol_icon("⬟"), "Polygon", self)
        self.action_tool_ruler = QAction(_make_symbol_icon("↔"), "Ruler", self)
        self.action_tool_select.setCheckable(True)
        self.action_tool_cell.setCheckable(True)
        self.action_tool_rect.setCheckable(True)
        self.action_tool_polygon.setCheckable(True)
        self.action_tool_ruler.setCheckable(True)
        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_group.addAction(self.action_tool_select)
        self.tool_group.addAction(self.action_tool_cell)
        self.tool_group.addAction(self.action_tool_rect)
        self.tool_group.addAction(self.action_tool_polygon)
        self.tool_group.addAction(self.action_tool_ruler)
        # Default to Rect mode
        self.action_tool_select.setChecked(False)
        self.action_tool_cell.setChecked(False)
        self.action_tool_rect.setChecked(True)
        self.action_tool_polygon.setChecked(False)
        layer_label = QLabel("Layer:")
        layer_label.setProperty("class", "sectionLabel")
        layer_label.setStyleSheet("QLabel { color:#6b6b6b; font-weight:600; }")
        layer_label_act = QWidgetAction(self)
        layer_label_act.setDefaultWidget(layer_label)
        self.action_type_dump = QAction(_make_color_icon(COLOR_DUMP), "Dump", self)
        self.action_type_foundation = QAction(_make_color_icon(COLOR_FOUNDATION), "Foundation", self)
        self.action_type_obstacle = QAction(_make_color_icon(COLOR_OBSTACLE), "Obstacle", self)
        self.action_type_nodump = QAction(_make_color_icon(COLOR_NODUMP), "No-Dump", self)
        self.action_type_eraser = QAction(_make_cross_icon(size=18, color=QColor(200,0,0)), "Eraser", self)
        for a in (self.action_type_dump, self.action_type_foundation, self.action_type_obstacle, self.action_type_nodump, self.action_type_eraser):
            a.setCheckable(True)
        self.type_group = QActionGroup(self)
        self.type_group.setExclusive(True)
        self.type_group.addAction(self.action_type_dump)
        self.type_group.addAction(self.action_type_foundation)
        self.type_group.addAction(self.action_type_obstacle)
        self.type_group.addAction(self.action_type_nodump)
        self.type_group.addAction(self.action_type_eraser)
        self.action_type_dump.setChecked(True)
        self.toolbar.addAction(mode_label_act)
        self.toolbar.addActions([self.action_tool_select, self.action_tool_cell, self.action_tool_rect, self.action_tool_polygon, self.action_tool_ruler])
        self.toolbar.addSeparator()
        self.toolbar.addAction(layer_label_act)
        self.toolbar.addActions([
            self.action_type_dump,
            self.action_type_foundation,
            self.action_type_obstacle,
            self.action_type_nodump,
            self.action_type_eraser,
        ])
        self.toolbar.addSeparator()
        self.action_clear = QAction("Clear", self)
        self.toolbar.addAction(self.action_clear)
        self.action_tool_select.triggered.connect(self.on_tool_select)
        self.action_tool_cell.triggered.connect(self.on_tool_cell)
        self.action_tool_rect.triggered.connect(self.on_tool_rect)
        self.action_tool_polygon.triggered.connect(self.on_tool_polygon)
        self.action_tool_ruler.triggered.connect(self.on_tool_ruler)
        self.action_type_dump.triggered.connect(lambda: self.on_type_change("dump"))
        self.action_type_foundation.triggered.connect(lambda: self.on_type_change("foundation"))
        self.action_type_obstacle.triggered.connect(lambda: self.on_type_change("obstacle"))
        self.action_type_nodump.triggered.connect(lambda: self.on_type_change("nodump"))
        self.action_type_eraser.triggered.connect(lambda: self.on_type_change("eraser"))
        self.action_clear.triggered.connect(self.on_clear_paint)

        # Zoom control in left toolbar (centered like the buttons)
        spacer_top = QWidget()
        spacer_top.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        spacer_top_act = QWidgetAction(self)
        spacer_top_act.setDefaultWidget(spacer_top)
        self.toolbar.addAction(spacer_top_act)

        zoom_container = QWidget()
        zoom_layout = QVBoxLayout()
        zoom_layout.setContentsMargins(4, 4, 4, 4)
        zoom_layout.setSpacing(6)
        zoom_layout.setAlignment(Qt.AlignHCenter)
        zoom_label = QLabel("Zoom")
        zoom_label.setAlignment(Qt.AlignHCenter)
        zoom_label.setStyleSheet("QLabel { color:#6b6b6b; font-weight:600; }")
        self.canvas_zoom_slider = QSlider(Qt.Vertical)
        self.canvas_zoom_slider.setMinimum(10)
        self.canvas_zoom_slider.setMaximum(400)
        self.canvas_zoom_slider.setSingleStep(5)
        self.canvas_zoom_slider.setPageStep(10)
        self.canvas_zoom_slider.setTickInterval(10)
        self.canvas_zoom_slider.setTickPosition(QSlider.TicksRight)
        self.canvas_zoom_slider.setValue(100)
        self.canvas_zoom_slider.valueChanged.connect(self.on_canvas_zoom_changed)
        zoom_layout.addWidget(zoom_label)
        zoom_layout.addWidget(self.canvas_zoom_slider, 0, Qt.AlignHCenter)
        zoom_container.setLayout(zoom_layout)
        zoom_container_act = QWidgetAction(self)
        zoom_container_act.setDefaultWidget(zoom_container)
        self.toolbar.addAction(zoom_container_act)

        # No bottom spacer so the zoom block sits at the bottom of the toolbar

        # Bottom bar with PCL controls
        self.btn_load_geo = QPushButton("Load Geo Map")
        self.btn_load_foundation = QPushButton("Import Foundation")
        self.btn_export = QPushButton("Export")
        self.btn_export.setObjectName("exportBtn")

        # Resolution for loaded elevation maps (meters per cell), does not change global meters_per_tile
        self.map_res_spin = QDoubleSpinBox()
        self.map_res_spin.setRange(0.0001, 1000.0)
        self.map_res_spin.setDecimals(5)
        self.map_res_spin.setSingleStep(0.01)
        self.map_res_spin.setValue(0.1)

        # Offset spinboxes for manual positioning
        self.offset_x_spin = QSpinBox()
        self.offset_x_spin.setRange(-10000, 10000)
        self.offset_x_spin.setValue(0)
        self.offset_x_spin.valueChanged.connect(self.on_offset_manual_change)
        self.offset_y_spin = QSpinBox()
        self.offset_y_spin.setRange(-10000, 10000)
        self.offset_y_spin.setValue(0)
        self.offset_y_spin.valueChanged.connect(self.on_offset_manual_change)
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-180.0, 180.0)
        self.rotation_spin.setDecimals(1)
        self.rotation_spin.setSingleStep(1.0)
        self.rotation_spin.setValue(self.rotation_deg)
        self.rotation_spin.valueChanged.connect(self.on_rotation_changed)

        self.grid_size_combo = QComboBox()
        self.grid_size_combo.addItems(["32", "64", "128", "256"])
        self.grid_size_combo.setCurrentText(str(self.grid_size))
        self.grid_size_combo.currentTextChanged.connect(self.on_grid_size_change)

        self.meters_spin = QDoubleSpinBox()
        self.meters_spin.setRange(0.01, 1000.0)
        self.meters_spin.setDecimals(5)
        self.meters_spin.setSingleStep(0.01)
        self.meters_spin.setValue(self.meters_per_tile)
        self.meters_spin.valueChanged.connect(self.on_meters_per_tile_change)

        self.placement_combo = QComboBox()
        self.placement_combo.addItems(["Center", "Top-Left"])
        self.placement_combo.setCurrentText("Top-Left")

        # Drag-to-offset replaces sliders

        self.btn_load_geo.clicked.connect(self.on_load_geo_map)
        self.btn_load_foundation.clicked.connect(self.on_load_foundation)
        self.btn_export.clicked.connect(self.on_export)
        self.placement_combo.currentTextChanged.connect(self.on_placement_changed)

        # Connect rectangle callback
        self.scene.on_rectangle_committed = self.on_rectangle_committed
        self.scene.on_mask_changed = self._on_mask_changed
        self.scene.get_current_depth = lambda: self.depth_spin.value()
        self.scene.get_elevation = lambda: self.last_placed_elevation
        self.scene.on_group_selected = self.on_group_selected

        # Build main layout with right sidebar (resizable via splitter)

        # Sidebar: Foundation controls and profile/3D tabs
        self.sidebar = QWidget()
        self.sidebar.setMinimumWidth(360)
        # Allow horizontal resizing; remove hard max width
        side_layout = QVBoxLayout()
        side_layout.setContentsMargins(10, 10, 10, 10)
        side_layout.setSpacing(8)
        # Canvas zoom control moved to left toolbar

        # Foundation section title (moved below zoom)
        title = QLabel("Foundation")
        title.setStyleSheet("QLabel{font-weight:700;color:#333;font-size:14px;}")
        side_layout.addWidget(title)
        # Visibility toggles (placed right below Foundation title)
        self.chk_show_overlay = QCheckBox("Show overlay")
        self.chk_show_overlay.setChecked(True)
        self.chk_show_overlay.stateChanged.connect(self.on_toggle_overlay)
        side_layout.addWidget(self.chk_show_overlay)
        self.chk_show_background = QCheckBox("Show elevation map")
        self.chk_show_background.setChecked(True)
        self.chk_show_background.stateChanged.connect(self.on_toggle_background)
        side_layout.addWidget(self.chk_show_background)
        self.chk_show_desired_elevation = QCheckBox("Show desired elevation")
        self.chk_show_desired_elevation.setChecked(False)
        self.chk_show_desired_elevation.setEnabled(False)  # Disabled until a bag file with desired_elevation is loaded
        self.chk_show_desired_elevation.stateChanged.connect(self.on_toggle_desired_elevation)
        side_layout.addWidget(self.chk_show_desired_elevation)

        # Overlay controls (moved before depth/height settings)
        overlay_title = QLabel("Overlay")
        overlay_title.setStyleSheet("QLabel{font-weight:700;color:#333;font-size:14px;}")
        side_layout.addWidget(overlay_title)
        overlay_btns = QHBoxLayout()
        self.btn_load_overlay = QPushButton("Load Image…")
        self.btn_load_overlay.clicked.connect(self.on_load_overlay)
        overlay_btns.addWidget(self.btn_load_overlay)
        # Add orthophoto loader next to image loader
        self.btn_load_orthophoto = QPushButton("Load Orthophoto")
        self.btn_load_orthophoto.clicked.connect(self.on_load_orthophoto)
        overlay_btns.addWidget(self.btn_load_orthophoto)
        side_layout.addLayout(overlay_btns)
        # Overlay rotation/scale
        ov_row1 = QHBoxLayout()
        ov_row1.addWidget(QLabel("Rotation:"))
        self.overlay_rot_slider = QSlider(Qt.Horizontal)
        self.overlay_rot_slider.setMinimum(-180)
        self.overlay_rot_slider.setMaximum(180)
        self.overlay_rot_slider.setValue(0)
        self.overlay_rot_slider.valueChanged.connect(self.on_overlay_transform_changed)
        ov_row1.addWidget(self.overlay_rot_slider)
        side_layout.addLayout(ov_row1)
        ov_row2 = QHBoxLayout()
        ov_row2.addWidget(QLabel("Scale:"))
        self.overlay_scale_slider = QSlider(Qt.Horizontal)
        self.overlay_scale_slider.setMinimum(10)   # 0.10x
        self.overlay_scale_slider.setMaximum(500)  # 5.00x
        self.overlay_scale_slider.setValue(100)
        self.overlay_scale_slider.valueChanged.connect(self.on_overlay_transform_changed)
        ov_row2.addWidget(self.overlay_scale_slider)
        side_layout.addLayout(ov_row2)

        # Settings title
        settings_title2 = QLabel("Settings")
        settings_title2.setStyleSheet("QLabel{font-weight:700;color:#333;font-size:14px;}")
        side_layout.addWidget(settings_title2)

        depth_row = QHBoxLayout()
        depth_row.addWidget(QLabel("Depth (m):"))
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(0.0, 50.0)
        self.depth_spin.setDecimals(2)
        self.depth_spin.setSingleStep(0.05)
        self.depth_spin.setValue(1.00)
        self.depth_spin.valueChanged.connect(self.update_foundation_profile)
        self.depth_spin.valueChanged.connect(self.update_3d_view)
        depth_row.addWidget(self.depth_spin)
        side_layout.addLayout(depth_row)
        
        # Group depth control (only visible when group is selected)
        self.group_depth_container = QWidget()
        self.group_depth_container.setVisible(False)
        group_depth_layout = QHBoxLayout()
        group_depth_layout.setContentsMargins(0, 0, 0, 0)
        group_depth_layout.addWidget(QLabel("Group Depth (m):"))
        self.group_depth_spin = QDoubleSpinBox()
        self.group_depth_spin.setRange(0.0, 50.0)
        self.group_depth_spin.setDecimals(2)
        self.group_depth_spin.setSingleStep(0.05)
        self.group_depth_spin.setValue(1.00)
        self.group_depth_spin.valueChanged.connect(self.on_group_depth_changed)
        group_depth_layout.addWidget(self.group_depth_spin)
        self.group_depth_container.setLayout(group_depth_layout)
        side_layout.addWidget(self.group_depth_container)

        # Height scale (vertical exaggeration)
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Height scale:"))
        self.height_scale_spin = QDoubleSpinBox()
        self.height_scale_spin.setRange(0.1, 10.0)
        self.height_scale_spin.setDecimals(2)
        self.height_scale_spin.setSingleStep(0.1)
        self.height_scale_spin.setValue(1.0)
        self.height_scale_spin.valueChanged.connect(self.update_3d_view)
        scale_row.addWidget(self.height_scale_spin)
        side_layout.addLayout(scale_row)

        row = QHBoxLayout()
        self.lbl_min = QLabel("Min dig: -")
        self.lbl_max = QLabel("Max dig: -")
        row.addWidget(self.lbl_min)
        row.addWidget(self.lbl_max)
        self.btn_flatten_max = QPushButton("Flatten to max")
        self.btn_flatten_max.setToolTip("Set the selected foundation floor to the deepest dig depth.")
        self.btn_flatten_max.clicked.connect(self.flatten_dig_plane)
        row.addWidget(self.btn_flatten_max)
        side_layout.addLayout(row)


        self.tabs = QTabWidget()
        # Profile tab (show X and Y profiles stacked)
        profile_tab = QWidget()
        prof_layout = QVBoxLayout()
        # X-direction profile
        self.profile_view_x = QGraphicsView()
        self.profile_view_x.setMinimumHeight(160)
        self.profile_view_x.setRenderHints(self.profile_view_x.renderHints())
        self.profile_scene_x = QGraphicsScene()
        self.profile_view_x.setScene(self.profile_scene_x)
        prof_layout.addWidget(self.profile_view_x)
        # Y-direction profile
        self.profile_view_y = QGraphicsView()
        self.profile_view_y.setMinimumHeight(160)
        self.profile_view_y.setRenderHints(self.profile_view_y.renderHints())
        self.profile_scene_y = QGraphicsScene()
        self.profile_view_y.setScene(self.profile_scene_y)
        prof_layout.addWidget(self.profile_view_y)
        profile_tab.setLayout(prof_layout)
        self.tabs.addTab(profile_tab, "Profile")
        # 3D tab
        self.gl_view = None
        self.gl_surface = None
        self.gl_plane = None
        self.gl_walls = []  # Initialize walls list (walls are part of main mesh, but list kept for compatibility)
        self.gl_contours = []  # Initialize contours list
        self.chk_show_plane = QCheckBox("Show foundation plane")
        self.chk_show_plane.setChecked(True)
        self.chk_show_plane.stateChanged.connect(self.update_3d_view)
        if HAS_GL:
            gl_tab = QWidget()
            gl_layout = QVBoxLayout()
            self.gl_view = gl.GLViewWidget()
            self.gl_view.setMinimumHeight(260)
            self.gl_view.opts['distance'] = 25
            self.gl_view.opts['elevation'] = 25
            self.gl_view.opts['azimuth'] = 35
            # Single ground grid (created once)
            self.gl_grid = gl.GLGridItem()
            self.gl_grid.setSize(40, 40)
            self.gl_grid.setSpacing(1, 1)
            self.gl_grid.translate(0, 0, 0)
            self.gl_view.addItem(self.gl_grid)
            gl_layout.addWidget(self.gl_view)
            gl_tab.setLayout(gl_layout)
            self.tabs.addTab(gl_tab, "3D")
        else:
            gl_tab = QWidget()
            gl_layout = QVBoxLayout()
            gl_layout.addWidget(QLabel("pyqtgraph not installed. 3D view unavailable."))
            gl_tab.setLayout(gl_layout)
            self.tabs.addTab(gl_tab, "3D")
        # Set 3D view as default tab
        self.tabs.setCurrentIndex(1)  # 3D tab is at index 1
        # Refresh 3D when switching to the 3D tab
        def _on_tab_changed(idx: int) -> None:
            try:
                if self.tabs.tabText(idx) == "3D":
                    self.update_3d_view()
            except Exception:
                pass
        self.tabs.currentChanged.connect(_on_tab_changed)

        side_layout.addWidget(self.tabs, stretch=1)
        self.sidebar.setLayout(side_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.view)
        splitter.addWidget(self.sidebar)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        # Bottom bar remains the same; embed both
        center = QVBoxLayout()
        center.addWidget(splitter, stretch=1)
        # create bottom bar (existing code reused)
        bottom_bar = self._build_bottom_bar()
        try:
            bottom_bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            bottom_bar.setMaximumHeight(64)
        except Exception:
            pass
        center.addWidget(bottom_bar, stretch=0)

        container = QWidget()
        container.setLayout(center)
        self.setCentralWidget(container)
        self.resize(1200, 920)

        blank = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
        self.scene.set_background_from_array(blank)

        # After UI built, apply button styles
        self._apply_type_button_styles()
        self._apply_tool_button_styles()
        self._apply_canvas_zoom()

    # ----- Painting handlers -----
    def on_tool_cell(self) -> None:
        self.scene.tool_mode = "cell"
        self.scene._clear_polygon()
        self.scene._select_foundation_group(None)

    def on_tool_rect(self) -> None:
        self.scene.tool_mode = "rect"
        self.scene._clear_polygon()
        self.scene._select_foundation_group(None)

    def on_tool_polygon(self) -> None:
        self.scene.tool_mode = "polygon"
        self.scene._select_foundation_group(None)

    def on_tool_select(self) -> None:
        self.scene.tool_mode = "select"
        self.scene._clear_polygon()
        self.scene._clear_ruler()
        # Don't deselect on select tool - allow selection to remain

    def on_tool_ruler(self) -> None:
        self.scene.tool_mode = "ruler"
        self.scene._clear_polygon()
        self.scene._select_foundation_group(None)

    def on_type_change(self, type_key: str) -> None:
        self.scene.current_type = type_key
        # Update polygon preview color if in polygon mode
        if self.scene.tool_mode == "polygon" and len(self.scene._polygon_points) > 0:
            self.scene._update_polygon_preview()

    def on_clear_paint(self) -> None:
        self.scene.clear_paint()

    # ----- Overlay handlers -----
    def on_load_overlay(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open overlay image",
            os.getcwd(),
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
        try:
            self.scene.load_overlay_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Overlay Error", f"Failed to load overlay:\n{exc}")

    def on_toggle_overlay(self, state: int) -> None:
        self.scene.set_overlay_visible(state == Qt.Checked)

    def on_toggle_background(self, state: int) -> None:
        self.scene.set_background_visible(state == Qt.Checked)
    
    def on_toggle_desired_elevation(self, state: int) -> None:
        """Toggle between showing elevation, desired_elevation, and previous_desired_elevation."""
        if state == Qt.Checked:
            # Check if we have any desired elevation to show
            if (self.desired_elevation_canvas is None and 
                self.previous_desired_elevation_canvas is None):
                QMessageBox.information(
                    self,
                    "No Desired Elevation",
                    "No desired_elevation layer available.\n\n"
                    "This layer is created when you export a modified map."
                )
                self.chk_show_desired_elevation.setChecked(False)
                return
        # _apply_current_placement will check the toggle state and use the appropriate canvas
        self._apply_current_placement()

    def on_overlay_transform_changed(self, _: int) -> None:
        self.scene.overlay_rotation_deg = float(self.overlay_rot_slider.value())
        self.scene.overlay_scale = float(self.overlay_scale_slider.value()) / 100.0
        self.scene.apply_overlay_transform()

    def on_load_orthophoto(self) -> None:
        """Fetch Swiss orthophoto (swisstopo) for current geo map region and load as overlay."""
        try:
            if self.last_placed_elevation is None:
                QMessageBox.warning(self, "No Geo Map", "Load a geo-referenced map first.")
                return
            H, W = self.last_placed_elevation.shape
            if H < 2 or W < 2:
                QMessageBox.warning(self, "Invalid Map", "Elevation map is too small.")
                return

            if not hasattr(self, 'georef_config') or not isinstance(self.georef_config, dict):
                QMessageBox.warning(self, "No Georef Config", "GNSS reference not loaded.")
                return
            gnss = self.georef_config.get('gnss', {})
            if not gnss.get('useGnssReference', False):
                QMessageBox.warning(self, "Georef Disabled", "useGnssReference is false in config.")
                return
            ref_lat = float(gnss.get('referenceLatitude'))
            ref_lon = float(gnss.get('referenceLongitude'))
            ref_alt = float(gnss.get('referenceAltitude', 0.0))
            # Derive exact footprint from bag: center ENU and original resolution/size
            # Center ENU from GridMap info.pose.position
            # Swap X and Y for center and negate the new X to match transposed array: (cx, cy) = (-bag_y, bag_x)
            cx = 0.0
            cy = 0.0
            try:
                if hasattr(self, 'georef_gridmap_center') and isinstance(self.georef_gridmap_center, dict):
                    bag_x = float(self.georef_gridmap_center.get('x', 0.0))
                    bag_y = float(self.georef_gridmap_center.get('y', 0.0))
                    cx = -bag_y  # new x = -bag_y (transpose + negate)
                    cy = bag_x   # new y = bag_x (transpose)
            except Exception:
                cx = 0.0
                cy = 0.0

            # Original size and resolution
            if self.original_gridmap_resolution is None or not hasattr(self, 'original_gridmap_size'):
                raise ValueError("Original GridMap size/resolution not available from bag")
            oH, oW = getattr(self, 'original_gridmap_size', (None, None))
            if not (isinstance(oH, int) and isinstance(oW, int) and oH and oW):
                raise ValueError("Original GridMap size invalid")
            r = float(self.original_gridmap_resolution)
            # Use displayed grid dimensions for bbox (what's actually shown)
            # Convert grid size to meters using current meters_per_tile
            fetch_width_m = float(W) * self.meters_per_tile
            fetch_height_m = float(H) * self.meters_per_tile

            # ENU bbox around true center (using displayed grid size)
            half_w = 0.5 * fetch_width_m
            half_h = 0.5 * fetch_height_m
            x_min = cx - half_w
            x_max = cx + half_w
            y_min = cy - half_h
            y_max = cy + half_h

            # Build corners in ENU and convert to WGS84 bbox (exact elevation footprint)
            enu_tl = (cx - half_w, cy + half_h)
            enu_tr = (cx + half_w, cy + half_h)
            enu_bl = (cx - half_w, cy - half_h)
            enu_br = (cx + half_w, cy - half_h)
            try:
                from utils.swisstopo_fetcher import enu_to_wgs84_small_angle, fetch_swissimage_by_bbox_wgs84
            except Exception:
                from .utils.swisstopo_fetcher import enu_to_wgs84_small_angle, fetch_swissimage_by_bbox_wgs84  # type: ignore
            # Convert corners and center to WGS84
            center_lat, center_lon = enu_to_wgs84_small_angle(ref_lat, ref_lon, cx, cy)
            tl_lat, tl_lon = enu_to_wgs84_small_angle(ref_lat, ref_lon, enu_tl[0], enu_tl[1])
            tr_lat, tr_lon = enu_to_wgs84_small_angle(ref_lat, ref_lon, enu_tr[0], enu_tr[1])
            bl_lat, bl_lon = enu_to_wgs84_small_angle(ref_lat, ref_lon, enu_bl[0], enu_bl[1])
            br_lat, br_lon = enu_to_wgs84_small_angle(ref_lat, ref_lon, enu_br[0], enu_br[1])
            # Debug prints for corner lat/lon
            try:
                print("Orthophoto ENU center (cx, cy):", cx, cy)
                print(f"Orthophoto center WGS84 (lat, lon): {center_lat:.8f}, {center_lon:.8f}")
                print("Orthophoto corners WGS84 (lat, lon):")
                print(f"  TL: {tl_lat:.8f}, {tl_lon:.8f}")
                print(f"  TR: {tr_lat:.8f}, {tr_lon:.8f}")
                print(f"  BL: {bl_lat:.8f}, {bl_lon:.8f}")
                print(f"  BR: {br_lat:.8f}, {br_lon:.8f}")
            except Exception:
                pass
            min_lat = min(tl_lat, br_lat)
            max_lat = max(tl_lat, br_lat)
            min_lon = min(tl_lon, br_lon)
            max_lon = max(tl_lon, br_lon)
            # No expansion: request exactly the elevation footprint
            expand = 1.0
            lat_pad = (max_lat - min_lat) * (expand - 1.0) * 0.5
            lon_pad = (max_lon - min_lon) * (expand - 1.0) * 0.5
            bbox_wgs84 = (min_lat - lat_pad, min_lon - lon_pad, max_lat + lat_pad, max_lon + lon_pad)

            # Request image size based on displayed grid dimensions
            # This ensures the image matches what's actually shown
            quality_factor = 3
            req_w = max(1, int(W * quality_factor))
            req_h = max(1, int(H * quality_factor))
            img = fetch_swissimage_by_bbox_wgs84(bbox_wgs84, size_px=(req_w, req_h), format_str="image/jpeg")

            qimg = QImage(img.tobytes(), img.width, img.height, img.width * 4, QImage.Format_RGBA8888)
            pix = QPixmap.fromImage(qimg)

            # Scale to scene to match the displayed grid size (what's actually shown)
            # The bbox uses original dimensions, but display should match the grid
            target_w_px = int(W * CELL_SIZE)
            target_h_px = int(H * CELL_SIZE)
            scaled = pix.scaled(target_w_px, target_h_px, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

            if self.scene.overlay_item is None:
                self.scene.overlay_item = QGraphicsPixmapItem(scaled)
                self.scene.overlay_item.setZValue(0.5)
                self.scene.overlay_item.setOpacity(0.7)
                self.scene.overlay_item.setFlag(QGraphicsPixmapItem.ItemIsMovable, True)
                self.scene.overlay_item.setFlag(QGraphicsPixmapItem.ItemIsSelectable, True)
                self.scene.overlay_item.setTransformOriginPoint(scaled.rect().center())
                # Overlay aligns directly with grid (no expansion, so no offset needed)
                self.scene.overlay_item.setPos(0.0, 0.0)
                self.scene.addItem(self.scene.overlay_item)
            else:
                self.scene.overlay_item.setPixmap(scaled)
                self.scene.overlay_item.setTransformOriginPoint(scaled.rect().center())
                # Overlay aligns directly with grid (no expansion, so no offset needed)
                self.scene.overlay_item.setPos(0.0, 0.0)

            # Draw/refresh an outline of the exact elevation square on top of the overlay
            try:
                if hasattr(self.scene, 'overlay_outline_item') and self.scene.overlay_outline_item is not None:
                    try:
                        self.scene.removeItem(self.scene.overlay_outline_item)
                    except Exception:
                        pass
                    self.scene.overlay_outline_item = None
                from PyQt5.QtGui import QPen, QColor
                from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsEllipseItem
                outline = QGraphicsRectItem(0, 0, W * CELL_SIZE, H * CELL_SIZE)
                outline.setZValue(2.0)
                outline.setPen(QPen(QColor(255, 0, 0, 220), 2))
                outline.setBrush(Qt.NoBrush)
                self.scene.addItem(outline)
                self.scene.overlay_outline_item = outline
                # Mark reference point directly on the image (overlay local coords)
                # Clean previous markers
                if hasattr(self.scene, 'overlay_ref_item') and self.scene.overlay_ref_item is not None:
                    try:
                        self.scene.overlay_ref_item.setParentItem(None)
                        self.scene.removeItem(self.scene.overlay_ref_item)
                    except Exception:
                        pass
                    self.scene.overlay_ref_item = None
                if hasattr(self.scene, 'overlay_ref_hline') and self.scene.overlay_ref_hline is not None:
                    try:
                        self.scene.overlay_ref_hline.setParentItem(None)
                        self.scene.removeItem(self.scene.overlay_ref_hline)
                    except Exception:
                        pass
                    self.scene.overlay_ref_hline = None
                if hasattr(self.scene, 'overlay_ref_vline') and self.scene.overlay_ref_vline is not None:
                    try:
                        self.scene.overlay_ref_vline.setParentItem(None)
                        self.scene.removeItem(self.scene.overlay_ref_vline)
                    except Exception:
                        pass
                    self.scene.overlay_ref_vline = None
                # Reference at center of fetched image (cx,cy)
                ref_radius = 8.0
                ref_x_local = target_w_px * 0.5
                ref_y_local = target_h_px * 0.5
                ref_marker = QGraphicsEllipseItem(ref_x_local - ref_radius, ref_y_local - ref_radius, ref_radius*2, ref_radius*2)
                ref_marker.setZValue(2.1)
                pen = QPen(QColor(0, 200, 255, 255), 3)
                ref_marker.setPen(pen)
                ref_marker.setBrush(QColor(0, 200, 255, 80))
                # Parent to overlay so it moves/scales with image
                ref_marker.setParentItem(self.scene.overlay_item)
                self.scene.overlay_ref_item = ref_marker
                # Crosshair
                from PyQt5.QtWidgets import QGraphicsLineItem
                cross_len = 20
                h_line = QGraphicsLineItem(ref_x_local - cross_len, ref_y_local, ref_x_local + cross_len, ref_y_local)
                v_line = QGraphicsLineItem(ref_x_local, ref_y_local - cross_len, ref_x_local, ref_y_local + cross_len)
                h_line.setZValue(2.1)
                v_line.setZValue(2.1)
                h_line.setPen(QPen(QColor(0, 200, 255, 255), 2))
                v_line.setPen(QPen(QColor(0, 200, 255, 255), 2))
                h_line.setParentItem(self.scene.overlay_item)
                v_line.setParentItem(self.scene.overlay_item)
                self.scene.overlay_ref_hline = h_line
                self.scene.overlay_ref_vline = v_line
            except Exception:
                pass

            if hasattr(self, 'chk_show_overlay'):
                self.chk_show_overlay.setChecked(True)
            self.scene.set_overlay_visible(True)
            self.scene.apply_overlay_transform()

            QMessageBox.information(self, "Orthophoto", "Swiss orthophoto loaded for current map region.")
        except Exception as exc:
            QMessageBox.critical(self, "Orthophoto Error", f"Failed to load orthophoto image:\n{exc}")

    def _apply_canvas_zoom(self) -> None:
        try:
            self.view.resetTransform()
            self.view.scale(self.canvas_zoom, self.canvas_zoom)
        except Exception:
            pass

    def on_canvas_zoom_changed(self, value: int) -> None:
        try:
            self.canvas_zoom = max(0.01, float(value) / 100.0)
            self._apply_canvas_zoom()
        except Exception:
            pass

    def _on_background_dragged(self, dx_tiles: int, dy_tiles: int) -> None:
        # Update offsets and re-apply placement
        self.offset_x += int(dx_tiles)
        self.offset_y += int(dy_tiles)
        # Update spinboxes to reflect new offsets
        if hasattr(self, 'offset_x_spin'):
            self.offset_x_spin.blockSignals(True)
            self.offset_x_spin.setValue(self.offset_x)
            self.offset_x_spin.blockSignals(False)
        if hasattr(self, 'offset_y_spin'):
            self.offset_y_spin.blockSignals(True)
            self.offset_y_spin.setValue(self.offset_y)
            self.offset_y_spin.blockSignals(False)
        self._apply_current_placement()

    def on_offset_manual_change(self, _: int) -> None:
        """Handle manual offset changes from spinboxes."""
        if hasattr(self, 'offset_x_spin') and hasattr(self, 'offset_y_spin'):
            self.offset_x = int(self.offset_x_spin.value())
            self.offset_y = int(self.offset_y_spin.value())
            # Update live based on current placement mode
            self._apply_current_placement()

    

    # ----- Helpers -----
    def _update_offset_ranges(self) -> None:
        pass

    # ----- Settings handlers -----
    def rebuild_scene(self, new_grid_size: int, keep_background: bool = True) -> None:
        self.scene = GridScene(new_grid_size)
        self.scene.meters_per_tile = self.meters_per_tile
        self.scene.on_background_moved = self._on_background_dragged
        self.view.setScene(self.scene)
        blank = np.ones((new_grid_size, new_grid_size), dtype=np.float32)
        self.scene.set_background_from_array(blank)
        if self.last_pcl_canvas is not None:
            self._apply_current_placement()
        self._update_offset_ranges()
        # Reattach callbacks after scene rebuild
        self.scene.on_rectangle_committed = self.on_rectangle_committed
        self.scene.on_mask_changed = self._on_mask_changed
        self.scene.get_current_depth = lambda: self.depth_spin.value()
        self.scene.get_elevation = lambda: self.last_placed_elevation
        self.scene.on_group_selected = self.on_group_selected
        # Re-apply current canvas zoom after scene rebuild
        self._apply_canvas_zoom()

    def on_grid_size_change(self, text: str) -> None:
        try:
            new_size = int(text)
        except ValueError:
            return
        if new_size == self.grid_size:
            return
        self.grid_size = new_size
        self.rebuild_scene(self.grid_size)

    def on_meters_per_tile_change(self, value: float) -> None:
        self.meters_per_tile = float(value)
        self.scene.meters_per_tile = self.meters_per_tile
        if self.last_pcl_canvas is not None:
            self._apply_current_placement()

    def on_placement_changed(self, _: str) -> None:
        self._apply_current_placement()

    def on_offset_changed(self, _: int) -> None:
        pass

    # ----- File loaders -----
    def on_load_map(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open elevation map (.npy)",
            os.getcwd(),
            "NumPy (*.npy)"
        )
        if not path:
            return
        try:
            # Load raw array (meters). Use user-provided map resolution to scale to current meters_per_tile without changing it.
            arr = np.load(path)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=2) if arr.shape[2] > 1 else arr[:, :, 0]
            if arr.ndim != 2:
                raise ValueError("NPY must be 2D (HxW) or 3D (HxWxC) array")
            
            map_res = float(self.map_res_spin.value())  # meters per cell for this map
            if map_res <= 0:
                raise ValueError("Resolution must be > 0")
            H, W = int(arr.shape[0]), int(arr.shape[1])  # cells in the loaded map
            # Convert: cells * meters_per_cell = meters, then meters / meters_per_tile = tiles
            meters_h = H * map_res
            meters_w = W * map_res
            mpt = float(self.meters_per_tile)
            out_h = max(1, int(round(meters_h / mpt)))
            out_w = max(1, int(round(meters_w / mpt)))
            print(f"Load Map: {H}×{W} cells × {map_res} m/cell = {meters_h}×{meters_w} meters")
            print(f"  → {meters_h}×{meters_w} m ÷ {mpt} m/tile = {out_h}×{out_w} tiles")
            # Resize to calculated tile size (preserves physical scale)
            if not np.isfinite(arr).any():
                raise ValueError("Loaded array contains no finite values")
            a_min = float(np.nanmin(arr))
            a_max = float(np.nanmax(arr))
            arr_filled = np.where(np.isfinite(arr), arr, a_min).astype(np.float32)
            if a_max - a_min < 1e-8:
                resized = np.full((out_h, out_w), a_min, dtype=np.float32)
            else:
                tmp_norm = (arr_filled - a_min) / (a_max - a_min)
                img = Image.fromarray((tmp_norm * 255.0).astype(np.uint8))
                img_resized = img.resize((out_w, out_h), Image.BILINEAR)
                resized = np.array(img_resized, dtype=np.float32) / 255.0
                resized = resized * (a_max - a_min) + a_min
            # Store as canvas (unrotated) and use placement system to display on grid
            self._store_base_canvas(resized, apply_rotation=False)
            self._store_base_desired_canvas(None, apply_rotation=False)
            self._store_base_previous_desired_canvas(None, apply_rotation=False)
            self._store_base_original_elevation_array(resized.copy(), apply_rotation=False)
            self._store_base_original_desired_array(None, apply_rotation=False)
            self._store_base_original_previous_desired_array(None, apply_rotation=False)
            self._apply_rotation_to_bases()
            canvas_shape = self.last_pcl_canvas.shape if self.last_pcl_canvas is not None else resized.shape
            self._set_default_offsets(canvas_shape)
            # Clear placement params (will be set when placement is applied)
            self.placement_params = None
            if hasattr(self, 'placement_combo'):
                self.placement_combo.blockSignals(True)
                self.placement_combo.setCurrentText("Top-Left")
                self.placement_combo.blockSignals(False)
            self._update_offset_ranges()
            self._apply_current_placement()
            self.update_foundation_profile()
            self.update_3d_view()
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to load map:\n{exc}")

    def on_load_foundation(self) -> None:
        """Import foundation from STL or OBJ file."""
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Foundation Mesh",
            os.getcwd(),
            "3D Mesh Files (*.stl *.obj);;STL Files (*.stl);;OBJ Files (*.obj);;All Files (*.*)"
        )
        if not file_path:
            return
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # Parse mesh file
            vertices = None
            faces = None
            
            if file_ext == '.stl':
                if not HAS_STL:
                    QMessageBox.critical(
                        self,
                        "Missing Dependency",
                        "numpy-stl library is required to load STL files.\n\n"
                        "Install with: pip install numpy-stl"
                    )
                    return
                
                # Load STL file
                stl_mesh = mesh.Mesh.from_file(file_path)
                # STL vertices are in the vectors (points of triangles)
                all_points = stl_mesh.vectors.reshape(-1, 3)
                
                # Get unique vertices and create mapping
                vertices_unique, vertex_indices = np.unique(
                    all_points.round(decimals=6), 
                    axis=0, 
                    return_inverse=True
                )
                vertices = vertices_unique
                
                # Create faces from triangles (each triangle is a face)
                num_triangles = len(stl_mesh.vectors)
                faces = []
                for i in range(num_triangles):
                    # Map triangle vertices to unique vertex indices
                    tri_start = i * 3
                    face = [
                        vertex_indices[tri_start],
                        vertex_indices[tri_start + 1],
                        vertex_indices[tri_start + 2]
                    ]
                    faces.append(face)
                faces = np.array(faces, dtype=np.int32)
                
            elif file_ext == '.obj':
                # Parse OBJ file manually
                vertices = []
                faces = []
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('v '):
                            # Vertex
                            parts = line.split()
                            if len(parts) >= 4:
                                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                vertices.append([x, y, z])
                        elif line.startswith('f '):
                            # Face (we'll use this for better footprint calculation)
                            parts = line.split()[1:]
                            face_verts = []
                            for part in parts:
                                # Handle format like "1" or "1/2/3" or "1//3"
                                v_idx = int(part.split('/')[0]) - 1  # OBJ indices start at 1
                                if v_idx >= 0:
                                    face_verts.append(v_idx)
                            if len(face_verts) >= 3:
                                faces.append(face_verts)
                
                if vertices:
                    vertices = np.array(vertices, dtype=np.float32)
                if faces:
                    faces = np.array(faces, dtype=np.int32)
            else:
                QMessageBox.critical(self, "Import Error", f"Unsupported file format: {file_ext}")
                return
            
            if vertices is None or len(vertices) == 0:
                raise ValueError("No vertices found in mesh file")
            
            # Check if coordinates look georeferenced
            # Swiss coordinates are typically 6-7 digits, WGS84 lat/lon are -180 to 180
            # Local ENU coordinates are usually smaller (hundreds to thousands of meters)
            x_coords = vertices[:, 0]
            y_coords = vertices[:, 1]
            x_range = x_coords.max() - x_coords.min()
            y_range = y_coords.max() - y_coords.min()
            x_mean = x_coords.mean()
            y_mean = y_coords.mean()
            
            # Heuristic: detect coordinate system
            # Swiss coordinates are typically 6-7 digits (200000-2800000 range)
            # WGS84 lat/lon are -180 to 180
            # Local ENU coordinates are usually smaller (hundreds to thousands of meters)
            is_georeferenced = False
            coordinate_system = None
            coord_info = []
            
            # Print coordinate statistics for debugging
            coord_info.append(f"X: min={x_coords.min():.2f}, max={x_coords.max():.2f}, mean={x_mean:.2f}")
            coord_info.append(f"Y: min={y_coords.min():.2f}, max={y_coords.max():.2f}, mean={y_mean:.2f}")
            coord_info.append(f"Z: min={vertices[:, 2].min():.2f}, max={vertices[:, 2].max():.2f}")
            
            # Check if coordinates look like Swiss (CH1903+)
            # Swiss coordinates: X typically 200000-2800000, Y typically 4800000-7400000
            if (x_mean > 200000 and x_mean < 2800000) or (y_mean > 4800000 and y_mean < 7400000):
                is_georeferenced = True
                coordinate_system = "swiss"
                coord_info.append(f"Detected: Swiss CH1903+ / LV95 (EPSG:2056)")
            # Check for other projected coordinate systems (UTM, etc.) - check before WGS84
            elif abs(x_mean) > 10000 and abs(x_mean) < 10000000:
                is_georeferenced = True
                coordinate_system = "projected"
                coord_info.append(f"Detected: Possibly UTM or other projected coordinate system")
            # Check if coordinates look like WGS84 lat/lon
            # WGS84: X (longitude) -180 to 180, Y (latitude) -90 to 90
            # IMPORTANT: Values < ~10 are almost certainly local coordinates (meters), not lat/lon
            # Real lat/lon values are typically > 10 for most populated areas
            elif (abs(x_mean) >= 10 and abs(x_mean) < 180 and 
                  abs(y_mean) >= 10 and abs(y_mean) < 90 and 
                  abs(x_coords.min()) >= -180 and abs(x_coords.max()) <= 180 and
                  abs(y_coords.min()) >= -90 and abs(y_coords.max()) <= 90):
                is_georeferenced = True
                coordinate_system = "wgs84"
                coord_info.append(f"Detected: WGS84 (EPSG:4326) - Lat/Lon")
            else:
                # Local coordinates or ungeoreferenced (values typically < 10 or in reasonable meter range)
                coord_info.append(f"Detected: Local ENU or ungeoreferenced (meters)")
            
            # Log coordinate information
            print(f"\nSTL Coordinate Analysis:")
            for info in coord_info:
                print(f"  {info}")
            
            # Load georeferencing config if available (always load for potential ENU assumption)
            use_geo_ref = False
            geo_ref_status = "Not georeferenced"
            config_loaded = False
            if yaml is not None:
                from pathlib import Path
                script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
                cwd = Path(os.getcwd())
                possible_config_paths = [
                    script_dir / "map_georeference_config.yaml",
                    cwd / "map_georeference_config.yaml",
                    Path(file_path).parent / "map_georeference_config.yaml",
                ]
                
                config_path = None
                for test_path in possible_config_paths:
                    if test_path.exists():
                        config_path = test_path
                        break
                
                if config_path:
                    with open(str(config_path), 'r') as f:
                        config = yaml.safe_load(f)
                    config_loaded = True
                    if config and 'gnss' in config:
                        # Store full config (keep 'gnss' nesting)
                        self.georef_config = config
                        self._configure_gnss_usage(config.get('gnss', {}).get('useGnssReference', False))
                        
                        if is_georeferenced and config.get('gnss', {}).get('useGnssReference', False):
                            use_geo_ref = True
                            ref_lat = config['gnss']['referenceLatitude']
                            ref_lon = config['gnss']['referenceLongitude']
                            ref_alt = config['gnss']['referenceAltitude']
                            ref_heading = config['gnss']['referenceHeading']
                            geo_ref_status = f"Georeferenced ({coordinate_system}) - using config"
                            print(f"  ✓ Found georeference config, converting to ENU")
                        elif is_georeferenced:
                            geo_ref_status = f"Detected {coordinate_system} but config not active"
                            print(f"  ⚠ Coordinate system detected but config not valid/active")
                        else:
                            # Config loaded but coordinates are local - might be ENU
                            print(f"  ✓ Found georeference config (will assume ENU if coordinates are small)")
                else:
                    if is_georeferenced:
                        geo_ref_status = f"Detected {coordinate_system} but config file not found"
                        print(f"  ⚠ Coordinate system detected but config file not found")
            elif is_georeferenced:
                geo_ref_status = f"Detected {coordinate_system} but yaml library not available"
                print(f"  ⚠ Coordinate system detected but PyYAML not available")
            else:
                geo_ref_status = "Not georeferenced - will center on grid"
                print(f"  → Will center mesh on grid")
            
            # Convert coordinates to local ENU if georeferenced
            if use_geo_ref and coordinate_system == "swiss":
                # Convert Swiss coordinates to WGS84, then to ENU
                try:
                    import pyproj
                    # Swiss CH1903+ to WGS84
                    swiss_to_wgs84 = pyproj.Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
                    # Then WGS84 to ENU (approximate)
                    vertices_wgs84 = swiss_to_wgs84.transform(x_coords, y_coords)
                    # Convert to ENU using approximate method
                    # This is a simplified conversion - for better accuracy, use proper geodetic calculations
                    lat_rad = np.radians(ref_lat)
                    lon_rad = np.radians(ref_lon)
                    earth_radius = 6378137.0  # meters
                    
                    # Approximate ENU conversion for small distances
                    dlat = np.radians(vertices_wgs84[1] - ref_lat)
                    dlon = np.radians(vertices_wgs84[0] - ref_lon)
                    
                    e = earth_radius * dlon * np.cos(lat_rad)
                    n = earth_radius * dlat
                    u = vertices[:, 2] - ref_alt
                    
                    vertices_local = np.column_stack([e, n, u])
                except Exception as e:
                    print(f"Warning: Could not convert Swiss coordinates: {e}, centering on grid")
                    use_geo_ref = False
            elif use_geo_ref and coordinate_system == "wgs84":
                # Convert WGS84 to ENU
                try:
                    lat_rad = np.radians(ref_lat)
                    earth_radius = 6378137.0  # meters
                    
                    dlat = np.radians(y_coords - ref_lat)
                    dlon = np.radians(x_coords - ref_lon)
                    
                    e = earth_radius * dlon * np.cos(lat_rad)
                    n = earth_radius * dlat
                    u = vertices[:, 2] - ref_alt
                    
                    vertices_local = np.column_stack([e, n, u])
                except Exception as e:
                    print(f"Warning: Could not convert WGS84 coordinates: {e}, centering on grid")
                    use_geo_ref = False
            else:
                # Use coordinates as-is (assumed to be local ENU or ungeoreferenced)
                vertices_local = vertices.copy()
            
            # If not georeferenced or conversion failed, decide how to handle
            if not use_geo_ref:
                # Check if coordinates might already be ENU relative to reference point
                # If coordinates are small and positive (typical for ENU offsets), ask user
                # For now, we'll assume they're already ENU if they're small and positive
                # and we have a config file available
                x_mean_local = vertices_local[:, 0].mean()
                y_mean_local = vertices_local[:, 1].mean()
                
                # If coordinates are small (< 1000m) and we have a config, assume they're ENU
                # Otherwise, center on grid
                assume_enu = False
                if (abs(x_mean_local) < 1000 and abs(y_mean_local) < 1000 and 
                    yaml is not None and hasattr(self, 'georef_config')):
                    # Coordinates might already be ENU relative to reference point
                    # Use them as-is (don't center)
                    assume_enu = True
                    print(f"  → Assuming local coordinates are ENU relative to reference point")
                    print(f"  → Using coordinates as-is (not centering)")
                
                if not assume_enu:
                    # Center the mesh on the grid
                    x_center = vertices_local[:, 0].mean()
                    y_center = vertices_local[:, 1].mean()
                    grid_center_x = (self.grid_size - 1) * 0.5 * self.meters_per_tile
                    grid_center_y = (self.grid_size - 1) * 0.5 * self.meters_per_tile
                    
                    # Translate to grid center
                    vertices_local[:, 0] = vertices_local[:, 0] - x_center + grid_center_x
                    vertices_local[:, 1] = vertices_local[:, 1] - y_center + grid_center_y
                    print(f"  → Centered mesh on grid (local coordinates)")
            
            # Project mesh to 2D grid to create foundation mask
            # Get 2D footprint (XY plane)
            footprint_x = vertices_local[:, 0]
            footprint_y = vertices_local[:, 1]
            
            # Convert to grid cell coordinates
            cell_x = (footprint_x / self.meters_per_tile).astype(int)
            cell_y = (footprint_y / self.meters_per_tile).astype(int)
            
            # Create foundation mask and optional depth map
            # Clear existing foundation
            self.scene.foundation_mask[:, :] = 0
            self.scene.foundation_depth_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            
            # Check if mesh has varying Z coordinates (variable depth)
            z_coords = vertices_local[:, 2]
            z_min, z_max = z_coords.min(), z_coords.max()
            z_range = z_max - z_min
            has_variable_depth = z_range > 0.01  # More than 1cm variation
            
            if has_variable_depth:
                print(f"  → Mesh has variable depth: Z range = {z_range:.2f}m (min={z_min:.2f}, max={z_max:.2f})")
                print(f"  → Will use per-cell depth from mesh Z coordinates")
            else:
                print(f"  → Mesh has uniform depth: Z = {z_min:.2f}m")
                print(f"  → Will use uniform depth from depth spinbox")
            
            # Store absolute Z values for all imported cells (for recalculation when base height changes)
            all_cell_absolute_z = {}  # Dict: (cx, cy) -> average absolute Z
            
            # If we have faces, use them for better coverage
            if faces is not None and len(faces) > 0:
                # Rasterize faces to grid cells with depth information
                # Process faces in batches to avoid too many individual updates
                cells_to_fill = {}  # Dict: (cx, cy) -> list of Z values for averaging
                
                for face in faces:
                    if len(face) >= 3:
                        # Get face vertices in world coordinates (not grid cells yet)
                        face_verts = vertices_local[face[:3]]
                        face_x = face_verts[:, 0]
                        face_y = face_verts[:, 1]
                        face_z = face_verts[:, 2]
                        
                        # Convert to grid cell coordinates
                        face_cells_x = (face_x / self.meters_per_tile).astype(int)
                        face_cells_y = (face_y / self.meters_per_tile).astype(int)
                        
                        # Find bounding box of face
                        x_min, x_max = max(0, face_cells_x.min()), min(self.grid_size - 1, face_cells_x.max())
                        y_min, y_max = max(0, face_cells_y.min()), min(self.grid_size - 1, face_cells_y.max())
                        
                        # Fill cells inside triangle
                        for cy in range(y_min, y_max + 1):
                            for cx in range(x_min, x_max + 1):
                                # Convert cell center to world coordinates for point-in-triangle test
                                cell_center_x = (cx + 0.5) * self.meters_per_tile
                                cell_center_y = (cy + 0.5) * self.meters_per_tile
                                
                                # Point-in-triangle test using barycentric coordinates
                                p = np.array([cell_center_x, cell_center_y])
                                v0 = np.array([face_x[0], face_y[0]])
                                v1 = np.array([face_x[1], face_y[1]])
                                v2 = np.array([face_x[2], face_y[2]])
                                
                                # Barycentric coordinates
                                denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
                                if abs(denom) < 1e-10:
                                    continue
                                
                                a = ((v1[1] - v2[1]) * (p[0] - v2[0]) + (v2[0] - v1[0]) * (p[1] - v2[1])) / denom
                                b = ((v2[1] - v0[1]) * (p[0] - v2[0]) + (v0[0] - v2[0]) * (p[1] - v2[1])) / denom
                                c = 1 - a - b
                                
                                if 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1:
                                    # Interpolate Z using barycentric coordinates
                                    z_interp = a * face_z[0] + b * face_z[1] + c * face_z[2]
                                    cell_key = (cx, cy)
                                    if cell_key not in cells_to_fill:
                                        cells_to_fill[cell_key] = []
                                    cells_to_fill[cell_key].append(z_interp)
                
                # Batch update foundation mask and depth map
                # For imported foundations, use a common base height (max elevation in foundation area) for all tiles
                for (cx, cy), z_values in cells_to_fill.items():
                    if 0 <= cy < self.grid_size and 0 <= cx < self.grid_size:
                        self.scene.foundation_mask[cy, cx] = 1
                        # Average Z values if multiple faces contribute to this cell
                        avg_bottom_z = np.mean(z_values)
                        # Store absolute Z for this cell (for recalculation when base height changes)
                        all_cell_absolute_z[(cx, cy)] = avg_bottom_z
                
                # Calculate common base height: max elevation in foundation area
                common_base_height = 0.0
                if self.last_placed_elevation is not None:
                    elev_values = []
                    for (cx, cy) in all_cell_absolute_z.keys():
                        if 0 <= cy < self.grid_size and 0 <= cx < self.grid_size:
                            elev_val = self.last_placed_elevation[cy, cx]
                            if np.isfinite(elev_val):
                                elev_values.append(float(elev_val))
                    if elev_values:
                        common_base_height = max(elev_values)
                
                # Convert to relative depth using common base height for all tiles
                for (cx, cy), avg_bottom_z in all_cell_absolute_z.items():
                    if 0 <= cy < self.grid_size and 0 <= cx < self.grid_size:
                        # Convert to relative depth: common base height - bottom Z (same base for all tiles)
                        relative_depth = common_base_height - avg_bottom_z
                        # Store as relative depth (positive means below base height)
                        if self.scene.foundation_depth_map is None:
                            self.scene.foundation_depth_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
                        self.scene.foundation_depth_map[cy, cx] = max(0.0, relative_depth)
            else:
                # No faces available, use point cloud approach
                # Mark cells that contain vertices and store Z values
                cell_z_map = {}  # Dict: (cx, cy) -> list of Z values
                for i in range(len(cell_x)):
                    cx, cy = cell_x[i], cell_y[i]
                    if 0 <= cy < self.grid_size and 0 <= cx < self.grid_size:
                        self.scene.foundation_mask[cy, cx] = 1
                        cell_key = (cx, cy)
                        if cell_key not in cell_z_map:
                            cell_z_map[cell_key] = []
                        cell_z_map[cell_key].append(vertices_local[i, 2])
                
                # Store average Z per cell, use common base height for all tiles
                for (cx, cy), z_values in cell_z_map.items():
                    if 0 <= cy < self.grid_size and 0 <= cx < self.grid_size:
                        avg_bottom_z = np.mean(z_values)
                        # Store absolute Z for this cell (for recalculation when base height changes)
                        all_cell_absolute_z[(cx, cy)] = avg_bottom_z
                
                # Calculate common base height: max elevation in foundation area
                common_base_height = 0.0
                if self.last_placed_elevation is not None:
                    elev_values = []
                    for (cx, cy) in all_cell_absolute_z.keys():
                        if 0 <= cy < self.grid_size and 0 <= cx < self.grid_size:
                            elev_val = self.last_placed_elevation[cy, cx]
                            if np.isfinite(elev_val):
                                elev_values.append(float(elev_val))
                    if elev_values:
                        common_base_height = max(elev_values)
                
                # Convert to relative depth using common base height for all tiles
                for (cx, cy), avg_bottom_z in all_cell_absolute_z.items():
                    if 0 <= cy < self.grid_size and 0 <= cx < self.grid_size:
                        # Convert to relative depth: common base height - bottom Z (same base for all tiles)
                        relative_depth = common_base_height - avg_bottom_z
                        if self.scene.foundation_depth_map is None:
                            self.scene.foundation_depth_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
                        self.scene.foundation_depth_map[cy, cx] = max(0.0, relative_depth)
                
                # Fill convex hull of points
                try:
                    from scipy.spatial import ConvexHull
                    points_2d = np.column_stack([cell_x, cell_y])
                    hull = ConvexHull(points_2d)
                    # Fill inside convex hull
                    from scipy.spatial import Delaunay
                    tri = Delaunay(points_2d[hull.vertices])
                    
                    # Find all grid cells inside convex hull
                    grid_y, grid_x = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size), indexing='ij')
                    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
                    inside = tri.find_simplex(grid_points) >= 0
                    inside = inside.reshape(self.grid_size, self.grid_size)
                    self.scene.foundation_mask[inside] = 1
                except ImportError:
                    # scipy not available, just use the points
                    pass
            
            # Collect all foundation cells for this import
            foundation_cells = []
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if self.scene.foundation_mask[y, x] == 1:
                        foundation_cells.append((x, y))
                        # Clear other masks for these cells
                        self.scene.dump_mask[y, x] = 0
                        self.scene.obstacle_mask[y, x] = 0
                        self.scene.nodump_mask[y, x] = 0
            
            # Create a foundation group with outline
            if foundation_cells:
                # Create outline polygon around the group
                outline_polygon = self.scene._create_group_outline(foundation_cells)
                
                # Store depth map for this group (always store if depth_map exists, even for imported)
                # For imported foundations, this stores the variable bottom mesh Z as relative depths
                group_depth_map = None
                if self.scene.foundation_depth_map is not None:
                    # Create a copy of depth values for this group
                    group_depth_map = {}
                    for x, y in foundation_cells:
                        if 0 <= y < self.scene.grid_size and 0 <= x < self.scene.grid_size:
                            depth_val = self.scene.foundation_depth_map[y, x]
                            # Store as relative depth (already converted during import)
                            group_depth_map[(x, y)] = depth_val
                
                # Store absolute Z values for recalculation when base height changes
                group_absolute_z = {}
                for x, y in foundation_cells:
                    if (x, y) in all_cell_absolute_z:
                        group_absolute_z[(x, y)] = all_cell_absolute_z[(x, y)]
                
                # Calculate common base height: max elevation in foundation area
                common_base_height = 0.0
                if self.last_placed_elevation is not None:
                    elev_values = []
                    for x, y in foundation_cells:
                        if 0 <= y < self.scene.grid_size and 0 <= x < self.scene.grid_size:
                            elev_val = self.last_placed_elevation[y, x]
                            if np.isfinite(elev_val):
                                elev_values.append(float(elev_val))
                    if elev_values:
                        common_base_height = max(elev_values)
                
                # Create group object
                group = {
                    'cells': foundation_cells,
                    'outline': outline_polygon,
                    'outline_item': None,  # Will be created
                    'depth_map': group_depth_map,
                    'absolute_z': group_absolute_z,  # Store absolute Z for recalculation
                    'id': len(self.scene.foundation_groups),  # Unique ID
                    'is_imported': True  # Mark as imported
                }
                
                # Set group depth to common base height (max elevation in foundation area)
                if hasattr(self, 'group_depth_spin') and common_base_height > 0:
                    self.group_depth_spin.blockSignals(True)
                    self.group_depth_spin.setValue(common_base_height)
                    self.group_depth_spin.blockSignals(False)
                    # Select the group so the depth box is visible
                    self.scene._select_foundation_group(group)
                
                # Add outline to scene
                group['outline_item'] = self.scene._draw_group_outline(group)
                
                # Add to groups list
                self.scene.foundation_groups.append(group)
            
            # Update all cell brushes at once
            if foundation_cells:
                self.scene._batch_set_cell_type(foundation_cells, "foundation")
            
            # Update foundation profile and 3D view
            if callable(getattr(self.scene, 'on_mask_changed', None)):
                self.scene.on_mask_changed()
            
            self.update_foundation_profile()
            self.update_3d_view()
            
            # Calculate mesh complexity
            num_faces_info = ""
            if faces is not None:
                num_faces_info = f"\n  Faces: {len(faces)}"
                if len(faces) > 10000:
                    num_faces_info += " (high detail - may be slow)"
                elif len(faces) > 5000:
                    num_faces_info += " (medium detail)"
            
            num_cells = self.scene.foundation_mask.sum()
            cells_info = f"  Foundation cells: {num_cells}"
            if num_cells > 500:
                cells_info += " (large footprint)"
            
            QMessageBox.information(
                self,
                "Import Successful",
                f"Imported foundation from {file_ext.upper()} file:\n"
                f"  Vertices: {len(vertices)}{num_faces_info}\n"
                f"  Coordinate system: {geo_ref_status}\n"
                f"  {cells_info}\n\n"
                f"Coordinate info printed to console.\n\n"
                f"Tip: Use 'select' mode to drag the foundation outline."
            )
            
        except Exception as exc:
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import foundation:\n{exc}\n\n{error_details}"
            )

    def on_load_geo_map(self) -> None:
        """Load a GridMap from a ROS bag file or georeferenced npy file."""
        if yaml is None:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                "yaml library is required to load georeferenced maps.\n\n"
                "Install with: pip install pyyaml"
            )
            return

        # Load bag file or npy file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open GridMap bag file or georeferenced npy file",
            os.getcwd(),
            "ROS Bag Files (*.bag *.mcap);;ROS1 Bag Files (*.bag);;ROS2 MCAP Files (*.mcap);;NumPy Files (*.npy);;All Files (*.*)"
        )
        if not file_path:
            return
        
        file_ext = os.path.splitext(file_path)[1].lower()

        # Load config file
        from pathlib import Path
        
        # Try multiple possible locations for config file
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        cwd = Path(os.getcwd())
        file_path_for_parent = Path(file_path) if isinstance(file_path, (str, Path)) else Path(str(file_path))
        
        possible_config_paths = [
            script_dir / "map_georeference_config.yaml",
            cwd / "map_georeference_config.yaml",
            file_path_for_parent.parent / "map_georeference_config.yaml",
        ]
        
        config_path = None
        for test_path in possible_config_paths:
            if test_path.exists():
                config_path = test_path
                break
        
        if config_path is None:
            QMessageBox.critical(
                self,
                "Config Error",
                f"Georeference config file not found.\n\n"
                f"Tried:\n" + "\n".join(str(p) for p in possible_config_paths) + "\n\n"
                "Please ensure map_georeference_config.yaml exists."
            )
            return

        try:
            with open(str(config_path), 'r') as f:
                config = yaml.safe_load(f)
            
            if not config or 'gnss' not in config:
                raise ValueError("Config file missing 'gnss' section")
            
            use_gnss_ref = bool(config.get('gnss', {}).get('useGnssReference', False))
            if not use_gnss_ref:
                QMessageBox.warning(
                    self,
                    "Config Warning",
                    "GNSS reference is disabled in config. Using default values."
                )
            
            ref_lat = config['gnss']['referenceLatitude']
            ref_lon = config['gnss']['referenceLongitude']
            ref_alt = config['gnss']['referenceAltitude']
            ref_heading = config['gnss']['referenceHeading']
            
            # Store full config (keep 'gnss' nesting)
            self.georef_config = config
            self._configure_gnss_usage(use_gnss_ref)
            
            # Handle .npy files directly
            if file_ext == '.npy':
                # Load npy file directly
                elev_array = np.load(file_path).astype(np.float32)
                
                # For npy files, we need to determine resolution
                # Default to meters_per_tile or ask user
                resolution = self.meters_per_tile
                
                # Store file path for export
                self.original_bag_path = file_path
                self.original_gridmap_msg = None  # No GridMap message for npy files
                self.original_gridmap_resolution = resolution
                self.original_gridmap_conn_info = None  # No connection info for npy files
                # Clear desired_elevation (npy files don't have it)
                self._store_base_desired_canvas(None, apply_rotation=False)
                self._store_base_previous_desired_canvas(None, apply_rotation=False)
                self._store_base_original_desired_array(None, apply_rotation=False)
                self._store_base_original_previous_desired_array(None, apply_rotation=False)
                # Disable checkbox for npy files
                if hasattr(self, 'chk_show_desired_elevation'):
                    self.chk_show_desired_elevation.setEnabled(False)
                    self.chk_show_desired_elevation.setChecked(False)
                
                # Set GridMap center to (0, 0, 0) for npy files (origin at GNSS reference)
                self.georef_gridmap_center = {
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.0
                }
                
                # Use the existing placement/resize logic
                H, W = elev_array.shape
                meters_h = H * resolution
                meters_w = W * resolution
                
                # Resize to match current grid resolution
                mpt = self.meters_per_tile
                out_h = max(1, int(round(meters_h / mpt)))
                out_w = max(1, int(round(meters_w / mpt)))
                
                # Resize elevation array
                if not np.isfinite(elev_array).any():
                    raise ValueError("Elevation array contains no finite values")
                
                a_min = float(np.nanmin(elev_array))
                a_max = float(np.nanmax(elev_array))
                arr_filled = np.where(np.isfinite(elev_array), elev_array, a_min).astype(np.float32)
                
                if a_max - a_min < 1e-8:
                    resized = np.full((out_h, out_w), a_min, dtype=np.float32)
                else:
                    tmp_norm = (arr_filled - a_min) / (a_max - a_min)
                    img = Image.fromarray((tmp_norm * 255.0).astype(np.uint8))
                    img_resized = img.resize((out_w, out_h), Image.BILINEAR)
                    resized = np.array(img_resized, dtype=np.float32) / 255.0
                    resized = resized * (a_max - a_min) + a_min
            
                # Store as canvas/original bases
                self._store_base_original_elevation_array(elev_array.copy(), apply_rotation=False)
                self._store_base_canvas(resized, apply_rotation=False)
                self._apply_rotation_to_bases()
                canvas_shape = self.last_pcl_canvas.shape if self.last_pcl_canvas is not None else resized.shape
                self._set_default_offsets(canvas_shape)
                # Clear placement params (will be set when placement is applied)
                self.placement_params = None
                
                # Update UI
                if hasattr(self, 'map_res_spin'):
                    self.map_res_spin.blockSignals(True)
                    self.map_res_spin.setValue(resolution)
                    self.map_res_spin.blockSignals(False)
                
                if hasattr(self, 'placement_combo'):
                    self.placement_combo.blockSignals(True)
                    self.placement_combo.setCurrentText("Top-Left")
                    self.placement_combo.blockSignals(False)
                
                self._update_offset_ranges()
                self._apply_current_placement()
                self.update_foundation_profile()
                self.update_3d_view()
                
                QMessageBox.information(
                    self,
                    "Load Successful",
                    f"Loaded georeferenced npy file:\n"
                    f"  Size: {H}×{W} cells ({meters_h:.1f}×{meters_w:.1f}m)\n"
                    f"  Resolution: {resolution} m/cell\n"
                    f"  Elevation range: {a_min:.2f}m to {a_max:.2f}m\n"
                    f"  Resized to: {out_h}×{out_w} tiles"
                )
                return
            
            # Handle .bag and .mcap files
            if not HAS_ROSBAGS:
                QMessageBox.critical(
                    self,
                    "Missing Dependency",
                    "rosbags library is required to load bag/MCAP files.\n\n"
                    "Install with: pip install rosbags"
                )
                return
            
            # Load GridMap from bag - convert to Path object (as in working tools)
            bag_path_obj = Path(file_path)
            if not bag_path_obj.exists():
                raise ValueError(f"Bag file not found: {file_path}")
            
            # Use Path object directly (as in our working tools)
            with AnyReader([bag_path_obj]) as reader:
                conns = [c for c in reader.connections if c.msgtype in ("grid_map_msgs/msg/GridMap", "grid_map_msgs/GridMap")]
                if not conns:
                    raise ValueError("No grid_map_msgs/GridMap topic found in bag file")
                
                conn = conns[0]
                msg = None
                for _, _, raw in reader.messages(connections=[conn]):
                    msg = reader.deserialize(raw, conn.msgtype)
                    break  # Get first message
                
                if msg is None:
                    raise ValueError("No messages found in grid_map topic")
                
                # Store original message and bag path for exporting
                self.original_bag_path = file_path
                self.original_gridmap_msg = msg
                # Store connection info for writing (msgdef, rihs01, msgtype, typestore)
                # Extract msgdef.data if msgdef is a MessageDefinition object
                msgdef_val = getattr(conn, 'msgdef', None)
                if msgdef_val is not None and hasattr(msgdef_val, 'data'):
                    # Extract the string data from MessageDefinition object
                    msgdef_str = msgdef_val.data
                else:
                    msgdef_str = msgdef_val
                
                # Get typestore from reader (preferred) or connection
                reader_typestore = getattr(reader, 'typestore', None)
                conn_typestore = getattr(conn, 'typestore', None) if hasattr(conn, 'typestore') else None
                typestore = reader_typestore if reader_typestore is not None else conn_typestore
                
                # Try to get md5sum from typestore if available (needed for rosbag1.Writer)
                md5sum = None
                if typestore is not None and msgdef_str:
                    try:
                        # Try to generate msgdef and md5sum from typestore
                        _, md5sum = typestore.generate_msgdef(conn.msgtype)
                    except Exception:
                        # If that fails, try to get md5sum from connection
                        md5sum = getattr(conn, 'md5sum', None)
                
                self.original_gridmap_conn_info = {
                    'msgtype': conn.msgtype,
                    'msgdef': msgdef_str,  # Store as string
                    'md5sum': md5sum,  # Store md5sum for rosbag1.Writer
                    'rihs01': getattr(conn, 'rihs01', None),
                    'typestore': typestore  # Use reader's typestore (preferred)
                }
                
                # Extract GridMap info
                info = getattr(msg, "info", None)
                if info is None:
                    raise ValueError("GridMap message missing 'info' field")
                
                resolution = getattr(info, "resolution", None)
                if resolution is None:
                    raise ValueError("GridMap missing resolution")
                
                self.original_gridmap_resolution = resolution
                
                pose = getattr(info, "pose", None)
                pos = getattr(pose, "position", None) if pose else None
                if pos is None:
                    raise ValueError("GridMap missing pose.position")
                
                center_x = getattr(pos, 'x', 0.0)
                center_y = getattr(pos, 'y', 0.0)
                center_z = getattr(pos, 'z', 0.0)
                
                # Store GridMap center for georeferencing
                self.georef_gridmap_center = {
                    'x': center_x,
                    'y': center_y,
                    'z': center_z
                }
                
                # Extract layers
                layers = list(getattr(msg, "layers", []))
                try:
                    layer_names = ", ".join(layers) if layers else "<none>"
                    print(f"Loaded layers: {layer_names}")
                except Exception:
                    print(f"Loaded layers: {len(layers)} entries (could not format names)")
                if "elevation" not in layers:
                    raise ValueError("GridMap missing 'elevation' layer")
                
                # Extract elevation data
                # GridMap data structure: data is a list of matrices (one per layer)
                # Each matrix might be a ROS message type (Float32MultiArray) or numpy array
                elev_array = None
                
                # Get the data list
                data = getattr(msg, "data", [])
                if not data:
                    raise ValueError("GridMap has no data")
                
                # Find elevation layer index
                elev_idx = layers.index("elevation")
                if elev_idx >= len(data):
                    raise ValueError(f"Elevation layer index {elev_idx} out of range (data has {len(data)} items)")
                
                elev_matrix = data[elev_idx]
                
                print(f"Elevation matrix type: {type(elev_matrix)}")
                print(f"Elevation matrix attributes: {dir(elev_matrix)[:10]}")
                
                # Handle different data types
                if isinstance(elev_matrix, np.ndarray):
                    # Already a numpy array
                    elev_array = elev_matrix.astype(np.float32)
                elif 'Float32MultiArray' in str(type(elev_matrix)) or hasattr(elev_matrix, 'data'):
                    # ROS message type (e.g., Float32MultiArray)
                    # Access the .data attribute which contains the array
                    raw_data = elev_matrix.data
                    print(f"Raw data type: {type(raw_data)}, length: {len(raw_data) if hasattr(raw_data, '__len__') else 'N/A'}")
                    
                    # Get shape from layout if available
                    shape = None
                    if hasattr(elev_matrix, 'layout') and elev_matrix.layout is not None:
                        if hasattr(elev_matrix.layout, 'dim'):
                            dims = elev_matrix.layout.dim
                            if len(dims) >= 2:
                                shape = (dims[0].size, dims[1].size)
                                print(f"Shape from layout: {shape}")
                    
                    # Convert data to numpy array
                    # Float32MultiArray.data is typically a list or array-like
                    if isinstance(raw_data, np.ndarray):
                        elev_array = raw_data.astype(np.float32)
                    elif isinstance(raw_data, (list, tuple)):
                        elev_array = np.array(raw_data, dtype=np.float32)
                    elif hasattr(raw_data, '__iter__'):
                        # Try to convert iterable to list then array
                        elev_array = np.array([float(x) for x in raw_data], dtype=np.float32)
                    else:
                        raise ValueError(f"Cannot extract data from {type(elev_matrix)}, data type: {type(raw_data)}")
                    
                    print(f"Extracted array shape: {elev_array.shape}, size: {elev_array.size}")
                    
                    # Reshape if shape information is available
                    if shape is not None and elev_array.size == shape[0] * shape[1]:
                        elev_array = elev_array.reshape(shape)
                        print(f"Reshaped to: {elev_array.shape}")
                        # Store original GridMap cell size (H, W) before any resizing for display
                        try:
                            self.original_gridmap_size = (int(elev_array.shape[0]), int(elev_array.shape[1]))
                        except Exception:
                            self.original_gridmap_size = (None, None)
                    elif shape is not None:
                        # Shape mismatch - try to infer from array size
                        print(f"Warning: Expected shape {shape} but array size is {elev_array.size}, trying to infer shape")
                        # Try to infer shape from GridMap info
                        if hasattr(info, 'length') and hasattr(info.length, 'x') and hasattr(info.length, 'y'):
                            length_x = getattr(info.length, 'x', None)
                            length_y = getattr(info.length, 'y', None)
                            if length_x is not None and length_y is not None:
                                expected_size = int(length_y / resolution) * int(length_x / resolution)
                                if elev_array.size == expected_size:
                                    height = int(length_y / resolution)
                                    width = int(length_x / resolution)
                                    elev_array = elev_array.reshape((height, width))
                                    print(f"Inferred shape from GridMap length: {elev_array.shape}")
                                    try:
                                        self.original_gridmap_size = (int(height), int(width))
                                    except Exception:
                                        self.original_gridmap_size = (None, None)
                                else:
                                    # Just reshape to square if possible
                                    side = int(np.sqrt(elev_array.size))
                                    if side * side == elev_array.size:
                                        elev_array = elev_array.reshape((side, side))
                                        print(f"Reshaped to square: {elev_array.shape}")
                                        try:
                                            self.original_gridmap_size = (int(side), int(side))
                                        except Exception:
                                            self.original_gridmap_size = (None, None)
                    elif elev_array.ndim == 1:
                        # 1D array - need to infer 2D shape
                        # Try to get from GridMap info
                        if hasattr(info, 'length') and hasattr(info.length, 'x') and hasattr(info.length, 'y'):
                            length_x = getattr(info.length, 'x', None)
                            length_y = getattr(info.length, 'y', None)
                            if length_x is not None and length_y is not None:
                                height = int(length_y / resolution)
                                width = int(length_x / resolution)
                                if elev_array.size == height * width:
                                    elev_array = elev_array.reshape((height, width))
                                    print(f"Inferred 2D shape: {elev_array.shape}")
                                    try:
                                        self.original_gridmap_size = (int(height), int(width))
                                    except Exception:
                                        self.original_gridmap_size = (None, None)
                elif hasattr(elev_matrix, 'matrix'):
                    # GridMap Matrix type
                    elev_array = np.array(elev_matrix.matrix, dtype=np.float32)
                else:
                    # Try direct conversion
                    try:
                        elev_array = np.array(elev_matrix, dtype=np.float32)
                    except:
                        raise ValueError(f"Cannot convert elevation data from type: {type(elev_matrix)}")
                
                if elev_array is None:
                    raise ValueError("Could not extract elevation data from GridMap")
                
                # Handle NaN/infinite values
                elev_array = np.where(np.isfinite(elev_array), elev_array, 0.0)
                
                # Get dimensions
                H, W = elev_array.shape[0], elev_array.shape[1]

                def _extract_layer_array(layer_name: str) -> Optional[np.ndarray]:
                    """Extracts a layer from GridMap data as a numpy array matching elevation shape if possible."""
                    if layer_name not in layers:
                        return None
                    idx = layers.index(layer_name)
                    if idx >= len(data):
                        return None
                    layer_msg = data[idx]
                    layer_arr: Optional[np.ndarray] = None
                    
                    if isinstance(layer_msg, np.ndarray):
                        layer_arr = layer_msg.astype(np.float32)
                    elif 'Float32MultiArray' in str(type(layer_msg)) or hasattr(layer_msg, 'data'):
                        raw_data = getattr(layer_msg, 'data', None)
                        if raw_data is None:
                            return None
                        if isinstance(raw_data, np.ndarray):
                            layer_arr = raw_data.astype(np.float32)
                        elif isinstance(raw_data, (list, tuple)):
                            layer_arr = np.array(raw_data, dtype=np.float32)
                        elif hasattr(raw_data, '__iter__'):
                            try:
                                layer_arr = np.array([float(x) for x in raw_data], dtype=np.float32)
                            except Exception:
                                layer_arr = None
                        if layer_arr is not None and hasattr(layer_msg, 'layout') and getattr(layer_msg, 'layout') is not None:
                            layout = layer_msg.layout
                            if hasattr(layout, 'dim'):
                                dims = layout.dim
                                if len(dims) >= 2:
                                    shape = (int(dims[0].size), int(dims[1].size))
                                    if layer_arr.size == shape[0] * shape[1]:
                                        layer_arr = layer_arr.reshape(shape)
                    elif hasattr(layer_msg, 'matrix'):
                        try:
                            layer_arr = np.array(layer_msg.matrix, dtype=np.float32)
                        except Exception:
                            layer_arr = None
                    else:
                        try:
                            layer_arr = np.array(layer_msg, dtype=np.float32)
                        except Exception:
                            layer_arr = None
                    
                    if layer_arr is None:
                        return None
                    if layer_arr.ndim == 1 and layer_arr.size == H * W:
                        layer_arr = layer_arr.reshape((H, W))
                    return layer_arr

                def _print_binary_layer_stats(layer_name: str, friendly_name: str) -> None:
                    layer_arr = _extract_layer_array(layer_name)
                    if layer_arr is None:
                        return
                    arr = np.where(np.isfinite(layer_arr), layer_arr, 0.0)
                    if arr.size == 0:
                        return
                    if arr.ndim != 2 or arr.shape != (H, W):
                        flat = arr.ravel()
                        ones = int(np.count_nonzero(np.isclose(flat, 1.0, atol=1e-6)))
                        print(f"  Layer '{friendly_name}' loaded (shape {arr.shape}), cells == 1: {ones}")
                    else:
                        ones = int(np.count_nonzero(np.isclose(arr, 1.0, atol=1e-6)))
                        print(f"  Layer '{friendly_name}' loaded ({arr.shape[0]}×{arr.shape[1]}), cells == 1: {ones}")

                _print_binary_layer_stats("occupancy", "occupancy")
                _print_binary_layer_stats("dump_zone", "dump_zone")
                
                # Extract existing desired_elevation if it exists and save it as "previous"
                previous_desired_elev_array = None
                if "desired_elevation" in layers:
                    desired_elev_idx = layers.index("desired_elevation")
                    if desired_elev_idx < len(data):
                        desired_elev_matrix = data[desired_elev_idx]
                        
                        # Extract desired_elevation data using same logic as elevation
                        if isinstance(desired_elev_matrix, np.ndarray):
                            previous_desired_elev_array = desired_elev_matrix.astype(np.float32)
                        elif 'Float32MultiArray' in str(type(desired_elev_matrix)) or hasattr(desired_elev_matrix, 'data'):
                            raw_data = desired_elev_matrix.data
                            
                            # Get shape from layout if available
                            desired_shape = None
                            if hasattr(desired_elev_matrix, 'layout') and desired_elev_matrix.layout is not None:
                                if hasattr(desired_elev_matrix.layout, 'dim'):
                                    dims = desired_elev_matrix.layout.dim
                                    if len(dims) >= 2:
                                        desired_shape = (dims[0].size, dims[1].size)
                            
                            if isinstance(raw_data, np.ndarray):
                                previous_desired_elev_array = raw_data.astype(np.float32)
                            elif isinstance(raw_data, (list, tuple)):
                                previous_desired_elev_array = np.array(raw_data, dtype=np.float32)
                            elif hasattr(raw_data, '__iter__'):
                                previous_desired_elev_array = np.array([float(x) for x in raw_data], dtype=np.float32)
                            
                            # Reshape using layout information if available
                            if previous_desired_elev_array is not None and desired_shape is not None:
                                if previous_desired_elev_array.size == desired_shape[0] * desired_shape[1]:
                                    previous_desired_elev_array = previous_desired_elev_array.reshape(desired_shape)
                        
                        # Reshape to match elevation shape
                        if previous_desired_elev_array is not None:
                            if previous_desired_elev_array.ndim == 1:
                                if previous_desired_elev_array.size == H * W:
                                    previous_desired_elev_array = previous_desired_elev_array.reshape((H, W))
                                else:
                                    previous_desired_elev_array = None
                            elif previous_desired_elev_array.ndim == 2:
                                if previous_desired_elev_array.shape != (H, W):
                                    if previous_desired_elev_array.size == H * W:
                                        previous_desired_elev_array = previous_desired_elev_array.reshape((H, W))
                                    else:
                                        previous_desired_elev_array = None
                        
                        # Handle NaN/infinite values
                        if previous_desired_elev_array is not None:
                            previous_desired_elev_array = np.where(np.isfinite(previous_desired_elev_array), previous_desired_elev_array, 0.0)
                
                # Always use elevation as desired_elevation (overwrite any existing desired_elevation)
                # This ensures desired_elevation starts as a copy of elevation (no digging needed initially)
                if previous_desired_elev_array is not None:
                    print("Info: Found existing desired_elevation in bag - saving as previous_desired_elevation")
                print("Info: Using elevation as desired_elevation (overwriting any existing desired_elevation layer)")
                desired_elev_array = elev_array.copy()
                
                print(f"Load Geo Map: {H}×{W} cells × {resolution} m/cell")
                print(f"  GridMap center (ENU): x={center_x:.3f}m, y={center_y:.3f}m, z={center_z:.3f}m")
                print(f"  GNSS reference: lat={ref_lat}, lon={ref_lon}, alt={ref_alt}, heading={ref_heading}")
                
                # Calculate physical size in meters
                meters_h = H * resolution
                meters_w = W * resolution
                
                # Convert to current meters_per_tile grid
                mpt = float(self.meters_per_tile)
                out_h = max(1, int(round(meters_h / mpt)))
                out_w = max(1, int(round(meters_w / mpt)))
                
                print(f"  → {meters_h}×{meters_w} m ÷ {mpt} m/tile = {out_h}×{out_w} tiles")
                
                # Resize elevation array to match grid resolution
                if not np.isfinite(elev_array).any():
                    raise ValueError("Elevation array contains no finite values")
                
                a_min = float(np.nanmin(elev_array))
                a_max = float(np.nanmax(elev_array))
                arr_filled = np.where(np.isfinite(elev_array), elev_array, a_min).astype(np.float32)
                
                if a_max - a_min < 1e-8:
                    resized = np.full((out_h, out_w), a_min, dtype=np.float32)
                else:
                    # Normalize, resize, denormalize
                    tmp_norm = (arr_filled - a_min) / (a_max - a_min)
                    img = Image.fromarray((tmp_norm * 255.0).astype(np.uint8))
                    img_resized = img.resize((out_w, out_h), Image.BILINEAR)
                    resized = np.array(img_resized, dtype=np.float32) / 255.0
                    resized = resized * (a_max - a_min) + a_min
                
                # Store original elevation array (before resizing) for export
                self._store_base_original_elevation_array(elev_array.copy(), apply_rotation=False)
                
                # Store and resize previous_desired_elevation if it exists (from loaded bag)
                if previous_desired_elev_array is not None and previous_desired_elev_array.shape == elev_array.shape:
                    # Store original previous_desired_elevation array (before resizing)
                    self._store_base_original_previous_desired_array(previous_desired_elev_array.copy(), apply_rotation=False)
                    
                    # Resize previous_desired_elevation using its own min/max (not elevation's)
                    # desired_elevation has different values (lower in foundation areas), so needs its own normalization
                    prev_min = float(np.nanmin(previous_desired_elev_array))
                    prev_max = float(np.nanmax(previous_desired_elev_array))
                    prev_arr_filled = np.where(np.isfinite(previous_desired_elev_array), previous_desired_elev_array, prev_min).astype(np.float32)
                    
                    if prev_max - prev_min < 1e-8:
                        prev_resized = np.full((out_h, out_w), prev_min, dtype=np.float32)
                    else:
                        # Normalize using previous_desired_elevation's own range
                        prev_tmp_norm = (prev_arr_filled - prev_min) / (prev_max - prev_min)
                        prev_img = Image.fromarray((prev_tmp_norm * 255.0).astype(np.uint8))
                        prev_img_resized = prev_img.resize((out_w, out_h), Image.BILINEAR)
                        prev_resized = np.array(prev_img_resized, dtype=np.float32) / 255.0
                        prev_resized = prev_resized * (prev_max - prev_min) + prev_min
                    
                    self._store_base_previous_desired_canvas(prev_resized, apply_rotation=False)
                else:
                    # No previous desired_elevation - clear it
                    self._store_base_previous_desired_canvas(None, apply_rotation=False)
                    self._store_base_original_previous_desired_array(None, apply_rotation=False)
                
                # Resize and store desired_elevation (always exists now, as copy of elevation)
                if desired_elev_array is not None and desired_elev_array.shape == elev_array.shape:
                    # Store original desired_elevation array (before resizing) for export
                    self._store_base_original_desired_array(desired_elev_array.copy(), apply_rotation=False)
                    
                    # Resize desired_elevation the same way as elevation
                    desired_arr_filled = np.where(np.isfinite(desired_elev_array), desired_elev_array, a_min).astype(np.float32)
                    
                    if a_max - a_min < 1e-8:
                        desired_resized = np.full((out_h, out_w), a_min, dtype=np.float32)
                    else:
                        # Normalize, resize, denormalize (same as elevation)
                        desired_tmp_norm = (desired_arr_filled - a_min) / (a_max - a_min)
                        desired_img = Image.fromarray((desired_tmp_norm * 255.0).astype(np.uint8))
                        desired_img_resized = desired_img.resize((out_w, out_h), Image.BILINEAR)
                        desired_resized = np.array(desired_img_resized, dtype=np.float32) / 255.0
                        desired_resized = desired_resized * (a_max - a_min) + a_min
                    
                    self._store_base_desired_canvas(desired_resized, apply_rotation=False)
                else:
                    # This shouldn't happen since we always set desired_elev_array = elev_array.copy()
                    self._store_base_desired_canvas(None, apply_rotation=False)
                    self._store_base_original_desired_array(None, apply_rotation=False)
                
                # Store resized canvas (unrotated) and apply current rotation
                self._store_base_canvas(resized, apply_rotation=False)
                self._apply_rotation_to_bases()

                # Update checkbox state and label based on availability
                # Enable if we have either desired_elevation or previous_desired_elevation
                if hasattr(self, 'chk_show_desired_elevation'):
                    has_any_desired = (self.desired_elevation_canvas is not None or 
                                      self.previous_desired_elevation_canvas is not None)
                    self.chk_show_desired_elevation.setEnabled(has_any_desired)
                    if not has_any_desired:
                        self.chk_show_desired_elevation.setChecked(False)
                    else:
                        # Update label to indicate what will be shown
                        if self.previous_desired_elevation_canvas is not None:
                            self.chk_show_desired_elevation.setText("Show previous desired elevation")
                        elif self.desired_elevation_canvas is not None:
                            self.chk_show_desired_elevation.setText("Show desired elevation")
                
                # Store as canvas (same as regular map loading)
            canvas_shape = self.last_pcl_canvas.shape if self.last_pcl_canvas is not None else resized.shape
            self._set_default_offsets(canvas_shape)
            # Clear placement params (will be set when placement is applied)
            self.placement_params = None
                
            # Update map resolution spinbox to match GridMap resolution
            if hasattr(self, 'map_res_spin'):
                self.map_res_spin.blockSignals(True)
                self.map_res_spin.setValue(resolution)
                self.map_res_spin.blockSignals(False)
                
                # Reset offsets
                # Reset placement
            if hasattr(self, 'placement_combo'):
                self.placement_combo.blockSignals(True)
                self.placement_combo.setCurrentText("Top-Left")
                self.placement_combo.blockSignals(False)
                
            self._update_offset_ranges()
            self._apply_current_placement()
            self.update_foundation_profile()
            self.update_3d_view()
                
            QMessageBox.information(
                self,
                "Load Successful",
                f"Loaded georeferenced map:\n"
                f"  Size: {H}×{W} cells ({meters_h:.1f}×{meters_w:.1f}m)\n"
                f"  Resolution: {resolution} m/cell\n"
                f"  Elevation range: {a_min:.2f}m to {a_max:.2f}m\n"
                f"  Resized to: {out_h}×{out_w} tiles"
            )
                
        except Exception as exc:
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load georeferenced map:\n{exc}\n\n"
                f"Error type: {exc.__class__.__name__}\n\n"
                f"Full traceback:\n{error_details}"
            )

    def on_load_pcl(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open point cloud (.npy Nx3 or .csv)",
            os.getcwd(),
            "PointCloud (*.npy *.csv *.txt)"
        )
        if not path:
            return
        try:
            pts = load_pointcloud(path)
            canvas, _ = rasterize_pointcloud(pts, self.meters_per_tile)
            self._store_base_canvas(canvas, apply_rotation=False)
            self._apply_rotation_to_bases()
            canvas_shape = self.last_pcl_canvas.shape if self.last_pcl_canvas is not None else canvas.shape
            self._set_default_offsets(canvas_shape)
            self._update_offset_ranges()
            self._apply_current_placement()
        except Exception as exc:
            QMessageBox.critical(self, "PCL Load Error", f"Failed to load point cloud:\n{exc}")

    # ----- Placement application -----
    def _apply_current_placement(self) -> None:
        # Use desired_elevation or previous_desired_elevation canvas if toggle is on
        canvas_to_use = None
        if (hasattr(self, 'chk_show_desired_elevation') and 
            self.chk_show_desired_elevation.isChecked()):
            # Prefer previous_desired_elevation if it exists (from loaded bag), otherwise use desired_elevation
            if self.previous_desired_elevation_canvas is not None:
                canvas_to_use = self.previous_desired_elevation_canvas
            elif self.desired_elevation_canvas is not None:
                canvas_to_use = self.desired_elevation_canvas
        elif self.last_pcl_canvas is not None:
            canvas_to_use = self.last_pcl_canvas
        
        if canvas_to_use is None:
            return
        h, w = canvas_to_use.shape
        self._current_canvas_shape = (h, w)
        mode_text = self.placement_combo.currentText().strip().lower()
        if "top" in mode_text and "left" in mode_text:
            mode = "topleft"
        elif "center" in mode_text:
            mode = "center"
        else:
            mode = "topleft"  # Default to topleft
        raw_offx = getattr(self, 'offset_x', 0)
        raw_offy = getattr(self, 'offset_y', 0)
        base_offx = int(np.floor(w / 2.0 - self.grid_size / 2.0))
        base_offy = int(np.floor(h / 2.0 - self.grid_size / 2.0))
        offx = raw_offx - base_offx
        offy = raw_offy - base_offy
        print(f"_apply_current_placement: mode_text='{self.placement_combo.currentText()}' -> mode='{mode}', raw_off=({raw_offx},{raw_offy}), eff_off=({offx},{offy})")
        # Debug: report placement parameters
        try:
            # Compute placement start consistent with apply_placement()
            if mode == "topleft":
                start_x_dbg = int(np.floor(w / 2.0 - self.grid_size / 2.0) + offx)
                start_y_dbg = int(np.floor(h / 2.0 - self.grid_size / 2.0) + offy)
            elif mode == "center":
                start_x_dbg = int(offx + w / 2)
                start_y_dbg = int(offy + h / 2)
            else:
                start_x_dbg = int(offx)
                start_y_dbg = int(offy)
            print(f"Placement: mode={mode} off=({offx},{offy}) src={w}x{h} grid={self.grid_size} start=({start_x_dbg},{start_y_dbg})")
        except Exception:
            pass
        finite_vals = canvas_to_use[np.isfinite(canvas_to_use)]
        fill_val = float(finite_vals.min()) if finite_vals.size > 0 else 0.0
        placed = apply_placement(canvas_to_use, self.grid_size, mode, offx, offy, fill_val)
        # Debug: verify first element mapping
        if mode == "topleft" and offx == 0 and offy == 0:
            print(f"_apply_current_placement: canvas[0,0]={canvas_to_use[0,0]}, placed[0,0]={placed[0,0]}, match={placed[0,0]==canvas_to_use[0,0]}")
            print(f"  placed array range: min={placed.min()}, max={placed.max()}, unique_values={len(np.unique(placed))}")
        self.last_placed_elevation = placed
        
        # Store placement parameters for exact reverse mapping in export
        if mode == "topleft":
            actual_start_x = int(np.floor(w / 2.0 - self.grid_size / 2.0) + offx)
            actual_start_y = int(np.floor(h / 2.0 - self.grid_size / 2.0) + offy)
        elif mode == "center":
            actual_start_x = int(offx + w / 2)
            actual_start_y = int(offy + h / 2)
        else:
            actual_start_x = int(offx)
            actual_start_y = int(offy)
        self.placement_params = {
            'canvas_h': h,
            'canvas_w': w,
            'mode': mode,
            'offset_x': offx,
            'offset_y': offy,
            'start_x': actual_start_x,
            'start_y': actual_start_y,
            'grid_size': self.grid_size
        }
        print(f"_apply_current_placement: stored placement params: canvas={w}x{h}, mode={mode}, start=({actual_start_x},{actual_start_y})")
        norm = self._normalize01(placed)
        # Debug: verify normalized first element
        if mode == "topleft" and offx == 0 and offy == 0:
            print(f"_apply_current_placement: norm[0,0]={norm[0,0]}, norm range: min={norm.min()}, max={norm.max()}")
        self.scene.set_background_from_array(norm)
        # Draw/update marker for GridMap center (info.pose.position)
        try:
            # Center of source in its own indices
            src_center_x = int(round(w / 2.0))
            src_center_y = int(round(h / 2.0))
            # Find where that lands in the output grid
            grid_j = int(round(src_center_x - start_x_dbg))  # column (x)
            grid_i = int(round(src_center_y - start_y_dbg))  # row (y)
            self._draw_map_center_marker(grid_j, grid_i)
        except Exception:
            pass
        self.update_foundation_profile()
        self.update_3d_view()

    def _set_default_offsets(self, canvas_shape: Tuple[int, int]) -> None:
        """Initialize offset spin boxes so the source center aligns with grid center."""
        self._current_canvas_shape = canvas_shape
        h, w = canvas_shape
        self.offset_x = int(np.floor(w / 2.0 - self.grid_size / 2.0))
        self.offset_y = int(np.floor(h / 2.0 - self.grid_size / 2.0))
        if hasattr(self, 'offset_x_spin'):
            self.offset_x_spin.blockSignals(True)
            self.offset_x_spin.setValue(self.offset_x)
            self.offset_x_spin.blockSignals(False)
        if hasattr(self, 'offset_y_spin'):
            self.offset_y_spin.blockSignals(True)
            self.offset_y_spin.setValue(self.offset_y)
            self.offset_y_spin.blockSignals(False)

    def on_rotation_changed(self, value: float) -> None:
        """Handle rotation adjustments (degrees)."""
        self.rotation_deg = float(value)
        self._apply_rotation_to_bases()
        self._apply_current_placement()
        self.update_foundation_profile()
        self.update_3d_view()

    def _draw_map_center_marker(self, grid_col: int, grid_row: int) -> None:
        # Remove existing marker items if any
        try:
            for it in self._map_center_marker_items:
                try:
                    self.scene.removeItem(it)
                except Exception:
                    pass
        except Exception:
            pass
        self._map_center_marker_items = []
        # Validate within grid bounds
        if grid_col < 0 or grid_row < 0 or grid_col >= self.grid_size or grid_row >= self.grid_size:
            return
        # Scene position at cell center
        cx = grid_col * CELL_SIZE + CELL_SIZE * 0.5
        cy = grid_row * CELL_SIZE + CELL_SIZE * 0.5
        # Draw cyan circle and crosshair
        radius = max(4.0, CELL_SIZE * 0.25)
        pen = QPen(QColor(0, 200, 255))
        pen.setWidth(2)
        circ = self.scene.addEllipse(cx - radius, cy - radius, radius * 2.0, radius * 2.0, pen, QBrush(Qt.NoBrush))
        circ.setZValue(3.0)
        line_h = self.scene.addLine(cx - radius * 1.2, cy, cx + radius * 1.2, cy, pen)
        line_h.setZValue(3.0)
        line_v = self.scene.addLine(cx, cy - radius * 1.2, cx, cy + radius * 1.2, pen)
        line_v.setZValue(3.0)
        self._map_center_marker_items.extend([circ, line_h, line_v])

    def update_foundation_profile(self) -> None:
        # Clear scenes
        if hasattr(self, 'profile_scene_x'):
            self.profile_scene_x.clear()
        if hasattr(self, 'profile_scene_y'):
            self.profile_scene_y.clear()
        self.lbl_min.setText("Min dig: -")
        self.lbl_max.setText("Max dig: -")
        if self.last_placed_elevation is None:
            return
        # Use all painted foundation tiles rather than last rectangle only
        mask = self.scene.foundation_mask
        if mask is None or mask.sum() == 0:
            return
        yy, xx = np.where(mask == 1)
        ymin, ymax = int(yy.min()), int(yy.max())
        xmin, xmax = int(xx.min()), int(xx.max())
        sub_elev = self.last_placed_elevation[ymin:ymax+1, xmin:xmax+1]
        sub_mask = mask[ymin:ymax+1, xmin:xmax+1].astype(bool)
        if sub_elev.size == 0:
            return
        default_depth = float(self.depth_spin.value())
        surface_source = getattr(self.scene, 'foundation_original_elevation', None)
        if surface_source is None or surface_source.shape != self.last_placed_elevation.shape:
            surface_source = self.last_placed_elevation
        surface_sub = surface_source[ymin:ymax+1, xmin:xmax+1]
        max_in_mask = float(surface_sub[sub_mask].max()) if sub_mask.any() else float(surface_sub.max())

        depth_map_scene = getattr(self.scene, 'foundation_depth_map', None)
        depth_values = []

        def _rel_depth_from_value(stored_val: float, gx: int, gy: int) -> float:
            if abs(stored_val) <= 1e-6:
                return default_depth
            if stored_val > 100 and surface_source is not None:
                surface_z = surface_source[gy, gx]
                return max(0.0, surface_z - stored_val)
            return max(0.0, float(stored_val))

        if depth_map_scene is not None:
            for gx, gy in zip(xx, yy):
                if 0 <= gy < depth_map_scene.shape[0] and 0 <= gx < depth_map_scene.shape[1]:
                    val = float(depth_map_scene[gy, gx])
                    depth_values.append(_rel_depth_from_value(val, gx, gy))
        if not depth_values:
            depth_values = [default_depth] * len(yy)

        min_depth = min(depth_values) if depth_values else 0.0
        max_depth = max(depth_values) if depth_values else 0.0
        self.lbl_min.setText(f"Min dig: {min_depth:.2f} m")
        self.lbl_max.setText(f"Max dig: {max_depth:.2f} m")

        # Use deepest depth when drawing flat bottom reference
        bottom_ref_depth = max_depth if depth_values else default_depth
        bottom_height = max_in_mask - bottom_ref_depth
        depth_label_value = bottom_ref_depth
        def compute_profile_x(elev: np.ndarray, msk: np.ndarray) -> np.ndarray:
            vals = []
            for c in range(elev.shape[1]):
                col_vals = elev[:, c]
                col_mask = msk[:, c]
                vals.append(float(col_vals[col_mask].mean()) if col_mask.any() else np.nan)
            return np.array(vals, dtype=float)

        def compute_profile_y(elev: np.ndarray, msk: np.ndarray) -> np.ndarray:
            vals = []
            for r in range(elev.shape[0]):
                row_vals = elev[r, :]
                row_mask = msk[r, :]
                vals.append(float(row_vals[row_mask].mean()) if row_mask.any() else np.nan)
            return np.array(vals, dtype=float)

        def fill_profile_nan(profile_arr: np.ndarray) -> np.ndarray:
            prof = profile_arr.copy()
            if np.isnan(prof).all():
                return prof
            idx = np.where(~np.isnan(prof))[0]
            first, last = idx[0], idx[-1]
            prof[:first] = prof[first]
            prof[last+1:] = prof[last]
            for i in range(first+1, last):
                if np.isnan(prof[i]):
                    j = i+1
                    while j <= last and np.isnan(prof[j]):
                        j += 1
                    prof[i:j] = np.linspace(prof[i-1], prof[j], j - i + 1)[1:]
            return prof

        def draw_profile(scene: QGraphicsScene, profile_arr: np.ndarray, title_text: str) -> None:
            scene.clear()
            profile_local = fill_profile_nan(profile_arr)
            if np.isnan(profile_local).all():
                return
            w = profile_local.shape[0]
            view_w, view_h = 300, 180
            margin = 20
            y_min = min(float(np.nanmin(profile_local)), bottom_height)
            y_max = max(float(np.nanmax(profile_local)), bottom_height)
            if y_max - y_min < 1e-6:
                y_max = y_min + 1.0
            def x_to_px(i: int) -> float:
                return margin + (view_w - 2*margin) * (i / max(w-1, 1))
            def y_to_px(val: float) -> float:
                t = (val - y_min) / (y_max - y_min)
                return view_h - margin - t * (view_h - 2*margin)
            # Title
            title_item = scene.addSimpleText(title_text)
            title_item.setBrush(QBrush(QColor(30, 30, 30)))
            title_item.setPos(margin, 2)
            # Axes
            pen_axis = QPen(QColor(200, 200, 200))
            scene.addLine(margin, margin, margin, view_h - margin, pen_axis)
            scene.addLine(margin, view_h - margin, view_w - margin, view_h - margin, pen_axis)
            # Bottom line
            pen_bottom = QPen(QColor(0, 140, 255))
            pen_bottom.setStyle(Qt.DashLine)
            yb = y_to_px(bottom_height)
            scene.addLine(margin, yb, view_w - margin, yb, pen_bottom)
            # Text annotations: Max/Min/Depth/Bottom
            annot_brush = QBrush(QColor(50, 50, 50))
            t_max = scene.addSimpleText(f"Max elev: {y_max:.2f} m")
            t_max.setBrush(annot_brush)
            t_max.setPos(margin + 4, margin + 2)
            t_min = scene.addSimpleText(f"Min elev: {y_min:.2f} m")
            t_min.setBrush(annot_brush)
            t_min_rect = t_min.boundingRect()
            t_min.setPos(margin + 4, view_h - margin - t_min_rect.height() - 2)
            t_depth = scene.addSimpleText(f"Depth: {depth_label_value:.2f} m")
            t_depth.setBrush(annot_brush)
            t_depth_rect = t_depth.boundingRect()
            t_depth.setPos(view_w - margin - t_depth_rect.width() - 4, margin + 2)
            t_bottom = scene.addSimpleText(f"Bottom: {bottom_height:.2f} m")
            t_bottom.setBrush(QBrush(QColor(20, 80, 160)))
            t_bottom_rect = t_bottom.boundingRect()
            by = yb - t_bottom_rect.height() - 2
            if by < margin:
                by = yb + 2
            t_bottom.setPos(view_w - margin - t_bottom_rect.width() - 4, by)
            # Profile polyline
            pen_prof = QPen(QColor(60, 60, 60))
            last_x = x_to_px(0)
            last_y = y_to_px(float(profile_local[0]))
            for i in range(1, w):
                x = x_to_px(i)
                y = y_to_px(float(profile_local[i]))
                scene.addLine(last_x, last_y, x, y, pen_prof)
                last_x, last_y = x, y
            # Shade dig area
            brush_shade = QBrush(QColor(255, 0, 0, 60))
            for i in range(w-1):
                x1 = x_to_px(i)
                x2 = x_to_px(i+1)
                y1 = y_to_px(float(max(profile_local[i], bottom_height)))
                yb1 = y_to_px(bottom_height)
                y2 = y_to_px(float(max(profile_local[i+1], bottom_height)))
                yb2 = y_to_px(bottom_height)
                poly = scene.addPolygon(QPolygonF([QPointF(x1, y1), QPointF(x2, y2), QPointF(x2, yb2), QPointF(x1, yb1)]))
                poly.setBrush(brush_shade)
                poly.setPen(QPen(Qt.NoPen))

        # Compute and draw X and Y profiles
        prof_x = compute_profile_x(sub_elev, sub_mask)
        prof_y = compute_profile_y(sub_elev, sub_mask)
        draw_profile(self.profile_scene_x, prof_x, "Profile X (columns)")
        draw_profile(self.profile_scene_y, prof_y, "Profile Y (rows)")

    def flatten_dig_plane(self) -> None:
        """Flatten the selected foundation floor to the maximum dig depth (deepest bottom)."""
        if self.last_placed_elevation is None:
            return
        mask = getattr(self.scene, 'foundation_mask', None)
        if mask is None:
            return
        selected_group = self.scene.selected_foundation_group
        if selected_group is not None and selected_group.get('cells'):
            cells = [(x, y) for (x, y) in selected_group['cells']
                     if 0 <= x < self.scene.grid_size and 0 <= y < self.scene.grid_size]
        else:
            yy, xx = np.where(mask == 1)
            cells = list(zip(xx.tolist(), yy.tolist()))
        if not cells:
            return

        if self.scene.foundation_depth_map is None:
            self.scene.foundation_depth_map = np.zeros((self.scene.grid_size, self.scene.grid_size), dtype=np.float32)

        default_depth = float(self.depth_spin.value())
        surface_source = getattr(self.scene, 'foundation_original_elevation', None)
        if surface_source is None or surface_source.shape != self.last_placed_elevation.shape:
            surface_source = self.last_placed_elevation

        def _relative_depth_for_cell(x: int, y: int) -> float:
            """Return relative depth (meters) for cell."""
            candidate = None
            if selected_group is not None:
                group_depth_map = selected_group.get('depth_map')
                if group_depth_map is not None and (x, y) in group_depth_map:
                    candidate = group_depth_map[(x, y)]
            if candidate is None and self.scene.foundation_depth_map is not None:
                candidate = self.scene.foundation_depth_map[y, x]
            if candidate is None or abs(candidate) <= 1e-6:
                return default_depth
            if candidate > 100 and surface_source is not None:
                surface_z = surface_source[y, x]
                return max(0.0, surface_z - candidate)
            return float(candidate)
        bottom_heights = {}
        for x, y in cells:
            if not (0 <= x < self.scene.grid_size and 0 <= y < self.scene.grid_size):
                continue
            rel_depth = _relative_depth_for_cell(x, y)
            surface_z = surface_source[y, x] if surface_source is not None else self.last_placed_elevation[y, x]
            bottom_heights[(x, y)] = surface_z - rel_depth

        if not bottom_heights:
            return

        target_bottom = max(bottom_heights.values())

        cells_set = set(bottom_heights.keys())

        for (x, y) in cells_set:
            self.scene.foundation_mask[y, x] = 1
            if surface_source is not None:
                surface_z = surface_source[y, x]
            else:
                surface_z = self.last_placed_elevation[y, x]
            target_depth = max(0.0, surface_z - target_bottom)
            self.scene.foundation_depth_map[y, x] = target_depth
        print(f"Flatten: set {len(cells_set)} cells to bottom {target_bottom:.3f} m (absolute)")
        if hasattr(self, 'desired_elevation_canvas') and self.desired_elevation_canvas is not None:
            des_canvas = self.desired_elevation_canvas
            updated = 0
            for (x, y) in cells_set:
                if 0 <= y < des_canvas.shape[0] and 0 <= x < des_canvas.shape[1]:
                    des_canvas[y, x] = target_bottom
                    updated += 1
            print(f"  Flatten verification: updated desired canvas cells={updated}")

        def _update_group_maps(group: dict) -> None:
            if group is None:
                return
            group_cells = group.get('cells', [])
            if not group_cells:
                return
            group_map = group.get('depth_map')
            if group_map is None:
                group_map = {}
                group['depth_map'] = group_map
            abs_map = group.get('absolute_z')
            for cell in group_cells:
                if cell not in cells_set:
                    continue
                x, y = cell
                surface_z = surface_source[y, x] if surface_source is not None else self.last_placed_elevation[y, x]
                dive_depth = max(0.0, surface_z - target_bottom)
                group_map[cell] = dive_depth
                if abs_map is not None:
                    abs_map[cell] = target_bottom

        if selected_group is not None:
            _update_group_maps(selected_group)
        else:
            for group in getattr(self.scene, 'foundation_groups', []):
                if not group or not group.get('cells'):
                    continue
                group_cell_set = set(group['cells'])
                if group_cell_set & cells_set:
                    _update_group_maps(group)

        # Refresh displays
        if callable(getattr(self.scene, 'on_mask_changed', None)):
            self.scene.on_mask_changed()
        else:
            self.update_foundation_profile()
            self.update_3d_view()

        if selected_group is not None:
            self.on_group_selected(selected_group)

    # ----- Export -----
    def on_export(self) -> None:
        out_dir = QFileDialog.getExistingDirectory(self, "Select export folder", os.getcwd())
        if not out_dir:
            return
        try:
            target = self.export_target_combo.currentText().lower()
            if target.startswith("terra"):
                self._export_to_terra(out_dir)
            else:
                # ROS1 or ROS2 export
                self._export_to_isaac(out_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{exc}")

    def _export_to_terra(self, out_dir: str) -> None:
        # Create Terra folder structure: map/images/, map/occupancy/, map/dumpability/, map/actions/
        map_dir = os.path.join(out_dir, "map")
        images_dir = os.path.join(map_dir, "images")
        occupancy_dir = os.path.join(map_dir, "occupancy")
        dumpability_dir = os.path.join(map_dir, "dumpability")
        actions_dir = os.path.join(map_dir, "actions")
        metadata_dir = os.path.join(map_dir, "metadata")
        
        for d in [images_dir, occupancy_dir, dumpability_dir, actions_dir, metadata_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Terra color conventions from color_dict (RGB)
        terra_colors = {
            "neutral": [220, 220, 220],  # Light Gray
            "digging": [255, 255, 255],  # White
            "dumping": [90, 191, 20],  # Green 
            "nondumpable": [255, 0, 0],  # Red
            "obstacle": [0, 0, 255],  # Blue
            "dirt": [19, 69, 139],  # Brown
        }
        
        H, W = self.grid_size, self.grid_size
        
        # 1. Occupancy: obstacles (blue) on neutral background
        occ_img = np.full((H, W, 3), terra_colors["neutral"], dtype=np.uint8)  # neutral gray background
        occ_mask = self.scene.obstacle_mask.astype(bool)
        occ_img[occ_mask] = terra_colors["obstacle"]  # blue obstacles
        occ_img_rgb = Image.fromarray(occ_img, 'RGB')
        occ_img_rgb.save(os.path.join(occupancy_dir, "map.png"))
        
        # 2. Dumpability: non-dumpable areas (red) on neutral background
        dmp_img = np.full((H, W, 3), terra_colors["neutral"], dtype=np.uint8)  # neutral gray background
        dmp_mask = self.scene.nodump_mask.astype(bool)
        dmp_img[dmp_mask] = terra_colors["nondumpable"]  # red non-dumpable
        dmp_img_rgb = Image.fromarray(dmp_img, 'RGB')
        dmp_img_rgb.save(os.path.join(dumpability_dir, "map.png"))
        
        # 3. Images: foundation (white/digging) and dump zones (green) on neutral background
        img_terra = np.full((H, W, 3), terra_colors["neutral"], dtype=np.uint8)  # neutral gray background
        foundation_mask = self.scene.foundation_mask.astype(bool)
        dump_mask = self.scene.dump_mask.astype(bool)
        img_terra[foundation_mask] = terra_colors["digging"]  # white for foundations/digging
        img_terra[dump_mask] = terra_colors["dumping"]  # green for dump zones
        img_terra_pil = Image.fromarray(img_terra, 'RGB')
        img_terra_pil.save(os.path.join(images_dir, "map.png"))
        
        # 4. Actions: neutral background (dirt would be brown, but empty for now)
        action_img = np.full((H, W, 3), terra_colors["neutral"], dtype=np.uint8)  # neutral gray background
        action_img_rgb = Image.fromarray(action_img, 'RGB')
        action_img_rgb.save(os.path.join(actions_dir, "map.png"))
        
        # Also export Terra arrays (.npy) matching converter semantics
        # images → int8 in {-1,0,1}: foundation=-1, dump=1, neutral/other=0
        img_terra_arr = np.zeros((H, W), dtype=np.int8)
        img_terra_arr[foundation_mask] = -1
        img_terra_arr[dump_mask] = 1
        np.save(os.path.join(images_dir, "img_1.npy"), img_terra_arr)

        # occupancy → bool: obstacle True, else False
        occ_bool = occ_mask.astype(np.bool_)
        np.save(os.path.join(occupancy_dir, "img_1.npy"), occ_bool)

        # dumpability → bool: dumpable True, nondump False
        dumpable_bool = (~dmp_mask).astype(np.bool_)
        np.save(os.path.join(dumpability_dir, "img_1.npy"), dumpable_bool)

        # actions → int8, currently zeros
        actions_arr = np.zeros((H, W), dtype=np.int8)
        np.save(os.path.join(actions_dir, "img_1.npy"), actions_arr)

        # 5. Metadata
        meta = {
            "target": "terra",
            "grid_size": int(self.grid_size),
            "meters_per_tile": float(self.meters_per_tile),
        }
        with open(os.path.join(metadata_dir, "map.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        
        QMessageBox.information(self, "Export", f"Exported Terra files to:\n{os.path.join(out_dir, 'map')}")

    def _export_to_isaac(self, out_dir: str) -> None:
        # Export modified elevation as npy and create new bag file
        if self.last_placed_elevation is None:
            QMessageBox.warning(self, "Export", "No elevation grid available to export.")
            return

        grid_region_mask = None
        grid_region_bounds = None
        grid_region_array = None

        # Use original elevation array if available (for bag file export), otherwise use resized
        # Keep a copy of the original elevation for desired_elevation calculation
        original_elev_for_desired = None
        if hasattr(self, '_base_original_elevation_array') and self._base_original_elevation_array is not None:
            elev = self._base_original_elevation_array.copy()
            original_elev_for_desired = self._base_original_elevation_array.copy()  # Keep original for desired_elevation
            orig_H, orig_W = elev.shape
            # Get resized canvas shape (before placement)
            # Use the same canvas that was used in apply_placement
            # Check desired_elevation_canvas first (if it was used), then last_pcl_canvas
            if hasattr(self, 'desired_elevation_canvas') and self.desired_elevation_canvas is not None:
                canvas_to_use_export = self.desired_elevation_canvas
            elif hasattr(self, 'last_pcl_canvas') and self.last_pcl_canvas is not None:
                canvas_to_use_export = self.last_pcl_canvas
            else:
                # Fallback - this shouldn't happen if map was loaded correctly
                canvas_to_use_export = self.last_placed_elevation
                print(f"  WARNING: Using last_placed_elevation as canvas (should be resized canvas)")
            
            resized_H, resized_W = canvas_to_use_export.shape
            print(f"  Export canvas: {resized_H}×{resized_W} (desired_elev={hasattr(self, 'desired_elevation_canvas') and self.desired_elevation_canvas is not None}, last_pcl={hasattr(self, 'last_pcl_canvas') and self.last_pcl_canvas is not None})")
            
            # Determine base (unrotated) canvas size
            base_canvas_H = resized_H
            base_canvas_W = resized_W
            if hasattr(self, '_base_desired_canvas') and self._base_desired_canvas is not None:
                base_canvas_H, base_canvas_W = self._base_desired_canvas.shape
            elif hasattr(self, '_base_canvas') and self._base_canvas is not None:
                base_canvas_H, base_canvas_W = self._base_canvas.shape
            print(f"  Base canvas size: {base_canvas_H}×{base_canvas_W}, Rotated canvas size: {resized_H}×{resized_W}")
            
            # Calculate mapping ratio (use base canvas size)
            ratio_y = orig_H / base_canvas_H
            ratio_x = orig_W / base_canvas_W
            cells_per_tile = ratio_y * ratio_x
            
            foundation_mask_placed = self.scene.foundation_mask.astype(bool)
            placed_H, placed_W = foundation_mask_placed.shape  # This is grid_size x grid_size
            num_placed_tiles = np.count_nonzero(foundation_mask_placed)
            occupancy_array = None
            dump_zone_array = None
            print(f"Export: Mapping {num_placed_tiles} placed tiles ({placed_H}×{placed_W}) from resized canvas ({resized_H}×{resized_W}) to original grid ({orig_H}×{orig_W})")
            
            # Use stored placement parameters if available (exact values used during placement)
            if hasattr(self, 'placement_params') and self.placement_params is not None:
                params = self.placement_params
                mode = params['mode']
                offx = params['offset_x']
                offy = params['offset_y']
                start_x = params['start_x']
                start_y = params['start_y']
                stored_canvas_w = params['canvas_w']
                stored_canvas_h = params['canvas_h']
                print(f"  Using stored placement params: canvas={stored_canvas_w}×{stored_canvas_h}, mode={mode}, start=({start_x},{start_y})")
                # Verify canvas dimensions match
                if stored_canvas_w != resized_W or stored_canvas_h != resized_H:
                    print(f"  WARNING: Canvas dimensions mismatch! Stored: {stored_canvas_w}×{stored_canvas_h}, Current: {resized_W}×{resized_H}")
                # Debug: also calculate what it would be with current canvas to compare
                if mode == "topleft":
                    calc_start_x = int(np.floor(resized_W / 2.0 - placed_W / 2.0) + offx)
                    calc_start_y = int(np.floor(resized_H / 2.0 - placed_H / 2.0) + offy)
                    print(f"  Comparison: stored start=({start_x},{start_y}), calculated with current canvas=({calc_start_x},{calc_start_y}), diff=({start_x-calc_start_x},{start_y-calc_start_y})")
                    # Test: verify what apply_placement would calculate with stored canvas dimensions
                    test_start_x = int(np.floor(stored_canvas_w / 2.0 - placed_W / 2.0) + offx)
                    test_start_y = int(np.floor(stored_canvas_h / 2.0 - placed_H / 2.0) + offy)
                    print(f"  Verification: apply_placement with stored canvas would give start=({test_start_x},{test_start_y}), stored start=({start_x},{start_y}), match=({start_x==test_start_x},{start_y==test_start_y})")
            else:
                # Fallback: recalculate (shouldn't happen if placement was done)
                print(f"  WARNING: No stored placement params, recalculating...")
                mode_text = getattr(self, 'placement_combo', None)
                if mode_text is not None:
                    mode_text = mode_text.currentText().lower()
                else:
                    mode_text = "topleft"
                
                if "center" in mode_text:
                    mode = "center"
                else:
                    mode = "topleft"  # Default
                
                offx = getattr(self, 'offset_x', 0)
                offy = getattr(self, 'offset_y', 0)
                
                # Calculate placement start (same as apply_placement function)
                if mode == "topleft":
                    start_x = int(np.floor(resized_W / 2.0 - placed_W / 2.0) + offx)
                    start_y = int(np.floor(resized_H / 2.0 - placed_H / 2.0) + offy)
                elif mode == "center":
                    start_x = int(offx + resized_W / 2)
                    start_y = int(offy + resized_H / 2)
                else:
                    start_x = int(offx)
                    start_y = int(offy)
                print(f"  Calculated placement: mode={mode}, offset=({offx},{offy}), start=({start_x},{start_y})")
            
            # Get mpt and resolution for debug output
            mpt = float(self.meters_per_tile)
            orig_resolution = getattr(self, 'original_gridmap_resolution', None)
            if orig_resolution is None:
                orig_resolution = 0.1  # Default fallback
            
            # Debug: verify mapping for cell (0,0) - should map to canvas (start_y, start_x)
            if foundation_mask_placed.size > 0:
                # Find first foundation cell
                test_cy, test_cx = None, None
                for cy in range(min(5, placed_H)):
                    for cx in range(min(5, placed_W)):
                        if foundation_mask_placed[cy, cx]:
                            test_cy, test_cx = cy, cx
                            break
                    if test_cy is not None:
                        break
                if test_cy is not None:
                    test_resized_cy = start_y + test_cy
                    test_resized_cx = start_x + test_cx
                    print(f"  Debug mapping: placed[{test_cy},{test_cx}] -> canvas[{test_resized_cy},{test_resized_cx}] (start_y={start_y}, start_x={start_x})")
                    # Also check what it maps to in original grid
                    test_ox = int(round((test_resized_cy * mpt) / orig_resolution))
                    test_oy = int(round((test_resized_cx * mpt) / orig_resolution))
                    print(f"  Debug mapping: canvas[{test_resized_cy},{test_resized_cx}] -> original[{test_oy},{test_ox}] (mpt={mpt}, resolution={orig_resolution})")
            
            def _center_crop_or_pad(arr: np.ndarray, target_h: int, target_w: int, fill_value: float = 0.0) -> np.ndarray:
                arr_h, arr_w = arr.shape
                if arr_h == target_h and arr_w == target_w:
                    return arr
                result = np.full((target_h, target_w), fill_value, dtype=arr.dtype)
                # Determine overlap region
                src_y = max((arr_h - target_h) // 2, 0)
                src_x = max((arr_w - target_w) // 2, 0)
                dst_y = max((target_h - arr_h) // 2, 0)
                dst_x = max((target_w - arr_w) // 2, 0)
                copy_h = min(arr_h, target_h)
                copy_w = min(arr_w, target_w)
                result[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = arr[src_y:src_y + copy_h, src_x:src_x + copy_w]
                return result

            rotation_deg = getattr(self, 'rotation_deg', 0.0)
            transpose_bg = getattr(self.scene, 'transpose_background', True)

            def map_mask_to_canvas_then_original(mask_placed, label):
                if mask_placed is None:
                    print(f"  NOTE: No {label} mask available to map.")
                    return np.zeros((orig_H, orig_W), dtype=bool), 0
                mask_bool = np.asarray(mask_placed, dtype=bool)
                if mask_bool.shape != (placed_H, placed_W):
                    print(f"  WARNING: {label} mask shape {mask_bool.shape} != placed grid {(placed_H, placed_W)}; skipping mapping.")
                    return np.zeros((orig_H, orig_W), dtype=bool), 0
                canvas_mask_rot = np.zeros((resized_H, resized_W), dtype=bool)
                for cy in range(placed_H):
                    for cx in range(placed_W):
                        if not mask_bool[cy, cx]:
                            continue
                        # Account for optional display transpose (arr.T in GridScene)
                        if transpose_bg:
                            resized_cy = start_y + cx
                            resized_cx = start_x + cy
                        else:
                            resized_cy = start_y + cy
                            resized_cx = start_x + cx
                        if 0 <= resized_cy < resized_H and 0 <= resized_cx < resized_W:
                            canvas_mask_rot[resized_cy, resized_cx] = True
                canvas_mask = canvas_mask_rot
                if abs(rotation_deg) > 1e-6:
                    try:
                        mask_img = Image.fromarray((canvas_mask_rot.astype(np.float32) * 255.0).astype(np.uint8), mode='L')
                        unrot = mask_img.rotate(-rotation_deg, resample=Image.BILINEAR, expand=True)
                        unrot_arr = (np.array(unrot, dtype=np.float32) / 255.0) > 0.5
                        canvas_mask = _center_crop_or_pad(unrot_arr.astype(bool), base_canvas_H, base_canvas_W, False)
                    except Exception as e:
                        print(f"  WARNING: Failed to unrotate {label} mask: {e}")
                        canvas_mask = _center_crop_or_pad(canvas_mask, base_canvas_H, base_canvas_W, False)
                else:
                    if canvas_mask.shape != (base_canvas_H, base_canvas_W):
                        canvas_mask = _center_crop_or_pad(canvas_mask, base_canvas_H, base_canvas_W, False)
                mapped_mask = np.zeros((orig_H, orig_W), dtype=bool)
                mapped_cells = 0
                source_tiles = int(np.count_nonzero(mask_bool))
                for cy in range(base_canvas_H):
                    for cx in range(base_canvas_W):
                        if not canvas_mask[cy, cx]:
                            continue
                        oy_start = int(round(cy * ratio_y))
                        oy_end = int(round((cy + 1) * ratio_y))
                        ox_start = int(round(cx * ratio_x))
                        ox_end = int(round((cx + 1) * ratio_x))
                        if ox_end <= ox_start:
                            ox_end = ox_start + 1
                        if oy_end <= oy_start:
                            oy_end = oy_start + 1
                        oy_start = max(0, min(oy_start, orig_H - 1))
                        oy_end = max(0, min(oy_end, orig_H))
                        ox_start = max(0, min(ox_start, orig_W - 1))
                        ox_end = max(0, min(ox_end, orig_W))
                        mapped_mask[oy_start:oy_end, ox_start:ox_end] = True
                        mapped_cells += (oy_end - oy_start) * (ox_end - ox_start)
                nonzero = int(np.count_nonzero(mapped_mask))
                print(f"  Mapped {mapped_cells} {label} cells to original grid (result non-zero cells: {nonzero}, placed tiles: {source_tiles})")
                if nonzero == 0 and source_tiles > 0:
                    print(f"  WARNING: {label} mask is empty after mapping! This indicates a coordinate mapping issue.")
                return mapped_mask, nonzero
            
            foundation_mask, _ = map_mask_to_canvas_then_original(foundation_mask_placed, "foundation")
            foundation_array = np.zeros((orig_H, orig_W), dtype=np.float32)
            foundation_array[foundation_mask] = 1.0

            # Map full grid region (square workspace) to original coordinates for metadata + layer export
            try:
                full_grid_mask = np.ones_like(foundation_mask_placed, dtype=bool)
                grid_region_mask, _ = map_mask_to_canvas_then_original(full_grid_mask, "grid region")
                if grid_region_mask is not None:
                    grid_region_array = grid_region_mask.astype(np.float32)
                grid_coords = np.argwhere(grid_region_mask)
                if grid_coords.size > 0:
                    row_min = int(grid_coords[:, 0].min())
                    row_max = int(grid_coords[:, 0].max())
                    col_min = int(grid_coords[:, 1].min())
                    col_max = int(grid_coords[:, 1].max())
                    grid_region_bounds = {
                        "row_start": row_min,
                        "row_end": row_max + 1,
                        "col_start": col_min,
                        "col_end": col_max + 1,
                        "row_start_m": float(row_min * orig_resolution),
                        "row_end_m": float((row_max + 1) * orig_resolution),
                        "col_start_m": float(col_min * orig_resolution),
                        "col_end_m": float((col_max + 1) * orig_resolution),
                    }
                    print(f"  Grid region bounds (original grid coords): rows {row_min}-{row_max}, cols {col_min}-{col_max}")
                else:
                    print("  WARNING: Grid region mask empty after mapping.")
            except Exception as grid_exc:
                print(f"  WARNING: Failed to map grid region: {grid_exc}")
            
            # Map obstacle mask to occupancy grid using the same transformation
            obstacle_mask_placed = getattr(self.scene, 'obstacle_mask', None)
            if obstacle_mask_placed is not None:
                obstacle_mask_bool = obstacle_mask_placed.astype(bool)
                obstacle_mask_original, obstacle_original_tiles = map_mask_to_canvas_then_original(obstacle_mask_bool, "obstacle/occupancy")
                occupancy_array = np.zeros((orig_H, orig_W), dtype=np.float32)
                occupancy_array[obstacle_mask_original] = 1.0
                print(f"  Export: occupancy array has {int(np.count_nonzero(occupancy_array))} occupied cells")
            else:
                print("  NOTE: No obstacle mask available; occupancy layer will not be updated.")
            
            # Map dump zone mask using the same transformation
            dump_mask_placed = getattr(self.scene, 'dump_mask', None)
            if dump_mask_placed is not None:
                dump_mask_bool = dump_mask_placed.astype(bool)
                dump_mask_original, dump_original_tiles = map_mask_to_canvas_then_original(dump_mask_bool, "dump zone")
                dump_zone_array = np.zeros((orig_H, orig_W), dtype=np.float32)
                dump_zone_array[dump_mask_original] = 1.0
                print(f"  Export: dump zone array has {int(np.count_nonzero(dump_zone_array))} dump cells")
            else:
                print("  NOTE: No dump mask available; dump zone layer will not be updated.")
            
            # Map depth map from placed grid to original (same logic as foundation mask)
            depth_map_placed = getattr(self.scene, 'foundation_depth_map', None)
            depth_map = None
            if depth_map_placed is not None:
                canvas_depth_rot = np.zeros((resized_H, resized_W), dtype=np.float32)
                for cy in range(placed_H):
                    for cx in range(placed_W):
                        if foundation_mask_placed[cy, cx] and depth_map_placed[cy, cx] != 0:
                            if transpose_bg:
                                resized_cy = start_y + cx
                                resized_cx = start_x + cy
                            else:
                                resized_cy = start_y + cy
                                resized_cx = start_x + cx
                            if 0 <= resized_cy < resized_H and 0 <= resized_cx < resized_W:
                                canvas_depth_rot[resized_cy, resized_cx] = depth_map_placed[cy, cx]
                canvas_depth = canvas_depth_rot
                if abs(rotation_deg) > 1e-6:
                    try:
                        depth_img = Image.fromarray(canvas_depth_rot.astype(np.float32), mode='F')
                        unrot_depth = depth_img.rotate(-rotation_deg, resample=Image.BILINEAR, expand=True)
                        canvas_depth = np.array(unrot_depth, dtype=np.float32)
                    except Exception as e:
                        print(f"  WARNING: Failed to unrotate depth map: {e}")
                        canvas_depth = canvas_depth_rot
                canvas_depth = _center_crop_or_pad(canvas_depth, base_canvas_H, base_canvas_W, 0.0)
                depth_map = np.zeros((orig_H, orig_W), dtype=depth_map_placed.dtype)
                for cy in range(base_canvas_H):
                    for cx in range(base_canvas_W):
                        value = canvas_depth[cy, cx]
                        if abs(value) <= 1e-6:
                            continue
                        oy_start = int(round(cy * ratio_y))
                        oy_end = int(round((cy + 1) * ratio_y))
                        ox_start = int(round(cx * ratio_x))
                        ox_end = int(round((cx + 1) * ratio_x))
                        if ox_end <= ox_start:
                            ox_end = ox_start + 1
                        if oy_end <= oy_start:
                            oy_end = oy_start + 1
                        oy_start = max(0, min(oy_start, orig_H - 1))
                        oy_end = max(0, min(oy_end, orig_H))
                        ox_start = max(0, min(ox_start, orig_W - 1))
                        ox_end = max(0, min(ox_end, orig_W))
                        depth_map[oy_start:oy_end, ox_start:ox_end] = value
        else:
            # Fallback to resized array
            elev = self.last_placed_elevation.copy()
            original_elev_for_desired = self.last_placed_elevation.copy()  # Keep original for desired_elevation
            foundation_mask = self.scene.foundation_mask.astype(bool)
            depth_map = getattr(self.scene, 'foundation_depth_map', None)
            grid_region_mask = np.ones_like(elev, dtype=bool)
            grid_region_array = grid_region_mask.astype(np.float32)
            grid_region_bounds = {
                "row_start": 0,
                "row_end": int(elev.shape[0]),
                "col_start": 0,
                "col_end": int(elev.shape[1]),
                "row_start_m": 0.0,
                "row_end_m": float(elev.shape[0] * getattr(self, 'original_gridmap_resolution', self.meters_per_tile)),
                "col_start_m": 0.0,
                "col_end_m": float(elev.shape[1] * getattr(self, 'original_gridmap_resolution', self.meters_per_tile)),
            }
        
        # Apply foundation excavation: lower heights in foundation regions
        print(f"Export: foundation_mask has {np.count_nonzero(foundation_mask)} cells")
        elev_before = elev.copy()
        if foundation_mask.any():
            # Check if we have a depth map (from STL/OBJ import or drawn tiles)
            # depth_map is already set above (either mapped from resized or from scene)
            has_depth_map = (depth_map is not None and 
                            np.any(np.abs(depth_map[foundation_mask]) > 1e-6))
            print(f"Export: has_depth_map = {has_depth_map}")
            
            # Default depth from spinbox (used if no stored depth for a cell)
            default_depth = float(self.depth_spin.value())
            
            if has_depth_map:
                # Use per-cell depth values
                for cy in range(elev.shape[0]):
                    for cx in range(elev.shape[1]):
                        if foundation_mask[cy, cx]:
                            surface_z = elev[cy, cx]
                            stored_value = depth_map[cy, cx]
                            
                            if abs(stored_value) > 1e-6:
                                # Check if stored value is absolute Z (from STL) or relative depth (from drawn tiles)
                                # Absolute Z values are typically > 100 (elevation in meters)
                                # Relative depth values are typically < 50 (depth in meters)
                                if stored_value > 100:
                                    # Absolute Z from STL import, use directly
                                    elev[cy, cx] = stored_value
                                else:
                                    # Relative depth (meters below surface), calculate bottom
                                    bottom_height = surface_z - stored_value
                                    elev[cy, cx] = bottom_height
                            else:
                                # No stored depth, use default from spinbox
                                bottom_height = surface_z - default_depth
                                elev[cy, cx] = bottom_height
            else:
                # No depth map at all, use uniform depth from spinbox
                # Create a flat bottom by using the maximum elevation in the foundation area
                depth = default_depth
                foundation_elev = elev[foundation_mask]
                if foundation_elev.size > 0:
                    max_in_foundation = float(foundation_elev.max())
                    # Uniform bottom height: highest point minus depth
                    bottom_height = max_in_foundation - depth
                    # Set all foundation cells to the same bottom height (flat bottom)
                    elev[foundation_mask] = bottom_height
            
            # Check if modifications were actually applied
            elev_after = elev.copy()
            diff = np.abs(elev_after - elev_before)
            num_changed = np.count_nonzero(diff > 1e-6)
            print(f"Export: Modified {num_changed} cells in elevation array")
        else:
            print(f"Export: WARNING - foundation_mask is empty, no modifications applied!")

        # Export elevation array as npy file
        # Georeferencing info is already in map_georeference_config.yaml
        elev_npy_path = os.path.join(out_dir, "elevation_modified.npy")
        # Remove existing file if it exists to allow overwriting
        if os.path.exists(elev_npy_path):
            os.remove(elev_npy_path)
        np.save(elev_npy_path, elev.astype(np.float32))

        # Export new bag file if original bag was loaded (skip for npy files)
        bag_path = None
        if self.original_bag_path and self.original_gridmap_msg and HAS_ROSBAGS:
            # Only create bag file if original was a bag or mcap file
            original_ext = os.path.splitext(self.original_bag_path)[1].lower()
            if original_ext in ('.bag', '.mcap'):
                try:
                    # Check rosbags version first
                    try:
                        import importlib.metadata
                        rosbags_version = importlib.metadata.version('rosbags')
                    except:
                        try:
                            import pkg_resources
                            rosbags_version = pkg_resources.get_distribution('rosbags').version
                        except:
                            rosbags_version = 'unknown'
                    
                    print(f"rosbags version: {rosbags_version}")
                    
                    # Determine export format from combo box (default to ROS1 bag)
                    export_format = "ros1_bag"
                    target_text = self.export_target_combo.currentText().lower()
                    if "mcap" in target_text or "ros2" in target_text:
                        export_format = "ros2_mcap"
                    elif "ros1" in target_text or "bag" in target_text:
                        export_format = "ros1_bag"
                    
                    # Try to import writers based on selected format
                    Writer = None
                    use_anywriter = False
                    use_rosbag1 = False
                    use_rosbag2 = False
                    storage_plugin = None  # For rosbag2: StoragePlugin.MCAP or StoragePlugin.SQLITE3
                    
                    if export_format == "ros2_mcap":
                        # For MCAP format, use rosbag2.Writer with MCAP storage plugin
                        try:
                            from rosbags.rosbag2 import Writer, StoragePlugin
                            Writer = Writer
                            use_rosbag2 = True
                            storage_plugin = StoragePlugin.MCAP
                            print("Using rosbag2.Writer with MCAP storage format")
                        except ImportError:
                            # Fallback to AnyWriter if available
                            try:
                                from rosbags.highlevel import AnyWriter
                                Writer = AnyWriter
                                use_anywriter = True
                                print("Using AnyWriter for MCAP (fallback)")
                            except ImportError as e:
                                raise ImportError(
                                    f"MCAP export requires rosbag2.Writer. "
                                    f"Error: {e}"
                                )
                    else:
                        # For ROS1 bag format, prefer rosbag1.Writer
                        try:
                            from rosbags.rosbag1 import Writer
                            use_rosbag1 = True
                            print("Using rosbag1.Writer (single .bag file)")
                        except ImportError:
                            # Try AnyWriter (for rosbags < 0.11)
                            try:
                                from rosbags.highlevel import AnyWriter
                                Writer = AnyWriter
                                use_anywriter = True
                                print("Using AnyWriter from rosbags.highlevel")
                            except ImportError:
                                # Fallback to rosbag2.Writer (creates directory structure)
                                try:
                                    from rosbags.rosbag2 import Writer, StoragePlugin
                                    use_rosbag2 = True
                                    storage_plugin = StoragePlugin.SQLITE3  # Default for rosbag2
                                    print("Using rosbag2.Writer (directory format)")
                                except ImportError as e:
                                    raise ImportError(
                                        f"No writer available in rosbags (version: {rosbags_version}). "
                                        f"Tried: rosbag1.Writer, AnyWriter, rosbag2.Writer. "
                                        f"Error: {e}"
                                    )
                    
                    if Writer is None:
                        raise ImportError(
                            f"No writer found in rosbags (version: {rosbags_version}). "
                            f"Bag file writing requires rosbags with writer support."
                        )
                    
                    from pathlib import Path
                    import time
                    
                    # Create output bag file name with appropriate extension
                    # Strip all extensions to avoid double extensions
                    original_basename = os.path.basename(self.original_bag_path)
                    # Remove common bag extensions (.bag, .mcap, .db3)
                    for ext in ['.bag', '.mcap', '.db3']:
                        if original_basename.lower().endswith(ext):
                            original_basename = original_basename[:-len(ext)]
                            break
                    else:
                        # If no known extension, use splitext to remove any extension
                        original_basename = os.path.splitext(original_basename)[0]
                    
                    if export_format == "ros2_mcap":
                        bag_path = os.path.join(out_dir, f"{original_basename}_modified.mcap")
                    else:
                        bag_path = os.path.join(out_dir, f"{original_basename}_modified.bag")
                    
                    # Remove existing bag file/directory to allow overwriting
                    bag_path_obj = Path(bag_path)
                    if bag_path_obj.exists():
                        if bag_path_obj.is_dir():
                            # rosbag2 creates a directory structure
                            import shutil
                            shutil.rmtree(bag_path)
                        else:
                            # rosbag1 creates a single file
                            os.remove(bag_path)
                    
                    # Get original message structure
                    msg = self.original_gridmap_msg
                    info = getattr(msg, "info", None)
                    layers = list(getattr(msg, "layers", []))
                    
                    # Find desired_elevation layer index (this is what we want to modify)
                    # Keep elevation layer unchanged
                    if "desired_elevation" not in layers:
                        layers.append("desired_elevation")
                        desired_elev_idx = len(layers) - 1
                    else:
                        desired_elev_idx = layers.index("desired_elevation")
                    
                    # Compute desired_elevation from original elevation minus depth
                    # desired_elevation should represent the desired depth (negative for digging)
                    # Start with original elevation (before foundation modifications)
                    if original_elev_for_desired is None:
                        # Fallback: use current elev (shouldn't happen)
                        original_elev_for_desired = elev.copy()
                    
                    desired_elev_array = original_elev_for_desired.copy().astype(np.float32)
                    
                    # Default depth from spinbox
                    default_depth = float(self.depth_spin.value())
                    
                    # Apply foundation depths to desired_elevation
                    if foundation_mask.any():
                        # Check if we have a depth map
                        has_depth_map = (depth_map is not None and 
                                        np.any(np.abs(depth_map[foundation_mask]) > 1e-6))
                        
                        if has_depth_map:
                            # Use per-cell depth values
                            for cy in range(desired_elev_array.shape[0]):
                                for cx in range(desired_elev_array.shape[1]):
                                    if foundation_mask[cy, cx]:
                                        stored_value = depth_map[cy, cx]
                                        if abs(stored_value) > 1e-6:
                                            # Check if stored value is absolute Z or relative depth
                                            if stored_value > 100:
                                                # Absolute Z from STL import
                                                # desired_elevation = absolute Z (already set from original)
                                                pass  # Keep original elevation
                                            else:
                                                # Relative depth (meters below surface)
                                                # desired_elevation = original - depth (negative for digging)
                                                desired_elev_array[cy, cx] = original_elev_for_desired[cy, cx] - stored_value
                                        else:
                                            # No stored depth, use default from spinbox
                                            desired_elev_array[cy, cx] = original_elev_for_desired[cy, cx] - default_depth
                        else:
                            # No depth map, use uniform depth from spinbox for all foundation cells
                            # Create a flat bottom by using the maximum elevation in the foundation area
                            foundation_original_elev = original_elev_for_desired[foundation_mask]
                            if foundation_original_elev.size > 0:
                                max_in_foundation = float(foundation_original_elev.max())
                                # Uniform bottom height: highest point minus depth
                                uniform_bottom_height = max_in_foundation - default_depth
                                # Set all foundation cells to the same bottom height (flat bottom)
                                desired_elev_array[foundation_mask] = uniform_bottom_height
                            else:
                                # Fallback: use per-cell depth (shouldn't happen)
                                desired_elev_array[foundation_mask] = original_elev_for_desired[foundation_mask] - default_depth
                    
                    if foundation_mask.any():
                        foundation_desired_vals = desired_elev_array[foundation_mask]
                        if foundation_desired_vals.size > 0:
                            min_desired = float(np.min(foundation_desired_vals))
                            max_desired = float(np.max(foundation_desired_vals))
                            mean_desired = float(np.mean(foundation_desired_vals))
                            tol = 0.01  # 1 cm tolerance to verify flat regions
                            bottom_count = int(np.count_nonzero(np.isclose(foundation_desired_vals, min_desired, atol=tol)))
                            print(
                                "  Desired elevation (foundation area): "
                                f"min={min_desired:.3f} m, max={max_desired:.3f} m, "
                                f"mean={mean_desired:.3f} m, cells={foundation_desired_vals.size}"
                            )
                            print(
                                f"  Flat bottom verification: {bottom_count} cells within ±{tol:.3f} m of {min_desired:.3f} m"
                            )
                    else:
                        print("  Desired elevation: foundation mask empty, no stats to report.")
                    
                    # Convert to flat array for export
                    desired_elev_flat = desired_elev_array.ravel()
                    
                    # Prepare occupancy payload if available
                    layer_targets = [{
                        'name': "desired_elevation",
                        'idx': desired_elev_idx,
                        'flat': desired_elev_flat,
                        'shape': desired_elev_array.shape
                    }]
                    have_dig = foundation_array is not None and foundation_array.size > 0 and foundation_mask.any()
                    have_dump = dump_zone_array is not None

                    if have_dig:
                        dig_layer_name = "dig_zone"
                        if dig_layer_name not in layers:
                            layers.append(dig_layer_name)
                    if have_dump:
                        if "dump_zone" not in layers:
                            layers.append("dump_zone")
                    if occupancy_array is not None:
                        obstacle_layer_name = "obstacles"
                        if obstacle_layer_name not in layers:
                            layers.append(obstacle_layer_name)
                        obstacle_idx = layers.index(obstacle_layer_name)
                        target_idx = None
                        if have_dig:
                            target_idx = layers.index("dig_zone")
                        if have_dump:
                            dump_idx_tmp = layers.index("dump_zone")
                            target_idx = dump_idx_tmp if target_idx is None else min(target_idx, dump_idx_tmp)
                        if target_idx is not None and obstacle_idx > target_idx:
                            layer_name = layers.pop(obstacle_idx)
                            layers.insert(target_idx, layer_name)
                            obstacle_idx = target_idx
                        obstacle_flat = occupancy_array.astype(np.float32).ravel()
                        layer_targets.append({
                            'name': obstacle_layer_name,
                            'idx': obstacle_idx,
                            'flat': obstacle_flat,
                            'shape': occupancy_array.shape
                        })
                    if have_dig:
                        dig_idx = layers.index("dig_zone")
                        dig_flat = foundation_array.astype(np.float32).ravel()
                        layer_targets.append({
                            'name': "dig_zone",
                            'idx': dig_idx,
                            'flat': dig_flat,
                            'shape': foundation_array.shape
                        })
                    if have_dump:
                        dump_idx = layers.index("dump_zone")
                        dump_flat = dump_zone_array.astype(np.float32).ravel()
                        layer_targets.append({
                            'name': "dump_zone",
                            'idx': dump_idx,
                            'flat': dump_flat,
                            'shape': dump_zone_array.shape
                        })
                    if grid_region_array is not None:
                        grid_region_layer_name = "grid_region_mask"
                        if grid_region_layer_name not in layers:
                            layers.append(grid_region_layer_name)
                        grid_region_idx = layers.index(grid_region_layer_name)
                        grid_region_flat = grid_region_array.astype(np.float32).ravel()
                        layer_targets.append({
                            'name': grid_region_layer_name,
                            'idx': grid_region_idx,
                            'flat': grid_region_flat,
                            'shape': grid_region_array.shape
                        })
                    
                    import copy as copy_module

                    def _clone_layer_structure(template_layer):
                        if template_layer is None:
                            return None
                        try:
                            new_layer = copy_module.deepcopy(template_layer)
                            if hasattr(new_layer, 'data'):
                                # Reset data to empty list to avoid sharing references
                                if isinstance(new_layer.data, np.ndarray):
                                    new_layer.data = np.zeros_like(new_layer.data)
                                else:
                                    new_layer.data = []
                            return new_layer
                        except Exception:
                            return None

                    def build_modified_message():
                        msg_copy = copy_module.deepcopy(msg)
                        if hasattr(msg_copy, 'layers'):
                            msg_copy.layers = layers
                        
                        if hasattr(msg_copy, 'data') and layer_targets:
                            data_list = msg_copy.data
                            elev_idx = layers.index("elevation") if "elevation" in layers else 0
                            elev_layer_data = data_list[elev_idx] if elev_idx < len(data_list) else None
                            
                            max_target_idx = max(t['idx'] for t in layer_targets)
                            while len(data_list) <= max_target_idx:
                                new_layer = _clone_layer_structure(elev_layer_data)
                                if new_layer is None and len(data_list) > 0:
                                    new_layer = _clone_layer_structure(data_list[0])
                                if new_layer is None:
                                    new_layer = np.zeros_like(elev_array)
                                data_list.append(new_layer)
                            
                            for target in layer_targets:
                                layer_idx = target['idx']
                                flat_vals = target['flat']
                                shape = target['shape']
                                if flat_vals is None or layer_idx >= len(data_list):
                                    continue
                                target_layer = data_list[layer_idx]
                                if hasattr(target_layer, 'data'):
                                    target_layer.data = flat_vals
                                    if (hasattr(target_layer, 'layout') and target_layer.layout and
                                            hasattr(target_layer.layout, 'dim') and len(target_layer.layout.dim) >= 2):
                                        target_layer.layout.dim[0].size = shape[0]
                                        target_layer.layout.dim[1].size = shape[1]
                                        if (elev_layer_data is not None and hasattr(elev_layer_data, 'layout') and
                                                elev_layer_data.layout and hasattr(elev_layer_data.layout, 'dim')):
                                            elev_dims = elev_layer_data.layout.dim
                                            for i in range(min(len(target_layer.layout.dim), len(elev_dims))):
                                                if hasattr(elev_dims[i], 'label'):
                                                    target_layer.layout.dim[i].label = elev_dims[i].label
                                                if hasattr(elev_dims[i], 'stride'):
                                                    target_layer.layout.dim[i].stride = elev_dims[i].stride
                                elif isinstance(target_layer, np.ndarray):
                                    data_list[layer_idx] = flat_vals.reshape(shape).astype(np.float32)
                        return msg_copy
                    
                    # Write to new bag file
                    # Get message type and definition from stored connection info
                    conn_info = getattr(self, 'original_gridmap_conn_info', None)
                    if conn_info:
                        msgtype = conn_info.get('msgtype', "grid_map_msgs/msg/GridMap")
                        msgdef = conn_info.get('msgdef', None)
                        md5sum = conn_info.get('md5sum', None)
                        rihs01 = conn_info.get('rihs01', None)
                        typestore = conn_info.get('typestore', None)
                    else:
                        msgtype = "grid_map_msgs/msg/GridMap"  # Default type
                        msgdef = None
                        md5sum = None
                        rihs01 = None
                        typestore = None
                    
                    if use_rosbag1:
                        # rosbag1.Writer API (creates single .bag file)
                        # Keep original message type format (may be ROS2 format even in ROS1 bag)
                        writer = Writer(Path(bag_path))
                        writer.open()
                        try:
                            # Add connection for grid_map topic
                            # rosbag1.Writer requires either typestore OR (msgdef + md5sum) pair
                            # Prefer msgdef + md5sum to avoid typestore.generate_msgdef issues
                            conn_kwargs = {
                                'topic': "grid_map",
                                'msgtype': msgtype
                            }
                            # Use msgdef + md5sum pair if available (avoids typestore.generate_msgdef)
                            if msgdef is not None and len(str(msgdef)) > 0 and md5sum is not None:
                                conn_kwargs['msgdef'] = msgdef
                                conn_kwargs['md5sum'] = md5sum
                                # Don't pass typestore when using msgdef + md5sum
                            elif typestore is not None:
                                conn_kwargs['typestore'] = typestore
                            else:
                                raise ValueError("Need either (msgdef + md5sum) or typestore to write bag file")
                            conn = writer.add_connection(**conn_kwargs)
                            
                            # Serialize and write the modified message
                            timestamp = int(time.time() * 1e9)  # nanoseconds
                            
                            msg_copy = build_modified_message()
                            
                            # Serialize the message to bytes using typestore
                            # rosbag1 uses ROS1 serialization, not CDR
                            serialization_typestore = typestore if typestore is not None else (conn.typestore if hasattr(conn, 'typestore') else None)
                            
                            if serialization_typestore is None:
                                raise ValueError("No typestore available for serialization")
                            
                            # Use typestore's serialize_ros1 method for rosbag1 format
                            # Use original msgtype (may be ROS2 format, but typestore knows it)
                            try:
                                serialized_mv = serialization_typestore.serialize_ros1(msg_copy, msgtype)
                                # Convert memoryview to bytes
                                serialized_data = bytes(serialized_mv)
                            except Exception as ser_error:
                                print(f"ROS1 serialization failed: {ser_error}")
                                import traceback
                                traceback.print_exc()
                                raise ValueError(f"Could not serialize message: {ser_error}")
                            
                            # Write the message
                            writer.write(conn, timestamp, serialized_data)
                        finally:
                            writer.close()
                    elif use_anywriter:
                        # AnyWriter API (rosbags < 0.11)
                        with Writer([Path(bag_path)]) as writer:
                            # Add connection for grid_map topic
                            conn_kwargs = {
                                'topic': "grid_map",
                                'msgtype': msgtype,
                                'serialization_format': 'cdr',
                                'offered_qos_profiles': None
                            }
                            if msgdef is not None:
                                conn_kwargs['msgdef'] = msgdef
                            if rihs01 is not None:
                                conn_kwargs['rihs01'] = rihs01
                            if typestore is not None:
                                conn_kwargs['typestore'] = typestore
                            conn = writer.add_connection(**conn_kwargs)
                            
                            # Serialize and write the modified message
                            timestamp = int(time.time() * 1e9)  # nanoseconds
                            
                            msg_copy = build_modified_message()
                            
                            # Write the message
                            writer.write(conn, timestamp, msg_copy)
                    elif use_rosbag2:
                        # rosbag2.Writer API (rosbags >= 0.11)
                        # Writer requires path and version in __init__
                        # For MCAP, use storage_plugin=StoragePlugin.MCAP, for SQLite use StoragePlugin.SQLITE3
                        if storage_plugin is None:
                            from rosbags.rosbag2 import StoragePlugin
                            storage_plugin = StoragePlugin.SQLITE3
                        writer = Writer(Path(bag_path), version=9, storage_plugin=storage_plugin)
                        writer.open()
                        try:
                            # Add connection for grid_map topic
                            conn_kwargs = {
                                'topic': "grid_map",
                                'msgtype': msgtype,
                                'serialization_format': 'cdr'
                            }
                            
                            # Prioritize typestore (preferred method)
                            if typestore is not None:
                                conn_kwargs['typestore'] = typestore
                                print(f"Using typestore for connection")
                            else:
                                # Fallback to msgdef + rihs01 if typestore not available
                                if msgdef is not None and len(str(msgdef)) > 0:
                                    conn_kwargs['msgdef'] = msgdef
                                    print(f"Using msgdef (length: {len(str(msgdef))})")
                                if rihs01 is not None and len(str(rihs01)) > 0:
                                    conn_kwargs['rihs01'] = rihs01
                                    print(f"Using rihs01")
                            
                            print(f"Calling add_connection with keys: {list(conn_kwargs.keys())}")
                            conn = writer.add_connection(**conn_kwargs)
                            
                            # Serialize and write the modified message
                            timestamp = int(time.time() * 1e9)  # nanoseconds
                            
                            msg_copy = build_modified_message()
                            
                            # Serialize the message to bytes using typestore
                            # Use stored typestore or get from connection
                            serialization_typestore = typestore if typestore is not None else (conn.typestore if hasattr(conn, 'typestore') else None)
                            
                            if serialization_typestore is None:
                                raise ValueError("No typestore available for serialization")
                            
                            # Use typestore's serialize_cdr method
                            try:
                                serialized_mv = serialization_typestore.serialize_cdr(msg_copy, msgtype, little_endian=True)
                                # Convert memoryview to bytes
                                serialized_data = bytes(serialized_mv)
                            except Exception as ser_error:
                                print(f"CDR serialization failed: {ser_error}")
                                import traceback
                                traceback.print_exc()
                                raise ValueError(f"Could not serialize message: {ser_error}")
                            
                            # Write the message
                            writer.write(conn, timestamp, serialized_data)
                        finally:
                            writer.close()
                    
                except ImportError as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"Bag file export failed (ImportError): {error_details}")
                    # Get version info if available
                    try:
                        import rosbags
                        version_info = f"rosbags version: {getattr(rosbags, '__version__', 'unknown')}"
                    except:
                        version_info = "rosbags version: unknown"
                    
                    QMessageBox.warning(
                        self,
                        "Export Warning",
                        f"Could not create bag file. rosbags library writing support not available.\n\n"
                        f"{version_info}\n"
                        f"Error: {str(e)}\n\n"
                        f"✓ Elevation saved to: {elev_npy_path}\n\n"
                        f"Note: Bag file writing requires rosbags >= 0.10.0 with AnyWriter support.\n"
                        f"You can use the npy file directly or manually create a bag file using ROS tools."
                    )
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"Bag file export failed: {error_details}")
                    QMessageBox.warning(
                        self,
                        "Export Warning",
                        f"Could not create bag file: {str(e)}\n\n"
                        f"Elevation saved to: {elev_npy_path}"
                    )
        
        # Show success message
        files_exported = [elev_npy_path]
        if bag_path:
            files_exported.append(bag_path)
        
        message = "Exported Isaac Sim files:\n" + "\n".join(files_exported)
        if hasattr(self, 'georef_config'):
            message += "\n\nNote: Georeferencing info is in map_georeference_config.yaml"
        QMessageBox.information(self, "Export", message)

    # ----- Utils -----
    @staticmethod
    def _load_to_array(path: str) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            data = np.load(path)
            if data.ndim == 3:
                data = data.mean(axis=2)
            return data.astype(np.float32)
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.float32)

    @staticmethod
    def _normalize01(arr: np.ndarray) -> np.ndarray:
        amin = float(np.min(arr))
        amax = float(np.max(arr))
        if amax - amin < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - amin) / (amax - amin)).astype(np.float32)

    @staticmethod
    def _compute_fill_value(arr: np.ndarray, default: float = 0.0) -> float:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return default
        return float(finite.min())

    def _rotate_array(self, arr: Optional[np.ndarray], angle_deg: float, fill_value: Optional[float] = None) -> Optional[np.ndarray]:
        if arr is None:
            return None
        if abs(angle_deg) < 1e-6:
            return arr.copy()
        safe_fill = float(fill_value if fill_value is not None else self._compute_fill_value(arr, 0.0))
        base = np.where(np.isfinite(arr), arr, safe_fill).astype(np.float32)
        try:
            img = Image.fromarray(base, mode='F')
            rotated_img = img.rotate(angle_deg, resample=Image.BILINEAR, expand=True)
            rotated_arr = np.array(rotated_img, dtype=np.float32)
            mask = np.ones_like(base, dtype=np.float32)
            mask_img = Image.fromarray(mask, mode='F')
            rotated_mask = mask_img.rotate(angle_deg, resample=Image.BILINEAR, expand=True)
            mask_arr = np.array(rotated_mask, dtype=np.float32)
            rotated_arr = np.where(mask_arr > 1e-3, rotated_arr, safe_fill)
            return rotated_arr
        except Exception:
            return base.copy()

    def _store_base_canvas(self, canvas: Optional[np.ndarray], apply_rotation: bool = True) -> None:
        self._base_canvas = canvas.copy() if canvas is not None else None
        if apply_rotation:
            self._apply_rotation_to_bases()

    def _store_base_desired_canvas(self, canvas: Optional[np.ndarray], apply_rotation: bool = True) -> None:
        self._base_desired_canvas = canvas.copy() if canvas is not None else None
        if apply_rotation:
            self._apply_rotation_to_bases()

    def _store_base_previous_desired_canvas(self, canvas: Optional[np.ndarray], apply_rotation: bool = True) -> None:
        self._base_previous_desired_canvas = canvas.copy() if canvas is not None else None
        if apply_rotation:
            self._apply_rotation_to_bases()

    def _store_base_original_elevation_array(self, arr: Optional[np.ndarray], apply_rotation: bool = True) -> None:
        self._base_original_elevation_array = arr.copy() if arr is not None else None
        if apply_rotation:
            self._apply_rotation_to_bases()

    def _store_base_original_desired_array(self, arr: Optional[np.ndarray], apply_rotation: bool = True) -> None:
        self._base_original_desired_array = arr.copy() if arr is not None else None
        if apply_rotation:
            self._apply_rotation_to_bases()

    def _store_base_original_previous_desired_array(self, arr: Optional[np.ndarray], apply_rotation: bool = True) -> None:
        self._base_original_previous_desired_array = arr.copy() if arr is not None else None
        if apply_rotation:
            self._apply_rotation_to_bases()

    def _apply_rotation_to_bases(self) -> None:
        angle = float(getattr(self, "rotation_deg", 0.0))

        def rotate_or_none(base: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if base is None:
                return None
            fill = self._compute_fill_value(base, 0.0)
            return self._rotate_array(base, angle, fill)

        self.last_pcl_canvas = rotate_or_none(self._base_canvas)
        self.desired_elevation_canvas = rotate_or_none(self._base_desired_canvas)
        self.previous_desired_elevation_canvas = rotate_or_none(self._base_previous_desired_canvas)
        self.original_elevation_array = (
            self._base_original_elevation_array.copy()
            if self._base_original_elevation_array is not None else None
        )
        self.original_desired_elevation_array = (
            self._base_original_desired_array.copy()
            if self._base_original_desired_array is not None else None
        )
        self.original_previous_desired_elevation_array = (
            self._base_original_previous_desired_array.copy()
            if self._base_original_previous_desired_array is not None else None
        )

    def _apply_type_button_styles(self) -> None:
        dump_btn = self.toolbar.widgetForAction(self.action_type_dump)
        foundation_btn = self.toolbar.widgetForAction(self.action_type_foundation)
        obstacle_btn = self.toolbar.widgetForAction(self.action_type_obstacle)
        nodump_btn = self.toolbar.widgetForAction(self.action_type_nodump)
        eraser_btn = self.toolbar.widgetForAction(self.action_type_eraser)
        if dump_btn is not None:
            dump_btn.setStyleSheet(
                f"QToolButton{{background: rgba(0,200,0,0.10); border:1px solid rgba(0,200,0,0.35); border-radius:8px;}}"
                f"QToolButton:hover{{background: rgba(0,200,0,0.20);}}"
                f"QToolButton:checked{{background: rgba(0,200,0,0.30); border:2px solid rgba(0,200,0,0.8);}}"
            )
        if foundation_btn is not None:
            foundation_btn.setStyleSheet(
                f"QToolButton{{background: rgba(150,40,220,0.10); border:1px solid rgba(150,40,220,0.35); border-radius:8px;}}"
                f"QToolButton:hover{{background: rgba(150,40,220,0.20);}}"
                f"QToolButton:checked{{background: rgba(150,40,220,0.30); border:2px solid rgba(150,40,220,0.8);}}"
            )
        if obstacle_btn is not None:
            obstacle_btn.setStyleSheet(
                f"QToolButton{{background: rgba(0,0,0,0.08); border:1px solid rgba(0,0,0,0.35); border-radius:8px;}}"
                f"QToolButton:hover{{background: rgba(0,0,0,0.16);}}"
                f"QToolButton:checked{{background: rgba(0,0,0,0.24); border:2px solid rgba(0,0,0,0.8);}}"
            )
        if nodump_btn is not None:
            nodump_btn.setStyleSheet(
                f"QToolButton{{background: rgba(120,120,120,0.10); border:1px solid rgba(120,120,120,0.35); border-radius:8px;}}"
                f"QToolButton:hover{{background: rgba(120,120,120,0.20);}}"
                f"QToolButton:checked{{background: rgba(120,120,120,0.30); border:2px solid rgba(120,120,120,0.8);}}"
            )
        if eraser_btn is not None:
            eraser_btn.setStyleSheet(
                "QToolButton{background: rgba(0,0,0,0.04); border:1px solid rgba(0,0,0,0.12); border-radius:8px;}"
                "QToolButton:hover{background: rgba(0,0,0,0.08);}"
                "QToolButton:checked{background: rgba(0,0,0,0.12); border:2px solid rgba(0,0,0,0.35);}"
            )

    def _apply_tool_button_styles(self) -> None:
        cell_btn = self.toolbar.widgetForAction(self.action_tool_cell)
        rect_btn = self.toolbar.widgetForAction(self.action_tool_rect)
        polygon_btn = self.toolbar.widgetForAction(self.action_tool_polygon)
        ruler_btn = self.toolbar.widgetForAction(self.action_tool_ruler)
        if cell_btn is not None:
            cell_btn.setStyleSheet(
                "QToolButton{background: rgba(0,0,0,0.04); border:1px solid rgba(0,0,0,0.12); border-radius:8px;}"
                "QToolButton:hover{background: rgba(0,0,0,0.08);}"
                "QToolButton:checked{background: rgba(0,0,0,0.12); border:2px solid rgba(0,0,0,0.35);}"
            )
        if rect_btn is not None:
            rect_btn.setStyleSheet(
                "QToolButton{background: rgba(0,0,0,0.04); border:1px solid rgba(0,0,0,0.12); border-radius:8px;}"
                "QToolButton:hover{background: rgba(0,0,0,0.08);}"
                "QToolButton:checked{background: rgba(0,0,0,0.12); border:2px solid rgba(0,0,0,0.35);}"
            )
        if polygon_btn is not None:
            polygon_btn.setStyleSheet(
                "QToolButton{background: rgba(0,0,0,0.04); border:1px solid rgba(0,0,0,0.12); border-radius:8px;}"
                "QToolButton:hover{background: rgba(0,0,0,0.08);}"
                "QToolButton:checked{background: rgba(0,0,0,0.12); border:2px solid rgba(0,0,0,0.35);}"
            )
        if ruler_btn is not None:
            ruler_btn.setStyleSheet(
                "QToolButton{background: rgba(0,0,0,0.04); border:1px solid rgba(0,0,0,0.12); border-radius:8px;}"
                "QToolButton:hover{background: rgba(0,0,0,0.08);}"
                "QToolButton:checked{background: rgba(0,0,0,0.12); border:2px solid rgba(0,0,0,0.35);}"
            )

    def _configure_gnss_usage(self, enabled: bool) -> None:
        """Toggle whether grid assets should be transposed/rotated for GNSS alignment."""
        self.use_gnss_reference = bool(enabled)
        if hasattr(self, 'scene') and self.scene is not None:
            self.scene.transpose_background = self.use_gnss_reference

    def _build_bottom_bar(self) -> QWidget:
        # Ensure required widgets exist
        if not hasattr(self, 'btn_load_geo'):
            self.btn_load_geo = QPushButton("Load Geo Map")
            self.btn_load_geo.clicked.connect(self.on_load_geo_map)
        if not hasattr(self, 'btn_export'):
            self.btn_export = QPushButton("Export")
            self.btn_export.setObjectName("exportBtn")
            self.btn_export.clicked.connect(self.on_export)
        if not hasattr(self, 'grid_size_combo'):
            self.grid_size_combo = QComboBox()
            self.grid_size_combo.addItems(["32", "64", "128", "256"])
            self.grid_size_combo.setCurrentText(str(self.grid_size))
            self.grid_size_combo.currentTextChanged.connect(self.on_grid_size_change)
        if not hasattr(self, 'meters_spin'):
            self.meters_spin = QDoubleSpinBox()
            self.meters_spin.setRange(0.01, 1000.0)
            self.meters_spin.setDecimals(5)
            self.meters_spin.setSingleStep(0.01)
            self.meters_spin.setValue(self.meters_per_tile)
            self.meters_spin.valueChanged.connect(self.on_meters_per_tile_change)
        if not hasattr(self, 'placement_combo'):
            self.placement_combo = QComboBox()
            self.placement_combo.addItems(["Center", "Top-Left"])
            self.placement_combo.setCurrentText("Top-Left")
            self.placement_combo.currentTextChanged.connect(self.on_placement_changed)
        if not hasattr(self, 'rotation_spin'):
            self.rotation_spin = QDoubleSpinBox()
            self.rotation_spin.setRange(-180.0, 180.0)
            self.rotation_spin.setDecimals(1)
            self.rotation_spin.setSingleStep(1.0)
            self.rotation_spin.setValue(self.rotation_deg)
            self.rotation_spin.valueChanged.connect(self.on_rotation_changed)
        # Add export target selector (includes format selection)
        if not hasattr(self, 'export_target_combo'):
            self.export_target_combo = QComboBox()
            self.export_target_combo.addItems(["Terra", "ROS1 (.bag)", "ROS2 (.mcap)"])
        # Offset sliders removed (drag background to reposition)

        bottom_bar = QWidget()
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(10, 8, 10, 8)
        bottom_layout.setSpacing(10)
        bottom_bar.setLayout(bottom_layout)
        bottom_bar.setStyleSheet(
            """
            QWidget { background: #ffffff; border-top: 1px solid #e5e5e5; }
            QPushButton { background: #f7f7f9; border: 1px solid #e3e3e7; padding: 6px 10px; border-radius: 8px; }
            QPushButton:hover { background: #f0f0f3; }
            QPushButton#exportBtn { background: #0b7cff; color: #ffffff; border: 1px solid #0b7cff; }
            QPushButton#exportBtn:hover { background: #096ae0; border-color: #096ae0; }
            QLabel { color:#4a4a4a; }
            QComboBox, QDoubleSpinBox { background: #ffffff; border: 1px solid #e3e3e7; border-radius: 6px; padding: 2px 6px; }
            QSlider::groove:horizontal { height: 6px; background: #e5e7eb; border-radius: 3px; }
            QSlider::handle:horizontal { background: #0b7cff; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #bfd7ff; border-radius: 3px; }
            """
        )
        bottom_layout.addWidget(self.btn_load_geo)
        bottom_layout.addWidget(self.btn_load_foundation)
        bottom_layout.addWidget(QLabel("Map res (m/cell):"))
        bottom_layout.addWidget(self.map_res_spin)
        bottom_layout.addSpacing(12)
        bottom_layout.addWidget(QLabel("Grid Size:"))
        bottom_layout.addWidget(self.grid_size_combo)
        bottom_layout.addSpacing(12)
        bottom_layout.addWidget(QLabel("Meters/Tile:"))
        bottom_layout.addWidget(self.meters_spin)
        bottom_layout.addSpacing(12)
        bottom_layout.addWidget(QLabel("Placement:"))
        bottom_layout.addWidget(self.placement_combo)
        bottom_layout.addSpacing(12)
        bottom_layout.addWidget(QLabel("Offset X:"))
        bottom_layout.addWidget(self.offset_x_spin)
        bottom_layout.addWidget(QLabel("Offset Y:"))
        bottom_layout.addWidget(self.offset_y_spin)
        bottom_layout.addWidget(QLabel("Rotation (deg):"))
        bottom_layout.addWidget(self.rotation_spin)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(QLabel("Export to:"))
        bottom_layout.addWidget(self.export_target_combo)
        bottom_layout.addWidget(self.btn_export)
        return bottom_bar

    def on_rectangle_committed(self, rect_cells: Tuple[int, int, int, int]) -> None:
        self.foundation_rect = rect_cells
        self.update_foundation_profile()

    # ----- 3D view -----
    def _clear_3d_items(self) -> None:
        if not HAS_GL or self.gl_view is None:
            return
        # Clear all items from the view first
        for attr in ("gl_surface", "gl_plane"):
            itm = getattr(self, attr, None)
            if itm is not None:
                try:
                    self.gl_view.removeItem(itm)
                except Exception:
                    pass
                setattr(self, attr, None)
        if hasattr(self, 'gl_walls') and self.gl_walls:
            for itm in self.gl_walls:
                try:
                    self.gl_view.removeItem(itm)
                except Exception:
                    pass
            self.gl_walls = []
        # Also clear any contour lines
        if hasattr(self, 'gl_contours') and self.gl_contours:
            for itm in self.gl_contours:
                try:
                    self.gl_view.removeItem(itm)
                except Exception:
                    pass
            self.gl_contours = []
        # Force a repaint to ensure everything is cleared
        if self.gl_view is not None:
            try:
                self.gl_view.update()
            except:
                pass

    def update_3d_view(self) -> None:
        if not HAS_GL or self.gl_view is None:
            return
        if self.last_placed_elevation is None:
            self._clear_3d_items()
            return
        elev = self.last_placed_elevation
        finite_vals = elev[np.isfinite(elev)]
        base_val = float(finite_vals.min()) if finite_vals.size > 0 else 0.0
        z_full = np.nan_to_num(elev, nan=base_val, posinf=base_val, neginf=base_val).astype(np.float32)
        z_full = np.ascontiguousarray(z_full)
        mask = getattr(self.scene, 'foundation_mask', None)
        self._clear_3d_items()
        try:
            # Use full elevation mesh
            H, W = z_full.shape
            if H < 2 or W < 2:
                self._clear_3d_items()
                return
            
            # Vertical exaggeration
            scale = float(self.height_scale_spin.value()) if hasattr(self, 'height_scale_spin') else 1.0
            
            # Create carved elevation: full mesh with foundation areas carved out
            z_carved = z_full.copy()
            foundation_mask = mask.astype(bool) if mask is not None else np.zeros((H, W), dtype=bool)
            
            # Only carve and add walls if there are foundation cells
            # If foundation mask is empty, z_carved will just be z_full (original elevation)
            if foundation_mask.sum() > 0:
                # Calculate bottom depth for foundation areas
                # Check if we have a depth map (from STL/OBJ import or drawn tiles)
                depth_map = getattr(self.scene, 'foundation_depth_map', None)
                has_depth_map = (depth_map is not None and 
                                np.any(np.abs(depth_map[foundation_mask]) > 1e-6))
                
                # Default depth from spinbox (used if no stored depth for a cell)
                default_depth = float(self.depth_spin.value())
                
                if has_depth_map:
                    # Use per-cell depth values (stored when tile was drawn or from imported mesh)
                    # For imported foundations, prefer group's depth_map which has variable depth
                    # Build a lookup: cell -> group depth if available
                    cell_to_group_depth = {}
                    for group in getattr(self.scene, 'foundation_groups', []):
                        group_depth_map = group.get('depth_map')
                        if group_depth_map is not None:
                            for x, y in group.get('cells', []):
                                if (x, y) in group_depth_map:
                                    cell_to_group_depth[(x, y)] = group_depth_map[(x, y)]
                    
                    # For each cell, use stored depth if available, otherwise use default
                    for cy in range(H):
                        for cx in range(W):
                            if foundation_mask[cy, cx]:
                                surface_z = z_full[cy, cx]
                                
                                # Prefer group depth map for imported foundations (variable depth)
                                stored_depth = None
                                if (cx, cy) in cell_to_group_depth:
                                    stored_depth = cell_to_group_depth[(cx, cy)]
                                elif depth_map is not None:
                                    stored_depth = depth_map[cy, cx]
                                
                                if stored_depth is not None and stored_depth > 1e-6:
                                    # Check if absolute Z (> 100) or relative depth
                                    if stored_depth > 100:
                                        # Absolute Z from STL - use directly as bottom
                                        z_carved[cy, cx] = stored_depth
                                    else:
                                        # Relative depth (meters below surface)
                                        bottom = surface_z - stored_depth
                                        z_carved[cy, cx] = bottom
                                else:
                                    # No stored depth, use default from spinbox
                                    bottom = surface_z - default_depth
                                    z_carved[cy, cx] = bottom
                else:
                    # No depth map at all, use uniform depth from spinbox
                    depth = default_depth
                    # Find max elevation within foundation mask
                    foundation_elev = z_full[foundation_mask]
                    if foundation_elev.size > 0:
                        top_max = float(foundation_elev.max())
                        bottom = top_max - depth
                        # Carve out foundation: set foundation areas to bottom height
                        z_carved[foundation_mask] = bottom
            else:
                # No foundation cells - use original elevation (z_carved is already z_full.copy())
                # No carving needed, z_carved already equals z_full
                pass
            
            # Apply vertical exaggeration
            z_scaled = z_carved * scale
            z_scaled = np.ascontiguousarray(z_scaled.astype(np.float32))
            
            # Compute center of entire mesh for camera positioning
            x_centroid = (W - 1) * 0.5 * self.meters_per_tile
            y_centroid = (H - 1) * 0.5 * self.meters_per_tile
            z_center = float(z_scaled.mean())
            
            # Translate so center is at origin
            tx = -x_centroid
            # Flip Y axis to match 2D (rows increase downwards): use negative Y for increasing i
            # To keep mesh centered at origin with y = ty - i*dy, set ty = +y_centroid
            ty = y_centroid
            
            # Camera: view entire mesh
            try:
                size_m = max((W-1), (H-1)) * self.meters_per_tile
                self.gl_view.setCameraPosition(
                center=pg.Vector(0, 0, z_center),
                distance=max(1.2, size_m * 1.2),
                elevation=28,
                azimuth=30
            )
            except Exception:
                pass
            
            # Build mesh for full elevation with carved foundation
            dx = self.meters_per_tile
            dy = self.meters_per_tile
            vertices = []
            faces = []
            colors_v = []
            index_map = {}

            def vkey(i, j, kind):
                return (i, j, kind)

            def add_vertex(i, j, zval, kind, col_rgba):
                key = vkey(i, j, kind)
                idx = index_map.get(key)
                if idx is None:
                    x = tx + j * dx
                    y = ty - i * dy  # Flip Y axis to match 2D
                    z = float(zval)
                    idx = len(vertices)
                    vertices.append([x, y, z])
                    colors_v.append(col_rgba)
                    index_map[key] = idx
                return idx
            
            # Height-based colormap for terrain (green to brown/yellow)
            z_min = float(z_scaled.min())
            z_max = float(z_scaled.max())
            z_range = z_max - z_min if z_max > z_min else 1.0
            
            def height_to_color(z_val, is_foundation=False):
                """Convert height to color: green (low) -> yellow -> brown (high)"""
                if is_foundation:
                    # Foundation: distinct blue-gray color
                    return (0.4, 0.5, 0.7, 1.0)
                # Normalize height to [0, 1]
                t = (z_val - z_min) / z_range
                t = max(0.0, min(1.0, t))
                # Green (low) -> Yellow -> Brown (high)
                if t < 0.5:
                    # Green to yellow
                    r = t * 2.0
                    g = 0.7 + t * 0.3
                    b = 0.3 - t * 0.3
                else:
                    # Yellow to brown
                    r = 0.6 + (t - 0.5) * 0.4
                    g = 0.7 - (t - 0.5) * 0.4
                    b = 0.2 - (t - 0.5) * 0.2
                return (max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b)), 1.0)
            
            # Build full mesh triangles with height-based coloring
            # Process events periodically to keep UI responsive
            total_triangles = (H - 1) * (W - 1)
            update_freq = max(1, total_triangles // 20)  # Update UI every 5% progress
            
            for i in range(H - 1):
                for j in range(W - 1):
                    # Determine if vertices are in foundation
                    c00_in_foundation = foundation_mask[i, j] if foundation_mask.size > 0 else False
                    c10_in_foundation = foundation_mask[i+1, j] if foundation_mask.size > 0 else False
                    c01_in_foundation = foundation_mask[i, j+1] if foundation_mask.size > 0 else False
                    c11_in_foundation = foundation_mask[i+1, j+1] if foundation_mask.size > 0 else False
                    
                    # Get colors based on height and foundation status
                    col00 = height_to_color(z_scaled[i, j], c00_in_foundation)
                    col10 = height_to_color(z_scaled[i+1, j], c10_in_foundation)
                    col01 = height_to_color(z_scaled[i, j+1], c01_in_foundation)
                    col11 = height_to_color(z_scaled[i+1, j+1], c11_in_foundation)
                    
                    # First triangle - interpolate colors
                    i0 = add_vertex(i, j, z_scaled[i, j], 't', col00)
                    i1 = add_vertex(i+1, j, z_scaled[i+1, j], 't', col10)
                    i2 = add_vertex(i, j+1, z_scaled[i, j+1], 't', col01)
                    faces.append([i0, i1, i2])
                    
                    # Second triangle - interpolate colors
                    i3 = add_vertex(i+1, j, z_scaled[i+1, j], 't', col10)
                    i4 = add_vertex(i+1, j+1, z_scaled[i+1, j+1], 't', col11)
                    i5 = add_vertex(i, j+1, z_scaled[i, j+1], 't', col01)
                    faces.append([i3, i4, i5])
                    
                    # Process events periodically to keep UI responsive
                    if (i * (W - 1) + j) % update_freq == 0:
                        QApplication.processEvents()
            
            # Helper function for terrain color (used in walls)
            def get_terrain_color(z_val):
                return height_to_color(z_val, False)
            
            # Add walls along foundation boundary (inside edge of foundation)
            # Only add walls if there are foundation cells
            wall_color = (0.6, 0.6, 0.6, 1.0)
            if foundation_mask.sum() > 0:
                # Initialize walls list if needed
                if not hasattr(self, 'gl_walls'):
                    self.gl_walls = []
                # Get depth map for per-cell depth values
                depth_map = getattr(self.scene, 'foundation_depth_map', None)
                default_depth = float(self.depth_spin.value())
                
                # Build lookup for group depth maps (for imported foundations with variable depth)
                cell_to_group_depth = {}
                for group in getattr(self.scene, 'foundation_groups', []):
                    group_depth_map = group.get('depth_map')
                    if group_depth_map is not None:
                        for x, y in group.get('cells', []):
                            if (x, y) in group_depth_map:
                                cell_to_group_depth[(x, y)] = group_depth_map[(x, y)]
                
                # Helper function to get bottom Z for a cell
                def get_bottom_z(cy, cx):
                    if foundation_mask[cy, cx]:
                        surface_z = z_full[cy, cx]
                        # Prefer group depth map for imported foundations
                        stored_depth = None
                        if (cx, cy) in cell_to_group_depth:
                            stored_depth = cell_to_group_depth[(cx, cy)]
                        elif depth_map is not None and abs(depth_map[cy, cx]) > 1e-6:
                            stored_depth = depth_map[cy, cx]
                        
                        if stored_depth is not None and abs(stored_depth) > 1e-6:
                            # Check if absolute Z (from STL) or relative depth
                            if stored_depth > 100:
                                return stored_depth
                            else:
                                return surface_z - stored_depth
                        else:
                            return surface_z - default_depth
                    return surface_z
                
                # Vertical borders (left/right edges of foundation)
                for i in range(H - 1):
                    for j in range(1, W):
                        # Check if we're at the boundary: inside foundation on one side, outside on the other
                        right_in = foundation_mask[i, j] and foundation_mask[i+1, j]
                        left_in = foundation_mask[i, j-1] and foundation_mask[i+1, j-1]
                        
                        # Wall on right edge of foundation (foundation on left j-1, terrain on right j)
                        # Wall connects terrain surface (right side) down to carved bottom (left side)
                        if left_in and not right_in:
                            # Left side (foundation): carved bottom (per-cell depth)
                            z_bottom_left = get_bottom_z(i, j-1) * scale
                            z_bottom_right = get_bottom_z(i+1, j-1) * scale
                            # Right side (terrain): original elevation
                            z_top_left = z_full[i, j] * scale
                            z_top_right = z_full[i+1, j] * scale
                            
                            ia_top = add_vertex(i, j, z_top_left, 't', get_terrain_color(z_top_left))
                            ib_top = add_vertex(i+1, j, z_top_right, 't', get_terrain_color(z_top_right))
                            ia_bot = add_vertex(i, j-1, z_bottom_left, 'b', wall_color)
                            ib_bot = add_vertex(i+1, j-1, z_bottom_right, 'b', wall_color)
                            faces.append([ia_top, ib_top, ia_bot])
                            faces.append([ib_top, ib_bot, ia_bot])
                        
                        # Wall on left edge of foundation (foundation on right j, terrain on left j-1)
                        # Wall connects terrain surface (left side) down to carved bottom (right side)
                        if right_in and not left_in:
                            # Right side (foundation): carved bottom (per-cell depth)
                            z_bottom_left = get_bottom_z(i, j) * scale
                            z_bottom_right = get_bottom_z(i+1, j) * scale
                            # Left side (terrain): original elevation
                            z_top_left = z_full[i, j-1] * scale
                            z_top_right = z_full[i+1, j-1] * scale
                            
                            ia_top = add_vertex(i, j-1, z_top_left, 't', get_terrain_color(z_top_left))
                            ib_top = add_vertex(i+1, j-1, z_top_right, 't', get_terrain_color(z_top_right))
                            ia_bot = add_vertex(i, j, z_bottom_left, 'b', wall_color)
                            ib_bot = add_vertex(i+1, j, z_bottom_right, 'b', wall_color)
                            faces.append([ia_top, ib_top, ia_bot])
                            faces.append([ib_top, ib_bot, ia_bot])
                
                # Horizontal borders (top/bottom edges of foundation)
                for i in range(1, H):
                    for j in range(W - 1):
                        down_in = foundation_mask[i, j] and foundation_mask[i, j+1]
                        up_in = foundation_mask[i-1, j] and foundation_mask[i-1, j+1]
                        
                        # Wall on bottom edge of foundation (foundation on top i-1, terrain on bottom i)
                        # Wall connects terrain surface (bottom side) down to carved bottom (top side)
                        if up_in and not down_in:
                            # Top side (foundation): carved bottom (per-cell depth)
                            z_bottom_left = get_bottom_z(i-1, j) * scale
                            z_bottom_right = get_bottom_z(i-1, j+1) * scale
                            # Bottom side (terrain): original elevation
                            z_top_left = z_full[i, j] * scale
                            z_top_right = z_full[i, j+1] * scale
                            
                            ia_top = add_vertex(i, j, z_top_left, 't', get_terrain_color(z_top_left))
                            ib_top = add_vertex(i, j+1, z_top_right, 't', get_terrain_color(z_top_right))
                            ia_bot = add_vertex(i-1, j, z_bottom_left, 'b', wall_color)
                            ib_bot = add_vertex(i-1, j+1, z_bottom_right, 'b', wall_color)
                            faces.append([ia_top, ib_top, ia_bot])
                            faces.append([ib_top, ib_bot, ia_bot])
                        
                        # Wall on top edge of foundation (foundation on bottom i, terrain on top i-1)
                        # Wall connects terrain surface (top side) down to carved bottom (bottom side)
                        if down_in and not up_in:
                            # Bottom side (foundation): carved bottom (per-cell depth)
                            z_bottom_left = get_bottom_z(i, j) * scale
                            z_bottom_right = get_bottom_z(i, j+1) * scale
                            # Top side (terrain): original elevation
                            z_top_left = z_full[i-1, j] * scale
                            z_top_right = z_full[i-1, j+1] * scale
                            
                            ia_top = add_vertex(i-1, j, z_top_left, 't', get_terrain_color(z_top_left))
                            ib_top = add_vertex(i-1, j+1, z_top_right, 't', get_terrain_color(z_top_right))
                            ia_bot = add_vertex(i, j, z_bottom_left, 'b', wall_color)
                            ib_bot = add_vertex(i, j+1, z_bottom_right, 'b', wall_color)
                        faces.append([ia_top, ib_top, ia_bot])
                        faces.append([ib_top, ib_bot, ia_bot])
            
            # Create and add mesh
                if vertices:
                    md = MeshData(
                        vertexes=np.array(vertices, dtype=np.float32),
                        faces=np.array(faces, dtype=np.int32),
                        vertexColors=np.array(colors_v, dtype=np.float32)
                    )
                    mesh_item = gl.GLMeshItem(meshdata=md, smooth=False, drawEdges=False, drawFaces=True)
                    mesh_item.setGLOptions('opaque')
                    self.gl_view.addItem(mesh_item)
                    self.gl_surface = mesh_item
                
                # Add contour lines (height lines) - optimized and with progress updates
                contour_lines = []
                contour_color = (0.0, 0.0, 0.0, 0.6)  # Semi-transparent black
                # Reduce number of contour lines for large meshes to improve performance
                max_contours = 15 if H * W > 10000 else 20
                contour_interval = max(0.5, (z_max - z_min) / max_contours)
                contour_levels = np.arange(z_min, z_max + contour_interval, contour_interval)
                
                # Process events before starting contour generation
                QApplication.processEvents()
                
                for level_idx, level in enumerate(contour_levels):
                    level_scaled = level
                    # Find contour points by checking grid edges
                    contour_points = []
                    
                    # Check horizontal edges
                    for i in range(H):
                        for j in range(W - 1):
                            z1 = z_scaled[i, j]
                            z2 = z_scaled[i, j+1]
                            if (z1 <= level_scaled <= z2) or (z2 <= level_scaled <= z1):
                                # Interpolate position
                                if abs(z2 - z1) > 1e-6:
                                    t = (level_scaled - z1) / (z2 - z1)
                                else:
                                    t = 0.5
                                x = tx + (j + t) * dx
                                y = ty - i * dy
                                contour_points.append([x, y, level_scaled])
                    
                    # Check vertical edges
                    for i in range(H - 1):
                        for j in range(W):
                            z1 = z_scaled[i, j]
                            z2 = z_scaled[i+1, j]
                            if (z1 <= level_scaled <= z2) or (z2 <= level_scaled <= z1):
                                # Interpolate position
                                if abs(z2 - z1) > 1e-6:
                                    t = (level_scaled - z1) / (z2 - z1)
                                else:
                                    t = 0.5
                                x = tx + j * dx
                                y = ty - (i + t) * dy
                                contour_points.append([x, y, level_scaled])
                    
                    # Draw contour segments (connect nearby points)
                    if len(contour_points) >= 2:
                        # Simple approach: connect points that are close
                        points_array = np.array(contour_points, dtype=np.float32)
                        if points_array.size > 0:
                            # Draw as line segments - connect consecutive pairs
                            # For simplicity, draw all points as a connected line
                            # (in a full implementation, you'd trace closed contours)
                            if len(points_array) >= 2:
                                # Draw line segments between nearby points
                                for k in range(len(points_array) - 1):
                                    p1 = points_array[k]
                                    p2 = points_array[k + 1]
                                    # Only draw if points are reasonably close (same contour segment)
                                    dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                                    if dist < self.meters_per_tile * 2.0:  # Within 2 cells
                                        line_data = np.array([p1, p2], dtype=np.float32)
                                        line_item = gl.GLLinePlotItem(
                                            pos=line_data,
                                            color=contour_color,
                                            width=1.0,
                                            antialias=True
                                        )
                                        self.gl_view.addItem(line_item)
                                        contour_lines.append(line_item)
                    
                    # Process events every few contour levels
                    if level_idx % 3 == 0:
                        QApplication.processEvents()
                
                self.gl_contours = contour_lines
                if not hasattr(self, 'gl_walls'):
                    self.gl_walls = []
            else:
                # No mesh created (no vertices)
                self._clear_3d_items()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._clear_3d_items()

    def _on_mask_changed(self) -> None:
        self.update_foundation_profile()
        self.update_3d_view()
    
    def on_group_selected(self, group: Optional[dict]) -> None:
        """Handle foundation group selection - show/hide depth textbox."""
        if group is None:
            # No group selected - hide depth textbox
            if hasattr(self, 'group_depth_container'):
                self.group_depth_container.setVisible(False)
        else:
            # Group selected - show depth textbox and set to max depth of group
            if hasattr(self, 'group_depth_container'):
                self.group_depth_container.setVisible(True)
                # Calculate max depth for this group
                # Use group's depth_map if available (for imported), otherwise use scene depth_map
                max_depth = 0.0
                cells = group.get('cells', [])
                if cells:
                    # Get all depth values for this group
                    depth_values = []
                    
                    # Prefer group's depth_map if available (for imported foundations)
                    group_depth_map = group.get('depth_map')
                    if group_depth_map is not None:
                        for x, y in cells:
                            if (x, y) in group_depth_map:
                                depth_val = group_depth_map[(x, y)]
                                # Check if it's absolute Z (> 100) or relative depth
                                if depth_val > 100:
                                    # Absolute Z - convert to relative depth
                                    if self.last_placed_elevation is not None:
                                        surface_z = self.last_placed_elevation[y, x]
                                        relative_depth = surface_z - depth_val
                                        depth_values.append(max(0.0, relative_depth))
                                elif abs(depth_val) > 1e-6:
                                    # Relative depth
                                    depth_values.append(depth_val)
                    elif self.scene.foundation_depth_map is not None:
                        # Fall back to scene depth map
                        for x, y in cells:
                            if 0 <= x < self.scene.grid_size and 0 <= y < self.scene.grid_size:
                                depth_val = self.scene.foundation_depth_map[y, x]
                                # Check if it's absolute Z (> 100) or relative depth
                                if depth_val > 100:
                                    # Absolute Z - convert to relative depth
                                    if self.last_placed_elevation is not None:
                                        surface_z = self.last_placed_elevation[y, x]
                                        relative_depth = surface_z - depth_val
                                        depth_values.append(max(0.0, relative_depth))
                                elif abs(depth_val) > 1e-6:
                                    # Relative depth
                                    depth_values.append(depth_val)
                    
                    if depth_values:
                        max_depth = float(max(depth_values))
                
                # Set to max depth or default if no depth found
                if max_depth <= 1e-6:
                    max_depth = self.depth_spin.value()
                
                self.group_depth_spin.blockSignals(True)
                self.group_depth_spin.setValue(max_depth)
                self.group_depth_spin.blockSignals(False)
    
    def on_group_depth_changed(self, value: float) -> None:
        """Update depth for selected foundation group - delete and recreate with new depth."""
        if self.scene.selected_foundation_group is None:
            return
        
        group = self.scene.selected_foundation_group
        cells = group.get('cells', [])
        if not cells:
            return
        
        # Step 1: Temporarily remove foundation (clear mask and depth, restore terrain)
        for x, y in cells:
            if 0 <= y < self.scene.grid_size and 0 <= x < self.scene.grid_size:
                # Clear foundation mask
                self.scene.foundation_mask[y, x] = 0
                # Clear depth map
                if self.scene.foundation_depth_map is not None:
                    self.scene.foundation_depth_map[y, x] = 0
                # Restore original terrain elevation
                if self.scene.foundation_original_elevation is not None and callable(self.scene.get_elevation):
                    try:
                        elev = self.scene.get_elevation()
                        if elev is not None and elev.shape == self.scene.foundation_original_elevation.shape:
                            elev[y, x] = self.scene.foundation_original_elevation[y, x]
                    except:
                        pass
        
        # Step 2: Clear 3D view completely (removes all walls and foundation mesh)
        self._clear_3d_items()
        
        # Step 3: Update 3D view to show terrain without foundation
        self.update_3d_view()
        
        # Step 4: Calculate new depth values
        # Initialize depth map if needed
        if self.scene.foundation_depth_map is None:
            self.scene.foundation_depth_map = np.zeros((self.scene.grid_size, self.scene.grid_size), dtype=np.float32)
        
        # Get current depth values for this group
        # Prefer group's depth_map if available (for imported foundations)
        current_depths = {}
        max_current_depth = 0.0
        
        group_depth_map = group.get('depth_map')
        if group_depth_map is not None:
            # Use group's depth map (for imported foundations with variable depth)
            for x, y in cells:
                if (x, y) in group_depth_map:
                    depth_val = group_depth_map[(x, y)]
                    # Check if it's absolute Z (> 100) or relative depth
                    if depth_val > 100:
                        # Absolute Z - convert to relative depth
                        if self.last_placed_elevation is not None:
                            surface_z = self.last_placed_elevation[y, x]
                            relative_depth = surface_z - depth_val
                            current_depths[(x, y)] = max(0.0, relative_depth)
                        else:
                            current_depths[(x, y)] = 0.0
                    elif abs(depth_val) > 1e-6:
                        # Relative depth
                        current_depths[(x, y)] = depth_val
                    else:
                        current_depths[(x, y)] = 0.0
        elif self.scene.foundation_depth_map is not None:
            # Fall back to scene depth map
            for x, y in cells:
                if 0 <= x < self.scene.grid_size and 0 <= y < self.scene.grid_size:
                    depth_val = self.scene.foundation_depth_map[y, x]
                    # Check if it's absolute Z (> 100) or relative depth
                    if depth_val > 100:
                        # Absolute Z - convert to relative depth
                        if self.last_placed_elevation is not None:
                            surface_z = self.last_placed_elevation[y, x]
                            relative_depth = surface_z - depth_val
                            current_depths[(x, y)] = max(0.0, relative_depth)
                        else:
                            current_depths[(x, y)] = 0.0
                    elif abs(depth_val) > 1e-6:
                        # Relative depth
                        current_depths[(x, y)] = depth_val
                    else:
                        current_depths[(x, y)] = 0.0
        
        # Find max current depth
        if current_depths:
            max_current_depth = float(max(current_depths.values()))
        else:
            max_current_depth = 0.0
        
        # Calculate depth difference (additive change, not multiplicative)
        new_max_depth = float(value)
        depth_difference = new_max_depth - max_current_depth
        
        # Step 5: Re-apply foundation with new depth values (add difference to each cell)
        for x, y in cells:
            if 0 <= x < self.scene.grid_size and 0 <= y < self.scene.grid_size:
                # Restore foundation mask
                self.scene.foundation_mask[y, x] = 1
                # Calculate new depth: add difference to each cell's depth
                old_depth = current_depths.get((x, y), 0.0)
                new_depth = old_depth + depth_difference
                # Ensure depth is not negative
                new_depth = max(0.0, new_depth)
                # Update scene depth map
                self.scene.foundation_depth_map[y, x] = new_depth
        
        # Update group's depth map (for imported foundations) - this is the source of truth for variable depth
        if group.get('depth_map') is not None:
            for cell in cells:
                if cell in current_depths:
                    old_depth = current_depths[cell]
                    new_depth = old_depth + depth_difference
                    new_depth = max(0.0, new_depth)  # Ensure not negative
                    group['depth_map'][cell] = new_depth
        else:
            # Create depth map for group if it doesn't exist (for drawn groups that should have variable depth)
            group['depth_map'] = {}
            for x, y in cells:
                if (x, y) in current_depths:
                    old_depth = current_depths[(x, y)]
                    new_depth = old_depth + depth_difference
                    new_depth = max(0.0, new_depth)  # Ensure not negative
                    group['depth_map'][(x, y)] = new_depth
        
        # Step 6: Update 3D view to show foundation with new depth
        self.update_3d_view()
        self.update_foundation_profile()


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
