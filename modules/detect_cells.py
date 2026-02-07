#!/usr/bin/env python3
"""
Detect Individual Cells in Matrix Images
Analyzes each matrix image and detects individual cells with piece detection.
Creates visualizations showing detected cell boundaries and piece presence.

Features:
- Uses layout configuration to divide matrices into cells
- Detects piece presence in each cell (color/content analysis)
- Creates visualization with cell boundaries and piece markers
- Exports detection results with cell coordinates and piece info

Usage:
    python detect_cells.py
    python detect_cells.py --input training_output/matrices
    python detect_cells.py --output training_output/cell_detection
"""
import sys
from pathlib import Path

# Add parent directory to path so config can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import json
from typing import List, Tuple, Dict
import math
from config import MATRICES_DIR, CELL_DETECTION_DIR, COMPOSITES_DIR, LAYOUT_FILE, PIECE_DARKNESS_THRESHOLD, GRID_CELL_TOLERANCE


class CellDetector:
    """Detect individual cells in matrix images and visualize results"""
    
    def __init__(self, input_dir: str = None,
                 output_dir: str = None,
                 layout_file: str = None):
        """
        Initialize cell detector.
        
        Args:
            input_dir: Directory containing matrix images (default: from config)
            output_dir: Where to save detection visualizations (default: from config)
            layout_file: Board layout configuration file (default: from config)
        """
        self.input_dir = Path(input_dir or MATRICES_DIR)
        self.output_dir = Path(output_dir or CELL_DETECTION_DIR)
        self.layout_file = layout_file or LAYOUT_FILE
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate input directory
        if not self.input_dir.is_dir():
            raise ValueError(f"Input directory not found: {input_dir}")
        
        # Load layout
        self.layout = self._load_layout()
        
        # Matrix configurations (from layout)
        self.matrix_configs = {}
        for matrix_id in ["matrix_1", "matrix_2", "matrix_3", "matrix_4"]:
            matrix_layout = self.layout.get(matrix_id, {})
            self.matrix_configs[matrix_id] = {
                "rows": matrix_layout.get("rows", 5),
                "cols": matrix_layout.get("cols", 5),
                "name": matrix_layout.get("name", f"matrix_{matrix_id[-1]}"),
                "location": matrix_layout.get("location", "unknown"),
                "h_margin": matrix_layout.get("horizontal_margin_pixels", 0),
                "v_margin": matrix_layout.get("vertical_margin_pixels", 0)
            }
        
        self.stats = {
            'processed_matrices': 0,
            'total_cells_detected': 0,
            'total_pieces_detected': 0,
            'failed': 0,
            'errors': []
        }
    
    def _load_layout(self) -> dict:
        """Load board layout configuration"""
        try:
            with open(self.layout_file, 'r') as f:
                layout_data = json.load(f)
                return layout_data.get("board_layout", {})
        except Exception as e:
            print(f"⚠ Warning: Could not load layout file ({e}), using defaults")
            return {}
    
    def get_matrix_images(self) -> List[Path]:
        """Get all matrix_*.jpg files from input directory"""
        image_files = sorted([
            f for f in self.input_dir.glob("*_matrix_*.jpg")
        ])
        return image_files
    
    def detect_grid_lines(self, image: np.ndarray, expected_rows: int = None, 
                         expected_cols: int = None) -> Tuple[List[float], List[float]]:
        """
        Detect horizontal and vertical grid lines using edge detection and Hough transform.
        
        Args:
            image: Matrix image
            expected_rows: Expected number of rows (for sorting/filtering lines)
            expected_cols: Expected number of columns (for sorting/filtering lines)
            
        Returns:
            Tuple of (horizontal_y_positions, vertical_x_positions)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection with morphological operations
        edges = cv2.Canny(gray, 50, 150)
        
        h, w = image.shape[:2]
        
        # Detect horizontal lines using dilated edges
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w//4, 1))
        edges_h = cv2.dilate(edges, kernel_h, iterations=1)
        h_lines = cv2.HoughLines(edges_h, 1, np.pi/180, min(w//2, 500))
        
        # Detect vertical lines using dilated edges
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h//4))
        edges_v = cv2.dilate(edges, kernel_v, iterations=1)
        v_lines = cv2.HoughLines(edges_v, 1, np.pi/180, min(h//2, 500))
        
        horizontal_ys = []
        if h_lines is not None:
            for line in h_lines:
                rho, theta = line[0]
                # For nearly horizontal lines (theta ≈ π/2)
                sin_theta = np.sin(theta)
                if abs(sin_theta) > 0.5:  # Roughly horizontal
                    y = int(rho / sin_theta)
                    if 0 <= y < h:
                        horizontal_ys.append(y)
        
        vertical_xs = []
        if v_lines is not None:
            for line in v_lines:
                rho, theta = line[0]
                # For nearly vertical lines (theta ≈ 0)
                cos_theta = np.cos(theta)
                if abs(cos_theta) > 0.5:  # Roughly vertical
                    x = int(rho / cos_theta)
                    if 0 <= x < w:
                        vertical_xs.append(x)
        
        # Sort and remove duplicates (group nearby lines)
        horizontal_ys = self._group_nearby_values(sorted(list(set([int(y) for y in horizontal_ys]))), tolerance=GRID_CELL_TOLERANCE)
        vertical_xs = self._group_nearby_values(sorted(list(set([int(x) for x in vertical_xs]))), tolerance=GRID_CELL_TOLERANCE)
        
        # Ensure we have the correct number of lines
        if expected_rows and len(horizontal_ys) > expected_rows + 1:
            horizontal_ys = self._select_evenly_spaced(horizontal_ys, expected_rows + 1)
        
        if expected_cols and len(vertical_xs) > expected_cols + 1:
            vertical_xs = self._select_evenly_spaced(vertical_xs, expected_cols + 1)
        
        return horizontal_ys, vertical_xs
    
    def _group_nearby_values(self, values: List[float], tolerance: float = None) -> List[float]:
        """Group nearby values and keep the median of each group."""
        if tolerance is None:
            tolerance = GRID_CELL_TOLERANCE
        if not values:
            return values
        
        groups = []
        current_group = [values[0]]
        
        for val in values[1:]:
            if val - current_group[-1] <= tolerance:
                current_group.append(val)
            else:
                groups.append(int(np.median(current_group)))
                current_group = [val]
        
        if current_group:
            groups.append(int(np.median(current_group)))
        
        return groups
    
    def _select_evenly_spaced(self, values: List[float], target_count: int) -> List[float]:
        """Select target_count evenly spaced values from the list."""
        if len(values) <= target_count:
            return values
        
        # Use interpolation to select evenly spaced indices
        indices = np.linspace(0, len(values) - 1, target_count).astype(int)
        return [values[i] for i in indices]
    
    
    def detect_cells_by_grid(self, image: np.ndarray, rows: int, cols: int) -> List[Tuple[int, int, int, int, bool]]:
        """
        Detect cells by finding grid lines (better for matrices with many cells).
        
        Args:
            image: Matrix image
            rows: Expected number of rows
            cols: Expected number of columns
            
        Returns:
            List of (x, y, w, h, has_piece) tuples
        """
        h_lines, v_lines = self.detect_grid_lines(image, rows, cols)
        
        # If we don't have enough lines, fall back to rectangular division
        if len(h_lines) < rows + 1 or len(v_lines) < cols + 1:
            return None
        
        cells = []
        
        # Create cells from grid intersections
        for r in range(rows):
            if r + 1 >= len(h_lines):
                break
            y_start = int(h_lines[r])
            y_end = int(h_lines[r + 1])
            
            for c in range(cols):
                if c + 1 >= len(v_lines):
                    break
                x_start = int(v_lines[c])
                x_end = int(v_lines[c + 1])
                
                w = x_end - x_start
                h = y_end - y_start
                
                if w > 0 and h > 0:
                    cell_img = image[y_start:y_end, x_start:x_end]
                    has_piece = self.detect_piece_in_cell(cell_img)
                    cells.append((x_start, y_start, w, h, has_piece))
        
        return cells if len(cells) == rows * cols else None
    
    def detect_piece_in_cell(self, cell: np.ndarray) -> bool:
        """
        Detect if a cell contains a piece by analyzing color/content.
        
        Args:
            cell: Cell image
            
        Returns:
            True if piece detected, False otherwise
        """
        if cell.size == 0:
            return False
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for pieces (adjust based on actual piece colors)
        # Looking for darker colors (pieces) vs light background
        
        # Calculate saturation and value statistics
        h, s, v = cv2.split(hsv)
        
        # Piece detection heuristics:
        # 1. Check if cell has significant dark areas (low value)
        # 2. Check if cell has moderate-to-high saturation (colored piece)
        
        # Count pixels that are "dark" (potential piece pixels)
        dark_mask = v < PIECE_DARKNESS_THRESHOLD  # Dark pixels
        dark_pixels = np.count_nonzero(dark_mask)
        total_pixels = h.size
        
        # If more than 15% of pixels are dark, likely contains a piece
        return (dark_pixels / total_pixels) > 0.15
    
    def detect_cells_rectangular(self, image: np.ndarray, rows: int, cols: int,
                                 h_margin: int = 0, v_margin: int = 0) -> List[Tuple[int, int, int, int, bool]]:
        """
        Detect cells in a rectangular matrix.
        
        Args:
            image: Matrix image
            rows: Number of rows
            cols: Number of columns
            h_margin: Horizontal margin between cells
            v_margin: Vertical margin between cells
            
        Returns:
            List of (x, y, w, h, has_piece) tuples
        """
        height, width = image.shape[:2]
        
        # Account for margins
        total_h_margin = (cols - 1) * h_margin
        total_v_margin = (rows - 1) * v_margin
        
        available_width = width - total_h_margin
        available_height = height - total_v_margin
        
        cell_height = available_height // rows
        cell_width = available_width // cols
        
        cells = []
        for r in range(rows):
            for c in range(cols):
                # Calculate cell position
                y_start = r * (cell_height + v_margin)
                y_end = min(y_start + cell_height, height)
                x_start = c * (cell_width + h_margin)
                x_end = min(x_start + cell_width, width)
                
                cell_img = image[y_start:y_end, x_start:x_end]
                has_piece = self.detect_piece_in_cell(cell_img)
                
                cells.append((x_start, y_start, x_end - x_start, y_end - y_start, has_piece))
        
        return cells
    
    def detect_cells_triangular(self, image: np.ndarray, rows: int,
                               h_margin: int = 0, v_margin: int = 0) -> List[Tuple[int, int, int, int, bool]]:
        """
        Detect cells in a triangular matrix.
        
        Args:
            image: Triangular matrix image
            rows: Number of rows
            h_margin: Horizontal margin between cells
            v_margin: Vertical margin between cells
            
        Returns:
            List of (x, y, w, h, has_piece) tuples
        """
        height, width = image.shape[:2]
        
        # Account for row margins
        total_v_margin = (rows - 1) * v_margin
        available_height = height - total_v_margin
        cell_height = available_height // rows
        
        cells = []
        for r in range(rows):
            cells_in_row = r + 1
            
            # Account for column margins in this row
            total_h_margin_row = (cells_in_row - 1) * h_margin
            available_width = width - total_h_margin_row
            cell_width = available_width // cells_in_row
            
            y_start = r * (cell_height + v_margin)
            y_end = min(y_start + cell_height, height)
            
            # Right-aligned positioning
            total_cols = rows
            max_cell_width = width // total_cols
            
            for c in range(cells_in_row):
                x_start = (total_cols - cells_in_row) * max_cell_width + c * (max_cell_width + h_margin)
                x_end = min(x_start + max_cell_width, width)
                
                cell_img = image[y_start:y_end, x_start:x_end]
                has_piece = self.detect_piece_in_cell(cell_img)
                
                cells.append((x_start, y_start, x_end - x_start, y_end - y_start, has_piece))
        
        return cells
    
    def create_visualization(self, image: np.ndarray, cells: List[Tuple[int, int, int, int, bool]],
                           matrix_name: str) -> np.ndarray:
        """
        Create visualization showing detected cells with boundaries and piece markers.
        
        Args:
            image: Original matrix image
            cells: List of detected cells (x, y, w, h, has_piece)
            matrix_name: Name of the matrix for labeling
            
        Returns:
            Visualization image
        """
        vis = image.copy()
        
        # Colors
        empty_color = (100, 255, 100)  # Green for empty cells
        piece_color = (50, 50, 255)    # Red for cells with pieces
        border_thickness = 2
        
        cell_count = 0
        piece_count = 0
        
        for x, y, w, h, has_piece in cells:
            if w <= 0 or h <= 0:
                continue
            
            # Choose color based on piece presence
            color = piece_color if has_piece else empty_color
            
            # Draw cell rectangle
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, border_thickness)
            
            # Add a small indicator in the center if piece is present
            if has_piece:
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(vis, (center_x, center_y), 3, (0, 0, 255), -1)
                piece_count += 1
            
            cell_count += 1
        
        # Add text label with cell count
        text = f"{matrix_name}: {cell_count} cells, {piece_count} with pieces"
        cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return vis, cell_count, piece_count
    
    def process_matrix_image(self, image_path: Path) -> bool:
        """
        Process a single matrix image and create detection visualization.
        
        Args:
            image_path: Path to matrix image
            
        Returns:
            True if successful, False otherwise
        """
        # Parse filename
        parts = image_path.stem.split("_")
        
        if len(parts) < 3:
            self.stats['errors'].append(f"{image_path.name}: Could not parse filename")
            return False
        
        board_name = parts[0]
        matrix_num = parts[-1]
        matrix_type = f"matrix_{matrix_num}"
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            self.stats['errors'].append(f"{image_path.name}: Could not load")
            return False
        
        print(f"▶ {image_path.name:40}", end=" ")
        
        # Detect cells
        try:
            config = self.matrix_configs[matrix_type]
            h_margin = config.get("h_margin", 0)
            v_margin = config.get("v_margin", 0)
            matrix_name = config.get("name", "")
            
            # Use grid line detection for matrix_1 (has many cells)
            if matrix_type == "matrix_1":
                cells = self.detect_cells_by_grid(image, config["rows"], config["cols"])
                if cells is not None:
                    print(f"  [grid-based]", end=" ")
                else:
                    # Fall back to rectangular division if grid detection fails
                    print(f"  [rectangular-fallback]", end=" ")
                    cells = self.detect_cells_rectangular(image, config["rows"], config["cols"], h_margin, v_margin)
            # Check if matrix name contains "triangular" to determine shape
            elif "triangular" in matrix_name.lower():
                cells = self.detect_cells_triangular(image, config["rows"], h_margin, v_margin)
            else:
                cells = self.detect_cells_rectangular(image, config["rows"], config["cols"], h_margin, v_margin)
        
        except Exception as e:
            print(f"✗ Error detecting cells: {e}")
            self.stats['errors'].append(f"{image_path.name}: {str(e)}")
            self.stats['failed'] += 1
            return False
        
        # Create visualization
        try:
            matrix_name = self.matrix_configs[matrix_type]["name"]
            # Sanitize name for filename (lowercase, replace spaces with underscores)
            sanitized_name = matrix_name.lower().replace(" ", "_").replace("-", "_")
            vis, cell_count, piece_count = self.create_visualization(image, cells, matrix_name)
        except Exception as e:
            print(f"✗ Error creating visualization: {e}")
            self.stats['errors'].append(f"{image_path.name}: Visualization error")
            self.stats['failed'] += 1
            return False
        
        # Save visualization
        try:
            output_path = self.output_dir / f"{board_name}_matrix{matrix_num}_{sanitized_name}_detection.jpg"
            cv2.imwrite(str(output_path), vis)
        except Exception as e:
            print(f"✗ Error saving visualization: {e}")
            self.stats['errors'].append(f"{image_path.name}: Save error")
            self.stats['failed'] += 1
            return False
        
        self.stats['processed_matrices'] += 1
        self.stats['total_cells_detected'] += cell_count
        self.stats['total_pieces_detected'] += piece_count
        
        print(f"✓ {cell_count:3d} cells ({piece_count:2d} with pieces)")
        return True
    
    def create_composites(self) -> bool:
        """
        Create composite images grouping matrices by type across all boards.
        
        Returns:
            True if successful, False otherwise
        """
        composites_dir = Path(COMPOSITES_DIR)
        composites_dir.mkdir(parents=True, exist_ok=True)
        
        # Group detection images by matrix type
        images_by_matrix = {
            "matrix_1": [],
            "matrix_2": [],
            "matrix_3": [],
            "matrix_4": []
        }
        
        # Find all detection images (search in output_dir where visualizations are saved)
        for img_path in sorted(self.output_dir.glob("*_matrix*_detection.jpg")):
            # Parse filename: board{n}_matrix{m}_{name}_detection.jpg
            parts = img_path.stem.split("_")
            
            # Find matrix part (matrix1, matrix2, matrix3, matrix4)
            for part in parts:
                if part.startswith("matrix"):
                    if len(part) > 6:  # matrix1, matrix2, etc.
                        matrix_num = part[6]  # Get the digit
                        matrix_type = f"matrix_{matrix_num}"
                        if matrix_type in images_by_matrix:
                            images_by_matrix[matrix_type].append(img_path)
                    break
        
        # Create composite for each matrix type
        composites_created = 0
        for matrix_type in ["matrix_1", "matrix_2", "matrix_3", "matrix_4"]:
            images = images_by_matrix[matrix_type]
            
            if not images:
                continue
            
            # Load images
            loaded = []
            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is not None:
                    loaded.append(img)
            
            if not loaded:
                continue
            
            # Create grid composite
            h, w = loaded[0].shape[:2]
            num_images = len(loaded)
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
            
            padding = 10
            composite_w = cols * w + (cols + 1) * padding
            composite_h = rows * h + (rows + 1) * padding
            
            # Initialize with light gray background
            composite = np.ones((composite_h, composite_w, 3), dtype=np.uint8) * 240
            
            # Place images in grid
            for idx, img in enumerate(loaded):
                row = idx // cols
                col = idx % cols
                
                y = row * (h + padding) + padding
                x = col * (w + padding) + padding
                
                composite[y:y+h, x:x+w] = img
                
                # Add board label
                board_name = images[idx].stem.split("_")[0]
                cv2.putText(composite, board_name, (x + 5, y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Add title
            matrix_name = self.matrix_configs[matrix_type].get("name", matrix_type)
            title = f"{matrix_type}: {matrix_name}"
            
            # Add title background
            font_scale = 1.0
            thickness = 2
            text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            title_height = text_size[1] + 10
            
            title_img = np.ones((title_height + 10, composite_w, 3), dtype=np.uint8) * 50
            final_composite = np.vstack([title_img, composite])
            
            # Add title text
            cv2.putText(final_composite, title, (10, title_height),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Save composite
            sanitized_name = matrix_name.lower().replace(" ", "_").replace("-", "_")
            output_path = composites_dir / f"all_boards_{matrix_type}_{sanitized_name}_composite.jpg"
            cv2.imwrite(str(output_path), final_composite)
            composites_created += 1
        
        return composites_created > 0
    
    def run(self):
        """Process all matrix images"""
        print("\n" + "="*80)
        print("DETECTING CELLS IN MATRICES")
        print("="*80)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}\n")
        
        # Get matrix images
        matrix_images = self.get_matrix_images()
        
        if not matrix_images:
            print("✗ No matrix images found")
            return False
        
        print(f"Found {len(matrix_images)} matrix image(s):\n")
        for img in matrix_images:
            print(f"  • {img.name}")
        
        print("\nProcessing...\n")
        
        # Process each matrix image
        for img_path in matrix_images:
            self.process_matrix_image(img_path)
        
        # Print summary
        print("\n" + "="*80)
        print("CELL DETECTION SUMMARY")
        print("="*80)
        print(f"Processed matrices: {self.stats['processed_matrices']}/{len(matrix_images)}")
        print(f"Total cells detected: {self.stats['total_cells_detected']}")
        print(f"Total pieces detected: {self.stats['total_pieces_detected']}")
        print(f"Failed: {self.stats['failed']}")
        
        if self.stats['errors']:
            print("\nErrors:")
            for error in self.stats['errors']:
                print(f"  ✗ {error}")
        
        print(f"\n✓ Detection visualizations saved to: {self.output_dir}/")
        print("\nVisualization Legend:")
        print("  Green rectangles: Empty cells")
        print("  Red rectangles:   Cells with detected pieces")
        print("  Red circles:      Center marker for cells with pieces")
        
        # Generate composites
        print("\n" + "="*80)
        print("GENERATING COMPOSITE IMAGES")
        print("="*80)
        
        if self.create_composites():
            print(f"\n✓ Composite images saved to: {COMPOSITES_DIR}/")
        else:
            print("\n⊘ No composites created")
        
        return self.stats['failed'] == 0


def main():
    """Main entry point"""
    input_dir = "training_output/matrices"
    output_dir = None
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--input' and i + 1 < len(sys.argv) - 1:
            input_dir = sys.argv[i + 2]
        elif arg == '--output' and i + 1 < len(sys.argv) - 1:
            output_dir = sys.argv[i + 2]
        elif arg in ['-h', '--help']:
            print("Detect Cells in Matrix Images")
            print("\nUsage:")
            print("  python detect_cells.py")
            print("  python detect_cells.py --input training_output/matrices")
            print("  python detect_cells.py --output training_output/cell_detection")
            print("\nOptions:")
            print("  --input DIR    Input directory (default: training_output/matrices)")
            print("  --output DIR   Output directory (default: training_output/cell_detection)")
            print("  -h, --help     Show this help message")
            print("\nFeatures:")
            print("  - Detects individual cells based on matrix layout")
            print("  - Analyzes cell content to detect piece presence")
            print("  - Creates visualization with cell boundaries")
            print("  - Color-coded: green=empty, red=has piece")
            print("\nOutput:")
            print("  *_matrix*_detection.jpg - Visualization with detected cells")
            sys.exit(0)
    
    try:
        detector = CellDetector(input_dir, output_dir)
        success = detector.run()
        sys.exit(0 if success else 1)
    except ValueError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
