#!/usr/bin/env python3
"""
Unified Analysis and Comparison Module
Combines three post-processing analysis tasks:
1. Analyze game states - Extract cell states from detection visualizations
2. Compare cells with reference - Create side-by-side cell comparison composites
3. Visualize comparison results - Generate histogram similarity heatmaps

Pipeline flow: clean → detect_matrices → detect_cells → analyze_and_compare

Usage:
    python analyze_and_compare.py
    python analyze_and_compare.py --cleaned training_output/cleaned
    python analyze_and_compare.py --output training_output
"""
import sys
from pathlib import Path

# Add parent directory to path so config can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import json
from typing import Dict, List, Tuple
from config import CLEANED_DIR, LAYOUT_FILE, REFERENCE_IMAGE, CELL_DETECTION_DIR


class GameStateAnalyzer:
    """Convert cell detection results into game state matrices"""
    
    def __init__(self, input_dir: str = None,
                 layout_file: str = None,
                 output_dir: str = None,
                 cleaned_dir: str = None):
        """
        Initialize game state analyzer.
        
        Args:
            input_dir: Directory containing cell detection visualizations (default: from config)
            layout_file: Board layout configuration file (default: from config)
            output_dir: Where to save state files (default: training_output/states)
            cleaned_dir: Directory with cleaned board images for reference (default: from config)
        """
        self.input_dir = Path(input_dir or CELL_DETECTION_DIR)
        self.layout_file = Path(layout_file or LAYOUT_FILE)
        self.output_dir = Path(output_dir or "training_output/states")
        self.cleaned_dir = Path(cleaned_dir or CLEANED_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate input directory
        if not self.input_dir.is_dir():
            raise ValueError(f"Input directory not found: {input_dir}")
        
        # Load layout
        with open(self.layout_file, 'r') as f:
            self.layout = json.load(f)["board_layout"]
        
        # Load reference board image
        self.reference_board_name = REFERENCE_IMAGE.replace(".png", "")
        ref_path = self.cleaned_dir / REFERENCE_IMAGE
        if ref_path.exists():
            self.reference_board_image = cv2.imread(str(ref_path))
        else:
            self.reference_board_image = None
        
        # Extract reference cells
        self.reference_cells = {}
        if self.reference_board_image is not None:
            self._extract_reference_cells()
        
        self.stats = {
            'boards_processed': 0,
            'total_pieces': 0,
            'errors': []
        }
    
    def _extract_reference_cells(self):
        """Extract all cells from reference board image."""
        if self.reference_board_image is None:
            return
        
        for matrix_id in ["matrix_1", "matrix_2", "matrix_3", "matrix_4"]:
            matrix_config = self.layout.get(matrix_id, {})
            rows = matrix_config.get("rows", 5)
            cols = matrix_config.get("cols", 5)
            
            x1, y1, x2, y2 = matrix_config.get("pixel_coordinates", [0, 0, 0, 0])
            h_margin = matrix_config.get("horizontal_margin_pixels", 0)
            v_margin = matrix_config.get("vertical_margin_pixels", 0)
            
            if x2 == 0 or y2 == 0:
                continue
            
            matrix_region = self.reference_board_image[y1:y2, x1:x2]
            mat_h, mat_w = matrix_region.shape[:2]
            
            is_triangular = matrix_id == "matrix_2"
            
            cell_width = (mat_w - h_margin * max(0, cols - 1)) / cols if cols > 0 else mat_w
            cell_height = (mat_h - v_margin * max(0, rows - 1)) / rows if rows > 0 else mat_h
            
            for row in range(rows):
                for col in range(cols):
                    if is_triangular and col < (cols - 1 - row):
                        continue
                    
                    x_start = int(col * (cell_width + h_margin))
                    y_start = int(row * (cell_height + v_margin))
                    x_end = int(x_start + cell_width)
                    y_end = int(y_start + cell_height)
                    
                    x_start = max(0, min(x_start, mat_w - 1))
                    y_start = max(0, min(y_start, mat_h - 1))
                    x_end = max(x_start + 1, min(x_end, mat_w))
                    y_end = max(y_start + 1, min(y_end, mat_h))
                    
                    cell = matrix_region[y_start:y_end, x_start:x_end]
                    if cell.size > 0:
                        self.reference_cells[f"{matrix_id}_{row}_{col}"] = cell
    
    def get_board_matrix_images(self, board_name: str) -> Dict[str, Path]:
        """Get cell detection visualization files for a board."""
        matrix_images = {}
        
        for matrix_num in range(1, 5):
            matrix_id = f"matrix_{matrix_num}"
            pattern = f"{board_name}_matrix{matrix_num}_*_detection.jpg"
            matches = list(self.input_dir.glob(pattern))
            
            if matches:
                matrix_images[matrix_id] = matches[0]
        
        return matrix_images
    
    def extract_cell_states_from_visualization(self, image_path: Path, matrix_id: str) -> List[List[int]]:
        """
        Extract cell states by comparing each cell with reference using histogram similarity.
        
        State values:
        - 0 = high similarity to reference (>= 0.2)
        - 1 = low similarity to reference (< 0.2) or missing/invalid cells
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        matrix_config = self.layout.get(matrix_id, {})
        rows = matrix_config.get("rows", 5)
        cols = matrix_config.get("cols", 5)
        
        x1, y1, x2, y2 = matrix_config["pixel_coordinates"]
        h_margin = matrix_config.get("horizontal_margin_pixels", 0)
        v_margin = matrix_config.get("vertical_margin_pixels", 0)
        
        matrix_region = image[y1:y2, x1:x2]
        mat_h, mat_w = matrix_region.shape[:2]
        
        state = []
        is_triangular = matrix_id == "matrix_2"
        
        cell_width = (mat_w - h_margin * max(0, cols - 1)) / cols if cols > 0 else mat_w
        cell_height = (mat_h - v_margin * max(0, rows - 1)) / rows if rows > 0 else mat_h
        
        # Similarity threshold: cells with similarity >= this value are considered "same as reference" (state=0)
        similarity_threshold = 0.2
        
        for row in range(rows):
            state_row = []
            for col in range(cols):
                if is_triangular and col < (cols - 1 - row):
                    state_row.append(1)  # Missing cell in triangular matrix
                    continue
                
                x_start = int(col * (cell_width + h_margin))
                y_start = int(row * (cell_height + v_margin))
                x_end = int(x_start + cell_width)
                y_end = int(y_start + cell_height)
                
                x_start = max(0, min(x_start, mat_w - 1))
                y_start = max(0, min(y_start, mat_h - 1))
                x_end = max(x_start + 1, min(x_end, mat_w))
                y_end = max(y_start + 1, min(y_end, mat_h))
                
                cell = matrix_region[y_start:y_end, x_start:x_end]
                
                if cell.size == 0:
                    state_row.append(1)  # Empty/invalid cell defaults to 1
                    continue
                
                # Compare with reference cell if available
                try:
                    # Get reference cell (from reference board image)
                    ref_cell = self.reference_cells.get(f"{matrix_id}_{row}_{col}")
                    
                    if ref_cell is not None and ref_cell.size > 0:
                        # Calculate histogram similarity
                        h, w = ref_cell.shape[:2]
                        if cell.shape != ref_cell.shape:
                            cell_resized = cv2.resize(cell, (w, h))
                        else:
                            cell_resized = cell
                        
                        hist_cell = cv2.calcHist([cell_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                        hist_ref = cv2.calcHist([ref_cell], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                        
                        similarity = float(cv2.compareHist(hist_cell, hist_ref, cv2.HISTCMP_CORREL))
                        
                        # Assign state based on similarity threshold
                        # High similarity (>= threshold) = 0 (same as reference)
                        # Low similarity (< threshold) = 1 (different from reference)
                        state_row.append(0 if similarity >= similarity_threshold else 1)
                    else:
                        # No reference cell available, default to 1
                        state_row.append(1)
                
                except Exception as e:
                    # On error, default to 1
                    state_row.append(1)
            
            state.append(state_row)
        
        return state
    
    def analyze_board(self, board_name: str) -> Dict:
        """Analyze all matrices for a board and generate state."""
        board_state = {
            "board": board_name,
            "layout_file": str(self.layout_file),
            "matrices": {}
        }
        
        matrix_images = self.get_board_matrix_images(board_name)
        
        if not matrix_images:
            raise ValueError(f"No matrix detection files found for {board_name}")
        
        for matrix_id in ["matrix_1", "matrix_2", "matrix_3", "matrix_4"]:
            if matrix_id not in matrix_images:
                continue
            
            try:
                image_path = matrix_images[matrix_id]
                matrix_config = self.layout.get(matrix_id, {})
                
                state = self.extract_cell_states_from_visualization(image_path, matrix_id)
                
                board_state["matrices"][matrix_id] = {
                    "name": matrix_config.get("name", f"matrix_{matrix_id[-1]}"),
                    "rows": matrix_config.get("rows", 5),
                    "cols": matrix_config.get("cols", 5),
                    "state": state,
                    "description": "1=different/missing (similarity < 0.2), 0=same (similarity >= 0.2)"
                }
                
                different_count = sum(1 for row in state for cell in row if cell == 1)
                board_state["matrices"][matrix_id]["different_cells_count"] = different_count
                self.stats['total_pieces'] += different_count
            
            except Exception as e:
                error_msg = f"Error analyzing {board_name}/{matrix_id}: {e}"
                print(f"  ⚠ {error_msg}")
                self.stats['errors'].append(error_msg)
                continue
        
        return board_state
    
    def save_state(self, board_state: Dict) -> bool:
        """Save board state to file."""
        try:
            board_name = board_state["board"]
            output_file = self.output_dir / f"{board_name}_state.json"
            
            with open(output_file, 'w') as f:
                json.dump(board_state, f, indent=2)
            
            return True
        except Exception as e:
            print(f"  ✗ Error saving state: {e}")
            return False
    
    def print_state(self, board_state: Dict):
        """Print state for visualization"""
        board_name = board_state["board"]
        print(f"\n  {board_name}:")
        
        for matrix_id, matrix_data in board_state.get("matrices", {}).items():
            print(f"    {matrix_id}: {matrix_data.get('pieces_count', 0)} pieces")


class CellComparator:
    """Compare cells from boards with reference image cells"""
    
    def __init__(self, cleaned_dir: str = None,
                 reference_image: str = None,
                 layout_file: str = None,
                 output_dir: str = None):
        """Initialize cell comparator."""
        self.cleaned_dir = Path(cleaned_dir or CLEANED_DIR)
        self.layout_file = Path(layout_file or LAYOUT_FILE)
        self.output_dir = Path(output_dir or "training_output/cell_comparisons")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        ref_path = self.cleaned_dir / (reference_image or REFERENCE_IMAGE)
        if not ref_path.exists():
            raise ValueError(f"Reference image not found: {ref_path}")
        
        self.reference_image = cv2.imread(str(ref_path))
        if self.reference_image is None:
            raise ValueError(f"Could not load reference image: {ref_path}")
        
        with open(self.layout_file, 'r') as f:
            self.layout = json.load(f)["board_layout"]
        
        self.stats = {
            'boards_compared': 0,
            'composites_created': 0,
            'errors': []
        }
    
    def compare_cell_with_reference(self, board_cell: np.ndarray, ref_cell: np.ndarray) -> float:
        """Compare two cells using histogram similarity."""
        if board_cell.size == 0 or ref_cell.size == 0:
            return 0.0
        
        try:
            h, w = ref_cell.shape[:2]
            if board_cell.shape != ref_cell.shape:
                board_cell = cv2.resize(board_cell, (w, h))
            
            hist_ref = cv2.calcHist([ref_cell], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist_board = cv2.calcHist([board_cell], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            
            return float(cv2.compareHist(hist_ref, hist_board, cv2.HISTCMP_CORREL))
        except:
            return 0.0
    
    def extract_cells_from_matrix(self, image: np.ndarray, matrix_id: str) -> List[np.ndarray]:
        """Extract individual cells from a matrix region."""
        matrix_config = self.layout.get(matrix_id, {})
        if not matrix_config:
            return []
        
        rows = matrix_config.get("rows", 5)
        cols = matrix_config.get("cols", 5)
        x1, y1, x2, y2 = matrix_config["pixel_coordinates"]
        h_margin = matrix_config.get("horizontal_margin_pixels", 0)
        v_margin = matrix_config.get("vertical_margin_pixels", 0)
        
        matrix_region = image[y1:y2, x1:x2]
        mat_h, mat_w = matrix_region.shape[:2]
        
        cells = []
        is_triangular = matrix_id == "matrix_2"
        
        cell_width = (mat_w - h_margin * max(0, cols - 1)) / cols if cols > 0 else mat_w
        cell_height = (mat_h - v_margin * max(0, rows - 1)) / rows if rows > 0 else mat_h
        
        for row in range(rows):
            for col in range(cols):
                if is_triangular and col < (cols - 1 - row):
                    cells.append(None)
                    continue
                
                x_start = int(col * (cell_width + h_margin))
                y_start = int(row * (cell_height + v_margin))
                x_end = int(x_start + cell_width)
                y_end = int(y_start + cell_height)
                
                # Ensure bounds are within region and have minimum size
                x_start = max(0, min(x_start, mat_w - 1))
                y_start = max(0, min(y_start, mat_h - 1))
                x_end = max(x_start + 1, min(x_end, mat_w))
                y_end = max(y_start + 1, min(y_end, mat_h))
                
                # Ensure we have valid dimensions
                if x_end <= x_start or y_end <= y_start:
                    cells.append(np.ones((1, 1, 3), dtype=np.uint8) * 255)
                    continue
                
                cell = matrix_region[y_start:y_end, x_start:x_end]
                # Ensure cell has valid content
                if cell.size == 0:
                    cell = np.ones((int(cell_height), int(cell_width), 3), dtype=np.uint8) * 255
                cells.append(cell)
        
        return cells
    
    def create_cell_comparison_composite(self, ref_cells: List[np.ndarray], 
                                        board_cells: List[np.ndarray],
                                        matrix_id: str, board_name: str) -> np.ndarray:
        """Create a composite showing side-by-side cell comparisons with similarity heatmap."""
        matrix_config = self.layout.get(matrix_id, {})
        rows = matrix_config.get("rows", 5)
        cols = matrix_config.get("cols", 5)
        
        cell_size = None
        for cell in ref_cells:
            if cell is not None and cell.size > 0:
                h, w = cell.shape[:2]
                cell_size = (w, h)
                break
        
        if cell_size is None:
            cell_size = (80, 80)
        
        cell_w, cell_h = cell_size
        padding = 5
        label_h = 25
        heatmap_w = 40  # Width for similarity heatmap
        
        # Composite width: reference cell + board cell + heatmap + padding
        comp_w = (cell_w + padding) * 2 + heatmap_w + padding * 2
        comp_h = (cell_h + padding + label_h) * rows * cols + padding + label_h * 2
        
        composite = np.ones((comp_h, comp_w, 3), dtype=np.uint8) * 255
        
        title = f"{board_name} vs Reference - {matrix_id}"
        cv2.putText(composite, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.putText(composite, "Reference", (10, label_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(composite, board_name, (cell_w + padding + 10, label_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(composite, "Similarity", (cell_w * 2 + padding * 3 + 5, label_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        y_offset = label_h * 2 + padding
        is_triangular = matrix_id == "matrix_2"
        cell_idx = 0
        
        for row in range(rows):
            for col in range(cols):
                if is_triangular and col < (cols - 1 - row):
                    cell_idx += 1
                    continue
                
                ref_cell = ref_cells[cell_idx] if cell_idx < len(ref_cells) else None
                board_cell = board_cells[cell_idx] if cell_idx < len(board_cells) else None
                cell_idx += 1
                
                # Check if cells are valid (not None, have content)
                ref_cell_valid = ref_cell is not None and hasattr(ref_cell, 'size') and ref_cell.size > 0
                board_cell_valid = board_cell is not None and hasattr(board_cell, 'size') and board_cell.size > 0
                
                # Calculate similarity for valid cells
                if ref_cell_valid and board_cell_valid:
                    similarity = self.compare_cell_with_reference(board_cell, ref_cell)
                else:
                    similarity = -1.0  # Missing or invalid cell
                
                # Prepare cell for display
                if ref_cell_valid:
                    ref_cell_resized = cv2.resize(ref_cell, cell_size)
                else:
                    ref_cell_resized = np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 200
                
                if board_cell_valid:
                    board_cell_resized = cv2.resize(board_cell, cell_size)
                else:
                    board_cell_resized = np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 200
                
                y_end = y_offset + cell_h
                x_end = padding + cell_w
                
                # Place reference cell
                composite[y_offset:y_end, padding:x_end] = ref_cell_resized
                
                # Place board cell
                x_start = cell_w + padding * 2
                x_end = x_start + cell_w
                composite[y_offset:y_end, x_start:x_end] = board_cell_resized
                
                # Create similarity heatmap
                hm_x_start = x_end + padding
                hm_x_end = hm_x_start + heatmap_w
                
                if similarity >= 0:
                    # Color based on similarity: red (low) to green (high)
                    if similarity >= 0.8:
                        color = (100, 255, 100)  # Green
                    elif similarity >= 0.6:
                        color = (100, 200, 255)  # Yellow
                    elif similarity >= 0.4:
                        color = (100, 100, 255)  # Orange
                    else:
                        color = (50, 50, 255)    # Red
                    
                    # Fill heatmap area with color
                    composite[y_offset:y_end, hm_x_start:hm_x_end] = color
                    
                    # Draw border
                    cv2.rectangle(composite, (hm_x_start, y_offset), (hm_x_end, y_end), (0, 0, 0), 2)
                    
                    # Add similarity score text (centered, larger, white for visibility)
                    text = f"{similarity:.2f}"
                    font_scale = 0.5
                    font_thickness = 1
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    text_x = hm_x_start + (heatmap_w - text_size[0]) // 2
                    text_y = y_offset + (cell_h + text_size[1]) // 2
                    cv2.putText(composite, text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
                else:
                    # Missing cell
                    composite[y_offset:y_end, hm_x_start:hm_x_end] = (200, 200, 200)
                    cv2.putText(composite, "-", (hm_x_start + 12, y_offset + cell_h // 2 + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                
                # Add cell label
                label = f"[{row},{col}]"
                cv2.putText(composite, label, (10, y_end + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 50), 1)
                
                y_offset = y_end + padding + label_h
        
        # Add legend at bottom
        legend_y = y_offset + 10
        if legend_y < comp_h - 40:
            cv2.putText(composite, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            
            # Green
            cv2.rectangle(composite, (10, legend_y + 10), (25, legend_y + 20), (100, 255, 100), -1)
            cv2.putText(composite, ">= 0.8", (30, legend_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)
            
            # Yellow
            cv2.rectangle(composite, (75, legend_y + 10), (90, legend_y + 20), (100, 200, 255), -1)
            cv2.putText(composite, "0.6-0.8", (95, legend_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)
            
            # Orange
            cv2.rectangle(composite, (155, legend_y + 10), (170, legend_y + 20), (100, 100, 255), -1)
            cv2.putText(composite, "0.4-0.6", (175, legend_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)
            
            # Red
            cv2.rectangle(composite, (235, legend_y + 10), (250, legend_y + 20), (50, 50, 255), -1)
            cv2.putText(composite, "< 0.4", (255, legend_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)
        
        return composite
    
    def compare_board_with_reference(self, board_name: str) -> bool:
        """Compare all matrices of a board with reference."""
        try:
            board_path = self.cleaned_dir / f"{board_name}.png"
            if not board_path.exists():
                raise ValueError(f"Board image not found: {board_path}")
            
            board_image = cv2.imread(str(board_path))
            if board_image is None:
                raise ValueError(f"Could not load board image: {board_path}")
            
            for matrix_id in ["matrix_1", "matrix_2", "matrix_3", "matrix_4"]:
                try:
                    ref_cells = self.extract_cells_from_matrix(self.reference_image, matrix_id)
                    board_cells = self.extract_cells_from_matrix(board_image, matrix_id)
                    
                    if not ref_cells or not board_cells:
                        continue
                    
                    composite = self.create_cell_comparison_composite(
                        ref_cells, board_cells, matrix_id, board_name
                    )
                    
                    matrix_num = matrix_id.split("_")[1]
                    matrix_config = self.layout.get(matrix_id, {})
                    matrix_name = matrix_config.get("name", matrix_id).replace(" ", "_").lower()
                    
                    output_file = self.output_dir / f"{board_name}_vs_reference_{matrix_num}_{matrix_name}_comparison.jpg"
                    cv2.imwrite(str(output_file), composite)
                    self.stats['composites_created'] += 1
                
                except Exception as e:
                    error_msg = f"Error comparing {board_name}/{matrix_id}: {e}"
                    self.stats['errors'].append(error_msg)
                    continue
            
            return True
        
        except Exception as e:
            error_msg = f"Error processing {board_name}: {e}"
            self.stats['errors'].append(error_msg)
            return False


class ComparisonVisualizer:
    """Visualize cell comparison results with heatmaps"""
    
    def __init__(self, cleaned_dir: str = None,
                 reference_image: str = None,
                 layout_file: str = None,
                 output_dir: str = None):
        """Initialize visualizer."""
        self.cleaned_dir = Path(cleaned_dir or CLEANED_DIR)
        self.layout_file = Path(layout_file or LAYOUT_FILE)
        self.output_dir = Path(output_dir or "training_output/comparison_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        ref_path = self.cleaned_dir / (reference_image or REFERENCE_IMAGE)
        if not ref_path.exists():
            raise ValueError(f"Reference image not found: {ref_path}")
        
        self.reference_image = cv2.imread(str(ref_path))
        if self.reference_image is None:
            raise ValueError(f"Could not load reference image: {ref_path}")
        
        with open(self.layout_file, 'r') as f:
            self.layout = json.load(f)["board_layout"]
        
        self.stats = {'heatmaps_created': 0}
    
    def compare_cell_with_reference(self, board_cell: np.ndarray, ref_cell: np.ndarray) -> float:
        """Compare cells using histogram similarity"""
        if board_cell.size == 0 or ref_cell.size == 0:
            return 0.0
        
        try:
            h, w = ref_cell.shape[:2]
            if board_cell.shape != ref_cell.shape:
                board_cell = cv2.resize(board_cell, (w, h))
            
            hist_ref = cv2.calcHist([ref_cell], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist_board = cv2.calcHist([board_cell], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            
            return float(cv2.compareHist(hist_ref, hist_board, cv2.HISTCMP_CORREL))
        except:
            return 0.0
    
    def extract_cells_from_image(self, image: np.ndarray, matrix_id: str) -> List[np.ndarray]:
        """Extract cells from matrix region"""
        matrix_config = self.layout.get(matrix_id, {})
        rows = matrix_config.get("rows", 5)
        cols = matrix_config.get("cols", 5)
        x1, y1, x2, y2 = matrix_config["pixel_coordinates"]
        h_margin = matrix_config.get("horizontal_margin_pixels", 0)
        v_margin = matrix_config.get("vertical_margin_pixels", 0)
        
        matrix_region = image[y1:y2, x1:x2]
        mat_h, mat_w = matrix_region.shape[:2]
        
        cells = []
        is_triangular = matrix_id == "matrix_2"
        
        cell_width = (mat_w - h_margin * max(0, cols - 1)) / cols if cols > 0 else mat_w
        cell_height = (mat_h - v_margin * max(0, rows - 1)) / rows if rows > 0 else mat_h
        
        for row in range(rows):
            for col in range(cols):
                if is_triangular and col < (cols - 1 - row):
                    cells.append(None)
                    continue
                
                x_start = int(col * (cell_width + h_margin))
                y_start = int(row * (cell_height + v_margin))
                x_end = int(x_start + cell_width)
                y_end = int(y_start + cell_height)
                
                x_start = max(0, min(x_start, mat_w - 1))
                y_start = max(0, min(y_start, mat_h - 1))
                x_end = max(x_start + 1, min(x_end, mat_w))
                y_end = max(y_start + 1, min(y_end, mat_h))
                
                cell = matrix_region[y_start:y_end, x_start:x_end]
                cells.append(cell if cell.size > 0 else None)
        
        return cells
    
    def create_heatmap_for_matrix(self, similarities: List[float], 
                                 matrix_id: str, board_name: str) -> np.ndarray:
        """Create heatmap visualization of cell similarities"""
        matrix_config = self.layout.get(matrix_id, {})
        rows = matrix_config.get("rows", 5)
        cols = matrix_config.get("cols", 5)
        
        cell_size = 60
        padding = 5
        heatmap = np.ones((rows * (cell_size + padding) + 50, cols * (cell_size + padding) + 50, 3), dtype=np.uint8) * 255
        
        title = f"{board_name} - {matrix_id} Histogram Similarity"
        cv2.putText(heatmap, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        is_triangular = matrix_id == "matrix_2"
        cell_idx = 0
        
        for row in range(rows):
            for col in range(cols):
                x = padding + col * (cell_size + padding)
                y = 50 + row * (cell_size + padding)
                
                if is_triangular and col < (cols - 1 - row):
                    cv2.rectangle(heatmap, (x, y), (x + cell_size, y + cell_size), (200, 200, 200), -1)
                    cv2.putText(heatmap, "-", (x + 25, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)
                    cell_idx += 1
                    continue
                
                sim = similarities[cell_idx] if cell_idx < len(similarities) else 0.0
                cell_idx += 1
                
                if sim >= 0.8:
                    color = (100, 255, 100)
                elif sim >= 0.6:
                    color = (100, 200, 255)
                elif sim >= 0.4:
                    color = (100, 100, 255)
                else:
                    color = (50, 50, 255)
                
                cv2.rectangle(heatmap, (x, y), (x + cell_size, y + cell_size), color, -1)
                cv2.rectangle(heatmap, (x, y), (x + cell_size, y + cell_size), (0, 0, 0), 2)
                
                text = f"{sim:.2f}"
                cv2.putText(heatmap, text, (x + 8, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return heatmap
    
    def compare_board(self, board_name: str) -> bool:
        """Compare all matrices of a board"""
        try:
            board_path = self.cleaned_dir / f"{board_name}.png"
            if not board_path.exists():
                raise ValueError(f"Board image not found: {board_path}")
            
            board_image = cv2.imread(str(board_path))
            if board_image is None:
                raise ValueError(f"Could not load board image: {board_path}")
            
            report_height = 0
            report_width = 0
            heatmaps = {}
            
            for matrix_id in ["matrix_1", "matrix_2", "matrix_3", "matrix_4"]:
                try:
                    ref_cells = self.extract_cells_from_image(self.reference_image, matrix_id)
                    board_cells = self.extract_cells_from_image(board_image, matrix_id)
                    
                    similarities = []
                    for ref_cell, board_cell in zip(ref_cells, board_cells):
                        if ref_cell is None or board_cell is None:
                            similarities.append(-1.0)
                        else:
                            sim = self.compare_cell_with_reference(board_cell, ref_cell)
                            similarities.append(sim)
                    
                    heatmap = self.create_heatmap_for_matrix(similarities, matrix_id, board_name)
                    heatmaps[matrix_id] = heatmap
                    
                    report_height = max(report_height, heatmap.shape[0])
                    report_width += heatmap.shape[1] + 20
                
                except Exception as e:
                    continue
            
            if heatmaps:
                combined = np.ones((report_height, report_width, 3), dtype=np.uint8) * 255
                x_offset = 10
                
                for matrix_id in ["matrix_1", "matrix_2", "matrix_3", "matrix_4"]:
                    if matrix_id in heatmaps:
                        hm = heatmaps[matrix_id]
                        h, w = hm.shape[:2]
                        combined[0:h, x_offset:x_offset+w] = hm
                        x_offset += w + 10
                
                output_file = self.output_dir / f"{board_name}_comparison_heatmap.jpg"
                cv2.imwrite(str(output_file), combined)
                self.stats['heatmaps_created'] += 1
                return True
            
            return False
        
        except Exception as e:
            return False


def run_analysis_pipeline(cleaned_dir=CLEANED_DIR, output_dir="training_output") -> bool:
    """
    Run complete analysis and comparison pipeline.
    
    Args:
        cleaned_dir: Directory with cleaned board images
        output_dir: Base output directory
        
    Returns:
        True if all steps succeed
    """
    print("\n" + "="*80)
    print("ANALYZING AND COMPARING BOARD GAMES")
    print("="*80)
    
    # Step 1: Analyze game states
    print("\n▶ ANALYZING GAME STATES...")
    print("-" * 80)
    
    try:
        analyzer = GameStateAnalyzer(
            input_dir=f"{output_dir}/cell_detection",
            layout_file=None,
            output_dir=f"{output_dir}/states",
            cleaned_dir=cleaned_dir
        )
        
        # Find all boards
        cell_detection_files = list(Path(f"{output_dir}/cell_detection").glob("*_matrix*_*_detection.jpg"))
        board_names = set()
        for f in cell_detection_files:
            parts = f.stem.split("_")
            board_name = parts[0]
            board_names.add(board_name)
        
        board_names = sorted(board_names)
        
        all_states = {}
        for board_name in board_names:
            try:
                board_state = analyzer.analyze_board(board_name)
                analyzer.print_state(board_state)
                
                if analyzer.save_state(board_state):
                    all_states[board_name] = board_state
                    analyzer.stats['boards_processed'] += 1
            except Exception as e:
                analyzer.stats['errors'].append(str(e))
                continue
        
        print(f"✓ Boards processed: {analyzer.stats['boards_processed']}")
        print(f"✓ Total pieces detected: {analyzer.stats['total_pieces']}")
    
    except Exception as e:
        print(f"✗ Error during state analysis: {e}")
        return False
    
    # Step 2: Compare cells with reference
    print("\n▶ COMPARING CELLS WITH REFERENCE...")
    print("-" * 80)
    
    try:
        comparator = CellComparator(
            cleaned_dir=cleaned_dir,
            layout_file=None,
            output_dir=f"{output_dir}/cell_comparisons"
        )
        
        for board_name in board_names:
            if board_name == "board0":
                continue
            
            if comparator.compare_board_with_reference(board_name):
                comparator.stats['boards_compared'] += 1
        
        print(f"✓ Boards compared: {comparator.stats['boards_compared']}")
        print(f"✓ Comparison composites created: {comparator.stats['composites_created']}")
    
    except Exception as e:
        print(f"✗ Error during comparison: {e}")
        return False
    
    # Step 3: Visualize comparison results
    print("\n▶ VISUALIZING COMPARISON RESULTS...")
    print("-" * 80)
    
    try:
        visualizer = ComparisonVisualizer(
            cleaned_dir=cleaned_dir,
            layout_file=None,
            output_dir=f"{output_dir}/comparison_reports"
        )
        
        for board_name in board_names:
            if board_name == "board0":
                continue
            
            visualizer.compare_board(board_name)
        
        print(f"✓ Heatmaps created: {visualizer.stats['heatmaps_created']}")
    
    except Exception as e:
        print(f"✗ Error during visualization: {e}")
        return False
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS AND COMPARISON COMPLETE")
    print("="*80)
    print(f"\nOutput directories:")
    print(f"  {output_dir}/states/               - Game state JSON files")
    print(f"  {output_dir}/cell_comparisons/    - Cell comparison composites")
    print(f"  {output_dir}/comparison_reports/  - Similarity heatmaps")
    
    # Print all board states
    print("\n" + "="*80)
    print("GAME STATES - ALL BOARDS AND MATRICES")
    print("="*80)
    
    states_dir = Path(output_dir) / "states"
    state_files = sorted(states_dir.glob("*_state.json"))
    
    for state_file in state_files:
        try:
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            board_name = state_data.get("board", "unknown")
            print(f"\n{board_name.upper()}:")
            print("-" * 80)
            
            for matrix_id, matrix_info in state_data.get("matrices", {}).items():
                matrix_name = matrix_info.get("name", matrix_id)
                rows = matrix_info.get("rows", 0)
                cols = matrix_info.get("cols", 0)
                different_count = matrix_info.get("different_cells_count", 0)
                
                print(f"\n  {matrix_id}: {matrix_name} ({rows}×{cols})")
                print(f"    Different/Missing cells: {different_count}")
                
                # Print state grid
                state = matrix_info.get("state", [])
                print("    State grid:")
                for row in state:
                    row_str = "      [" + ", ".join(
                        str(cell).rjust(2) for cell in row
                    ) + "]"
                    print(row_str)
        
        except Exception as e:
            print(f"✗ Error reading {state_file}: {e}")
    
    print("\n" + "="*80)
    
    return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze game states and compare boards with reference"
    )
    parser.add_argument(
        "--cleaned",
        help="Cleaned images directory (default: from config)"
    )
    parser.add_argument(
        "--output",
        default="training_output",
        help="Output base directory (default: training_output)"
    )
    
    args = parser.parse_args()
    
    try:
        success = run_analysis_pipeline(
            cleaned_dir=args.cleaned or CLEANED_DIR,
            output_dir=args.output
        )
        sys.exit(0 if success else 1)
    
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
