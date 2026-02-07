#!/usr/bin/env python3
"""
Matrix Detection and Visualization Module
Loads board_layout.json to locate and visualize the 4 matrices on each board.
- Loads board_layout.json with matrix region definitions
- Processes cleaned board images to extract matrix regions
- Creates visual overlay showing detected matrices
- Saves matrix region crops and composite visualizations

Usage:
    python detect_matrices.py
    python detect_matrices.py --input training_data/cleaned
    python detect_matrices.py --output training_output/matrices
"""
import sys
from pathlib import Path

# Add parent directory to path so config can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import json
import numpy as np
from config import CLEANED_DIR, MATRICES_DIR, LAYOUT_FILE


class MatrixDetector:
    """Detects matrix regions on board images using predefined layout"""
    
    def __init__(self, layout_file: str = None, 
                 input_dir: str = None,
                 output_dir: str = None):
        """
        Initialize matrix detector.
        
        Args:
            layout_file: Path to board_layout.json with matrix definitions (default: from config)
            input_dir: Directory containing cleaned board images (default: from config)
            output_dir: Where to save matrix detections (default: from config)
        """
        self.layout_file = Path(layout_file or LAYOUT_FILE)
        self.input_dir = Path(input_dir or CLEANED_DIR)
        self.output_dir = Path(output_dir or MATRICES_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.layout = None
        self.reference_size = None
        self.matrices_info = {}
        
        # Load layout
        self._load_layout()
        
        self.stats = {
            'processed': 0,
            'failed': 0,
            'errors': []
        }
    
    def _load_layout(self):
        """Load board layout from JSON file"""
        if not self.layout_file.exists():
            raise ValueError(f"Layout file not found: {self.layout_file}")
        
        with open(self.layout_file, 'r') as f:
            data = json.load(f)
        
        self.layout = data.get('board_layout', {})
        self.reference_size = (
            self.layout.get('max_width', 1500),
            self.layout.get('max_height', 1500)
        )
        
        # Extract matrix information
        for i in range(1, 5):
            matrix_key = f"matrix_{i}"
            if matrix_key in self.layout:
                matrix_info = self.layout[matrix_key]
                self.matrices_info[matrix_key] = {
                    'name': matrix_info.get('name', f'Matrix {i}'),
                    'location': matrix_info.get('location', ''),
                    'pixels': matrix_info.get('pixel_coordinates', []),
                    'percentages': matrix_info.get('percentage_coordinates', [])
                }
        
        print(f"✓ Loaded layout from: {self.layout_file}")
        print(f"  Reference size: {self.reference_size[0]}×{self.reference_size[1]}")
        print(f"  Matrices defined: {len(self.matrices_info)}")
    
    def scale_coordinates(self, image_shape, matrix_pixels):
        """
        Scale matrix pixel coordinates to match actual image size.
        
        Args:
            image_shape: (height, width) of actual image
            matrix_pixels: [x1, y1, x2, y2] pixel coordinates from layout
            
        Returns:
            [x1, y1, x2, y2] scaled to actual image size
        """
        if not matrix_pixels or len(matrix_pixels) < 4:
            return None
        
        img_h, img_w = image_shape[:2]
        ref_w, ref_h = self.reference_size
        
        # Calculate scale factors
        scale_x = img_w / ref_w
        scale_y = img_h / ref_h
        
        # Scale coordinates
        x1 = int(matrix_pixels[0] * scale_x)
        y1 = int(matrix_pixels[1] * scale_y)
        x2 = int(matrix_pixels[2] * scale_x)
        y2 = int(matrix_pixels[3] * scale_y)
        
        # Ensure within bounds
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        
        return [x1, y1, x2, y2]
    
    def extract_matrices(self, image: np.ndarray, image_name: str = "board"):
        """
        Extract all 4 matrices from board image.
        
        Args:
            image: Board image
            image_name: Name for saving results
            
        Returns:
            Dictionary with extracted matrix regions and visualization
        """
        matrices = {}
        visualization = image.copy()
        
        colors = {
            'matrix_1': (255, 0, 0),    # Blue
            'matrix_2': (0, 255, 0),    # Green
            'matrix_3': (0, 0, 255),    # Red
            'matrix_4': (255, 255, 0),  # Cyan
        }
        
        for matrix_key, info in self.matrices_info.items():
            coords = self.scale_coordinates(image.shape, info['pixels'])
            
            if coords is None:
                continue
            
            x1, y1, x2, y2 = coords
            w = x2 - x1
            h = y2 - y1
            
            # Extract matrix region
            matrix_region = image[y1:y2, x1:x2].copy()
            
            matrices[matrix_key] = {
                'region': matrix_region,
                'name': info['name'],
                'location': info['location'],
                'coordinates': (x1, y1, x2, y2),
                'dimensions': (w, h)
            }
            
            # Draw on visualization
            color = colors.get(matrix_key, (200, 200, 200))
            
            # Draw rectangle
            cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with background
            label = f"{matrix_key}: {info['name']}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = x1 + 5
            text_y = y1 - 5
            
            # Background rectangle for text
            cv2.rectangle(visualization, 
                         (text_x - 2, text_y - text_size[1] - 2),
                         (text_x + text_size[0] + 2, text_y + 2),
                         color, -1)
            
            # Text
            cv2.putText(visualization, label, (text_x, text_y),
                       font, font_scale, (255, 255, 255), thickness)
        
        return matrices, visualization
    
    def process_image(self, image_path: Path):
        """
        Process a single board image to detect and visualize matrices.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if successful, False otherwise
        """
        print(f"▶ {image_path.name}", end=" ")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print("✗ Could not load")
            self.stats['failed'] += 1
            self.stats['errors'].append(f"{image_path.name}: Could not load")
            return False
        
        # Extract matrices
        matrices, visualization = self.extract_matrices(image, image_path.stem)
        
        if not matrices:
            print("✗ Could not extract matrices")
            self.stats['failed'] += 1
            self.stats['errors'].append(f"{image_path.name}: No matrices extracted")
            return False
        
        # Save visualization
        vis_path = self.output_dir / f"{image_path.stem}_matrices_visualization.jpg"
        cv2.imwrite(str(vis_path), visualization)
        
        # Save individual matrix regions
        for matrix_key, data in matrices.items():
            region = data['region']
            region_path = self.output_dir / f"{image_path.stem}_{matrix_key}.jpg"
            cv2.imwrite(str(region_path), region)
        
        print(f"✓ {len(matrices)} matrices detected")
        self.stats['processed'] += 1
        
        return True
    
    def get_board_images(self):
        """Get all board images from input directory"""
        if not self.input_dir.is_dir():
            return []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = sorted([
            f for f in self.input_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        
        return images
    
    def run(self):
        """Process all board images"""
        print("\n" + "="*80)
        print("MATRIX DETECTION AND VISUALIZATION")
        print("="*80)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Layout file: {self.layout_file}\n")
        
        # Get board images
        board_images = self.get_board_images()
        
        if not board_images:
            print(f"✗ No images found in {self.input_dir}")
            return False
        
        print(f"Found {len(board_images)} image(s):\n")
        for img in board_images:
            print(f"  • {img.name}")
        
        print("\nProcessing...\n")
        
        # Process each image
        for img_path in board_images:
            self.process_image(img_path)
        
        # Print summary
        print("\n" + "="*80)
        print("MATRIX DETECTION SUMMARY")
        print("="*80)
        print(f"Processed: {self.stats['processed']}/{len(board_images)}")
        print(f"Failed: {self.stats['failed']}/{len(board_images)}")
        
        if self.stats['errors']:
            print("\nErrors:")
            for error in self.stats['errors']:
                print(f"  ✗ {error}")
        
        print(f"\n✓ Matrix detections saved to: {self.output_dir}/")
        print("  - *_matrices_visualization.jpg: Board with matrix regions highlighted")
        print("  - *_matrix_*.jpg: Individual matrix crops")
        
        return self.stats['failed'] == 0


def main():
    """Main entry point"""
    # Parse command line arguments
    layout_file = "board_layout.json"
    input_dir = "training_data/cleaned"
    output_dir = None
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--layout' and i + 1 < len(sys.argv) - 1:
            layout_file = sys.argv[i + 2]
        elif arg == '--input' and i + 1 < len(sys.argv) - 1:
            input_dir = sys.argv[i + 2]
        elif arg == '--output' and i + 1 < len(sys.argv) - 1:
            output_dir = sys.argv[i + 2]
        elif arg in ['-h', '--help']:
            print("Matrix Detection and Visualization Module")
            print("\nUsage:")
            print("  python detect_matrices.py")
            print("  python detect_matrices.py --input training_data/cleaned")
            print("  python detect_matrices.py --output training_output/matrices")
            print("\nOptions:")
            print("  --layout FILE    Path to board_layout.json (default: board_layout.json)")
            print("  --input DIR      Input directory with cleaned images (default: training_data/cleaned)")
            print("  --output DIR     Output directory for detections (default: training_output/matrices)")
            print("  -h, --help       Show this help message")
            sys.exit(0)
    
    try:
        # Run detector
        detector = MatrixDetector(layout_file, input_dir, output_dir)
        success = detector.run()
        sys.exit(0 if success else 1)
    except ValueError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
