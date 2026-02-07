# Board Game Computer Vision Pipeline

Automated detection of board game configurations from table images. Identifies multiple board layouts, extracts 4 matrices per board, detects individual cells, and visualizes piece positions.

## Features

- **Board Cleaning**: Normalizes source images with perspective correction using board0 as reference
- **Matrix Detection**: Extracts 4 distinct matrices from each board:
  - **Matrix 1**: 5×20 positioning grid (positioning area)
  - **Matrix 2**: 5-row triangular staircase staging area
  - **Matrix 3**: 5×5 piece state final area
  - **Matrix 4**: 1×7 negative scoring area
- **Cell Detection**: Identifies individual cells using grid line detection (Hough transform) for precise boundaries
- **Piece Detection**: Analyzes cell brightness to identify piece presence
- **Composite Visualization**: Groups same matrix types across all boards for pattern analysis

## Project Structure

```
BoardGameCV/
├── main.py                    # Pipeline orchestrator
├── board_layout.json          # Matrix configuration
├── requirements.txt           # Python dependencies
├── modules/
│   ├── clean_source_boards.py # Image normalization & perspective correction
│   ├── detect_matrices.py     # Matrix region extraction
│   └── detect_cells.py        # Cell detection & visualization
├── training_data/
│   └── source/                # Input board images (board0.png - board5.png)
└── training_output/           # Generated outputs
    ├── cleaned/               # Normalized board images
    ├── matrices/              # Extracted matrix regions
    ├── cell_detection/        # Cell visualizations with piece markers
    └── composites/            # Composite images by matrix type
```

## Installation

### Requirements
- Python 3.8+
- OpenCV (cv2)
- NumPy

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd BoardGameCV
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Full Pipeline

```bash
python main.py
```

This executes all 3 processing steps:
1. **Step 1**: Cleans source images from `training_data/source/`
2. **Step 2**: Detects matrices on cleaned images
3. **Step 3**: Detects cells and generates composite images

All results are saved to `training_output/` with automatic cleanup between runs.

### Custom Directories

```bash
python main.py --source /path/to/images --output /path/to/output
```

## Input Data

**Required**: Board images in `training_data/`:
- `board0.png` - Reference image (used for perspective correction)
- `board1.png` through `boardN.png` - Additional boards to process

**Format**: JPEG or PNG images of board game configurations

## Output Structure

### Step 1: Cleaned Images
- `training_output/cleaned/boardX.png`
  - Normalized to 1497×1500px
  - Perspective-corrected using board0 as reference
  - Ready for matrix detection

### Step 2: Matrix Detections
- `training_output/matrices/boardX_matrix_N.jpg`
  - Individual matrix region crops
- `training_output/matrices/boardX_matrices_visualization.jpg`
  - Visual annotation showing all 4 matrices

### Step 3: Cell Detections & Composites
- `training_output/cell_detection/boardX_matrixN_NAME_detection.jpg`
  - Cell boundaries with piece visualization
  - Green rectangles: empty cells
  - Red rectangles: cells with detected pieces
  - Red circles: center marker for pieces

- `training_output/composites/all_boards_matrix_N_NAME_composite.jpg`
  - All 6 boards' same matrix type combined in grid layout
  - Shows piece patterns across all boards

## Configuration

Edit `board_layout.json` to customize matrix definitions:

```json
{
  "board_layout": {
    "matrix_1": {
      "name": "Positioning Area",
      "rows": 5,
      "cols": 20,
      "pixel_coordinates": [x1, y1, x2, y2]
    }
  }
}
```

## Pipeline Modules

### clean_source_boards.py
Normalizes board images with perspective correction:
- Uses AKAZE feature matching to align boards to board0 reference
- Applies perspective transformation for consistent viewing angle
- Saves normalized images to `training_data/cleaned/`

### detect_matrices.py
Extracts matrix regions based on board_layout.json:
- Scales pixel coordinates to actual image size
- Crops and saves individual matrix images
- Creates visualization overlay showing all matrices

### detect_cells.py
Detects individual cells within matrices:
- **Grid detection** (matrix_1): Uses Canny edge detection + Hough transform for precise cell boundaries
- **Rectangular detection** (matrix_3, matrix_4): Uniform grid division
- **Triangular detection** (matrix_2): Right-aligned staircase layout
- **Piece detection**: Analyzes cell darkness to identify pieces
- **Composites**: Combines same matrix types from all boards

## Performance Notes

- Pipeline runs in ~10 seconds for 6 boards on standard hardware
- Grid detection improves matrix_1 accuracy for varying cell widths
- Consistent board orientation (90° angles) required for optimal results
- High contrast between pieces and background improves detection

## Troubleshooting

**No boards detected / cleaning step fails:**
- Ensure `board0.png` exists in `training_data/source/`
- Board0 is used as the reference for perspective transformation
- Check image contrast and board visibility

**Incorrect matrix extraction:**
- Verify `board_layout.json` has correct pixel coordinates
- Coordinates should match your board dimensions

**Pieces not detected:**
- Verify piece colors create dark areas in cells
- Check image lighting (avoid heavy shadows)

**Grid detection issues (matrix_1):**
- Ensure clear grid lines between cells
- Falls back to rectangular division if insufficient grid lines detected

## Dependencies

- **OpenCV** (cv2): Image processing, edge detection, Hough transform
- **NumPy**: Array operations, homography calculations

See `requirements.txt` for exact versions.

## License

MIT
