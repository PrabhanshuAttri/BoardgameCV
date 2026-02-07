# Board Game Computer Vision Project

## Project Overview
This project processes board game table images through a complete pipeline:
1. **Board Cleaning** - Normalizes images with perspective correction using board0 reference
2. **Matrix Detection** - Extracts 4 matrices per board based on board_layout.json
3. **Cell Detection** - Identifies individual cells and detects piece presence
4. **Composite Generation** - Creates visualization summaries grouping matrices across boards

Detects game state from 4 matrices on each board:
- **Matrix 1**: 5×20 positioning grid (positioning area)
- **Matrix 2**: 5-row triangular staircase staging area  
- **Matrix 3**: 5×5 piece state final area
- **Matrix 4**: 1×7 negative scoring area

## Key Files
### Pipeline
- `main.py` - Orchestrates all pipeline steps (cleanup → clean → detect_matrices → detect_cells)
- `board_layout.json` - Configuration defining matrix dimensions, positions, and names

### Modules (in `modules/`)
- `clean_source_boards.py` - Board image normalization with perspective correction (AKAZE feature matching)
- `detect_matrices.py` - Extracts matrix regions using pixel coordinates from board_layout.json
- `detect_cells.py` - Cell detection with 3 methods: grid-based (Hough), rectangular, triangular
  - Also generates composite images grouping same matrix types across boards

## Detection Methods
- **Grid Detection** (matrix_1): Canny edge detection + Hough transform for precise boundaries
- **Rectangular Detection** (matrix_3, matrix_4): Uniform grid subdivision
- **Triangular Detection** (matrix_2): Right-aligned staircase layout
- **Piece Detection**: HSV brightness analysis to identify pieces in cells

## Input/Output Structure
```
training_data/                 → Input images (board0.png, board1.png, etc)
training_output/
  ├── cleaned/                  → Normalized images (1497px × 1500px)
  ├── matrices/                 → Matrix region crops
  ├── cell_detection/           → Cell visualizations with piece markers
  └── composites/               → Grid composites of same matrices across boards
```

## Configuration
- Edit `board_layout.json` to modify:
  - Matrix dimensions (rows, cols)
  - Pixel coordinates for matrix regions
  - Matrix names and descriptions
  - Cell margins for spacing

## Project Status
- [x] Project structure created
- [x] Board cleaning with perspective correction implemented
- [x] Matrix detection functional (all 6 boards)
- [x] Cell detection functional (all 24 matrices, 882 cells)
- [x] Grid-based detection for improved accuracy
- [x] Piece detection working
- [x] Composite visualization generation
- [x] Example usage and execution ready

## Running the Pipeline
```bash
python main.py
```

## Dependencies
- Python 3.8+
- OpenCV (cv2)
- NumPy
