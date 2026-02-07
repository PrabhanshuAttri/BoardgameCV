#!/usr/bin/env python3
"""
Configuration Management
Loads environment variables from .env file for pipeline settings.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Directories
SOURCE_DIR = os.getenv("SOURCE_DIR", "training_data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "training_output")
CLEANED_DIR = os.getenv("CLEANED_DIR", "training_output/cleaned")
MATRICES_DIR = os.getenv("MATRICES_DIR", "training_output/matrices")
CELL_DETECTION_DIR = os.getenv("CELL_DETECTION_DIR", "training_output/cell_detection")
COMPOSITES_DIR = os.getenv("COMPOSITES_DIR", "training_output/composites")

# Files & References
LAYOUT_FILE = os.getenv("LAYOUT_FILE", "board_layout.json")
REFERENCE_IMAGE = os.getenv("REFERENCE_IMAGE", "board0.png")

# Image dimensions (normalized size)
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "1497"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "1500"))

# Detection settings
MIN_DETECTION_CONFIDENCE = float(os.getenv("MIN_DETECTION_CONFIDENCE", "0.15"))
PIECE_DARKNESS_THRESHOLD = int(os.getenv("PIECE_DARKNESS_THRESHOLD", "150"))
GRID_CELL_TOLERANCE = int(os.getenv("GRID_CELL_TOLERANCE", "5"))

# Display settings
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
VERBOSE = os.getenv("VERBOSE", "True").lower() == "true"


def get_config():
    """Return all configuration values as a dictionary."""
    return {
        "source_dir": SOURCE_DIR,
        "output_dir": OUTPUT_DIR,
        "cleaned_dir": CLEANED_DIR,
        "matrices_dir": MATRICES_DIR,
        "cell_detection_dir": CELL_DETECTION_DIR,
        "composites_dir": COMPOSITES_DIR,
        "layout_file": LAYOUT_FILE,
        "reference_image": REFERENCE_IMAGE,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "min_detection_confidence": MIN_DETECTION_CONFIDENCE,
        "piece_darkness_threshold": PIECE_DARKNESS_THRESHOLD,
        "grid_cell_tolerance": GRID_CELL_TOLERANCE,
        "debug_mode": DEBUG_MODE,
        "verbose": VERBOSE,
    }


if __name__ == "__main__":
    """Display current configuration when run as script."""
    print("Current Configuration:")
    print("=" * 60)
    for key, value in get_config().items():
        print(f"  {key:<30} = {value}")
    print("=" * 60)
