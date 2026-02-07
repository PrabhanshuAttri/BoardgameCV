#!/usr/bin/env python3
"""
Main Pipeline Runner
Runs the complete board game image processing pipeline in sequence:
1. Clean source board images (remove margins, normalize, dewarp)
2. Detect matrices on cleaned images
3. Detect cells and visualize piece presence
   - Also generates composite images by matrix type
4. Analyze and compare
   - Extract game states from cell detection
   - Compare cells with reference image (board0)
   - Visualize histogram similarity heatmaps

Usage:
    python main.py
    python main.py --source training_data
    python main.py --output training_output
"""
import sys
import subprocess
import shutil
from pathlib import Path
from config import SOURCE_DIR, OUTPUT_DIR, CLEANED_DIR


def run_pipeline(source_dir=SOURCE_DIR, output_dir=OUTPUT_DIR):
    """
    Run the complete pipeline in sequence.
    
    Args:
        source_dir: Source directory with board images (default: from .env)
        output_dir: Output directory for results (default: from .env)
    
    Returns:
        True if all steps succeed, False otherwise
    """
    print("\n" + "="*80)
    print("BOARD GAME IMAGE PROCESSING PIPELINE")
    print("="*80)
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Cleanup: Remove previous outputs
    print("Cleaning up previous outputs...\n")
    
    cleanup_dirs = [
        Path(output_dir)
    ]
    
    for cleanup_dir in cleanup_dirs:
        if cleanup_dir.exists():
            try:
                shutil.rmtree(cleanup_dir)
                print(f"  ✓ Removed: {cleanup_dir}")
            except Exception as e:
                print(f"  ⚠ Could not remove {cleanup_dir}: {e}")
    
    print()
    
    # Step 1: Clean source boards
    print("="*80)
    print("STEP 1: CLEANING SOURCE BOARD IMAGES")
    print("="*80)
    
    result = subprocess.run(
        [sys.executable, "modules/clean_source_boards.py", "--source", source_dir, "--output", CLEANED_DIR],
        cwd=Path.cwd()
    )
    
    if result.returncode != 0:
        print("\n✗ Step 1 failed: clean_source_boards.py")
        return False
    
    print("\n✓ Step 1 completed successfully")
    
    # Step 2: Detect matrices
    print("\n" + "="*80)
    print("STEP 2: DETECTING MATRICES")
    print("="*80)
    
    result = subprocess.run(
        [sys.executable, "modules/detect_matrices.py", "--input", CLEANED_DIR],
        cwd=Path.cwd()
    )
    
    if result.returncode != 0:
        print("\n✗ Step 2 failed: detect_matrices.py")
        return False
    
    print("\n✓ Step 2 completed successfully")
    
    # Step 3: Detect cells visually
    print("\n" + "="*80)
    print("STEP 3: DETECTING CELLS WITH VISUALIZATION")
    print("="*80)
    
    result = subprocess.run(
        [sys.executable, "modules/detect_cells.py"],
        cwd=Path.cwd()
    )
    
    if result.returncode != 0:
        print("\n✗ Step 3 failed: detect_cells.py")
        return False
    
    print("\n✓ Step 3 completed successfully")
    
    # Step 4: Analyze and compare (merged analysis pipeline)
    print("\n" + "="*80)
    print("STEP 4: ANALYZING AND COMPARING BOARDS")
    print("="*80)
    
    result = subprocess.run(
        [sys.executable, "modules/analyze_and_compare.py", "--cleaned", CLEANED_DIR, "--output", output_dir],
        cwd=Path.cwd()
    )
    
    if result.returncode != 0:
        print("\n✗ Step 4 failed: analyze_and_compare.py")
        return False
    
    print("\n✓ Step 4 completed successfully")
    
    # Pipeline complete
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("✓ All processing steps completed successfully")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/cleaned/             - Cleaned and normalized board images")
    print(f"  {output_dir}/matrices/            - Detected matrix regions")
    print(f"  {output_dir}/cell_detection/      - Cell detection visualizations")
    print(f"  {output_dir}/composites/          - Composite images by matrix type")
    print(f"  {output_dir}/states/              - Game state JSON files")
    print(f"  {output_dir}/cell_comparisons/    - Side-by-side cell comparison composites")
    print(f"  {output_dir}/comparison_reports/  - Histogram similarity heatmap visualizations")
    
    return True


def main():
    """Main entry point"""
    source_dir = SOURCE_DIR
    output_dir = OUTPUT_DIR
    
    # Parse command line arguments
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--source' and i + 1 < len(sys.argv) - 1:
            source_dir = sys.argv[i + 2]
        elif arg == '--output' and i + 1 < len(sys.argv) - 1:
            output_dir = sys.argv[i + 2]
        elif arg in ['-h', '--help']:
            print("Board Game Image Processing Pipeline")
            print("\nUsage:")
            print("  python main.py")
            print(f"  python main.py --source {SOURCE_DIR}")
            print(f"  python main.py --output {OUTPUT_DIR}")
            print("\nOptions:")
            print(f"  --source DIR   Source directory (default: {SOURCE_DIR})")
            print(f"  --output DIR   Output directory (default: {OUTPUT_DIR})")
            print("  -h, --help     Show this help message")
            print("\nPipeline Steps:")
            print("  1. Clean source boards - Detect and crop board regions")
            print("  2. Detect matrices - Locate 4 matrices on each board")
            print("  3. Detect cells - Visualize cell boundaries and piece detection")
            print("     (generates composite images by matrix type)")
            print("  4. Analyze and compare - Game state extraction and board comparison")
            print("     - Extract game states from cell detection")
            print("     - Compare with reference board (board0)")
            print("     - Generate histogram similarity heatmaps")
            print("\nOutput:")
            print("  cleaned/              - Cleaned board images without margins")
            print("  matrices/             - Matrix regions visualization and crops")
            print("  cell_detection/       - Cell detection visualizations with piece markers")
            print("  composites/           - Composite images grouping matrices by type")
            print("  states/               - Game state JSON files")
            print("  cell_comparisons/     - Side-by-side cell comparison composites")
            print("  comparison_reports/   - Histogram similarity heatmap visualizations")
            sys.exit(0)
    
    try:
        success = run_pipeline(source_dir, output_dir)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
