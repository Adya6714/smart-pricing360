#!/usr/bin/env python3
"""
Script to organize output folders with consistent naming conventions.
Each notebook should have its own output directory with matching prefix.

Usage:
    python organize_outputs.py
"""

import shutil
from pathlib import Path

project_root = Path(__file__).parent

# Define notebook-specific output directories
NOTEBOOK_OUTPUTS = {
    "01_mm_all": {
        "out_dir": "outputs_01_mm_all",
        "cache_dirs": [
            "data/processed/cache_01_mm_all_text",
            "data/processed/cache_01_mm_all_clip"
        ]
    },
    "01_mm_fast_v2": {
        "out_dir": "outputs_01_mm_fast_v2",
        "cache_dirs": [
            "data/processed/cache_01_mm_fast_v2"
        ]
    },
    "02_mm_streamlined": {
        "out_dir": "outputs_02_mm_streamlined",
        "cache_dirs": [
            "data/processed/cache_02_mm_streamlined"
        ]
    },
    "03_mm_minimal": {
        "out_dir": "outputs_03_mm_minimal",
        "cache_dirs": [
            "data/processed/cache_03_mm_minimal"
        ]
    }
}

def create_output_structure():
    """Create all required output directories"""
    print("Creating output directory structure...")
    
    for notebook_name, config in NOTEBOOK_OUTPUTS.items():
        # Create main output directory
        out_dir = project_root / config["out_dir"]
        for subdir in ["oof", "test_preds", "reports", "models"]:
            (out_dir / subdir).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {config['out_dir']}")
        
        # Create cache directories
        for cache_dir in config["cache_dirs"]:
            (project_root / cache_dir).mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {cache_dir}")
    
    print("\n✅ All directories created!")

def show_current_structure():
    """Show current output structure"""
    print("\nCurrent Output Structure:")
    print("=" * 60)
    
    for notebook_name, config in NOTEBOOK_OUTPUTS.items():
        print(f"\n📓 {notebook_name}.ipynb:")
        print(f"   Output:  {config['out_dir']}/")
        print(f"            ├── oof/")
        print(f"            ├── test_preds/")
        print(f"            ├── reports/")
        print(f"            └── models/")
        
        if config["cache_dirs"]:
            print(f"   Cache:")
            for cache in config["cache_dirs"]:
                print(f"            └── {cache}/")
    
    print("\n" + "=" * 60)

def main():
    print("=" * 60)
    print("OUTPUT ORGANIZATION SCRIPT")
    print("=" * 60)
    
    show_current_structure()
    
    response = input("\nCreate all directories? (y/n): ").lower().strip()
    
    if response == 'y':
        create_output_structure()
    else:
        print("\n❌ Cancelled")

if __name__ == "__main__":
    main()

