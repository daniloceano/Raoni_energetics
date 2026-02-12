#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run All Analysis Scripts for Raoni Energetics
==============================================

This script executes all analysis scripts in the correct sequence:
1. plot_hovmoller_individual.py - Hovmöller diagrams for all sources
2. plot_LEC_individual.py - LEC diagrams for all sources
3. plot_timeseries_comparison_multiplot.py - Multi-panel comparisons
4. plot_taylor_diagrams.py - Taylor diagram comparisons

Author: Danilo
Date: 2025
"""

import sys
import subprocess
from pathlib import Path
import logging

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger() -> logging.Logger:
    """Configure logging."""
    logger = logging.Logger("RunAllAnalysis")
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

logger = setup_logger()

# ============================================================================
# SCRIPT DEFINITIONS
# ============================================================================

SCRIPTS = [
    {
        "name": "Hovmöller Diagrams (Individual)",
        "file": "plot_hovmoller_individual.py",
        "description": "Generate time-pressure diagrams for each data source"
    },
    {
        "name": "LEC Diagrams (Individual)",
        "file": "plot_LEC_individual.py",
        "description": "Generate Lorenz Energy Cycle box diagrams for each source"
    },
    {
        "name": "Time Series Comparisons",
        "file": "plot_timeseries_comparison_multiplot.py",
        "description": "Generate multi-panel time series comparing all sources"
    },
    {
        "name": "Taylor Diagrams",
        "file": "plot_taylor_diagrams.py",
        "description": "Generate Taylor diagrams comparing models against ERA5"
    }
]

# ============================================================================
# EXECUTION FUNCTIONS
# ============================================================================

def run_script(script_info: dict, script_dir: Path) -> bool:
    """
    Run a single analysis script.
    
    Args:
        script_info: Dictionary with script information
        script_dir: Directory containing the scripts
        
    Returns:
        True if successful, False otherwise
    """
    script_file = script_dir / script_info["file"]
    
    if not script_file.exists():
        logger.error(f"❌ Script not found: {script_file}")
        return False
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"📊 Running: {script_info['name']}")
    logger.info(f"   {script_info['description']}")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # Run the script using Python
        result = subprocess.run(
            [sys.executable, str(script_file)],
            cwd=script_dir,
            capture_output=False,
            text=True,
            check=True
        )
        
        logger.info("")
        logger.info(f"✅ Completed: {script_info['name']}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error("")
        logger.error(f"❌ Error running {script_info['name']}")
        logger.error(f"   Exit code: {e.returncode}")
        return False
    except Exception as e:
        logger.error("")
        logger.error(f"❌ Unexpected error running {script_info['name']}: {e}")
        return False

def main():
    """
    Main execution function.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("🚀 RAONI ENERGETICS ANALYSIS - RUN ALL SCRIPTS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Total scripts to run: {len(SCRIPTS)}")
    logger.info("")
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Validate configuration first
    logger.info("🔍 Validating configuration...")
    try:
        # Import config to validate
        sys.path.insert(0, str(script_dir))
        from config import validate_config
        
        if not validate_config():
            logger.error("")
            logger.error("❌ Configuration validation failed!")
            logger.error("   Please check config.py and ensure all paths are correct")
            logger.error("")
            return 1
        
        logger.info("✅ Configuration is valid")
    except Exception as e:
        logger.error(f"❌ Error loading configuration: {e}")
        return 1
    
    # Run all scripts
    success_count = 0
    failed_scripts = []
    
    for i, script_info in enumerate(SCRIPTS, 1):
        logger.info(f"[{i}/{len(SCRIPTS)}] Starting: {script_info['name']}...")
        
        if run_script(script_info, script_dir):
            success_count += 1
        else:
            failed_scripts.append(script_info['name'])
            
            # Ask if user wants to continue
            logger.warning("")
            logger.warning("⚠️  Script failed! Do you want to continue with remaining scripts?")
            logger.warning("   Press Enter to continue, or Ctrl+C to abort...")
            try:
                input()
            except KeyboardInterrupt:
                logger.info("")
                logger.info("🛑 Aborted by user")
                break
    
    # Final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total scripts: {len(SCRIPTS)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(failed_scripts)}")
    
    if failed_scripts:
        logger.info("")
        logger.info("Failed scripts:")
        for script_name in failed_scripts:
            logger.info(f"  - {script_name}")
    
    logger.info("=" * 80)
    logger.info("")
    
    if success_count == len(SCRIPTS):
        logger.info("🎉 All scripts completed successfully!")
        return 0
    else:
        logger.warning(f"⚠️  {len(failed_scripts)} script(s) failed")
        return 1

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())
