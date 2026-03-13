#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run All Analysis Scripts for Raoni Energetics
==============================================

This script executes all analysis scripts in the correct sequence for both
analysis modes (fixed and semi_lagrangian):
  1. plot_hovmoller_individual.py - Hovmöller diagrams for all sources
  2. plot_LEC_individual.py - LEC diagrams for all sources
  3. plot_timeseries_comparison_multiplot.py - Multi-panel comparisons
  4. plot_taylor_diagrams.py - Taylor diagram comparisons

Figures are saved in separate subfolders per mode:
  Figures/fixed/
  Figures/semi_lagrangian/

Author: Danilo
Date: 2025
"""

import sys
import os
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
# MODE DEFINITIONS
# ============================================================================

# Both modes that will be executed in sequence.
# Each produces figures in Figures/<mode>/
ANALYSIS_MODES = ["fixed", "semi_lagrangian"]

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

def run_script(script_info: dict, script_dir: Path, mode: str) -> bool:
    """
    Run a single analysis script with a given ANALYSIS_MODE.

    Args:
        script_info: Dictionary with script information
        script_dir: Directory containing the scripts
        mode: One of 'fixed' or 'semi_lagrangian'

    Returns:
        True if successful, False otherwise
    """
    script_file = script_dir / script_info["file"]
    
    if not script_file.exists():
        logger.error(f"❌ Script not found: {script_file}")
        return False
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"📊 [{mode}] Running: {script_info['name']}")
    logger.info(f"   {script_info['description']}")
    logger.info("=" * 80)
    logger.info("")
    
    # Build environment with ANALYSIS_MODE set
    env = os.environ.copy()
    env["ANALYSIS_MODE"] = mode

    try:
        # Run the script using Python
        result = subprocess.run(
            [sys.executable, str(script_file)],
            cwd=script_dir,
            capture_output=False,
            text=True,
            check=True,
            env=env,
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
    Main execution function – runs all scripts for both analysis modes.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("🚀 RAONI ENERGETICS ANALYSIS - RUN ALL SCRIPTS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Modes to process : {', '.join(ANALYSIS_MODES)}")
    logger.info(f"Scripts per mode : {len(SCRIPTS)}")
    logger.info(f"Total script runs: {len(ANALYSIS_MODES) * len(SCRIPTS)}")
    logger.info("")

    # Get script directory
    script_dir = Path(__file__).parent

    # Accumulate results across modes
    mode_summaries = {}

    for mode in ANALYSIS_MODES:
        logger.info("")
        logger.info("#" * 80)
        logger.info(f"#  MODE: {mode.upper()}")
        logger.info("#" * 80)

        # Validate configuration for this mode
        logger.info(f"\n🔍 Validating configuration for mode '{mode}'...")
        try:
            sys.path.insert(0, str(script_dir))
            # Re-import config with the target mode active
            import importlib
            os.environ["ANALYSIS_MODE"] = mode
            import config as _cfg
            importlib.reload(_cfg)
            if not _cfg.validate_config():
                logger.error(f"❌ Configuration validation failed for mode '{mode}'!")
                logger.error("   Skipping this mode.")
                mode_summaries[mode] = {"success": 0, "failed": list(s["name"] for s in SCRIPTS)}
                continue
            logger.info(f"✅ Configuration valid for mode '{mode}'")
        except Exception as e:
            logger.error(f"❌ Error loading configuration for mode '{mode}': {e}")
            mode_summaries[mode] = {"success": 0, "failed": list(s["name"] for s in SCRIPTS)}
            continue

        # Run all scripts for this mode
        success_count = 0
        failed_scripts = []

        for i, script_info in enumerate(SCRIPTS, 1):
            logger.info(f"[{i}/{len(SCRIPTS)}] Starting: {script_info['name']}...")

            if run_script(script_info, script_dir, mode):
                success_count += 1
            else:
                failed_scripts.append(script_info["name"])

                logger.warning("")
                logger.warning("⚠️  Script failed! Do you want to continue with remaining scripts?")
                logger.warning("   Press Enter to continue, or Ctrl+C to abort...")
                try:
                    input()
                except KeyboardInterrupt:
                    logger.info("")
                    logger.info("🛑 Aborted by user")
                    mode_summaries[mode] = {"success": success_count, "failed": failed_scripts}
                    _print_final_summary(mode_summaries)
                    return 1

        mode_summaries[mode] = {"success": success_count, "failed": failed_scripts}

    _print_final_summary(mode_summaries)
    total_failed = sum(len(v["failed"]) for v in mode_summaries.values())
    return 0 if total_failed == 0 else 1


def _print_final_summary(mode_summaries: dict):
    """Print a per-mode and overall execution summary."""
    total_scripts = len(SCRIPTS)
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 EXECUTION SUMMARY")
    logger.info("=" * 80)
    for mode, result in mode_summaries.items():
        status = "✅" if not result["failed"] else "⚠️ "
        logger.info(
            f"{status} [{mode}]  {result['success']}/{total_scripts} scripts OK"
        )
        for name in result["failed"]:
            logger.info(f"       ❌ {name}")
    logger.info("=" * 80)
    logger.info("")

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())
