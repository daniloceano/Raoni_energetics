#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hovmöller Diagram Generator for Individual Cyclone Energy Analysis
===================================================================

This script generates Hovmöller diagrams (time-pressure plots) for energy terms
from cyclone analysis data for each data source individually.

Style: Publication-ready for Scientific Reports

Author: Danilo
Date: 2025
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from datetime import datetime

# Import configuration
from config import (
    DATA_SOURCES, 
    BASE_RESULTS_DIR,
    BASE_OUTPUT_DIR,
    PERIODS_FILES,
    ENERGY_TERMS,
    CONVERSION_TERMS,
    GENERATION_TERMS,
    HOVMOLLER_CONFIG,
    apply_scientific_reports_style
)

# Apply Scientific Reports style
apply_scientific_reports_style()

# Additional Hovmoller-specific configuration
HOVMOLLER_LOCAL_CONFIG = {
    # Phase colors and labels
    "phase_colors": {
        "Incipient": "#A8DADC",
        "Intensification": "#F4A261",
        "Mature": "#E63946",
        "Decay": "#457B9D"
    },
    "phase_abbreviations": {
        "Incipient": "Ic",
        "Intensification": "It",
        "Mature": "M",
        "Decay": "D"
    },
    
    # Plot styling (Scientific Reports)
    "figure_size": (180/25.4, 120/25.4),  # 180mm x 120mm
    "title_fontsize": 12,
    "label_fontsize": 11,
    "tick_fontsize": 9,
    "phase_fontsize": 9,
    "phase_fontweight": "bold",
    
    # Color schemes
    "sequential_cmap": "YlOrRd",  # For energy terms (progressive)
    "divergent_cmap": "RdBu_r",   # For conversion/generation terms
    
    # Contour settings
    "contour_levels": 15,
    "add_contour_lines": True,
    "contour_line_color": "black",
    "contour_line_alpha": 0.3,
    "contour_line_width": 0.5,
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger() -> logging.Logger:
    """
    Configure logging with emojis for an immersive experience.
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("HovmollerGenerator")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(levelname)s | %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = setup_logger()

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_pressure_level_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load pressure level data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with datetime index and pressure levels as columns
    """
    try:
        logger.info(f"   📂 Loading data from: {Path(filepath).name}")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Check if file has data (more than just header rows)
        if len(df) < 2:
            logger.info(f"      ⚠️  Skipping - insufficient data (only {len(df)} rows)")
            return None
        
        df.index = pd.to_datetime(df.index)
        
        # Convert column names to float (pressure levels)
        df.columns = df.columns.astype(float)
        
        logger.info(f"      ✅ Loaded {len(df)} time steps, {len(df.columns)} pressure levels")
        return df
        
    except Exception as e:
        logger.info(f"      ⚠️  Skipping - {str(e)[:100]}")
        return None

def load_periods(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load cyclone phase periods from CSV file.
    
    Args:
        filepath: Path to the periods CSV file
        
    Returns:
        DataFrame with phase information
    """
    if filepath is None or not Path(filepath).exists():
        logger.info(f"   ℹ️  No periods file available - skipping phase markers")
        return None
        
    try:
        logger.info(f"   📅 Loading phase periods from: {Path(filepath).name}")
        df = pd.read_csv(filepath, index_col=0)
        df['start'] = pd.to_datetime(df['start'])
        df['end'] = pd.to_datetime(df['end'])
        
        logger.info(f"      ✅ Loaded {len(df)} phase periods")
        return df
        
    except Exception as e:
        logger.error(f"      ❌ Error loading periods: {e}")
        return None

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def determine_colormap(term: str) -> Tuple[str, bool]:
    """
    Determine appropriate colormap and whether to use diverging normalization.
    
    Args:
        term: Energy term name (e.g., 'Ae', 'Ck', 'Gz')
        
    Returns:
        Tuple of (colormap name, use diverging norm)
    """
    if term in ENERGY_TERMS:
        return HOVMOLLER_LOCAL_CONFIG["sequential_cmap"], False
    else:
        return HOVMOLLER_LOCAL_CONFIG["divergent_cmap"], True

def add_phase_markers(ax, periods_df: Optional[pd.DataFrame], data_df: pd.DataFrame):
    """
    Add vertical lines and phase labels to indicate cyclone phases.
    
    Args:
        ax: Matplotlib axis object
        periods_df: DataFrame with phase information (None if not available)
        data_df: Data DataFrame to get time range
    """
    if periods_df is None:
        return
    
    # Get data time range
    data_start = data_df.index.min()
    data_end = data_df.index.max()
    
    logger.info("      📍 Adding phase markers...")
    
    # Filter periods that overlap with data
    relevant_periods = periods_df[
        (periods_df['end'] >= data_start) & (periods_df['start'] <= data_end)
    ]
    
    for idx, row in relevant_periods.iterrows():
        phase_name = idx.split()[0]  # Get phase type (Incipient, Intensification, etc.)
        start_time = row['start']
        end_time = row['end']
        
        # Add vertical line at phase boundary
        if start_time >= data_start:
            ax.axvline(start_time, color='black', linestyle='--', 
                      linewidth=1.5, alpha=0.7, zorder=10)
        
        # Add phase label in the middle of the phase period
        mid_time = start_time + (end_time - start_time) / 2
        
        if data_start <= mid_time <= data_end:
            # Get color for this phase
            color = HOVMOLLER_LOCAL_CONFIG["phase_colors"].get(phase_name, "#666666")
            abbrev = HOVMOLLER_LOCAL_CONFIG["phase_abbreviations"].get(phase_name, phase_name[:2])
            
            # Add text at the top of the plot
            ax.text(mid_time, ax.get_ylim()[1] * 0.95, abbrev,
                   fontsize=HOVMOLLER_LOCAL_CONFIG["phase_fontsize"],
                   fontweight=HOVMOLLER_LOCAL_CONFIG["phase_fontweight"],
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color, 
                            edgecolor='black', linewidth=1, alpha=0.8),
                   zorder=15)

def create_hovmoller(
    data: pd.DataFrame,
    term: str,
    periods_df: Optional[pd.DataFrame],
    output_path: str,
    source_name: str
) -> bool:
    """
    Create a Hovmöller diagram for a given energy term.
    
    Args:
        data: DataFrame with datetime index and pressure levels as columns
        term: Energy term name
        periods_df: DataFrame with phase periods (None if not available)
        output_path: Path to save the figure
        source_name: Name of the data source
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"      🎨 Creating Hovmöller diagram...")
        
        # Determine colormap
        cmap_name, use_diverging = determine_colormap(term)
        
        # Create figure
        fig, ax = plt.subplots(figsize=HOVMOLLER_LOCAL_CONFIG["figure_size"])
        
        # Prepare data for plotting
        times = data.index
        pressure_levels = data.columns.values
        values = data.T.values  # Transpose to have pressure as rows
        
        # Create meshgrid
        X, Y = np.meshgrid(times, pressure_levels)
        
        # Determine normalization
        if use_diverging:
            # For conversion/generation terms, center colormap at zero
            vmax = np.nanmax(np.abs(values))
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        else:
            # For energy terms, use simple normalization
            norm = None
        
        # Create filled contour plot
        contourf = ax.contourf(X, Y, values, 
                              levels=HOVMOLLER_LOCAL_CONFIG["contour_levels"],
                              cmap=cmap_name,
                              norm=norm,
                              extend='both')
        
        # Add contour lines if configured
        if HOVMOLLER_LOCAL_CONFIG["add_contour_lines"]:
            contour_lines = ax.contour(X, Y, values,
                                      levels=HOVMOLLER_LOCAL_CONFIG["contour_levels"],
                                      colors=HOVMOLLER_LOCAL_CONFIG["contour_line_color"],
                                      alpha=HOVMOLLER_LOCAL_CONFIG["contour_line_alpha"],
                                      linewidths=HOVMOLLER_LOCAL_CONFIG["contour_line_width"])
        
        # Add colorbar
        cbar = plt.colorbar(contourf, ax=ax, pad=0.02)
        cbar.set_label(r'{} [J$\cdot$m$^{{-2}}$]'.format(term), fontsize=HOVMOLLER_LOCAL_CONFIG["label_fontsize"])
        cbar.ax.tick_params(labelsize=HOVMOLLER_LOCAL_CONFIG["tick_fontsize"])
        
        # Add phase markers if available
        add_phase_markers(ax, periods_df, data)
        
        # Get display name for title
        source_info = next((v for v in DATA_SOURCES.values() if v["path"] == source_name), None)
        display_name = source_info["label"] if source_info else source_name
        
        # Labels and title
        ax.set_xlabel('Date (2021)', fontsize=HOVMOLLER_LOCAL_CONFIG["label_fontsize"])
        ax.set_ylabel('Pressure Level (hPa)', fontsize=HOVMOLLER_LOCAL_CONFIG["label_fontsize"])
        ax.set_title(f'{term} Hovmöller Diagram - {display_name}',
                    fontsize=HOVMOLLER_LOCAL_CONFIG["title_fontsize"], fontweight='bold')
        
        # Adjust tick parameters
        ax.tick_params(labelsize=HOVMOLLER_LOCAL_CONFIG["tick_fontsize"])
        
        # Invert y-axis (lower pressure at top)
        ax.set_ylim(max(pressure_levels), min(pressure_levels))
        
        # Format x-axis (time)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=12))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=HOVMOLLER_CONFIG["dpi"], bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"      ✅ Saved: {Path(output_path).name}")
        return True
        
    except Exception as e:
        logger.error(f"      ❌ Error creating Hovmöller for {term}: {e}")
        return False

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_data_source(source_name: str, base_dir: Path):
    """
    Process a single data source and generate all its Hovmöller diagrams.
    
    Args:
        source_name: Name of the data source directory
        base_dir: Base directory path
    """
    logger.info("=" * 70)
    logger.info(f"🔄 Processing: {source_name}")
    logger.info("=" * 70)
    
    # Setup paths
    results_dir = base_dir / BASE_RESULTS_DIR / source_name / "results_vertical_levels"
    output_dir = base_dir / BASE_OUTPUT_DIR / source_name / "hovmollers"
    
    # Check if results directory exists
    if not results_dir.exists():
        logger.error(f"❌ Results directory not found: {results_dir}")
        return 0
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")
    
    # Load periods if available
    periods_file = PERIODS_FILES.get(source_name)
    if periods_file:
        periods_file = base_dir / BASE_RESULTS_DIR / source_name / periods_file
    periods_df = load_periods(str(periods_file) if periods_file else None)
    
    # Find all pressure level CSV files
    csv_files = sorted(results_dir.glob("*_pressure_level.csv"))
    
    if not csv_files:
        # Try with plevels instead
        csv_files = sorted(results_dir.glob("*_plevels.csv"))
    
    if not csv_files:
        logger.warning(f"⚠️  No pressure-level CSV files found in {results_dir} — skipping Hovmöller for {source_name}")
        return 0
    
    logger.info(f"📊 Found {len(csv_files)} data files")
    logger.info("")
    
    # Process each file
    success_count = 0
    
    for csv_file in csv_files:
        # Extract term name from filename
        term = csv_file.stem.replace("_pressure_level", "").replace("_plevels", "")
        
        logger.info(f"   🔄 Processing {term}...")
        
        # Load data
        data = load_pressure_level_data(str(csv_file))
        
        if data is None:
            continue
        
        # Create output filename
        output_file = output_dir / f"hovmoller_{term}.png"
        
        # Create Hovmöller diagram
        if create_hovmoller(data, term, periods_df, str(output_file), source_name):
            success_count += 1
        
        logger.info("")
    
    logger.info(f"✨ Completed {source_name}: {success_count}/{len(csv_files)} diagrams generated")
    logger.info("")
    
    return success_count

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main execution function to generate all Hovmöller diagrams for all sources.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("🌀 HOVMÖLLER DIAGRAM GENERATOR - INDIVIDUAL SOURCES")
    logger.info("=" * 70)
    logger.info("")
    
    # Get base directory
    base_dir = Path(__file__).parent
    
    # Process each data source
    total_success = 0
    total_sources = len(DATA_SOURCES)
    
    for source_info in DATA_SOURCES.values():
        source_path = source_info["path"]
        success = process_data_source(source_path, base_dir)
        total_success += success
    
    # Final summary
    logger.info("=" * 70)
    logger.info(f"🎉 ALL COMPLETED: {total_success} total Hovmöller diagrams generated")
    logger.info(f"📂 Figures saved in: {base_dir / BASE_OUTPUT_DIR}")
    logger.info("=" * 70)
    logger.info("")

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
