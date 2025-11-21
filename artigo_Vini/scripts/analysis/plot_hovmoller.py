#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hovmöller Diagram Generator for Cyclone Energy Analysis
========================================================

This script generates Hovmöller diagrams (time-pressure plots) for energy terms
from cyclone analysis data. It visualizes the vertical structure evolution of
energy components throughout the cyclone lifecycle.

Author: Automated Script
Date: 2024
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

# ============================================================================
# CONFIGURATION SECTION - Easily customizable parameters
# ============================================================================

CONFIG = {
    # File paths
    "data_dir": "melissa_track/results_vertical_levels",
    "periods_file": "periods.csv",
    "output_dir": "Figures/hovmollers",
    
    # Energy terms to plot (will be automatically detected from files)
    "energy_terms": ["Ae", "Az", "Ke", "Kz"],
    "conversion_terms": ["Ce", "Ck", "Cz"],
    "generation_terms": ["Ge", "Gz"],
    
    # Color schemes
    "sequential_cmap": "YlOrRd",  # For energy terms (progressive)
    "divergent_cmap": "RdBu_r",   # For conversion/generation terms
    
    # Plot styling
    "figure_size": (14, 8),
    "dpi": 300,
    "title_fontsize": 16,
    "label_fontsize": 14,
    "tick_fontsize": 12,
    "phase_fontsize": 11,
    "phase_fontweight": "bold",
    
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
        logger.info(f"📂 Loading data from: {Path(filepath).name}")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        
        # Convert column names to float (pressure levels)
        df.columns = df.columns.astype(float)
        
        logger.info(f"   ✅ Loaded {len(df)} time steps, {len(df.columns)} pressure levels")
        return df
        
    except Exception as e:
        logger.error(f"   ❌ Error loading {filepath}: {e}")
        return None

def load_periods(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load cyclone phase periods from CSV file.
    
    Args:
        filepath: Path to the periods CSV file
        
    Returns:
        DataFrame with phase information
    """
    try:
        logger.info(f"📅 Loading phase periods from: {filepath}")
        df = pd.read_csv(filepath, index_col=0)
        df['start'] = pd.to_datetime(df['start'])
        df['end'] = pd.to_datetime(df['end'])
        
        logger.info(f"   ✅ Loaded {len(df)} phase periods")
        return df
        
    except Exception as e:
        logger.error(f"   ❌ Error loading periods: {e}")
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
    energy_terms = CONFIG["energy_terms"]
    
    if term in energy_terms:
        return CONFIG["sequential_cmap"], False
    else:
        return CONFIG["divergent_cmap"], True

def add_phase_markers(ax, periods_df: pd.DataFrame, data_df: pd.DataFrame):
    """
    Add vertical lines and phase labels to indicate cyclone phases.
    
    Args:
        ax: Matplotlib axis object
        periods_df: DataFrame with phase information
        data_df: Data DataFrame to get time range
    """
    if periods_df is None:
        return
    
    # Get data time range
    data_start = data_df.index.min()
    data_end = data_df.index.max()
    
    logger.info("   📍 Adding phase markers...")
    
    # Filter periods that overlap with data
    relevant_periods = periods_df[
        (periods_df['end'] >= data_start) & (periods_df['start'] <= data_end)
    ]
    
    for idx, row in relevant_periods.iterrows():
        phase_name = idx.split()[0]  # Get phase type (Incipient, Intensification, etc.)
        start_time = max(row['start'], data_start)
        end_time = min(row['end'], data_end)
        
        # Add vertical line at phase start
        ax.axvline(start_time, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Calculate middle position for label
        middle_time = start_time + (end_time - start_time) / 2
        
        # Get phase abbreviation and color
        abbrev = CONFIG["phase_abbreviations"].get(phase_name, phase_name[:2])
        color = CONFIG["phase_colors"].get(phase_name, "gray")
        
        # Add phase label at top of plot
        ax.text(middle_time, 1.05, abbrev,
                transform=ax.get_xaxis_transform(),
                ha='center', va='top',
                fontsize=CONFIG["phase_fontsize"],
                fontweight=CONFIG["phase_fontweight"],
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.7, edgecolor='none'))

def create_hovmoller(data: pd.DataFrame, term: str, periods_df: Optional[pd.DataFrame],
                     output_path: str) -> bool:
    """
    Create a Hovmöller diagram for a given energy term.
    
    Args:
        data: DataFrame with time index and pressure levels as columns
        term: Energy term name
        periods_df: DataFrame with phase information
        output_path: Path to save the figure
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"🎨 Creating Hovmöller diagram for {term}...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=CONFIG["figure_size"], dpi=CONFIG["dpi"])
        
        # Prepare data for plotting
        # Sort pressure levels in descending order (1000 hPa at bottom)
        pressure_levels = sorted(data.columns, reverse=True)
        times = data.index
        
        # Create meshgrid
        X, Y = np.meshgrid(times, pressure_levels)
        Z = data[pressure_levels].T.values
        
        # Determine colormap and normalization
        cmap_name, use_diverging = determine_colormap(term)
        
        if use_diverging:
            # Diverging colormap centered at zero
            vmax = np.nanmax(np.abs(Z))
            vmin = -vmax
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            # Sequential colormap
            vmin, vmax = np.nanmin(Z), np.nanmax(Z)
            norm = None
        
        # Create filled contour plot
        contourf = ax.contourf(X, Y, Z, levels=CONFIG["contour_levels"],
                               cmap=cmap_name, norm=norm, extend='neither')
        
        # Add contour lines if configured
        if CONFIG["add_contour_lines"]:
            contour = ax.contour(X, Y, Z, levels=CONFIG["contour_levels"],
                                colors=CONFIG["contour_line_color"],
                                alpha=CONFIG["contour_line_alpha"],
                                linewidths=CONFIG["contour_line_width"])
        
        # Add colorbar
        cbar = plt.colorbar(contourf, ax=ax, orientation='vertical', pad=0.02)
        
        # Get unit from TERM_DETAILS
        from utils import TERM_DETAILS
        unit = "J·m⁻²"  # Default unit
        for category, details in TERM_DETAILS.items():
            if term in details["terms"]:
                unit = details["unit"]
                break
        
        cbar.set_label(f'{term} ({unit})', fontsize=CONFIG["label_fontsize"])
        cbar.ax.tick_params(labelsize=CONFIG["tick_fontsize"])
        
        # Add phase markers
        add_phase_markers(ax, periods_df, data)
        
        # Configure axes
        ax.set_ylabel('Pressure (hPa)', fontsize=CONFIG["label_fontsize"])
        ax.set_xlabel('Time', fontsize=CONFIG["label_fontsize"])
        ax.set_title(f'Hovmöller Diagram - {term}', 
                    fontsize=CONFIG["title_fontsize"], fontweight='bold', pad=40)
        
        # Format y-axis (pressure)
        ax.tick_params(axis='both', labelsize=CONFIG["tick_fontsize"])
        ax.set_ylim(max(pressure_levels), min(pressure_levels))  # Invert y-axis
        
        # Format x-axis (time)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        
        logger.info(f"   ✅ Saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Error creating Hovmöller for {term}: {e}")
        return False

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main execution function to generate all Hovmöller diagrams.
    """
    logger.info("=" * 70)
    logger.info("🌀 HOVMÖLLER DIAGRAM GENERATOR")
    logger.info("=" * 70)
    logger.info("")
    
    # Get base directory
    base_dir = Path(__file__).parent
    data_dir = base_dir / CONFIG["data_dir"]
    output_dir = base_dir / CONFIG["output_dir"]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info("")
    
    # Load periods
    periods_file = base_dir / CONFIG["periods_file"]
    periods_df = load_periods(str(periods_file))
    logger.info("")
    
    # Find all pressure level CSV files
    csv_files = sorted(data_dir.glob("*_pressure_level.csv"))
    
    if not csv_files:
        logger.error(f"❌ No CSV files found in {data_dir}")
        return
    
    logger.info(f"📊 Found {len(csv_files)} data files")
    logger.info("")
    
    # Process each file
    success_count = 0
    total_count = len(csv_files)
    
    for csv_file in csv_files:
        # Extract term name from filename
        term = csv_file.stem.replace("_pressure_level", "")
        
        logger.info(f"🔄 Processing {term}...")
        
        # Load data
        data = load_pressure_level_data(str(csv_file))
        
        if data is None:
            continue
        
        # Create output filename
        output_file = output_dir / f"hovmoller_{term}.png"
        
        # Create Hovmöller diagram
        if create_hovmoller(data, term, periods_df, str(output_file)):
            success_count += 1
        
        logger.info("")
    
    # Summary
    logger.info("=" * 70)
    logger.info(f"✨ COMPLETED: {success_count}/{total_count} Hovmöller diagrams generated")
    logger.info(f"📂 Figures saved in: {output_dir}")
    logger.info("=" * 70)

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
