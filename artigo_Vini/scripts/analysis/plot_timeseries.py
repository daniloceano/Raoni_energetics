#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Plotter for Cyclone Energy Analysis
================================================

This script generates time series plots for energy terms from cyclone analysis.
Creates three separate figures:
1. Energy terms (Az, Ae, Kz, Ke)
2. Conversion and Generation/Dissipation terms
3. Boundary terms

Author: Automated Script
Date: 2024
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# ============================================================================
# CONFIGURATION SECTION - Easily customizable parameters
# ============================================================================

CONFIG = {
    # File paths
    "results_file": "melissa_track/melissa_track_results.csv",
    "periods_file": "periods.csv",
    "output_dir": "Figures/timeseries",
    
    # Terms to plot in each figure
    "energy_terms": ["Az", "Ae", "Kz", "Ke"],
    "conversion_gen_terms": ["Cz", "Ca", "Ck", "Ce", "Gz", "Ge"],
    "boundary_terms": ["BAz", "BAe", "BKz", "BKe", "BΦZ", "BΦE"],
    
    # Plot styling
    "figure_size": (16, 10),
    "dpi": 300,
    "title_fontsize": 18,
    "label_fontsize": 14,
    "tick_fontsize": 12,
    "legend_fontsize": 11,
    "phase_fontsize": 10,
    "phase_alpha": 0.15,
    
    # Colors for each term (can be customized)
    "colors": {
        # Energy terms
        "Az": "#2E86AB",  # Blue
        "Ae": "#A23B72",  # Purple
        "Kz": "#F18F01",  # Orange
        "Ke": "#C73E1D",  # Red
        
        # Conversion terms
        "Cz": "#06A77D",  # Green
        "Ca": "#D4B483",  # Tan
        "Ck": "#6A4C93",  # Purple
        "Ce": "#1B998B",  # Teal
        
        # Generation terms
        "Gz": "#FF6B6B",  # Light red
        "Ge": "#4ECDC4",  # Light teal
        
        # Boundary terms
        "BAz": "#3D5A80",   # Dark blue
        "BAe": "#98C1D9",   # Light blue
        "BKz": "#EE6C4D",   # Coral
        "BKe": "#293241",   # Dark gray
        "BΦZ": "#E0FBFC",   # Very light blue
        "BΦE": "#C6DABF",   # Light green
    },
    
    # Line styles
    "linewidth": 2,
    "marker": "o",
    "markersize": 6,
    "markevery": 1,
    
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
    
    # Grid
    "grid_alpha": 0.3,
    "grid_linestyle": "--",
    "grid_linewidth": 0.5,
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger() -> logging.Logger:
    """Configure logging with emojis."""
    logger = logging.getLogger("TimeSeriesPlotter")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s | %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = setup_logger()

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_results(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load results from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with datetime index
    """
    try:
        logger.info(f"📂 Loading results from: {Path(filepath).name}")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        
        logger.info(f"   ✅ Loaded {len(df)} time steps, {len(df.columns)} variables")
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

def add_phase_backgrounds(ax, periods_df: pd.DataFrame, data_df: pd.DataFrame):
    """
    Add colored background rectangles for cyclone phases.
    
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
    
    logger.info("   🎨 Adding phase backgrounds...")
    
    # Get y-axis limits
    ymin, ymax = ax.get_ylim()
    
    # Filter periods that overlap with data
    relevant_periods = periods_df[
        (periods_df['end'] >= data_start) & (periods_df['start'] <= data_end)
    ]
    
    for idx, row in relevant_periods.iterrows():
        phase_name = idx.split()[0]  # Get phase type
        start_time = max(row['start'], data_start)
        end_time = min(row['end'], data_end)
        
        # Get phase color
        color = CONFIG["phase_colors"].get(phase_name, "gray")
        
        # Add rectangle
        rect = Rectangle(
            (mdates.date2num(start_time), ymin),
            mdates.date2num(end_time) - mdates.date2num(start_time),
            ymax - ymin,
            facecolor=color,
            alpha=CONFIG["phase_alpha"],
            zorder=0
        )
        ax.add_patch(rect)
        
        # Add phase label at top
        middle_time = start_time + (end_time - start_time) / 2
        abbrev = CONFIG["phase_abbreviations"].get(phase_name, phase_name[:2])
        
        ax.text(middle_time, 0.98, abbrev,
                transform=ax.get_xaxis_transform(),
                ha='center', va='top',
                fontsize=CONFIG["phase_fontsize"],
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', 
                         facecolor=color, 
                         alpha=0.7, 
                         edgecolor='none'))

def create_timeseries_plot(data: pd.DataFrame, terms: List[str], title: str,
                           periods_df: Optional[pd.DataFrame], 
                           output_path: str, ylabel: str = "Value") -> bool:
    """
    Create a time series plot for given terms.
    
    Args:
        data: DataFrame with time series data
        terms: List of term names to plot
        title: Plot title
        periods_df: DataFrame with phase information
        output_path: Path to save the figure
        ylabel: Y-axis label
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"🎨 Creating time series plot: {title}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=CONFIG["figure_size"], dpi=CONFIG["dpi"])
        
        # Plot each term
        for term in terms:
            if term in data.columns:
                color = CONFIG["colors"].get(term, None)
                ax.plot(data.index, data[term], 
                       label=term,
                       color=color,
                       linewidth=CONFIG["linewidth"],
                       marker=CONFIG["marker"],
                       markersize=CONFIG["markersize"],
                       markevery=CONFIG["markevery"])
            else:
                logger.warning(f"   ⚠️  Term '{term}' not found in data")
        
        # Add phase backgrounds (do this after plotting so backgrounds are behind)
        ax.set_ylim(ax.get_ylim())  # Lock y-limits before adding backgrounds
        add_phase_backgrounds(ax, periods_df, data)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Configure axes
        ax.set_ylabel(ylabel, fontsize=CONFIG["label_fontsize"], fontweight='bold')
        ax.set_xlabel('Time', fontsize=CONFIG["label_fontsize"], fontweight='bold')
        ax.set_title(title, fontsize=CONFIG["title_fontsize"], fontweight='bold', pad=20)
        
        # Format x-axis (time)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
        
        # Format ticks
        ax.tick_params(axis='both', labelsize=CONFIG["tick_fontsize"])
        
        # Add grid
        ax.grid(True, 
               alpha=CONFIG["grid_alpha"], 
               linestyle=CONFIG["grid_linestyle"],
               linewidth=CONFIG["grid_linewidth"])
        
        # Add legend
        ax.legend(loc='best', 
                 fontsize=CONFIG["legend_fontsize"],
                 framealpha=0.9,
                 ncol=min(3, len(terms)))
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        
        logger.info(f"   ✅ Saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Error creating plot: {e}")
        return False

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function to generate all time series plots."""
    logger.info("=" * 70)
    logger.info("📈 TIME SERIES PLOTTER")
    logger.info("=" * 70)
    logger.info("")
    
    # Get base directory
    base_dir = Path(__file__).parent
    results_file = base_dir / CONFIG["results_file"]
    periods_file = base_dir / CONFIG["periods_file"]
    output_dir = base_dir / CONFIG["output_dir"]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info("")
    
    # Load data
    data = load_results(str(results_file))
    if data is None:
        logger.error("❌ Failed to load results file")
        return
    
    logger.info("")
    
    # Load periods
    periods_df = load_periods(str(periods_file))
    logger.info("")
    
    # Track success
    success_count = 0
    total_plots = 3
    
    # ========================================================================
    # FIGURE 1: Energy Terms
    # ========================================================================
    logger.info("🔄 Creating Figure 1: Energy Terms...")
    output_file = output_dir / "timeseries_energy.png"
    if create_timeseries_plot(
        data, 
        CONFIG["energy_terms"],
        "Energy Terms - Time Series",
        periods_df,
        str(output_file),
        ylabel="Energy (J·m⁻²)"
    ):
        success_count += 1
    logger.info("")
    
    # ========================================================================
    # FIGURE 2: Conversion and Generation Terms
    # ========================================================================
    logger.info("🔄 Creating Figure 2: Conversion and Generation Terms...")
    output_file = output_dir / "timeseries_conversion_generation.png"
    if create_timeseries_plot(
        data,
        CONFIG["conversion_gen_terms"],
        "Conversion and Generation Terms - Time Series",
        periods_df,
        str(output_file),
        ylabel="Rate (W·m⁻²)"
    ):
        success_count += 1
    logger.info("")
    
    # ========================================================================
    # FIGURE 3: Boundary Terms
    # ========================================================================
    logger.info("🔄 Creating Figure 3: Boundary Terms...")
    output_file = output_dir / "timeseries_boundary.png"
    if create_timeseries_plot(
        data,
        CONFIG["boundary_terms"],
        "Boundary Terms - Time Series",
        periods_df,
        str(output_file),
        ylabel="Transport (W·m⁻²)"
    ):
        success_count += 1
    logger.info("")
    
    # Summary
    logger.info("=" * 70)
    logger.info(f"✨ COMPLETED: {success_count}/{total_plots} time series plots generated")
    logger.info(f"📂 Figures saved in: {output_dir}")
    logger.info("=" * 70)

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
