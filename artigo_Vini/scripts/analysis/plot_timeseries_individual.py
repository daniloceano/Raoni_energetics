#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Plotter for Individual Cyclone Energy Analysis
===========================================================

This script generates time series plots for energy terms from cyclone analysis
for each data source individually. Creates three separate figures per source:
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
# CONFIGURATION SECTION
# ============================================================================

CONFIG = {
    # Base directories
    "base_results_dir": "../../LEC_Results",
    "base_output_dir": "../../Figures",
    
    # Data sources to process
    "data_sources": [
        "Raoni_ERA5_fixed",
        "WRF_sacoplamento-RAONI-6h_INTRP-Regular_processed_fixed",
        "WRF-cacoplamento_Raoni-6h_INTRP_Regular_processed_fixed"
    ],
    
    # Optional periods file for each source
    "periods_files": {
        "Raoni_ERA5_fixed": None,
        "WRF_sacoplamento-RAONI-6h_INTRP-Regular_processed_fixed": None,
        "WRF-cacoplamento_Raoni-6h_INTRP_Regular_processed_fixed": None
    },
    
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
    
    # Colors for each term
    "colors": {
        # Energy terms
        "Az": "#2E86AB",
        "Ae": "#A23B72",
        "Kz": "#F18F01",
        "Ke": "#C73E1D",
        
        # Conversion terms
        "Cz": "#06A77D",
        "Ca": "#D4B483",
        "Ck": "#6A4C93",
        "Ce": "#1B998B",
        
        # Generation terms
        "Gz": "#FF6B6B",
        "Ge": "#4ECDC4",
        
        # Boundary terms
        "BAz": "#3D5A80",
        "BAe": "#98C1D9",
        "BKz": "#EE6C4D",
        "BKe": "#293241",
        "BΦZ": "#E0FBFC",
        "BΦE": "#C6DABF",
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
    "grid_linestyle": ":",
    "grid_linewidth": 0.5,
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger() -> logging.Logger:
    """Configure logging."""
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
    """Load results CSV file."""
    try:
        logger.info(f"   📂 Loading results from: {Path(filepath).name}")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        logger.info(f"      ✅ Loaded {len(df)} time steps, {len(df.columns)} variables")
        return df
    except Exception as e:
        logger.error(f"      ❌ Error loading {filepath}: {e}")
        return None

def load_periods(filepath: str) -> Optional[pd.DataFrame]:
    """Load cyclone phase periods from CSV file."""
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

def add_phase_backgrounds(ax, periods_df: Optional[pd.DataFrame], 
                         data_df: pd.DataFrame):
    """Add phase background shading to plot."""
    if periods_df is None:
        return
    
    data_start = data_df.index.min()
    data_end = data_df.index.max()
    ymin, ymax = ax.get_ylim()
    
    logger.info("      📍 Adding phase backgrounds...")
    
    relevant_periods = periods_df[
        (periods_df['end'] >= data_start) & (periods_df['start'] <= data_end)
    ]
    
    for idx, row in relevant_periods.iterrows():
        phase_name = idx.split()[0]
        start_time = max(row['start'], data_start)
        end_time = min(row['end'], data_end)
        
        color = CONFIG["phase_colors"].get(phase_name, "gray")
        
        rect = Rectangle(
            (mdates.date2num(start_time), ymin),
            mdates.date2num(end_time) - mdates.date2num(start_time),
            ymax - ymin,
            facecolor=color,
            alpha=CONFIG["phase_alpha"],
            zorder=0
        )
        ax.add_patch(rect)
        
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
                           output_path: str, ylabel: str = "Value",
                           source_name: str = "") -> bool:
    """Create a time series plot for given terms."""
    try:
        logger.info(f"      🎨 Creating: {Path(output_path).name}")
        
        fig, ax = plt.subplots(figsize=CONFIG["figure_size"], dpi=CONFIG["dpi"])
        
        # Plot each term
        plotted_terms = []
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
                plotted_terms.append(term)
            else:
                logger.info(f"         ⚠️  Term '{term}' not found in data")
        
        if not plotted_terms:
            logger.info(f"         ⚠️  No terms found to plot")
            plt.close()
            return False
        
        # Add phase backgrounds
        ax.set_ylim(ax.get_ylim())
        add_phase_backgrounds(ax, periods_df, data)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Configure axes
        ax.set_ylabel(ylabel, fontsize=CONFIG["label_fontsize"], fontweight='bold')
        ax.set_xlabel('Time', fontsize=CONFIG["label_fontsize"], fontweight='bold')
        full_title = f"{title} - {source_name}" if source_name else title
        ax.set_title(full_title, fontsize=CONFIG["title_fontsize"], 
                    fontweight='bold', pad=20)
        
        # Format x-axis
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
                 ncol=min(3, len(plotted_terms)))
        
        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        
        logger.info(f"         ✅ Saved: {Path(output_path).name}")
        return True
        
    except Exception as e:
        logger.error(f"         ❌ Error creating plot: {e}")
        return False

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_data_source(source_name: str, base_dir: Path):
    """Process a single data source and generate all its time series plots."""
    logger.info("=" * 70)
    logger.info(f"🔄 Processing: {source_name}")
    logger.info("=" * 70)
    
    # Setup paths
    results_dir = base_dir / CONFIG["base_results_dir"] / source_name
    results_file = list(results_dir.glob("*_results.csv"))
    
    if not results_file:
        logger.error(f"❌ No results file found in {results_dir}")
        return 0
    
    results_file = results_file[0]
    output_dir = base_dir / CONFIG["base_output_dir"] / source_name / "timeseries"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")
    
    # Load data
    data = load_results(str(results_file))
    if data is None:
        return 0
    
    # Load periods if available
    periods_file = CONFIG["periods_files"].get(source_name)
    if periods_file:
        periods_file = results_dir / periods_file
    periods_df = load_periods(str(periods_file) if periods_file else None)
    
    logger.info("")
    
    # Track success
    success_count = 0
    
    # Figure 1: Energy Terms
    logger.info("   🔄 Creating Figure 1: Energy Terms...")
    output_file = output_dir / "timeseries_energy.png"
    if create_timeseries_plot(
        data, 
        CONFIG["energy_terms"],
        "Energy Terms - Time Series",
        periods_df,
        str(output_file),
        ylabel="Energy (J·m⁻²)",
        source_name=source_name
    ):
        success_count += 1
    
    # Figure 2: Conversion and Generation Terms
    logger.info("   🔄 Creating Figure 2: Conversion and Generation Terms...")
    output_file = output_dir / "timeseries_conversion_generation.png"
    if create_timeseries_plot(
        data,
        CONFIG["conversion_gen_terms"],
        "Conversion and Generation Terms - Time Series",
        periods_df,
        str(output_file),
        ylabel="Rate (W·m⁻²)",
        source_name=source_name
    ):
        success_count += 1
    
    # Figure 3: Boundary Terms
    logger.info("   🔄 Creating Figure 3: Boundary Terms...")
    output_file = output_dir / "timeseries_boundary.png"
    if create_timeseries_plot(
        data,
        CONFIG["boundary_terms"],
        "Boundary Terms - Time Series",
        periods_df,
        str(output_file),
        ylabel="Transport (W·m⁻²)",
        source_name=source_name
    ):
        success_count += 1
    
    logger.info("")
    logger.info(f"✨ Completed {source_name}: {success_count}/3 plots generated")
    logger.info("")
    
    return success_count

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("📈 TIME SERIES PLOTTER - INDIVIDUAL SOURCES")
    logger.info("=" * 70)
    logger.info("")
    
    base_dir = Path(__file__).parent
    
    total_success = 0
    
    for source in CONFIG["data_sources"]:
        success = process_data_source(source, base_dir)
        total_success += success
    
    # Final summary
    logger.info("=" * 70)
    logger.info(f"🎉 ALL COMPLETED: {total_success} total time series plots generated")
    logger.info(f"📂 Figures saved in: {base_dir / CONFIG['base_output_dir']}")
    logger.info("=" * 70)
    logger.info("")

if __name__ == "__main__":
    main()
