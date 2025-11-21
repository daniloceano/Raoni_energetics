#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Comparison Plotter: ERA5 vs WRF Coupled
====================================================

This script generates comparison time series plots between ERA5 and WRF Coupled.
Creates four separate figures:
1. Energy terms (Az, Ae, Kz, Ke)
2. Conversion terms (Ca, Ce, Ck, Cz)
3. Generation terms (Ge, Gz)
4. Boundary terms (BAz, BAe, BKz, BKe)

Line styles:
- ERA5: Solid lines (same colors as individual plots)
- WRF Coupled: Dashed lines (same colors as individual plots)

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

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

CONFIG = {
    # Base directories
    "base_results_dir": "../../LEC_Results",
    "base_output_dir": "../../Figures",
    
    # Data sources to compare
    "data_sources": {
        "ERA5": {
            "path": "Raoni_ERA5_fixed",
            "label": "ERA5",
            "linestyle": "-",  # Solid line
        },
        "WRF_coupled": {
            "path": "WRF_sacoplamento-RAONI-6h_INTRP-Regular_processed_fixed",
            "label": "WRF Coupled",
            "linestyle": "--",  # Dashed line
        }
    },
    
    # Resampling configuration
    "resample_era5": True,
    "resample_freq": "6h",  # WRF frequency (lowercase 'h' to avoid deprecation warning)
    
    # Terms to plot in each figure with colors from plot_timeseries.py
    "energy_terms": {
        "Az": {"color": "#2E86AB", "marker": "o", "label": "Az"},
        "Ae": {"color": "#A23B72", "marker": "s", "label": "Ae"},
        "Kz": {"color": "#F18F01", "marker": "^", "label": "Kz"},
        "Ke": {"color": "#C73E1D", "marker": "v", "label": "Ke"},
    },
    
    "conversion_terms": {
        "Ca": {"color": "#D4B483", "marker": "o", "label": "Ca"},
        "Ce": {"color": "#1B998B", "marker": "s", "label": "Ce"},
        "Ck": {"color": "#6A4C93", "marker": "^", "label": "Ck"},
        "Cz": {"color": "#06A77D", "marker": "v", "label": "Cz"},
    },
    
    "generation_terms": {
        "Ge": {"color": "#4ECDC4", "marker": "o", "label": "Ge"},
        "Gz": {"color": "#FF6B6B", "marker": "s", "label": "Gz"},
    },
    
    "boundary_terms": {
        "BAz": {"color": "#3D5A80", "marker": "o", "label": "BAz"},
        "BAe": {"color": "#98C1D9", "marker": "s", "label": "BAe"},
        "BKz": {"color": "#EE6C4D", "marker": "^", "label": "BKz"},
        "BKe": {"color": "#293241", "marker": "v", "label": "BKe"},
    },
    
    # Plot styling
    "figure_size": (18, 10),
    "dpi": 300,
    "title_fontsize": 18,
    "label_fontsize": 14,
    "tick_fontsize": 12,
    "legend_fontsize": 11,
    
    # Line styles
    "linewidth": 2.5,
    "markersize": 7,
    "markevery": 1,
    "alpha": 0.85,
    
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
    logger = logging.getLogger("ComparisonPlotterV2")
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

def load_results(filepath: str, source_name: str, source_key: str) -> Optional[pd.DataFrame]:
    """Load results CSV file."""
    try:
        logger.info(f"   📂 Loading {source_name}: {Path(filepath).name}")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        logger.info(f"      ✅ Loaded {len(df)} time steps")
        
        # Resample ERA5 data to match WRF frequency if configured
        if source_key == "ERA5" and CONFIG["resample_era5"]:
            original_len = len(df)
            df = df.resample(CONFIG["resample_freq"]).mean()
            logger.info(f"      🔄 Resampled ERA5 from {original_len} to {len(df)} time steps ({CONFIG['resample_freq']})")
        
        return df
    except Exception as e:
        logger.error(f"      ❌ Error loading {filepath}: {e}")
        return None

def load_all_data(base_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load data from all sources."""
    logger.info("=" * 70)
    logger.info("📥 LOADING DATA FROM SOURCES")
    logger.info("=" * 70)
    
    all_data = {}
    
    for source_key, source_info in CONFIG["data_sources"].items():
        source_path = source_info["path"]
        results_dir = base_dir / CONFIG["base_results_dir"] / source_path
        results_file = list(results_dir.glob("*_results.csv"))
        
        if not results_file:
            logger.error(f"❌ No results file found for {source_info['label']}")
            continue
        
        results_file = results_file[0]
        data = load_results(str(results_file), source_info["label"], source_key)
        
        if data is not None:
            all_data[source_key] = data
    
    logger.info("")
    logger.info(f"✅ Successfully loaded {len(all_data)}/{len(CONFIG['data_sources'])} data sources")
    logger.info("")
    
    return all_data

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_comparison_plot(all_data: Dict[str, pd.DataFrame], 
                          terms_config: Dict, 
                          title: str,
                          ylabel: str,
                          output_path: str) -> bool:
    """Create a comparison time series plot for given terms."""
    try:
        logger.info(f"   🎨 Creating: {Path(output_path).name}")
        
        fig, ax = plt.subplots(figsize=CONFIG["figure_size"], dpi=CONFIG["dpi"])
        
        plotted_any = False
        
        # Plot each term for each data source
        for term_name, term_config in terms_config.items():
            for source_key, source_data in all_data.items():
                if term_name not in source_data.columns:
                    logger.info(f"      ⚠️  Term '{term_name}' not found in {CONFIG['data_sources'][source_key]['label']}")
                    continue
                
                source_info = CONFIG["data_sources"][source_key]
                
                # Get color, marker, and linestyle
                color = term_config["color"]
                marker = term_config["marker"]
                linestyle = source_info["linestyle"]
                
                # Create label combining term and source
                label = f"{term_config['label']} - {source_info['label']}"
                
                # Plot the data
                ax.plot(source_data.index, 
                       source_data[term_name],
                       label=label,
                       color=color,
                       linestyle=linestyle,
                       marker=marker,
                       linewidth=CONFIG["linewidth"],
                       markersize=CONFIG["markersize"],
                       markevery=CONFIG["markevery"],
                       alpha=CONFIG["alpha"])
                
                plotted_any = True
        
        if not plotted_any:
            logger.info(f"      ⚠️  No data to plot")
            plt.close()
            return False
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Configure axes
        ax.set_ylabel(ylabel, fontsize=CONFIG["label_fontsize"], fontweight='bold')
        ax.set_xlabel('Time', fontsize=CONFIG["label_fontsize"], fontweight='bold')
        ax.set_title(title, fontsize=CONFIG["title_fontsize"], 
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
        
        # Add legend - organize by term, then source
        handles, labels = ax.get_legend_handles_labels()
        
        # Sort legend: group by term (ERA5 and WRF for each term together)
        # Split into terms
        legend_dict = {}
        for handle, label in zip(handles, labels):
            term = label.split(' - ')[0]
            if term not in legend_dict:
                legend_dict[term] = []
            legend_dict[term].append((handle, label))
        
        # Flatten back maintaining grouping
        sorted_handles = []
        sorted_labels = []
        for term in sorted(legend_dict.keys()):
            for handle, label in sorted(legend_dict[term], key=lambda x: x[1]):
                sorted_handles.append(handle)
                sorted_labels.append(label)
        
        ax.legend(sorted_handles, sorted_labels,
                 loc='upper left', 
                 bbox_to_anchor=(1.01, 1),
                 fontsize=CONFIG["legend_fontsize"],
                 framealpha=0.95,
                 ncol=1)
        
        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        
        logger.info(f"      ✅ Saved: {Path(output_path).name}")
        return True
        
    except Exception as e:
        logger.error(f"      ❌ Error creating plot: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 TIME SERIES COMPARISON: ERA5 vs WRF COUPLED")
    logger.info("=" * 70)
    logger.info("")
    
    base_dir = Path(__file__).parent
    
    # Load all data
    all_data = load_all_data(base_dir)
    
    if len(all_data) != 2:
        logger.error(f"❌ Expected 2 data sources, got {len(all_data)}. Exiting.")
        return
    
    # Create output directory
    output_dir = base_dir / CONFIG["base_output_dir"] / "Comparisons" / "ERA5_vs_WRF_detailed"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info("")
    
    # Track success
    success_count = 0
    
    # Figure 1: Energy Terms
    logger.info("=" * 70)
    logger.info("🔄 Creating Figure 1: Energy Terms Comparison")
    logger.info("=" * 70)
    output_file = output_dir / "ERA5_vs_WRF_energy.png"
    if create_comparison_plot(
        all_data,
        CONFIG["energy_terms"],
        "Energy Terms - ERA5 vs WRF Coupled",
        "Energy (J·m⁻²)",
        str(output_file)
    ):
        success_count += 1
    logger.info("")
    
    # Figure 2: Conversion Terms
    logger.info("=" * 70)
    logger.info("🔄 Creating Figure 2: Conversion Terms Comparison")
    logger.info("=" * 70)
    output_file = output_dir / "ERA5_vs_WRF_conversion.png"
    if create_comparison_plot(
        all_data,
        CONFIG["conversion_terms"],
        "Conversion Terms - ERA5 vs WRF Coupled",
        "Rate (W·m⁻²)",
        str(output_file)
    ):
        success_count += 1
    logger.info("")
    
    # Figure 3: Generation Terms
    logger.info("=" * 70)
    logger.info("🔄 Creating Figure 3: Generation Terms Comparison")
    logger.info("=" * 70)
    output_file = output_dir / "ERA5_vs_WRF_generation.png"
    if create_comparison_plot(
        all_data,
        CONFIG["generation_terms"],
        "Generation Terms - ERA5 vs WRF Coupled",
        "Rate (W·m⁻²)",
        str(output_file)
    ):
        success_count += 1
    logger.info("")
    
    # Figure 4: Boundary Terms
    logger.info("=" * 70)
    logger.info("🔄 Creating Figure 4: Boundary Terms Comparison")
    logger.info("=" * 70)
    output_file = output_dir / "ERA5_vs_WRF_boundary.png"
    if create_comparison_plot(
        all_data,
        CONFIG["boundary_terms"],
        "Boundary Terms - ERA5 vs WRF Coupled",
        "Transport (W·m⁻²)",
        str(output_file)
    ):
        success_count += 1
    logger.info("")
    
    # Final summary
    logger.info("=" * 70)
    logger.info(f"🎉 COMPLETED: {success_count}/4 comparison plots generated")
    logger.info(f"📂 Figures saved in: {output_dir}")
    logger.info("=" * 70)
    logger.info("")
    logger.info("📝 Plot legend:")
    logger.info("   • Solid lines (—): ERA5")
    logger.info("   • Dashed lines (--): WRF Coupled")
    logger.info("   • Colors match individual plot_timeseries.py colors")
    logger.info("=" * 70)
    logger.info("")

if __name__ == "__main__":
    main()
