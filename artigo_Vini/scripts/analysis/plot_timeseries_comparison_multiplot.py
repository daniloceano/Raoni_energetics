#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-panel Time Series Comparison for Cyclone Lorenz Energy Cycle
===================================================================

Creates publication-quality figures for Scientific Reports with four
multi-panel figures:
  i)   Energy terms (Az, Ae, Kz, Ke) - 4 subplots
  ii)  Conversion terms (Ca, Ce, Ck, Cz) - 4 subplots
  iii) Generation terms (Ge, Gz) - 2 subplots
  iv)  Boundary terms (BAz, BAe, BKz, BKe) - 4 subplots

Each subplot shows one term with all data sources overlaid.

Author: Danilo
Date: 2025
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# Import configuration
from config import (
    DATA_SOURCES,
    BASE_RESULTS_DIR,
    BASE_OUTPUT_DIR,
    RESAMPLE_ERA5,
    RESAMPLE_GFS,
    RESAMPLE_FREQ,
    FIGURE_TYPES,
    apply_scientific_reports_style
)

# Apply Scientific Reports style
apply_scientific_reports_style()

# Additional local configuration for plotting style
LOCAL_CONFIG = {
    "linewidth": 1.5,
    "markersize": 4,
    "markevery": 2,  # Plot marker every N points
    "alpha": 0.8,
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger() -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("MultiplotTimeseries")
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

def load_results(filepath: str, source_key: str) -> Optional[pd.DataFrame]:
    """Load results CSV file."""
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        
        # Resample all datasets to the configured frequency (e.g., 6h)
        try:
            df = df.resample(RESAMPLE_FREQ).mean()
        except Exception:
            # If resampling fails for any reason, continue with original data
            logger.debug(f"Resampling failed for {filepath}; using original frequency")
        
        return df
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def load_all_data(base_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load data from all sources."""
    logger.info("=" * 70)
    logger.info("📥 LOADING DATA FROM ALL SOURCES")
    logger.info("=" * 70)
    
    all_data = {}
    
    for source_key, source_info in DATA_SOURCES.items():
        source_path = source_info["path"]
        results_dir = base_dir / BASE_RESULTS_DIR / source_path
        
        if not results_dir.exists():
            logger.warning(f"   ⚠️  Directory not found: {source_path}")
            continue
            
        results_files = list(results_dir.glob("*_results.csv"))
        
        if not results_files:
            logger.warning(f"   ⚠️  No results file found for {source_info['label']}")
            continue
        
        results_file = results_files[0]
        logger.info(f"   📂 Loading {source_info['label']}: {results_file.name}")
        
        data = load_results(str(results_file), source_key)
        
        if data is not None:
            all_data[source_key] = data
            logger.info(f"      ✅ {len(data)} time steps")
    
    logger.info("")
    logger.info(f"✅ Loaded {len(all_data)}/{len(DATA_SOURCES)} data sources")
    logger.info("")
    
    return all_data


def find_common_time_range(all_data: Dict[str, pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Find the overlapping time range across all datasets."""
    start_times = []
    end_times = []
    
    for source_key, df in all_data.items():
        start_times.append(df.index.min())
        end_times.append(df.index.max())
    
    common_start = max(start_times)
    common_end = min(end_times)
    
    return common_start, common_end

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_multiplot_figure(all_data: Dict[str, pd.DataFrame],
                           fig_config: Dict,
                           fig_key: str,
                           output_path: Path) -> bool:
    """
    Create a multi-panel figure with one subplot per term.
    Each subplot shows all data sources for that term.
    """
    try:
        terms = fig_config["terms"]
        n_terms = len(terms)
        ncols = fig_config.get("ncols", 2)  # Default to 2 columns if not specified
        nrows = int(np.ceil(n_terms / ncols))
        
        # Choose figsize; for 'all_combined' compute height proportional to rows
        default_figsize = fig_config.get("figsize", (8, 6))
        width = default_figsize[0]
        # Default per-row height in inches (can be overridden in config via 'row_height')
        default_row_height = fig_config.get("row_height", 3.0)

        if fig_key == "all_combined":
            # prefer explicit combined size in config
            if "figsize_combined" in fig_config:
                figsize = fig_config["figsize_combined"]
            else:
                # scale height with number of rows to give vertical space between subplots
                figsize = (width, max(default_figsize[1], default_row_height * nrows))
        else:
            figsize = default_figsize

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        # Find common time range
        common_start, common_end = find_common_time_range(all_data)
        
        # Plot each term in its own subplot
        for idx, term in enumerate(terms):
            ax = axes[idx]
            
            has_data = False
            
            # Plot each data source
            for source_key, source_data in all_data.items():
                source_info = DATA_SOURCES[source_key]
                
                if term not in source_data.columns:
                    continue
                
                # Filter to common time range
                mask = (source_data.index >= common_start) & (source_data.index <= common_end)
                data_filtered = source_data.loc[mask, term]
                
                if data_filtered.empty:
                    continue
                
                ax.plot(
                    data_filtered.index,
                    data_filtered.values,
                    label=source_info["label"],
                    color=source_info["color"],
                    linestyle=source_info["linestyle"],
                    marker=source_info["marker"],
                    linewidth=LOCAL_CONFIG["linewidth"],
                    markersize=LOCAL_CONFIG["markersize"],
                    markevery=LOCAL_CONFIG["markevery"],
                    alpha=LOCAL_CONFIG["alpha"],
                    zorder=source_info["zorder"]
                )
                has_data = True
            
            if not has_data:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=10, color='gray')
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
            
            # Subplot title with term label
            term_title = fig_config.get("term_titles", {}).get(term, term)
            term_label = fig_config.get("term_labels", {}).get(term, term)
            ax.set_title(f"({chr(97+idx)}) {term_title} ({term_label})", 
                        fontsize=10, fontweight='bold', loc='left')
            
            # Y-axis label (only on left column)
            if idx % ncols == 0:
                ylabel = fig_config.get("ylabel", "Value")
                ax.set_ylabel(ylabel, fontsize=9)
            
            # X-axis formatting
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=12))
            
            # Only show x-label on bottom row
            if idx >= n_terms - ncols:
                ax.set_xlabel('Date (2021)', fontsize=9)
            
            # Rotate x-tick labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)

            # Move scientific notation offset (e.g. '1e6') to avoid overlapping title
            if fig_key == "energy":
                try:
                    ax.yaxis.get_offset_text().set_visible(True)
                except Exception:
                    pass
        
        # Hide unused subplots
        for idx in range(n_terms, len(axes)):
            axes[idx].set_visible(False)
        
        # Create shared legend at the bottom
        handles = []
        labels = []
        for source_key in all_data.keys():
            source_info = DATA_SOURCES[source_key]
            handle = Line2D([0], [0], 
                           color=source_info["color"],
                           linestyle=source_info["linestyle"],
                           marker=source_info["marker"],
                           linewidth=LOCAL_CONFIG["linewidth"],
                           markersize=LOCAL_CONFIG["markersize"])
            handles.append(handle)
            labels.append(source_info["label"])
        
        if 'Generation' in fig_config['title']:
            bbox_to_anchor = (0.5, -0.15)
        else:
            bbox_to_anchor = (0.5, -0.02)

        fig.legend(handles, labels, 
                  loc='lower center',
                  bbox_to_anchor=bbox_to_anchor,
                  ncol=len(all_data),
                  frameon=True,
                  fontsize=9)
        
        # Adjust layout; give more bottom space for combined figure
        bottom_pad = 0.16 if fig_key != "all_combined" else 0.08
        plt.subplots_adjust(bottom=bottom_pad, hspace=0.45, wspace=0.25)
        
        # Save figure (avoid bbox_inches='tight' - it interacts badly with
        # custom artist transforms and can produce multi-thousand-inch images)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, facecolor='white')
        
        plt.close()
        
        logger.info(f"   ✅ Saved: {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Error creating {fig_key} figure: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 MULTI-PANEL TIME SERIES COMPARISON (Scientific Reports Style)")
    logger.info("=" * 70)
    logger.info("")
    
    base_dir = Path(__file__).parent
    
    # Load all data
    all_data = load_all_data(base_dir)
    
    if not all_data:
        logger.error("❌ No data loaded. Exiting.")
        return
    
    # Create output directory
    output_dir = base_dir / (BASE_OUTPUT_DIR + "/Comparisons/multiplot")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info("")
    
    # Generate each figure type
    logger.info("=" * 70)
    logger.info("🎨 GENERATING MULTI-PANEL FIGURES")
    logger.info("=" * 70)
    
    success_count = 0
    
    for fig_key, fig_config in FIGURE_TYPES.items():
        logger.info(f"\n   📊 Creating: {fig_config['title']}")
        
        output_path = output_dir / f"timeseries_{fig_key}_multiplot.png"
        
        if create_multiplot_figure(all_data, fig_config, fig_key, output_path):
            success_count += 1
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"🎉 COMPLETED: {success_count}/{len(FIGURE_TYPES)} figures generated")
    logger.info(f"📂 Saved to: {output_dir}")
    logger.info("=" * 70)
    logger.info("")


if __name__ == "__main__":
    main()
