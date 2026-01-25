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

# ============================================================================
# SCIENTIFIC REPORTS STYLE CONFIGURATION
# ============================================================================

# Use a clean, publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')

# Scientific Reports typography
plt.rcParams.update({
    # Font settings (Scientific Reports uses sans-serif)
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    
    # Axes
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.linewidth': 0.8,
    'axes.labelweight': 'normal',
    'axes.spines.top': False,
    'axes.spines.right': False,
    
    # Ticks
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    
    # Legend
    'legend.fontsize': 8,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    
    # Figure
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    
    # Lines
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    
    # Grid
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'grid.linestyle': '-',
})

# ============================================================================
# DATA SOURCE CONFIGURATION
# ============================================================================

CONFIG = {
    # Base directories
    "base_results_dir": "../../LEC_Results",
    "base_output_dir": "../../Figures/Comparisons/multiplot",
    
    # Data sources with distinct colors
    "data_sources": {
        "ERA5": {
            "path": "Raoni_ERA5_fixed",
            "label": "ERA5",
            "color": "#2c3e50",      # Dark blue-gray
            "linestyle": "-",
            "marker": "o",
            "zorder": 6,
        },
        "GFS": {
            "path": "GFS_Raoni_processed_fixed",
            "label": "GFS",
            "color": "#27ae60",      # Green
            "linestyle": "-",
            "marker": "s",
            "zorder": 5,
        },
        "GFS_CPL_EXP": {
            "path": "WRF_sacoplamento-RAONI-6h_INTRP-Regular_processed_fixed",
            "label": "GFS_CPL_EXP",
            "color": "#e74c3c",      # Red (GFS coupled)
            "linestyle": "-",
            "marker": "^",
            "zorder": 4,
        },
        "GFS_DCP_EXP": {
            "path": "WRF-cacoplamento_Raoni-6h_INTRP_Regular_processed_fixed",
            "label": "GFS_DCP_EXP",
            "color": "#3498db",      # Blue (GFS decoupled)
            "linestyle": "--",
            "marker": "v",
            "zorder": 3,
        },
        "ERA5_CPL_EXP": {
            "path": "WRFacoplado-ERA5-RAONI-6h_INTRP-Regular_processed_fixed",
            "label": "ERA5_CPL_EXP",
            "color": "#9b59b6",      # Purple (ERA5 coupled)
            "linestyle": "-",
            "marker": "D",
            "zorder": 2,
        },
        "ERA5_DCP_EXP": {
            "path": "WRFsa-ERA5-RAONI-6h_INTRP-Regular_processed_fixed",
            "label": "ERA5_DCP_EXP",
            "color": "#f39c12",      # Orange (ERA5 decoupled)
            "linestyle": "--",
            "marker": "p",
            "zorder": 1,
        }
    },
    
    # Resampling configuration
    "resample_era5": True,
    "resample_gfs": True,
    "resample_freq": "6h",  # Match WRF frequency
    
    # Terms organized by figure type
    "figures": {
        "energy": {
            "title": "Energy Reservoirs",
            "ylabel": r"Energy (J$\cdot$m$^{-2}$)",
            "terms": ["Az", "Ae", "Kz", "Ke"],
            "term_labels": {
                "Az": r"$A_Z$",
                "Ae": r"$A_E$", 
                "Kz": r"$K_Z$",
                "Ke": r"$K_E$"
            },
            "term_titles": {
                "Az": "Zonal Available Potential Energy",
                "Ae": "Eddy Available Potential Energy",
                "Kz": "Zonal Kinetic Energy",
                "Ke": "Eddy Kinetic Energy"
            },
            "ncols": 2,
            "figsize": (180/25.4, 150/25.4),  # 180mm x 150mm (Scientific Reports max width)
        },
        "conversion": {
            "title": "Energy Conversion Terms",
            "ylabel": r"Rate (W$\cdot$m$^{-2}$)",
            "terms": ["Ca", "Ce", "Ck", "Cz"],
            "term_labels": {
                "Ca": r"$C_A$",
                "Ce": r"$C_E$",
                "Ck": r"$C_K$",
                "Cz": r"$C_Z$"
            },
            "term_titles": {
                "Ca": r"$A_Z \rightarrow A_E$",
                "Ce": r"$A_E \rightarrow K_E$",
                "Ck": r"$K_E \rightarrow K_Z$",
                "Cz": r"$K_Z \rightarrow A_Z$"
            },
            "ncols": 2,
            "figsize": (180/25.4, 150/25.4),
        },
        "generation": {
            "title": "Generation Terms",
            "ylabel": r"Rate (W$\cdot$m$^{-2}$)",
            "terms": ["Ge", "Gz"],
            "term_labels": {
                "Ge": r"$G_E$",
                "Gz": r"$G_Z$"
            },
            "term_titles": {
                "Ge": "Eddy Generation",
                "Gz": "Zonal Generation"
            },
            "ncols": 2,
            "figsize": (180/25.4, 80/25.4),  # Smaller height for 2 panels
        },
        "boundary": {
            "title": "Boundary Transport Terms",
            "ylabel": r"Transport (W$\cdot$m$^{-2}$)",
            "terms": ["BAz", "BAe", "BKz", "BKe"],
            "term_labels": {
                "BAz": r"$B_{A_Z}$",
                "BAe": r"$B_{A_E}$",
                "BKz": r"$B_{K_Z}$",
                "BKe": r"$B_{K_E}$"
            },
            "term_titles": {
                "BAz": "Zonal APE Boundary",
                "BAe": "Eddy APE Boundary",
                "BKz": "Zonal KE Boundary",
                "BKe": "Eddy KE Boundary"
            },
            "ncols": 2,
            "figsize": (180/25.4, 150/25.4),
        },
    },
    
    # Plot styling
    "linewidth": 1.5,
    "markersize": 4,
    "markevery": 2,  # Plot marker every N points
    "alpha": 0.9,
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
        
        # Resample high-frequency data to match WRF
        if source_key in ["ERA5", "GFS"]:
            if (source_key == "ERA5" and CONFIG["resample_era5"]) or \
               (source_key == "GFS" and CONFIG["resample_gfs"]):
                df = df.resample(CONFIG["resample_freq"]).mean()
        
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
    
    for source_key, source_info in CONFIG["data_sources"].items():
        source_path = source_info["path"]
        results_dir = base_dir / CONFIG["base_results_dir"] / source_path
        
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
    logger.info(f"✅ Loaded {len(all_data)}/{len(CONFIG['data_sources'])} data sources")
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
        ncols = fig_config["ncols"]
        nrows = int(np.ceil(n_terms / ncols))
        
        figsize = fig_config["figsize"]
        
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
                source_info = CONFIG["data_sources"][source_key]
                
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
                    linewidth=CONFIG["linewidth"],
                    markersize=CONFIG["markersize"],
                    markevery=CONFIG["markevery"],
                    alpha=CONFIG["alpha"],
                    zorder=source_info["zorder"]
                )
                has_data = True
            
            if not has_data:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=10, color='gray')
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
            
            # Subplot title with term label
            term_title = fig_config["term_titles"].get(term, term)
            term_label = fig_config["term_labels"].get(term, term)
            ax.set_title(f"({chr(97+idx)}) {term_title} ({term_label})", 
                        fontsize=10, fontweight='bold', loc='left')
            
            # Y-axis label (only on left column)
            if idx % ncols == 0:
                ax.set_ylabel(fig_config["ylabel"], fontsize=9)
            
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
        
        # Hide unused subplots
        for idx in range(n_terms, len(axes)):
            axes[idx].set_visible(False)
        
        # Create shared legend at the bottom
        handles = []
        labels = []
        for source_key in all_data.keys():
            source_info = CONFIG["data_sources"][source_key]
            handle = Line2D([0], [0], 
                           color=source_info["color"],
                           linestyle=source_info["linestyle"],
                           marker=source_info["marker"],
                           linewidth=CONFIG["linewidth"],
                           markersize=CONFIG["markersize"])
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
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12, hspace=0.35, wspace=0.25)
        
        # Save figure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in multiple formats
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        # Also save as PDF for publication
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
        
        plt.close()
        
        logger.info(f"   ✅ Saved: {output_path.name} (+ PDF)")
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
    output_dir = base_dir / CONFIG["base_output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info("")
    
    # Generate each figure type
    logger.info("=" * 70)
    logger.info("🎨 GENERATING MULTI-PANEL FIGURES")
    logger.info("=" * 70)
    
    success_count = 0
    
    for fig_key, fig_config in CONFIG["figures"].items():
        logger.info(f"\n   📊 Creating: {fig_config['title']}")
        
        output_path = output_dir / f"timeseries_{fig_key}_multiplot.png"
        
        if create_multiplot_figure(all_data, fig_config, fig_key, output_path):
            success_count += 1
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"🎉 COMPLETED: {success_count}/{len(CONFIG['figures'])} figures generated")
    logger.info(f"📂 Saved to: {output_dir}")
    logger.info("=" * 70)
    logger.info("")


if __name__ == "__main__":
    main()
