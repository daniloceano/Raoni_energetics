#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Comparison Plotter for Cyclone Energy Analysis
===========================================================

This script generates comparison time series plots across different data sources
(ERA5, WRF with coupling, WRF without coupling). Creates four separate figures:
1. Energy terms (Az, Ae, Kz, Ke)
2. Conversion terms (Ca, Ce, Ck, Cz)
3. Generation terms (Ge, Gz)
4. Boundary terms (BAz, BAe, BKz, BKe)

Color scheme:
- ERA5: Grayscale (black, gray shades)
- WRF with coupling: Blue shades
- WRF without coupling: Red shades

Each term has a unique marker that is consistent across all data sources.

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
            "colors": {
                "primary": "#000000",    # Black
                "secondary": "#4a4a4a",  # Dark gray
                "tertiary": "#757575",   # Medium gray
                "quaternary": "#a0a0a0", # Light gray
            }
        },
        "WRF_coupled": {
            "path": "WRF_sacoplamento-RAONI-6h_INTRP-Regular_processed_fixed",
            "label": "WRF Coupled",
            "colors": {
                "primary": "#0d47a1",    # Dark blue
                "secondary": "#1976d2",  # Medium blue
                "tertiary": "#42a5f5",   # Light blue
                "quaternary": "#90caf9", # Very light blue
            }
        },
        "WRF_uncoupled": {
            "path": "WRF-cacoplamento_Raoni-6h_INTRP_Regular_processed_fixed",
            "label": "WRF Uncoupled",
            "colors": {
                "primary": "#b71c1c",    # Dark red
                "secondary": "#d32f2f",  # Medium red
                "tertiary": "#ef5350",   # Light red
                "quaternary": "#e57373", # Very light red
            }
        }
    },
    
    # Comparison combinations to generate
    "comparisons": {
        "ERA5_vs_coupled": {
            "sources": ["ERA5", "WRF_coupled"],
            "label": "ERA5 vs WRF Coupled",
            "filename_suffix": "ERA5_vs_coupled"
        },
        "ERA5_vs_uncoupled": {
            "sources": ["ERA5", "WRF_uncoupled"],
            "label": "ERA5 vs WRF Uncoupled",
            "filename_suffix": "ERA5_vs_uncoupled"
        },
        "coupled_vs_uncoupled": {
            "sources": ["WRF_coupled", "WRF_uncoupled"],
            "label": "WRF Coupled vs Uncoupled",
            "filename_suffix": "coupled_vs_uncoupled"
        },
        "all_three": {
            "sources": ["ERA5", "WRF_coupled", "WRF_uncoupled"],
            "label": "All Sources",
            "filename_suffix": "all_sources"
        }
    },
    
    # Resampling configuration
    "resample_era5": True,
    "resample_freq": "6H",  # WRF frequency
    
    # Terms to plot in each figure (with assigned colors and markers)
    "energy_terms": {
        "Az": {"color_key": "primary", "marker": "o", "label": "Az"},
        "Ae": {"color_key": "secondary", "marker": "s", "label": "Ae"},
        "Kz": {"color_key": "tertiary", "marker": "^", "label": "Kz"},
        "Ke": {"color_key": "quaternary", "marker": "v", "label": "Ke"},
    },
    
    "conversion_terms": {
        "Ca": {"color_key": "primary", "marker": "o", "label": "Ca"},
        "Ce": {"color_key": "secondary", "marker": "s", "label": "Ce"},
        "Ck": {"color_key": "tertiary", "marker": "^", "label": "Ck"},
        "Cz": {"color_key": "quaternary", "marker": "v", "label": "Cz"},
    },
    
    "generation_terms": {
        "Ge": {"color_key": "primary", "marker": "o", "label": "Ge"},
        "Gz": {"color_key": "secondary", "marker": "s", "label": "Gz"},
    },
    
    "boundary_terms": {
        "BAz": {"color_key": "primary", "marker": "o", "label": "BAz"},
        "BAe": {"color_key": "secondary", "marker": "s", "label": "BAe"},
        "BKz": {"color_key": "tertiary", "marker": "^", "label": "BKz"},
        "BKe": {"color_key": "quaternary", "marker": "v", "label": "BKe"},
    },
    
    # Plot styling
    "figure_size": (18, 10),
    "dpi": 300,
    "title_fontsize": 18,
    "label_fontsize": 14,
    "tick_fontsize": 12,
    "legend_fontsize": 10,
    
    # Line styles
    "linewidth": 2,
    "markersize": 7,
    "markevery": 1,
    "alpha": 0.8,
    
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
    logger = logging.getLogger("ComparisonPlotter")
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
    logger.info("📥 LOADING DATA FROM ALL SOURCES")
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
                          output_path: str,
                          sources_to_plot: List[str] = None) -> bool:
    """Create a comparison time series plot for given terms."""
    try:
        logger.info(f"   🎨 Creating: {Path(output_path).name}")
        
        fig, ax = plt.subplots(figsize=CONFIG["figure_size"], dpi=CONFIG["dpi"])
        
        plotted_any = False
        
        # If sources_to_plot is specified, filter the data
        if sources_to_plot:
            data_to_plot = {k: v for k, v in all_data.items() if k in sources_to_plot}
        else:
            data_to_plot = all_data
        
        # Plot each term for each data source
        for source_key, source_data in data_to_plot.items():
            source_info = CONFIG["data_sources"][source_key]
            
            for term_name, term_config in terms_config.items():
                if term_name not in source_data.columns:
                    logger.info(f"      ⚠️  Term '{term_name}' not found in {source_info['label']}")
                    continue
                
                # Get color and marker for this term
                color = source_info["colors"][term_config["color_key"]]
                marker = term_config["marker"]
                
                # Create label combining source and term
                label = f"{source_info['label']} - {term_config['label']}"
                
                # Plot the data
                ax.plot(source_data.index, 
                       source_data[term_name],
                       label=label,
                       color=color,
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
        
        # Add legend - place outside plot area
        num_items = len([item for item in ax.get_legend_handles_labels()[0]])
        ncol = min(3, max(1, num_items // 6))
        
        ax.legend(loc='upper left', 
                 bbox_to_anchor=(1.01, 1),
                 fontsize=CONFIG["legend_fontsize"],
                 framealpha=0.9,
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
    logger.info("📊 TIME SERIES COMPARISON PLOTTER")
    logger.info("=" * 70)
    logger.info("")
    
    base_dir = Path(__file__).parent
    
    # Load all data
    all_data = load_all_data(base_dir)
    
    if not all_data:
        logger.error("❌ No data loaded. Exiting.")
        return
    
    # Create base output directory
    base_output_dir = base_dir / CONFIG["base_output_dir"] / "Comparisons" / "timeseries"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Base output directory: {base_output_dir}")
    logger.info("")
    
    # Track total success
    total_success = 0
    total_expected = len(CONFIG["comparisons"]) * 4  # 4 figures per comparison
    
    # Loop through each comparison combination
    for comp_key, comp_config in CONFIG["comparisons"].items():
        logger.info("=" * 70)
        logger.info(f"🔄 COMPARISON: {comp_config['label']}")
        logger.info("=" * 70)
        logger.info(f"   Sources: {', '.join([CONFIG['data_sources'][s]['label'] for s in comp_config['sources']])}")
        logger.info("")
        
        # Create subdirectory for this comparison
        output_dir = base_output_dir / comp_config['filename_suffix']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comp_success = 0
        
        # Figure 1: Energy Terms
        logger.info("   🔄 Figure 1/4: Energy Terms")
        output_file = output_dir / f"comparison_energy_{comp_config['filename_suffix']}.png"
        if create_comparison_plot(
            all_data,
            CONFIG["energy_terms"],
            f"Energy Terms - {comp_config['label']}",
            "Energy (J·m⁻²)",
            str(output_file),
            sources_to_plot=comp_config['sources']
        ):
            comp_success += 1
            total_success += 1
        
        # Figure 2: Conversion Terms
        logger.info("   🔄 Figure 2/4: Conversion Terms")
        output_file = output_dir / f"comparison_conversion_{comp_config['filename_suffix']}.png"
        if create_comparison_plot(
            all_data,
            CONFIG["conversion_terms"],
            f"Conversion Terms - {comp_config['label']}",
            "Rate (W·m⁻²)",
            str(output_file),
            sources_to_plot=comp_config['sources']
        ):
            comp_success += 1
            total_success += 1
        
        # Figure 3: Generation Terms
        logger.info("   🔄 Figure 3/4: Generation Terms")
        output_file = output_dir / f"comparison_generation_{comp_config['filename_suffix']}.png"
        if create_comparison_plot(
            all_data,
            CONFIG["generation_terms"],
            f"Generation Terms - {comp_config['label']}",
            "Rate (W·m⁻²)",
            str(output_file),
            sources_to_plot=comp_config['sources']
        ):
            comp_success += 1
            total_success += 1
        
        # Figure 4: Boundary Terms
        logger.info("   🔄 Figure 4/4: Boundary Terms")
        output_file = output_dir / f"comparison_boundary_{comp_config['filename_suffix']}.png"
        if create_comparison_plot(
            all_data,
            CONFIG["boundary_terms"],
            f"Boundary Terms - {comp_config['label']}",
            "Transport (W·m⁻²)",
            str(output_file),
            sources_to_plot=comp_config['sources']
        ):
            comp_success += 1
            total_success += 1
        
        logger.info("")
        logger.info(f"   ✨ {comp_config['label']}: {comp_success}/4 plots generated")
        logger.info("")
    
    # Final summary
    logger.info("=" * 70)
    logger.info(f"🎉 ALL COMPLETED: {total_success}/{total_expected} comparison plots generated")
    logger.info(f"📂 Figures saved in: {base_output_dir}")
    logger.info("=" * 70)
    logger.info("")

if __name__ == "__main__":
    main()
