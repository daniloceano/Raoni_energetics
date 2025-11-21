#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lorenz Energy Cycle (LEC) Diagram Plotter for Individual Sources
=================================================================

This script generates Lorenz Energy Cycle box-and-arrow diagrams for each 
data source individually. Creates diagrams showing:
- Energy boxes (∂Az/∂t, ∂Ae/∂t, ∂Kz/∂t, ∂Ke/∂t)
- Conversion arrows (Ca, Ce, Ck, Cz)
- Residual terms (RGz, RGe, RKz, RKe)
- Boundary terms (BAz, BAe, BKz, BKe)

Generates:
1. Example diagram with term labels
2. Daily mean diagrams
3. Optional period mean diagrams (if periods file provided)

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
import matplotlib.patches as patches

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
    
    # Terms to plot
    "energy_terms": ["∂Az/∂t", "∂Ae/∂t", "∂Kz/∂t", "∂Ke/∂t"],
    "conversion_terms": ["Cz", "Ca", "Ck", "Ce"],
    "residual_terms": ["RGz", "RGe", "RKz", "RKe"],
    "boundary_terms": ["BAz", "BAe", "BKz", "BKe"],
    
    # Box positions and size
    "positions": {
        "∂Az/∂t": (-0.5, 0.5),
        "∂Ae/∂t": (-0.5, -0.5),
        "∂Kz/∂t": (0.5, 0.5),
        "∂Ke/∂t": (0.5, -0.5),
    },
    "box_size": 0.4,
    
    # Plot styling
    "figure_size": (8, 8),
    "dpi": 300,
    "box_color": "skyblue",
    "box_edge_color": "black",
    "arrow_color": "#5C5850",
    "positive_color": "#386641",  # Dark green
    "negative_color": "#ae2012",  # Dark red
    
    # Edge width range (scaled by normalized term magnitude)
    "min_edge_width": 0,
    "max_edge_width": 5,
    
    # Font sizes
    "title_fontsize": 16,
    "term_fontsize": 16,
    "value_fontsize": 16,
    
    # Normalization
    "norm_clip_lower": 1.5,
    "norm_clip_upper": 15,
    "norm_scale": 50,  # For daily plots
    "norm_scale_periods": 10,  # For period plots
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger() -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("LECPlotter")
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
        
        # Rename columns by removing "(finite diff.)"
        df = df.rename(columns=lambda x: x.replace(" (finite diff.)", ""))
        
        logger.info(f"      ✅ Loaded {len(df)} time steps, {len(df.columns)} variables")
        return df
    except Exception as e:
        logger.error(f"      ❌ Error loading {filepath}: {e}")
        return None

def load_periods(filepath: str) -> Optional[pd.DataFrame]:
    """Load cyclone phase periods from CSV file."""
    if filepath is None or not Path(filepath).exists():
        logger.info(f"   ℹ️  No periods file available - skipping period means")
        return None
        
    try:
        logger.info(f"   📅 Loading phase periods from: {Path(filepath).name}")
        df = pd.read_csv(filepath, parse_dates=["start", "end"], index_col=0)
        logger.info(f"      ✅ Loaded {len(df)} phase periods")
        return df
    except Exception as e:
        logger.error(f"      ❌ Error loading periods: {e}")
        return None

# ============================================================================
# PLOTTING HELPER FUNCTIONS
# ============================================================================

def plot_boxes(ax, data, normalized_data, plot_example=False):
    """Plot energy boxes with values."""
    positions = CONFIG["positions"]
    size = CONFIG["box_size"]
    
    for term, pos in positions.items():
        term_value = data[term]
        normalized_value = normalized_data[term]
        
        # Scale edge width based on normalized value
        edge_width = (
            CONFIG["min_edge_width"] + 
            (CONFIG["max_edge_width"] - CONFIG["min_edge_width"]) * 
            normalized_value / 10
        )
        
        # Determine value text color
        value_text_color = CONFIG["positive_color"] if term_value >= 0 else CONFIG["negative_color"]
        
        # Draw box
        square = patches.Rectangle(
            (pos[0] - size / 2, pos[1] - size / 2),
            size, size,
            fill=True,
            color=CONFIG["box_color"],
            ec=CONFIG["box_edge_color"],
            linewidth=edge_width,
        )
        ax.add_patch(square)
        
        # Add text (term name for example, value for actual plots)
        if plot_example:
            ax.text(
                pos[0], pos[1], f"{term}",
                ha="center", va="center",
                fontsize=CONFIG["term_fontsize"],
                color="k", fontweight="bold",
            )
        else:
            ax.text(
                pos[0], pos[1], f"{term_value:.2f}",
                ha="center", va="center",
                fontsize=CONFIG["value_fontsize"],
                color=value_text_color, fontweight="bold",
            )

def plot_arrow(ax, start, end, term_value):
    """Draw an arrow from start to end point."""
    # Determine arrow size based on term value
    if np.abs(term_value) < 1:
        size = 3 + np.abs(term_value)
    elif np.abs(term_value) < 5:
        size = 3 + np.abs(term_value)
    elif np.abs(term_value) < 10:
        size = 3 + np.abs(term_value)
    else:
        size = 15 + np.abs(term_value) * 0.1
    
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            facecolor=CONFIG["arrow_color"],
            edgecolor=CONFIG["arrow_color"],
            width=size,
            headwidth=size * 3,
            headlength=size * 3,
        ),
    )

def plot_term_text_and_value(ax, start, end, term, term_value, 
                             offset=(0, 0), plot_example=False):
    """Plot text label and value for a term."""
    text_color = CONFIG["positive_color"] if term_value >= 0 else CONFIG["negative_color"]
    
    mid_point = (
        (start[0] + end[0]) / 2 + offset[0],
        (start[1] + end[1]) / 2 + offset[1],
    )
    
    # Adjust offsets for specific terms
    if term in ["Ca", "BAz", "BAe"]:
        offset_x = -0.05
    elif term in ["Ck", "BKz", "BKe"]:
        offset_x = 0.05
    else:
        offset_x = 0
    
    if term == "Ce":
        offset_y = -0.05
    elif term == "Cz":
        offset_y = 0.05
    else:
        offset_y = 0
    
    x_pos = mid_point[0] + offset_x
    y_pos = mid_point[1] + offset_y
    
    # Plot term text or value
    if plot_example:
        ax.text(x_pos, y_pos, term,
                ha="center", va="center",
                fontsize=CONFIG["term_fontsize"],
                color="k", fontweight="bold")
    else:
        ax.text(x_pos, y_pos, f"{term_value:.2f}",
                ha="center", va="center",
                color=text_color,
                fontsize=CONFIG["value_fontsize"],
                fontweight="bold")

def plot_term_arrows_and_text(ax, term, data, plot_example=False):
    """Plot arrows and text for a specific term."""
    term_value = data[term]
    positions = CONFIG["positions"]
    size = CONFIG["box_size"]
    
    # Define start and end points for each term type
    if term == "Cz":
        start = (positions["∂Az/∂t"][0] + size / 2, positions["∂Az/∂t"][1])
        end = (positions["∂Kz/∂t"][0] - size / 2, positions["∂Kz/∂t"][1])
        plot_term_text_and_value(ax, start, end, term, term_value, 
                                offset=(0, 0.1), plot_example=plot_example)
    
    elif term == "Ca":
        start = (positions["∂Az/∂t"][0], positions["∂Az/∂t"][1] - size / 2)
        end = (positions["∂Ae/∂t"][0], positions["∂Ae/∂t"][1] + size / 2)
        plot_term_text_and_value(ax, start, end, term, term_value,
                                offset=(-0.1, 0), plot_example=plot_example)
    
    elif term == "Ck":
        start = (positions["∂Kz/∂t"][0], positions["∂Ke/∂t"][1] + size / 2)
        end = (positions["∂Ke/∂t"][0], positions["∂Kz/∂t"][1] - size / 2)
        plot_term_text_and_value(ax, start, end, term, term_value,
                                offset=(0.1, 0), plot_example=plot_example)
    
    elif term == "Ce":
        start = (positions["∂Ae/∂t"][0] + size / 2, positions["∂Ke/∂t"][1])
        end = (positions["∂Ke/∂t"][0] - size / 2, positions["∂Ae/∂t"][1])
        plot_term_text_and_value(ax, start, end, term, term_value,
                                offset=(0, -0.1), plot_example=plot_example)
    
    elif term == "RGz":
        start = (positions["∂Az/∂t"][0], 1)
        end = (positions["∂Az/∂t"][0], positions["∂Az/∂t"][1] + size / 2)
        plot_term_text_and_value(ax, start, end, term, term_value,
                                offset=(0, 0.2), plot_example=plot_example)
    
    elif term == "RGe":
        start = (positions["∂Ae/∂t"][0], -1)
        end = (positions["∂Ae/∂t"][0], positions["∂Ae/∂t"][1] - size / 2)
        plot_term_text_and_value(ax, start, end, term, term_value,
                                offset=(0, -0.2), plot_example=plot_example)
    
    elif term == "RKz":
        start = (positions["∂Kz/∂t"][0], 1)
        end = (positions["∂Kz/∂t"][0], positions["∂Kz/∂t"][1] + size / 2)
        plot_term_text_and_value(ax, start, end, term, term_value,
                                offset=(0, 0.2), plot_example=plot_example)
    
    elif term == "RKe":
        start = (positions["∂Ke/∂t"][0], -1)
        end = (positions["∂Ke/∂t"][0], positions["∂Ke/∂t"][1] - size / 2)
        plot_term_text_and_value(ax, start, end, term, term_value,
                                offset=(0, -0.2), plot_example=plot_example)
    
    elif term in ["BAz", "BAe"]:
        refered_term = "∂Az/∂t" if term == "BAz" else "∂Ae/∂t"
        start = (-1, positions[refered_term][1])
        end = (positions[refered_term][0] - size / 2, positions[refered_term][1])
        plot_term_text_and_value(ax, start, end, term, term_value,
                                offset=(-0.23, 0), plot_example=plot_example)
    
    elif term in ["BKz", "BKe"]:
        refered_term = "∂Kz/∂t" if term == "BKz" else "∂Ke/∂t"
        start = (1, positions[refered_term][1])
        end = (positions[refered_term][0] + size / 2, positions[refered_term][1])
        plot_term_text_and_value(ax, start, end, term, term_value,
                                offset=(0.23, 0), plot_example=plot_example)
    
    # Swap start and end for negative values
    if term_value < 0:
        start, end = end, start
    
    # Plot arrow
    plot_arrow(ax, start, end, term_value)

def create_lec_plot(data, normalized_data, plot_example=False):
    """Create a single LEC diagram."""
    fig, ax = plt.subplots(figsize=CONFIG["figure_size"])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    
    # Plot energy boxes
    plot_boxes(ax, data, normalized_data, plot_example)
    
    # Add title (date) if not example
    if not plot_example:
        if isinstance(data.name, pd.Timestamp):
            date_str = data.name.strftime("%Y-%m-%d")
        else:
            date_str = str(data.name)
        ax.text(0, 0, date_str,
                fontsize=CONFIG["title_fontsize"],
                ha="center", va="center",
                fontweight="bold", color="black")
    
    # Plot all terms
    all_terms = (CONFIG["conversion_terms"] + CONFIG["residual_terms"] + 
                 CONFIG["boundary_terms"])
    for term in all_terms:
        plot_term_arrows_and_text(ax, term, data, plot_example)
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN PLOTTING FUNCTIONS
# ============================================================================

def plot_daily_means(df_results: pd.DataFrame, output_dir: Path, 
                     source_name: str) -> int:
    """Generate LEC diagrams for daily means."""
    logger.info("   🔄 Creating daily mean LEC diagrams...")
    
    # Group by day
    daily_means = df_results.groupby(pd.Grouper(freq="1D")).mean(numeric_only=True)
    
    # Normalize data
    df_not_energy = np.abs(daily_means.drop(columns=["Az", "Ae", "Kz", "Ke"], errors='ignore'))
    normalized_data = (
        (df_not_energy - df_not_energy.min().min()) /
        (df_not_energy.max().max() - df_not_energy.min().min())
    ) * CONFIG["norm_scale"]
    normalized_data = normalized_data.clip(
        lower=CONFIG["norm_clip_lower"],
        upper=CONFIG["norm_clip_upper"]
    )
    
    # Create example diagram
    logger.info("      📝 Creating example diagram...")
    example_data = (daily_means * 0) + 1
    fig = create_lec_plot(example_data.iloc[0], example_data.iloc[0], plot_example=True)
    example_path = output_dir / "LEC_example.png"
    fig.savefig(example_path, dpi=CONFIG["dpi"])
    plt.close()
    logger.info(f"         ✅ Saved: {example_path.name}")
    
    # Create daily diagrams
    success_count = 1  # Count example
    for date, data in daily_means.iterrows():
        normalized = normalized_data.loc[date]
        fig = create_lec_plot(data, normalized, plot_example=False)
        
        if isinstance(date, pd.Timestamp):
            figure_name = date.strftime("%Y-%m-%d")
        else:
            figure_name = str(date)
        
        figure_path = output_dir / f"LEC_{figure_name}.png"
        fig.savefig(figure_path, dpi=CONFIG["dpi"])
        plt.close()
        success_count += 1
    
    logger.info(f"      ✅ Created {success_count} LEC diagrams (1 example + {success_count-1} daily)")
    return success_count

def plot_period_means(df_results: pd.DataFrame, periods_df: pd.DataFrame,
                     output_dir: Path, source_name: str) -> int:
    """Generate LEC diagrams for period means."""
    logger.info("   🔄 Creating period mean LEC diagrams...")
    
    period_means_df = pd.DataFrame()
    
    # Calculate means for each period
    for period_name, row in periods_df.iterrows():
        start, end = row["start"], row["end"]
        df_period = df_results.loc[start:end]
        
        if not df_period.empty:
            period_mean = df_period.mean().rename(period_name)
            period_means_df = pd.concat(
                [period_means_df, pd.DataFrame(period_mean).transpose()]
            )
        else:
            logger.info(f"      ⚠️  No data for period: {period_name}")
    
    if period_means_df.empty:
        logger.info("      ⚠️  No period means to plot")
        return 0
    
    # Normalize data
    df_not_energy = np.abs(period_means_df.drop(columns=["Az", "Ae", "Kz", "Ke"], errors='ignore'))
    normalized_data = (df_not_energy - df_not_energy.min().mean()) / \
                     (df_not_energy.max().max() - df_not_energy.min().min())
    normalized_data = normalized_data.clip(
        lower=CONFIG["norm_clip_lower"],
        upper=CONFIG["norm_clip_upper"]
    )
    
    # Create period diagrams
    success_count = 0
    for period_name, data in period_means_df.iterrows():
        normalized = normalized_data.loc[period_name]
        fig = create_lec_plot(data, normalized, plot_example=False)
        
        figure_path = output_dir / f"LEC_{period_name}.png"
        fig.savefig(figure_path, dpi=CONFIG["dpi"])
        plt.close()
        success_count += 1
    
    logger.info(f"      ✅ Created {success_count} period LEC diagrams")
    return success_count

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_data_source(source_name: str, base_dir: Path):
    """Process a single data source and generate all its LEC diagrams."""
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
    output_dir = base_dir / CONFIG["base_output_dir"] / source_name / "LEC"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")
    
    # Load data
    df_results = load_results(str(results_file))
    if df_results is None:
        return 0
    
    # Load periods if available
    periods_file = CONFIG["periods_files"].get(source_name)
    if periods_file:
        periods_file = results_dir / periods_file
    periods_df = load_periods(str(periods_file) if periods_file else None)
    
    logger.info("")
    
    # Track success
    total_count = 0
    
    # Generate daily diagrams
    daily_count = plot_daily_means(df_results, output_dir, source_name)
    total_count += daily_count
    
    # Generate period diagrams if periods available
    if periods_df is not None:
        period_count = plot_period_means(df_results, periods_df, output_dir, source_name)
        total_count += period_count
    
    logger.info("")
    logger.info(f"✨ Completed {source_name}: {total_count} LEC diagrams generated")
    logger.info("")
    
    return total_count

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("🔷 LORENZ ENERGY CYCLE PLOTTER - INDIVIDUAL SOURCES")
    logger.info("=" * 70)
    logger.info("")
    
    base_dir = Path(__file__).parent
    
    total_success = 0
    
    for source in CONFIG["data_sources"]:
        success = process_data_source(source, base_dir)
        total_success += success
    
    # Final summary
    logger.info("=" * 70)
    logger.info(f"🎉 ALL COMPLETED: {total_success} total LEC diagrams generated")
    logger.info(f"📂 Figures saved in: {base_dir / CONFIG['base_output_dir']}")
    logger.info("=" * 70)
    logger.info("")

if __name__ == "__main__":
    main()
