#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taylor Diagram Generator for Cyclone Lorenz Energy Cycle Analysis
===================================================================

Creates publication-quality Taylor diagrams for Scientific Reports comparing
model outputs (GFS, CPL_EXP, CPC_EXP) against ERA5 reanalysis as reference.

Taylor diagrams display:
- Correlation coefficient (angular position)
- Centered RMS difference (distance from reference)
- Standard deviation (radial distance from origin)

Generates 5 figures:
  i)   Energy terms (Az, Ae, Kz, Ke)
  ii)  Conversion terms (Ca, Ce, Ck, Cz)
  iii) Generation terms (Ge, Gz)
  iv)  Boundary terms (BAz, BAe, BKz, BKe)
  v)   All terms combined

Author: Automated Script
Date: 2025
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# Import configuration
from config import (
    DATA_SOURCES,
    BASE_RESULTS_DIR,
    BASE_OUTPUT_DIR,
    REFERENCE_SOURCE,
    MODEL_SOURCES,
    RESAMPLE_ERA5,
    RESAMPLE_GFS,
    RESAMPLE_FREQ,
    FIGURE_TYPES,
    apply_scientific_reports_style,
    get_output_path
)

# Apply Scientific Reports style
apply_scientific_reports_style()

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger() -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("TaylorDiagramGenerator")
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
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def align_and_resample_data(data_dict: Dict[str, pd.DataFrame], 
                            ref_name: str) -> Dict[str, pd.DataFrame]:
    """
    Align all datasets to common time index and resample if needed.
    
    Args:
        data_dict: Dictionary with source names as keys and DataFrames as values
        ref_name: Name of reference source
        
    Returns:
        Dictionary with aligned DataFrames
    """
    aligned = {}
    
    # Get reference data
    ref_df = data_dict.get(ref_name)
    if ref_df is None:
        logger.error(f"Reference data '{ref_name}' not found")
        return {}
    
    # Find common time range
    all_starts = [df.index.min() for df in data_dict.values()]
    all_ends = [df.index.max() for df in data_dict.values()]
    common_start = max(all_starts)
    common_end = min(all_ends)
    
    logger.info(f"   Common time range: {common_start} to {common_end}")
    
    for name, df in data_dict.items():
        # Slice to common time range
        df_sliced = df[(df.index >= common_start) & (df.index <= common_end)]
        
        # Resample ERA5 and GFS to 6h if needed
        if RESAMPLE_ERA5 and name == "ERA5":
            df_sliced = df_sliced.resample(RESAMPLE_FREQ).mean()
        elif RESAMPLE_GFS and name == "GFS":
            df_sliced = df_sliced.resample(RESAMPLE_FREQ).mean()
        
        aligned[name] = df_sliced
    
    # Find common time points across all sources
    common_times = aligned[ref_name].index
    for name, df in aligned.items():
        common_times = common_times.intersection(df.index)
    
    logger.info(f"   Common time points: {len(common_times)}")
    
    # Filter to common times
    for name in aligned:
        aligned[name] = aligned[name].loc[common_times]
    
    return aligned


def compute_taylor_statistics(reference: np.ndarray, model: np.ndarray) -> Dict[str, float]:
    """
    Compute Taylor diagram statistics.
    
    Args:
        reference: Reference data array (ERA5)
        model: Model data array
        
    Returns:
        Dictionary with correlation, std_model, std_ref, centered_rms
    """
    # Remove NaN values
    mask = ~(np.isnan(reference) | np.isnan(model))
    ref_clean = reference[mask]
    mod_clean = model[mask]
    
    if len(ref_clean) < 3:
        return None
    
    # Standard deviations
    std_ref = np.std(ref_clean, ddof=1)
    std_mod = np.std(mod_clean, ddof=1)
    
    # Correlation coefficient
    if std_ref > 0 and std_mod > 0:
        correlation = np.corrcoef(ref_clean, mod_clean)[0, 1]
    else:
        correlation = 0
    
    # Centered RMS difference
    ref_anom = ref_clean - np.mean(ref_clean)
    mod_anom = mod_clean - np.mean(mod_clean)
    centered_rms = np.sqrt(np.mean((ref_anom - mod_anom) ** 2))
    
    return {
        "correlation": correlation,
        "std_model": std_mod,
        "std_ref": std_ref,
        "centered_rms": centered_rms,
        "normalized_std": std_mod / std_ref if std_ref > 0 else np.nan
    }


def compute_overall_taylor_statistics(ref_df: pd.DataFrame, mod_df: pd.DataFrame,
                                       terms: List[str]) -> Dict[str, float]:
    """
    Compute overall Taylor diagram statistics by concatenating all terms.
    
    This creates a single performance metric per model by treating all
    term time series as one concatenated series.
    
    Args:
        ref_df: Reference DataFrame (ERA5)
        mod_df: Model DataFrame
        terms: List of terms to include
        
    Returns:
        Dictionary with overall correlation, std_model, std_ref, centered_rms, normalized_std
    """
    # Collect all values from all terms
    ref_all = []
    mod_all = []
    
    for term in terms:
        if term in ref_df.columns and term in mod_df.columns:
            ref_vals = ref_df[term].values
            mod_vals = mod_df[term].values
            
            # Remove NaN pairs
            mask = ~(np.isnan(ref_vals) | np.isnan(mod_vals))
            ref_all.extend(ref_vals[mask])
            mod_all.extend(mod_vals[mask])
    
    ref_all = np.array(ref_all)
    mod_all = np.array(mod_all)
    
    if len(ref_all) < 3:
        return None
    
    # Standard deviations
    std_ref = np.std(ref_all, ddof=1)
    std_mod = np.std(mod_all, ddof=1)
    
    # Correlation coefficient
    if std_ref > 0 and std_mod > 0:
        correlation = np.corrcoef(ref_all, mod_all)[0, 1]
    else:
        correlation = 0
    
    # Centered RMS difference
    ref_anom = ref_all - np.mean(ref_all)
    mod_anom = mod_all - np.mean(mod_all)
    centered_rms = np.sqrt(np.mean((ref_anom - mod_anom) ** 2))
    
    return {
        "correlation": correlation,
        "std_model": std_mod,
        "std_ref": std_ref,
        "centered_rms": centered_rms,
        "normalized_std": std_mod / std_ref if std_ref > 0 else np.nan,
        "n_points": len(ref_all)
    }


def create_taylor_diagram_overall(overall_stats: Dict[str, Dict], 
                                   output_path: str) -> bool:
    """
    Create a Taylor diagram showing overall model performance.
    
    Each model is represented by a single point summarizing performance
    across all LEC terms.
    
    Args:
        overall_stats: Dict {model_name: stats}
        output_path: Output file path
        
    Returns:
        Success status
    """
    try:
        fig = plt.figure(figsize=FIGURE_TYPES["energy"]["figsize"])
        ax = fig.add_subplot(111, projection='polar')
        
        # Set theta to start at top and go clockwise
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        # Limit to 0-90 degrees (correlation 0-1)
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        
        # Plot reference point at correlation=1, normalized std=1
        ax.plot(0, 1.0, 'ko', markersize=14, label='ERA5 (Reference)', zorder=10)
        
        # Track max normalized std for axis limit
        max_norm_std = 1.5
        
        # Create legend handles
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   markersize=12, label='ERA5 (Reference)')
        ]
        
        # Plot each model
        for model_name, stats in overall_stats.items():
            if stats is None:
                continue
                
            model_config = MODEL_SOURCES.get(model_name, {})
            color = model_config.get("color", "#333333")
            marker = model_config.get("marker", "o")
            label = model_config.get("label", model_name)
            
            corr = stats['correlation']
            norm_std = stats['normalized_std']
            
            if np.isnan(corr) or np.isnan(norm_std):
                continue
            
            # Angle from correlation (radians)
            theta = np.arccos(np.clip(corr, -1, 1))
            
            # Plot point with larger marker
            ax.plot(theta, norm_std, marker, color=color, 
                   markersize=16, markeredgecolor='white', 
                   markeredgewidth=1.5, zorder=5)
            
            # Add annotation with correlation value
            ax.annotate(f'r={corr:.2f}', (theta, norm_std),
                       textcoords="offset points", xytext=(8, 8),
                       fontsize=9, color=color, fontweight='bold')
            
            max_norm_std = max(max_norm_std, norm_std * 1.1)
            
            # Add to legend
            legend_handles.append(
                Line2D([0], [0], marker=marker, color='w', 
                      markerfacecolor=color, markersize=12, 
                      markeredgecolor='white', markeredgewidth=1,
                      label=label)
            )
        
        # Configure radial axis
        ax.set_rlim(0, min(max_norm_std, 2.0))
        ax.set_rlabel_position(45)
        
        # Add correlation labels on angular axis
        corr_values = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
        ax.set_xticks([np.arccos(c) for c in corr_values])
        ax.set_xticklabels([f'{c:.2g}' for c in corr_values])
        
        # Add RMS circles
        rms_levels = [0.25, 0.5, 0.75, 1.0]
        for rms in rms_levels:
            phi = np.linspace(0, 2*np.pi, 100)
            x_circ = 1 + rms * np.cos(phi)
            y_circ = rms * np.sin(phi)
            
            r_circ = np.sqrt(x_circ**2 + y_circ**2)
            theta_circ = np.arctan2(y_circ, x_circ)
            
            valid = (theta_circ >= 0) & (theta_circ <= np.pi/2) & (r_circ <= 2.0) & (r_circ >= 0)
            if np.any(valid):
                ax.plot(theta_circ[valid], r_circ[valid], 'gray', 
                       linestyle=':', alpha=0.5, linewidth=1)
        
        # Add standard deviation reference circle
        theta_range = np.linspace(0, np.pi/2, 100)
        ax.plot(theta_range, np.ones_like(theta_range), 'k--', alpha=0.5, linewidth=1.5)
        
        # Labels
        ax.set_xlabel('Correlation Coefficient', fontsize=11, labelpad=15)
        ax.text(0.5, -0.12, 'Normalized Standard Deviation', transform=ax.transAxes, 
               ha='center', fontsize=11)
        
        # Title
        ax.set_title('Taylor Diagram: Overall Model Performance\n(All LEC Terms)', 
                    fontsize=12, fontweight='bold', pad=20)
        
        # Legend
        ax.legend(handles=legend_handles, loc='upper right', 
                 bbox_to_anchor=(1.35, 1.0), fontsize=10, framealpha=0.9)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.close()
        
        logger.info(f"      ✅ Saved: {Path(output_path).name}")
        return True
        
    except Exception as e:
        logger.error(f"      ❌ Error creating overall Taylor diagram: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# SIMPLIFIED TAYLOR DIAGRAM FUNCTION
# ============================================================================

def create_taylor_diagram_simple(stats_dict: Dict[str, Dict[str, Dict]], 
                                  fig_config: Dict,
                                  output_path: str,
                                  fig_name: str) -> bool:
    """
    Create a simplified Taylor diagram without complex axis transforms.
    
    Args:
        stats_dict: Nested dict {model: {term: stats}}
        fig_config: Figure configuration
        output_path: Output file path
        fig_name: Figure name for title
        
    Returns:
        Success status
    """
    try:
        fig = plt.figure(figsize=FIGURE_TYPES[fig_name].get("figsize", FIGURE_TYPES["energy"]["figsize"]))
        ax = fig.add_subplot(111, projection='polar')
        
        # Set theta to start at top and go clockwise
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        # Limit to 0-90 degrees (correlation 0-1)
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        
        # Get reference std (average across terms for normalization)
        ref_stds = []
        for model_stats in stats_dict.values():
            for term, stats in model_stats.items():
                if stats and 'std_ref' in stats:
                    ref_stds.append(stats['std_ref'])
        ref_std_mean = np.mean(ref_stds) if ref_stds else 1.0
        
        # Plot reference point at correlation=1, normalized std=1
        ax.plot(0, 1.0, 'ko', markersize=12, label='ERA5 (Reference)', zorder=10)
        
        # Track max normalized std for axis limit
        max_norm_std = 1.5
        
        # Term markers (different for each term)
        term_markers = {
            'Az': 'o', 'Ae': 's', 'Kz': '^', 'Ke': 'v',
            'Ca': 'D', 'Ce': 'p', 'Ck': 'h', 'Cz': '*',
            'Ge': 'X', 'Gz': 'P',
            'BAz': '8', 'BAe': 'H', 'BKz': '<', 'BKe': '>'
        }
        
        # Create legend handles
        model_handles = []
        term_handles = []
        
        # Plot each model
        for model_name, model_stats in stats_dict.items():
            model_config = MODEL_SOURCES.get(model_name, {})
            color = model_config.get("color", "#333333")
            
            for term, stats in model_stats.items():
                if stats is None:
                    continue
                    
                corr = stats['correlation']
                norm_std = stats['normalized_std']
                
                if np.isnan(corr) or np.isnan(norm_std):
                    continue
                
                # Angle from correlation (radians)
                theta = np.arccos(np.clip(corr, -1, 1))
                
                # Get marker for this term
                marker = term_markers.get(term, 'o')
                
                # Plot point
                ax.plot(theta, norm_std, marker, color=color, 
                       markersize=10, markeredgecolor='white', 
                       markeredgewidth=0.5, zorder=5)
                
                max_norm_std = max(max_norm_std, norm_std * 1.1)
        
        # Create legend entries
        for model_name, model_config in MODEL_SOURCES.items():
            if model_name in stats_dict:
                model_handles.append(
                    Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=model_config["color"],
                          markersize=10, label=model_config["label"])
                )
        
        # Add term markers to legend (for combined plot - show all terms)
        terms = fig_config["terms"]
        term_labels = fig_config["term_labels"]
        for term in terms:  # Show ALL terms, not just first 8
            if term in term_markers:
                term_handles.append(
                    Line2D([0], [0], marker=term_markers[term], color='gray',
                          markersize=8, linestyle='None', 
                          label=term_labels.get(term, term))
                )
        
        # Configure radial axis
        ax.set_rlim(0, min(max_norm_std, 2.5))
        ax.set_rlabel_position(45)
        
        # Add correlation labels on angular axis
        corr_values = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
        ax.set_xticks([np.arccos(c) for c in corr_values])
        ax.set_xticklabels([f'{c:.2g}' for c in corr_values])
        
        # Add RMS circles
        rms_levels = [0.5, 1.0, 1.5]
        theta_range = np.linspace(0, np.pi/2, 100)
        for rms in rms_levels:
            # RMS circle centered at reference point (theta=0, r=1)
            phi = np.linspace(0, 2*np.pi, 100)
            x_circ = 1 + rms * np.cos(phi)
            y_circ = rms * np.sin(phi)
            
            # Convert to polar
            r_circ = np.sqrt(x_circ**2 + y_circ**2)
            theta_circ = np.arctan2(y_circ, x_circ)
            
            # Only plot valid portion
            valid = (theta_circ >= 0) & (theta_circ <= np.pi/2) & (r_circ <= 2.5) & (r_circ >= 0)
            if np.any(valid):
                ax.plot(theta_circ[valid], r_circ[valid], 'gray', 
                       linestyle=':', alpha=0.4, linewidth=0.8)
                # Add label
                label_idx = np.argmin(np.abs(theta_circ - np.pi/4))
                if valid[label_idx]:
                    ax.annotate(f'RMS={rms:.1f}', (theta_circ[label_idx], r_circ[label_idx]),
                               fontsize=7, color='gray', alpha=0.6)
        
        # Add standard deviation reference circle
        ax.plot(theta_range, np.ones_like(theta_range), 'k--', alpha=0.4, linewidth=1)
        
        # Labels
        ax.set_xlabel('Correlation Coefficient', fontsize=10, labelpad=15)
        ax.text(0.5, -0.12, 'Normalized Standard Deviation', transform=ax.transAxes, 
               ha='center', fontsize=10)
        
        # Title
        ax.set_title(f'Taylor Diagram: {fig_config["title"]}', fontsize=12, 
                    fontweight='bold', pad=20)
        
        # Add legends
        if len(terms) <= 4:
            # Simple legend for small number of terms
            all_handles = model_handles + term_handles
            ax.legend(handles=all_handles, loc='upper right', 
                     bbox_to_anchor=(1.35, 1.0), fontsize=8, framealpha=0.9)
        else:
            # Two-column legend for combined plot
            legend1 = ax.legend(handles=model_handles, title='Models', 
                               loc='upper right', bbox_to_anchor=(1.3, 1.0),
                               fontsize=8, framealpha=0.9)
            ax.add_artist(legend1)
            ax.legend(handles=term_handles, title='Terms', 
                     loc='lower right', bbox_to_anchor=(1.3, 0.0),
                     fontsize=7, framealpha=0.9, ncol=2)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.close()
        
        logger.info(f"      ✅ Saved: {Path(output_path).name}")
        return True
        
    except Exception as e:
        logger.error(f"      ❌ Error creating Taylor diagram: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("🎯 Taylor Diagram Generator for Cyclone Lorenz Energy Cycle")
    logger.info("=" * 70)
    logger.info("")
    
    # Get base directory (artigo_Vini)
    base_dir = Path(__file__).parent.parent.parent.resolve()
    
    # Create output directory
    output_dir = base_dir / "Figures" / "Comparisons" / "taylor_diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info("")
    
    # Load reference data (ERA5)
    logger.info("📂 Loading reference data (ERA5)...")
    ref_config = REFERENCE_SOURCE
    ref_results_file = base_dir / "LEC_Results" / ref_config["path"]
    
    # Find the results CSV file
    ref_csv_files = list(ref_results_file.glob("*_results.csv"))
    if not ref_csv_files:
        logger.error(f"❌ No results CSV found in {ref_results_file}")
        return
    
    ref_df = load_results(str(ref_csv_files[0]))
    if ref_df is None:
        logger.error("❌ Failed to load reference data")
        return
    logger.info(f"   ✅ Loaded ERA5: {len(ref_df)} time steps")
    
    # Load model data
    logger.info("")
    logger.info("📂 Loading model data...")
    all_data = {"ERA5": ref_df}
    
    for model_name, model_config in MODEL_SOURCES.items():
        model_path = base_dir / "LEC_Results" / model_config["path"]
        csv_files = list(model_path.glob("*_results.csv"))
        
        if csv_files:
            model_df = load_results(str(csv_files[0]))
            if model_df is not None:
                all_data[model_name] = model_df
                logger.info(f"   ✅ Loaded {model_name}: {len(model_df)} time steps")
            else:
                logger.warning(f"   ⚠️ Failed to load {model_name}")
        else:
            logger.warning(f"   ⚠️ No results CSV found for {model_name}")
    
    # Align data to common time range
    logger.info("")
    logger.info("🔄 Aligning data to common time range...")
    aligned_data = align_and_resample_data(all_data, "ERA5")
    
    if not aligned_data:
        logger.error("❌ Failed to align data")
        return
    
    # Generate Taylor diagrams for each figure type
    logger.info("")
    logger.info("🎨 Generating Taylor diagrams...")
    
    success_count = 0
    
    # Only generate the "all_combined" figure (with all individual terms)
    # and the new "overall_performance" figure (one point per model)
    figures_to_generate = ["all_combined"]
    
    for fig_name in figures_to_generate:
        fig_config = FIGURE_TYPES[fig_name]
        logger.info("")
        logger.info(f"   📊 Creating Taylor diagram: {fig_config['title']}")
        
        # Compute statistics for each model and term
        stats_dict = {}
        
        for model_name in MODEL_SOURCES.keys():
            if model_name not in aligned_data:
                continue
                
            model_df = aligned_data[model_name]
            ref_df = aligned_data["ERA5"]
            
            stats_dict[model_name] = {}
            
            for term in fig_config["terms"]:
                if term in ref_df.columns and term in model_df.columns:
                    ref_values = ref_df[term].values
                    mod_values = model_df[term].values
                    
                    stats = compute_taylor_statistics(ref_values, mod_values)
                    stats_dict[model_name][term] = stats
                    
                    if stats:
                        logger.info(f"      {model_name}/{term}: corr={stats['correlation']:.3f}, "
                                   f"norm_std={stats['normalized_std']:.3f}")
                else:
                    logger.info(f"      ⚠️ {model_name}/{term}: Term not found")
                    stats_dict[model_name][term] = None
        
        # Create Taylor diagram
        output_file = output_dir / f"taylor_{fig_name}.png"
        
        if create_taylor_diagram_simple(stats_dict, fig_config, str(output_file), fig_name):
            success_count += 1
    
    # Generate the overall performance diagram (one point per model)
    logger.info("")
    logger.info("   📊 Creating Taylor diagram: Overall Model Performance")
    
    all_terms = FIGURE_TYPES["all_combined"]["terms"]
    overall_stats = {}
    
    for model_name in MODEL_SOURCES.keys():
        if model_name not in aligned_data:
            continue
            
        model_df = aligned_data[model_name]
        ref_df = aligned_data["ERA5"]
        
        stats = compute_overall_taylor_statistics(ref_df, model_df, all_terms)
        overall_stats[model_name] = stats
        
        if stats:
            logger.info(f"      {model_name} (overall): corr={stats['correlation']:.3f}, "
                       f"norm_std={stats['normalized_std']:.3f}, n={stats['n_points']}")
    
    output_file = output_dir / "taylor_overall_performance.png"
    if create_taylor_diagram_overall(overall_stats, str(output_file)):
        success_count += 1
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"✨ Completed: {success_count}/2 Taylor diagrams generated")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info("=" * 70)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
