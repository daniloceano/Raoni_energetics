#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration File for Raoni Energetics Analysis Scripts
=========================================================

This file contains all universal configurations used across analysis scripts:
- Experiment names and paths
- Display names and colors
- Plotting styles and parameters
- Energy cycle terms

To modify any configuration, edit this file and all analysis scripts will
automatically use the updated values.

Author: Danilo
Date: 2025
"""

from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# Get the directory containing this config file
_CONFIG_DIR = Path(__file__).parent.resolve()

# Base directories (absolute paths calculated from config location)
# scripts/analysis -> scripts -> artigo_Vini
_ARTIGO_VINI_DIR = _CONFIG_DIR.parent.parent

BASE_RESULTS_DIR = str(_ARTIGO_VINI_DIR / "LEC_Results")
BASE_OUTPUT_DIR = str(_ARTIGO_VINI_DIR / "Figures")

# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

# Data sources - actual folder names in LEC_Results
DATA_SOURCES = {
    "ERA5": {
        "path": "Raoni_ERA5_fixed",
        "label": "ERA5",
        "color": "#2627ff",      # Dark blue-gray
        "linestyle": "-",
        "marker": "s",
        "zorder": 6,
    },
    "GFS": {
        "path": "GFS_Raoni_processed_fixed",
        "label": "GFS",
        "color": "#db4848",      # Green
        "linestyle": "-",
        "marker": "s",
        "zorder": 5,
    },
    "GFS_CPL": {
        "path": "GFS_COAWST_Acoplado_processed_fixed",
        "label": "GFS_CPL",
        "color": "#ed8679",      # Red (GFS coupled)
        "linestyle": "-",
        "marker": "^",
        "zorder": 4,
    },
    "GFS_DCP": {
        "path": "GFS_COAWST_SEM_Acoplamento_processed_fixed",
        "label": "GFS_DCP",
        "color": "#ed8679",      # Blue (GFS decoupled)
        "linestyle": "--",
        "marker": "o",
        "zorder": 3,
    },
    "ERA5_CPL": {
        "path": "ERA5_COAWST_Acoplado_processed_fixed",
        "label": "ERA5_CPL",
        "color": "#74b8e5",      # Purple (ERA5 coupled)
        "linestyle": "-",
        "marker": "^",
        "zorder": 2,
    },
    "ERA5_DCP": {
        "path": "ERA5_COAWST_Sem_Aco_processed_fixed",
        "label": "ERA5_DCP",
        "color": "#74b8e5",      # Orange (ERA5 decoupled)
        "linestyle": "--",
        "marker": "o",
        "zorder": 1,
    }
}

# Optional periods files for each source (set to None if not available)
PERIODS_FILES = {
    "Raoni_ERA5_fixed": None,
    "GFS_Raoni_processed_fixed": None,
    "GFS_COAWST_Acoplado_processed_fixed": None,
    "GFS_COAWST_SEM_Acoplamento_processed_fixed": None,
    "ERA5_COAWST_Acoplado_processed_fixed": None,
    "ERA5_COAWST_Sem_Aco_processed_fixed": None,
}

# Reference dataset for Taylor diagrams
REFERENCE_SOURCE = {
    "key": "ERA5",
    "path": "Raoni_ERA5_fixed",
    "label": "ERA5",
}

# Model sources for Taylor diagrams (all except reference)
MODEL_SOURCES = {k: v for k, v in DATA_SOURCES.items() if k != "ERA5"}

# ============================================================================
# RESAMPLING CONFIGURATION
# ============================================================================

RESAMPLE_ERA5 = True
RESAMPLE_GFS = True
RESAMPLE_FREQ = "6h"  # Match WRF/COAWST frequency

# ============================================================================
# ENERGY CYCLE TERMS
# ============================================================================

# Energy reservoirs
ENERGY_TERMS = ["Az", "Ae", "Kz", "Ke"]
ENERGY_TERM_LABELS = {
    "Az": r"$A_Z$",
    "Ae": r"$A_E$",
    "Kz": r"$K_Z$",
    "Ke": r"$K_E$"
}
ENERGY_TERM_TITLES = {
    "Az": "Zonal Available Potential Energy",
    "Ae": "Eddy Available Potential Energy",
    "Kz": "Zonal Kinetic Energy",
    "Ke": "Eddy Kinetic Energy"
}

# Conversion terms
CONVERSION_TERMS = ["Ca", "Ce", "Ck", "Cz"]
CONVERSION_TERM_LABELS = {
    "Ca": r"$C_A$",
    "Ce": r"$C_E$",
    "Ck": r"$C_K$",
    "Cz": r"$C_Z$"
}
CONVERSION_TERM_TITLES = {
    "Ca": r"$A_Z \rightarrow A_E$",
    "Ce": r"$A_E \rightarrow K_E$",
    "Ck": r"$K_E \rightarrow K_Z$",
    "Cz": r"$K_Z \rightarrow A_Z$"
}

# Generation terms
GENERATION_TERMS = ["Ge", "Gz"]
GENERATION_TERM_LABELS = {
    "Ge": r"$G_E$",
    "Gz": r"$G_Z$"
}
GENERATION_TERM_TITLES = {
    "Ge": "Eddy Generation",
    "Gz": "Zonal Generation"
}

# Boundary flux terms
BOUNDARY_TERMS = ["BAz", "BAe", "BKz", "BKe"]
BOUNDARY_TERM_LABELS = {
    "BAz": r"$B_{A_Z}$",
    "BAe": r"$B_{A_E}$",
    "BKz": r"$B_{K_Z}$",
    "BKe": r"$B_{K_E}$"
}

# Residual terms
RESIDUAL_TERMS = ["RGz", "RGe", "RKz", "RKe"]

# Time derivative terms (for LEC diagrams)
TIME_DERIVATIVE_TERMS = ["∂Az/∂t", "∂Ae/∂t", "∂Kz/∂t", "∂Ke/∂t"]

# All terms combined
ALL_TERMS = ENERGY_TERMS + CONVERSION_TERMS + GENERATION_TERMS + BOUNDARY_TERMS

# All term labels combined
ALL_TERM_LABELS = {
    **ENERGY_TERM_LABELS,
    **CONVERSION_TERM_LABELS,
    **GENERATION_TERM_LABELS,
    **BOUNDARY_TERM_LABELS
}

# ============================================================================
# FIGURE CONFIGURATIONS
# ============================================================================

# Figure types with their respective terms
FIGURE_TYPES = {
    "energy": {
        "title": "Energy Reservoirs",
        "ylabel": r"Energy (J$\cdot$m$^{-2}$)",
        "terms": ENERGY_TERMS,
        "term_labels": ENERGY_TERM_LABELS,
        "term_titles": ENERGY_TERM_TITLES,
        "ncols": 2,
        "figsize": (8, 6),  # 180mm x 150mm (Scientific Reports max width)
    },
    "conversion": {
        "title": "Energy Conversion Terms",
        "ylabel": r"Rate (W$\cdot$m$^{-2}$)",
        "terms": CONVERSION_TERMS,
        "term_labels": CONVERSION_TERM_LABELS,
        "term_titles": CONVERSION_TERM_TITLES,
        "ncols": 2,
        "figsize": (180/25.4, 150/25.4),
    },
    "generation": {
        "title": "Generation Terms",
        "ylabel": r"Rate (W$\cdot$m$^{-2}$)",
        "terms": GENERATION_TERMS,
        "term_labels": GENERATION_TERM_LABELS,
        "term_titles": GENERATION_TERM_TITLES,
        "ncols": 2,
        "figsize": (180/25.4, 100/25.4),  # Smaller height for 2 terms
    },
    "boundary": {
        "title": "Boundary Flux Terms",
        "ylabel": r"Rate (W$\cdot$m$^{-2}$)",
        "terms": BOUNDARY_TERMS,
        "term_labels": BOUNDARY_TERM_LABELS,
        "term_titles": {},  # Add if needed
        "ncols": 2,
        "figsize": (180/25.4, 150/25.4),
    },
    "all_combined": {
        "title": "All Terms Combined",
        "terms": ALL_TERMS,
        "term_labels": ALL_TERM_LABELS,
        "figsize": (180/25.4, 180/25.4),  # Square for combined
    }
}

# ============================================================================
# SCIENTIFIC REPORTS STYLE
# ============================================================================

def apply_scientific_reports_style():
    """Apply Scientific Reports journal style to matplotlib."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
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
# LEC DIAGRAM CONFIGURATION
# ============================================================================

# Box positions for LEC diagram (Az, Ae, Kz, Ke)
LEC_BOX_POSITIONS = {
    "∂Az/∂t": (-0.5, 0.5),
    "∂Ae/∂t": (-0.5, -0.5),
    "∂Kz/∂t": (0.5, 0.5),
    "∂Ke/∂t": (0.5, -0.5),
}

# LEC diagram styling
LEC_CONFIG = {
    "box_size": 0.4,
    "figure_size": (8, 8),
    "dpi": 300,
    "box_color": "skyblue",
    "box_edge_color": "black",
    "arrow_color": "#5C5850",
    "positive_color": "#386641",  # Dark green
    "negative_color": "#ae2012",  # Dark red
    "min_edge_width": 0,
    "max_edge_width": 5,
    "title_fontsize": 16,
    "term_fontsize": 16,
    "value_fontsize": 16,
    "norm_clip_lower": 1.5,
    "norm_clip_upper": 15,
    "norm_scale": 50,  # For daily plots
    "norm_scale_periods": 10,  # For period plots
}

# ============================================================================
# HOVMÖLLER DIAGRAM CONFIGURATION
# ============================================================================

HOVMOLLER_CONFIG = {
    "dpi": 300,
    "cmap_positive": "YlOrRd",  # Yellow-Orange-Red for positive values
    "cmap_negative": "Blues_r",  # Reversed Blues for negative values
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_results_path(source_key):
    """Get the full path to results CSV for a given source."""
    source_info = DATA_SOURCES.get(source_key)
    if not source_info:
        raise ValueError(f"Unknown source key: {source_key}")
    
    path = source_info["path"]
    results_dir = Path(BASE_RESULTS_DIR) / path
    results_file = results_dir / f"{path}_results.csv"
    return results_file

def get_vertical_results_path(source_key, term):
    """Get the full path to vertical level results for a given source and term."""
    source_info = DATA_SOURCES.get(source_key)
    if not source_info:
        raise ValueError(f"Unknown source key: {source_key}")
    
    path = source_info["path"]
    vertical_dir = Path(BASE_RESULTS_DIR) / path / "results_vertical_levels"
    vertical_file = vertical_dir / f"{term}_plevels.csv"
    return vertical_file

def get_output_path(subfolder=""):
    """Get output path for figures."""
    output_path = Path(BASE_OUTPUT_DIR)
    if subfolder:
        output_path = output_path / subfolder
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate that all configured paths exist."""
    import sys
    errors = []
    
    # Check if results directories exist
    for key, source_info in DATA_SOURCES.items():
        results_dir = Path(BASE_RESULTS_DIR) / source_info["path"]
        if not results_dir.exists():
            errors.append(f"Results directory not found: {results_dir}")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease check the paths in config.py")
        return False
    
    return True

if __name__ == "__main__":
    """Run validation when executed directly."""
    print("Validating configuration...")
    if validate_config():
        print("✓ Configuration is valid!")
        print(f"\nConfigured experiments:")
        for key, info in DATA_SOURCES.items():
            print(f"  - {key}: {info['label']} ({info['path']})")
    else:
        print("✗ Configuration validation failed!")
