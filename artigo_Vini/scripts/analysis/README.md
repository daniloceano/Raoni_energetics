# Lorenz Energy Cycle (LEC) Analysis Scripts

This directory contains scripts for analyzing the Lorenz Energy Cycle of Cyclone Raoni (2021) using different model configurations and initial conditions.

## 🚀 Quick Start

**To run all analysis scripts at once:**
```bash
cd scripts/analysis
python run_all.py
```

This will execute all analysis scripts in sequence and generate all figures automatically.

## Overview

The analysis compares the energetics of **Cyclone Raoni** simulated by:
- **ERA5 reanalysis** (reference data)
- **GFS forecast model**
- **COAWST coupled model** (WRF+ROMS) with different configurations

## Experiments

### Reference Data
- **ERA5**: ERA5 reanalysis data for Cyclone Raoni

### Model Experiments

#### GFS Initial Conditions
- **GFS**: GFS forecast model standalone
- **GFS_CPL**: COAWST coupled with GFS initial conditions
  - Path: `GFS_COAWST_Acoplado_processed_fixed`
- **GFS_DCP**: COAWST decoupled (WRF only) with GFS initial conditions
  - Path: `GFS_COAWST_SEM_Acoplamento_processed_fixed`

#### ERA5 Initial Conditions
- **ERA5_CPL**: COAWST coupled with ERA5 initial conditions
  - Path: `ERA5_COAWST_Acoplado_processed_fixed`
- **ERA5_DCP**: COAWST decoupled (WRF only) with ERA5 initial conditions
  - Path: `ERA5_COAWST_Sem_Aco_processed_fixed`

## Directory Structure

```
artigo_Vini/
├── LEC_inputs/              # Input files for LEC calculations
│   ├── box_limits_raoni.txt
│   └── namelist_*
├── LEC_Results/             # LEC computation results
│   ├── Raoni_ERA5_fixed/
│   ├── GFS_Raoni_processed_fixed/
│   ├── GFS_COAWST_Acoplado_processed_fixed/           (GFS_CPL)
│   ├── GFS_COAWST_SEM_Acoplamento_processed_fixed/    (GFS_DCP)
│   ├── ERA5_COAWST_Acoplado_processed_fixed/          (ERA5_CPL)
│   └── ERA5_COAWST_Sem_Aco_processed_fixed/           (ERA5_DCP)
├── Figures/                 # Generated plots and figures
│   ├── Comparisons/
│   │   ├── multiplot/       # Multi-panel time series
│   │   └── taylor_diagrams/ # Taylor diagrams
│   └── [source_name]/
│       ├── hovmollers/      # Hovmöller diagrams
│       ├── LEC/             # LEC box diagrams
│       └── timeseries/      # Time series plots
└── scripts/
    └── analysis/            # This directory
        ├── config.py        # ⚙️  Central configuration file
        ├── run_all.py       # 🚀 Run all scripts
        ├── plot_hovmoller_individual.py
        ├── plot_LEC_individual.py
        ├── plot_timeseries_comparison_multiplot.py
        ├── plot_taylor_diagrams.py
        └── README.md        # This file
```

## ⚙️ Configuration

### `config.py` - Central Configuration File

All scripts now use a centralized configuration file (`config.py`) that contains:

- **Experiment names and paths**: Update once, apply everywhere
- **Colors and plot styles**: Consistent across all figures
- **Display names**: Easy renaming of experiments
- **Energy cycle terms**: Centralized term definitions
- **Figure configurations**: Standard layouts and sizes

**Key benefits:**
- ✅ Single source of truth for all configurations
- ✅ Easy to update experiment names
- ✅ Consistent styling across all figures
- ✅ No need to edit multiple files

**To modify configurations:**
1. Edit `config.py`
2. All scripts automatically use the updated values
3. No need to modify individual analysis scripts

**Example configuration sections:**
```python
# Experiment definitions
DATA_SOURCES = {
    "ERA5": {
        "path": "Raoni_ERA5_fixed",
        "label": "ERA5",
        "color": "#2c3e50",
        ...
    },
    ...
}

# Figure types with their terms
FIGURE_TYPES = {
    "energy": {
        "terms": ["Az", "Ae", "Kz", "Ke"],
        ...
    },
    ...
}
```

### Validating Configuration

To check if your configuration is valid:
```bash
cd scripts/analysis
python config.py
```

This will validate that all paths exist and show configured experiments.

---

## Analysis Scripts

### 🔄 `run_all.py` - Run All Scripts

Executes all analysis scripts in the correct sequence with error handling and progress reporting.

**Features:**
- Validates configuration before running
- Runs all scripts in order
- Shows progress and status for each script
- Handles errors gracefully
- Provides final summary

**Usage:**
```bash
cd scripts/analysis
python run_all.py
```

**Output:**
- All Hovmöller diagrams
- All LEC diagrams
- All comparison time series
- All Taylor diagrams

---

### 1. Individual Source Analysis

#### `plot_hovmoller_individual.py`
Creates **Hovmöller diagrams** (time-pressure plots) for each data source.

**Features:**
- Time-pressure evolution of energy terms
- Separate diagrams for each term and data source
- Optional phase markers (Incipient, Intensification, Mature, Decay)
- Publication-ready style (Scientific Reports)

**Energy Terms:**
- `Az`, `Ae`, `Kz`, `Ke` (Energy reservoirs)
- `Ca`, `Ce`, `Ck`, `Cz` (Conversion terms)
- `Ge`, `Gz` (Generation terms)

**Output:** `Figures/[source_name]/hovmollers/`

**Usage:**
```bash
cd scripts/analysis
python plot_hovmoller_individual.py
```

---

#### `plot_LEC_individual.py`
Creates **Lorenz Energy Cycle box-and-arrow diagrams** for each data source.

**Features:**
- Classic LEC box diagram showing:
  - Energy boxes: ∂Az/∂t, ∂Ae/∂t, ∂Kz/∂t, ∂Ke/∂t
  - Conversion arrows: Ca, Ce, Ck, Cz
  - Residual terms: RGz, RGe, RKz, RKe
  - Boundary terms: BAz, BAe, BKz, BKe
- Daily mean diagrams
- Optional period mean diagrams
- Arrow thickness proportional to term magnitude

**Output:** `Figures/[source_name]/LEC/`

**Usage:**
```bash
cd scripts/analysis
python plot_LEC_individual.py
```

---

### 2. Comparative Analysis

#### `plot_timeseries_comparison_multiplot.py`
Creates **multi-panel time series comparison** figures.

**Features:**
- Four publication-quality figures:
  1. Energy terms (Az, Ae, Kz, Ke) - 4 subplots
  2. Conversion terms (Ca, Ce, Ck, Cz) - 4 subplots
  3. Generation terms (Ge, Gz) - 2 subplots
  4. Boundary terms (BAz, BAe, BKz, BKe) - 4 subplots
- All data sources overlaid in each subplot
- Automatic time alignment to common periods
- Resampling to 6h frequency (matching WRF output)
- Scientific Reports style formatting

**Output:** `Figures/Comparisons/multiplot/`

**Usage:**
```bash
cd scripts/analysis
python plot_timeseries_comparison_multiplot.py
```

---

#### `plot_taylor_diagrams.py`
Creates **Taylor diagrams** comparing model outputs against ERA5 reference.

**Features:**
- Five figures showing model performance:
  1. Energy terms (Az, Ae, Kz, Ke)
  2. Conversion terms (Ca, Ce, Ck, Cz)
  3. Generation terms (Ge, Gz)
  4. Boundary terms (BAz, BAe, BKz, BKe)
  5. All terms combined (overall performance)
- Displays:
  - Correlation coefficient (angular position)
  - Standard deviation (radial distance)
  - Centered RMS difference (distance from reference)
- Compares: GFS, GFS_CPL, GFS_DCP, ERA5_CPL, ERA5_DCP vs ERA5

**Output:** `Figures/Comparisons/taylor_diagrams/`

**Usage:**
```bash
cd scripts/analysis
python plot_taylor_diagrams.py
```

---

## Configuration (Legacy Information)

**Note:** Configuration is now centralized in `config.py`. The information below is kept for reference.

### Color Scheme

**Data Sources:**
- ERA5: Dark blue-gray `#2c3e50`
- GFS: Green `#27ae60`
- GFS_CPL: Red `#e74c3c` (GFS + coupled)
- GFS_DCP: Blue `#3498db` (GFS + decoupled)
- ERA5_CPL: Purple `#9b59b6` (ERA5 + coupled)
- ERA5_DCP: Orange `#f39c12` (ERA5 + decoupled)

---

## Lorenz Energy Cycle Theory

### Energy Reservoirs
- **Az**: Zonal Available Potential Energy
- **Ae**: Eddy Available Potential Energy
- **Kz**: Zonal Kinetic Energy
- **Ke**: Eddy Kinetic Energy

### Energy Conversions
- **Ca** (Az → Ae): Zonal to Eddy APE conversion
- **Ce** (Ae → Ke): Eddy APE to Eddy KE conversion
- **Ck** (Ke → Kz): Eddy to Zonal KE conversion
- **Cz** (Kz → Az): Zonal KE to Zonal APE conversion

### Generation Terms
- **Gz**: Zonal generation (diabatic heating)
- **Ge**: Eddy generation (diabatic heating)

### Boundary Terms
- **BAz, BAe, BKz, BKe**: Boundary fluxes for each reservoir

---

## Requirements

```bash
# Python packages
numpy
pandas
matplotlib
scipy

# Install with:
pip install numpy pandas matplotlib scipy
# or
conda install numpy pandas matplotlib scipy
```

---

## Data Flow

1. **Raw model output** → Preprocessing scripts
2. **Preprocessed data** → LEC calculation (separate workflow)
3. **LEC results** (`LEC_Results/`) → **Analysis scripts** (this directory)
4. **Figures** (`Figures/`) → Publication/presentation

---

## Notes

- All scripts automatically create output directories if they don't exist
- Figures are saved in both PNG (300 dpi) and PDF formats
- Time series are aligned to common time periods across all sources
- ERA5 and GFS are resampled to 6h frequency to match WRF output
- Missing data or failed loads are gracefully handled with informative logging

---

## Publication Style

All figures follow **Scientific Reports** journal guidelines:
- Sans-serif fonts (Arial/Helvetica)
- Maximum width: 180mm
- High resolution: 300 dpi
- Clean, professional layout
- Consistent color schemes

---

## Questions or Issues?

Contact: Danilo (danilocoutodesouza@...)

---

## Citation

When using these scripts, please cite:
- Cyclone Raoni analysis (paper in preparation)
- Lorenz Energy Cycle methodology references

---

**Last Updated:** February 2026
