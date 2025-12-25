"""
Script para pre-processar dados GFS para calculos de energetica.

Este script le multiplos arquivos GRIB2 do GFS, concatena no tempo, e produz um 
arquivo NetCDF processado pronto para utilizar nos calculos de energetica.

OTIMIZADO: Usa open_mfdataset + Dask para leitura lazy. O recorte espacial e 
de niveis e feito ANTES do .compute(), minimizando uso de memoria.

Autor: Danilo
Data: 2025
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACOES
# ============================================================================

GFS_DATA_DIR = '../../data/GFS'
FILE_PATTERN = 'gfs.0p25.*.f000.grib2'
OUTFILE = '../../data/GFS/GFS_Raoni_processed.nc'
NAMELIST_FILE = '../../namelist_GFS'

# Arquivos a excluir (corrompidos ou incompletos)
EXCLUDE_FILES = ['gfs.0p25.2021070218.f000.grib2']

VARS_TO_KEEP = ['t', 'u', 'v', 'w', 'gh']

VAR_RENAME = {
    't': 'TT',
    'u': 'UU',
    'v': 'VV',
    'w': 'W',
    'gh': 'GHT',
}

UNITS_DICT = {
    'TT': 'K',
    'UU': 'm/s',
    'VV': 'm/s',
    'W': 'Pa/s',
    'GHT': 'm',
}

MULTIPLY_OMEGA_BY_MINUS_ONE = True

PLEVEL_MIN = 100
PLEVEL_MAX = 1000

# Limites espaciais (de box_limits_raoni.txt)
LAT_MIN = -48.0
LAT_MAX = -17.5
LON_MIN = -60.0
LON_MAX = -28.0

USE_COMPRESSION = True
GENERATE_PLOTS = True
PLOT_PLEVEL_INDEX = 5
LIST_VARS_ONLY = False

# ============================================================================
# FUNCOES
# ============================================================================

def convert_longitude_180(ds):
    """Converte longitude de [0, 360] para [-180, 180]."""
    lon_name = 'longitude'
    if lon_name not in ds.coords:
        return ds
    
    lon = ds[lon_name].values
    if lon.max() > 180:
        new_lon = np.where(lon > 180, lon - 360, lon)
        ds = ds.assign_coords({lon_name: new_lon})
        ds = ds.sortby(lon_name)
    return ds


def open_gfs_variable(files, var_name, verbose=True):
    """Abre uma variavel especifica usando open_mfdataset (lazy)."""
    if verbose:
        print(f"   Abrindo variavel '{var_name}'...", end=" ", flush=True)
    
    try:
        ds = xr.open_mfdataset(
            files,
            engine='cfgrib',
            combine='nested',
            concat_dim='time',
            parallel=True,
            backend_kwargs={
                'filter_by_keys': {
                    'typeOfLevel': 'isobaricInhPa',
                    'shortName': var_name
                },
                'indexpath': ''
            }
        )
        
        if verbose:
            print("OK")
        return ds
        
    except Exception as e:
        if verbose:
            print(f"FALHOU ({str(e)[:50]}...)")
        return None


def main():
    print("\nIniciando pre-processamento de dados GFS (com Dask)")
    print("=" * 70)
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / GFS_DATA_DIR
    
    file_pattern = str(data_dir / FILE_PATTERN)
    files = sorted(glob(file_pattern))
    
    # Excluir arquivos corrompidos
    files = [f for f in files if Path(f).name not in EXCLUDE_FILES]
    
    if not files:
        print(f"Nenhum arquivo encontrado: {file_pattern}")
        sys.exit(1)
    
    print(f"\nEncontrados {len(files)} arquivos GFS")
    
    print(f"\nConfiguracao:")
    print(f"   Variaveis: {', '.join(VARS_TO_KEEP)}")
    print(f"   Niveis: {PLEVEL_MIN} - {PLEVEL_MAX} hPa")
    print(f"   Recorte: lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN}, {LON_MAX}]")
    
    # PASSO 1: Abrir cada variavel (LAZY)
    print(f"\nAbrindo arquivos com Dask (leitura lazy)...")
    
    datasets = {}
    for var in VARS_TO_KEEP:
        ds = open_gfs_variable(files, var)
        if ds is not None:
            datasets[var] = ds
    
    if not datasets:
        print("Nenhuma variavel foi lida!")
        sys.exit(1)
    
    print(f"\n{len(datasets)} variaveis abertas: {list(datasets.keys())}")
    
    # PASSO 2: Encontrar niveis comuns
    print(f"\nEncontrando niveis de pressao comuns...")
    
    common_levels = None
    for var, ds in datasets.items():
        if 'isobaricInhPa' in ds.dims:
            levels = set(ds['isobaricInhPa'].values)
            if common_levels is None:
                common_levels = levels
            else:
                common_levels = common_levels.intersection(levels)
    
    common_levels = sorted(common_levels, reverse=True)
    print(f"   Niveis comuns: {common_levels}")
    
    selected_levels = [p for p in common_levels if PLEVEL_MIN <= p <= PLEVEL_MAX]
    print(f"   Niveis selecionados: {selected_levels}")
    
    # PASSO 3: Selecionar niveis e recortar (ainda LAZY)
    print(f"\nSelecionando niveis e recortando (lazy)...")
    
    processed_datasets = []
    for var, ds in datasets.items():
        ds = ds.sel(isobaricInhPa=selected_levels)
        ds = convert_longitude_180(ds)
        ds = ds.sel(
            latitude=slice(LAT_MAX, LAT_MIN),
            longitude=slice(LON_MIN, LON_MAX)
        )
        
        if 'time' not in ds.dims and 'valid_time' in ds.coords:
            ds = ds.rename({'valid_time': 'time'})
        if 'time' not in ds.dims and 'time' in ds.coords:
            ds = ds.expand_dims('time')
        
        processed_datasets.append(ds)
        print(f"   {var}: {dict(ds.dims)}")
    
    print(f"\nFazendo merge das variaveis (lazy)...")
    ds_merged = xr.merge(processed_datasets)
    
    del datasets, processed_datasets
    
    print(f"   Dataset merged: {dict(ds_merged.dims)}")
    
    # PASSO 4: Carregar na memoria
    print(f"\nCarregando dados na memoria (compute)...")
    print(f"   Isso pode demorar alguns minutos...")
    
    ds_final = ds_merged.compute()
    
    print("Dados carregados!")
    
    ds_final = ds_final.sortby('time')
    
    # PASSO 5: Processamento final
    print(f"\nProcessando dados...")
    
    if 'isobaricInhPa' in ds_final.dims:
        ds_final = ds_final.rename({'isobaricInhPa': 'plevels'})
        ds_final['plevels'].attrs['long_name'] = 'Pressure levels'
        ds_final['plevels'].attrs['units'] = 'hPa'
    
    rename_dict = {}
    for old_name, new_name in VAR_RENAME.items():
        if old_name in ds_final.data_vars:
            rename_dict[old_name] = new_name
            print(f"   {old_name} -> {new_name}")
    ds_final = ds_final.rename(rename_dict)
    
    omega_var = VAR_RENAME.get('w', 'W')
    if omega_var in ds_final and MULTIPLY_OMEGA_BY_MINUS_ONE:
        print(f"   Multiplicando {omega_var} por -1...")
        ds_final[omega_var] = ds_final[omega_var] * -1
    
    for var in ds_final.data_vars:
        ds_final[var] = ds_final[var].astype(np.float32)
    
    for var, unit in UNITS_DICT.items():
        if var in ds_final:
            ds_final[var].attrs['units'] = unit
    
    coords_to_drop = ['step', 'valid_time', 'surface', 'number']
    for coord in coords_to_drop:
        if coord in ds_final.coords:
            ds_final = ds_final.drop_vars(coord)
    
    ds_final.attrs['source'] = 'GFS 0.25 degree'
    ds_final.attrs['history'] = f'Processed on {pd.Timestamp.now()}'
    
    print(f"\nDataset processado:")
    print(ds_final)
    
    # PASSO 6: Salvar
    outpath = script_dir / OUTFILE
    outpath.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSalvando arquivo: {outpath}")
    
    if USE_COMPRESSION:
        encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_final.data_vars}
        ds_final.to_netcdf(outpath, encoding=encoding)
    else:
        ds_final.to_netcdf(outpath)
    print(f"Arquivo salvo: {outpath}")
    
    # PASSO 7: Namelist
    print(f"\nGerando namelist...")
    namelist_path = script_dir / NAMELIST_FILE
    
    standard_names = {
        'TT': 'Air Temperature',
        'UU': 'Eastward Wind Component',
        'VV': 'Northward Wind Component',
        'W': 'Omega Velocity',
        'GHT': 'Geopotential Height',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'time': 'Time',
        'plevels': 'Vertical Level'
    }
    
    with open(namelist_path, 'w') as f:
        f.write('standard_name;Variable;Units\n')
        for var in ['TT', 'UU', 'VV', 'W', 'GHT']:
            if var in ds_final:
                f.write(f'{standard_names.get(var, var)};{var};{UNITS_DICT.get(var, "")}\n')
        for coord in ['longitude', 'latitude', 'time', 'plevels']:
            if coord in ds_final.coords:
                f.write(f'{standard_names.get(coord, coord)};{coord};\n')
    
    print(f"Namelist salvo: {namelist_path}")
    
    print(f"\nProcessamento concluido!")
    print("=" * 70)


if __name__ == '__main__':
    main()
