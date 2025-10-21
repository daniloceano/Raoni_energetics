import xarray as xr
import numpy as np
import pandas as pd
import glob

# Pastas dos arquivos
#folders = ["cowast_files", "wrf_files"]
folders = ['wrf-era_files']
for folder in folders:
    file_list = sorted(glob.glob(f"{folder}/*.nc"))
    print(f"Processando {len(file_list)} arquivos em {folder}")
    ds = xr.open_mfdataset(file_list, combine='nested', concat_dim='Time')
    # Definir o nome do arquivo de saída
    fname = folder.split('_')[0].upper()  # "COWAST" ou "WRF"
    outfile = f"Raoni_{fname}_merged_processed.nc"

    # Selecionar apenas as variáveis de interesse
    variables_to_keep = ["T", "QVAPOR", "U", "V", "W", "GPH", "plevels"]
    ds_filtered = ds[variables_to_keep]
    # Usar só de 900 hPa a 100 hPa
    ds_filtered = ds_filtered.sel(plevels=slice(100, 900))

    # Garantir que 'Time' seja uma coordenada indexada
    if 'Time' not in ds_filtered.coords and 'Time' in ds:
        ds_filtered = ds_filtered.assign_coords(Time=ds['Time'])
    # Converter 'Times' para datetime e usar como coordenada 'Time'
    if 'Times' in ds:
        times_dt = pd.to_datetime([t.decode('utf-8') if isinstance(t, bytes) else t for t in ds['Times'].values], format='%Y-%m-%d_%H:%M:%S')
        ds_filtered = ds_filtered.assign_coords(Time=("Time", times_dt))
        # (Opcional) manter 'Times' como coordenada auxiliar, se desejar
        if 'Times' not in ds_filtered.coords:
            ds_filtered = ds_filtered.assign_coords(Times=ds['Times'])
        # Remover as coordenadas 'XTIME' e 'Times' se existirem
        if 'XTIME' in ds_filtered.coords:
            del ds_filtered.coords['XTIME']
        if 'Times' in ds_filtered.coords:
            del ds_filtered.coords['Times']

    # Renomear variáveis inválidas, se necessário
    rename_dict = {var: f"var_{var}" for var in ds_filtered.data_vars if isinstance(var, (int, float))}
    if rename_dict:
        ds_filtered = ds_filtered.rename(rename_dict)

    # Converter todas as variáveis para float32 para evitar problemas com NumPy 2.0
    for var in ds_filtered.data_vars:
        ds_filtered[var] = ds_filtered[var].astype(np.float32)

    # Inverter sinal de W
    if 'W' in ds_filtered:
        ds_filtered['W'] = -ds_filtered['W']

    # Adicionar unidades às variáveis
    units_dict = {
        "T": "K",  # Kelvin
        "QVAPOR": "kg/kg",  # Razão de mistura do vapor d'água
        "U": "m/s",  # Componente zonal do vento
        "V": "m/s",  # Componente meridional do vento
        "W": "Pa/s",  # Velocidade vertical
        "GPH": "m^2/s^2",  # Geopotencial
        "plevels": "hPa"  # Níveis de pressão
    }

    for var, unit in units_dict.items():
        if var in ds_filtered:
            ds_filtered[var].attrs["units"] = unit

    # Salvar o dataset processado com compactação
    ds_filtered.to_netcdf(outfile, mode='w')

    print(f"Dataset exportado com sucesso para '{outfile}'.")
