import xarray as xr
import numpy as np

# Abrir o dataset
infile = "Raoni_COWAST.nc"
outfile = f"{infile}_processed.nc"
ds = xr.open_dataset(infile)

# Selecionar apenas as variáveis de interesse
variables_to_keep = ["T", "QVAPOR", "U", "V", "W", "GPH", "plevels"]
ds_filtered = ds[variables_to_keep]

# Renomear variáveis inválidas, se necessário
rename_dict = {var: f"var_{var}" for var in ds_filtered.data_vars if isinstance(var, (int, float))}
if rename_dict:
    ds_filtered = ds_filtered.rename(rename_dict)

# Converter todas as variáveis para float32 para evitar problemas com NumPy 2.0
for var in ds_filtered.data_vars:
    ds_filtered[var] = ds_filtered[var].astype(np.float32)

# Adicionar unidades às variáveis
units_dict = {
    "T": "K",  # Kelvin
    "QVAPOR": "kg/kg",  # Razão de mistura do vapor d'água
    "U": "m/s",  # Componente zonal do vento
    "V": "m/s",  # Componente meridional do vento
    "W": "hPa/s",  # Velocidade vertical
    "GPH": "m",  # Altura geopotencial
    "plevels": "hPa"  # Níveis de pressão
}

for var, unit in units_dict.items():
    if var in ds_filtered:
        ds_filtered[var].attrs["units"] = unit

# Salvar o dataset processado com compactação
ds_filtered.to_netcdf(outfile, mode='w')

print(f"Dataset exportado com sucesso para '{outfile}'.")
