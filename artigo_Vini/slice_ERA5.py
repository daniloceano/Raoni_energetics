import xarray as xr

print("Iniciando o processamento do arquivo Raoni_ERA5.nc")
ds = xr.open_dataset("Raoni_ERA5.nc")

# Slice para manter apenas os níveis de pressão entre 900 hPa e 100 hPa
ds_filtered = ds.sel(pressure_level=slice(900, 100))

# Exporta o dataset filtrado para um novo arquivo NetCDF
ds_filtered.to_netcdf("Raoni_ERA5_filtered.nc")
print("Dataset filtrado e salvo como 'Raoni_ERA5_filtered.nc'.")