import xarray as xr
from glob import glob 

files = glob("era5_data_2021*.nc")

# Abrir os datasets
ds = xr.open_mfdataset(files)

# Remover coordenadas inuteis (number, expver)
ds = ds.drop_vars(["number", "expver"])

# Salvar o dataset concatenado
ds.to_netcdf("Raoni_ERA5.nc")