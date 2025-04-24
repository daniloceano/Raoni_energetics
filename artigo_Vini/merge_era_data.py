import xarray as xr

# Abrir os datasets
ds1 = xr.open_dataset("era5_data_20210625.nc")
ds2 = xr.open_dataset("era5_data_20210626.nc")
ds3 = xr.open_dataset("era5_data_20210627.nc")
ds4 = xr.open_dataset("era5_data_20210628.nc")
ds5 = xr.open_dataset("era5_data_20210629.nc")
ds6 = xr.open_dataset("era5_data_20210630.nc")
ds7 = xr.open_dataset("era5_data_20210701.nc")
ds8 = xr.open_dataset("era5_data_20210702.nc")

# Concatenar os datasets
ds = xr.concat([ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8], dim="Time")

# Salvar o dataset concatenado
ds.to_netcdf("era5_data_20210625_20210702.nc")