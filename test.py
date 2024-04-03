import xarray as xr

ds = xr.open_dataset("data_vOMARSAT/Raoni_isobaric.nc")
print(ds)