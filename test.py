import xarray as xr

ds = xr.open_dataset("Raoni_isobaric.nc")
print(ds)