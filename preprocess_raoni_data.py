import xarray as xr

# Abrir o dataset
ds = xr.open_dataset('dado_raoni_sem_acoplamento.nc')

# Selecionar apenas as variáveis desejadas
vars_to_keep = ['T', 'U', 'V', 'GPH', 'W']
ds_filtered = ds[vars_to_keep]

# Adicionar a variável de tempo
ds_filtered = ds_filtered.set_coords('time')
ds_filtered = ds_filtered.set_index(Time='time')
ds_filtered = ds_filtered.drop_vars(['XTIME', 'Times'])

# Opcional: salvar em um novo arquivo NetCDF
# Salvar com compressão (opcional)
encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_filtered.data_vars}
ds_filtered.to_netcdf('dado_raoni_filtrado.nc', encoding=encoding, mode='w')

print("Arquivo filtrado criado com sucesso!")
print(ds_filtered)